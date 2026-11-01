"""RB prefill/decode timing on synthetic token IDs. No task/confirmation scoring."""
import argparse
import gc
import json
import math
import os
from pathlib import Path
import statistics
import time

import torch

from bypass_model import BranchModel,BranchCache
from mig_profile_guard import main as profile_guard
from rb_train import make_model,PARAMETERS
from v1_lora import LoRAModel
from v5_latency import merge_lora
from v5_train import write_json


def forward(model,ids,mask,cache=None):
    if isinstance(model,BranchModel):
        output=model(ids,attention_mask=mask,cache=cache,use_cache=True)
        return output.logits,output.cache
    pos=(mask.cumsum(-1)-1).clamp_min(0)[:,-ids.shape[1]:]
    output=model(ids,attention_mask=mask,position_ids=pos,past_key_values=cache,use_cache=True)
    return output.logits,output.past_key_values


def state_bytes(cache):
    if not isinstance(cache,BranchCache):
        return {'M_c_bytes':0,'counts_bytes':0}
    return {'M_c_bytes':sum((s.M.numel()+s.c.numel())*4 for s in cache.memory.values() if s is not None),
            'counts_bytes':sum(s.counts.numel()*s.counts.element_size() for s in cache.memory.values() if s is not None)}


@torch.no_grad()
def measure(model,prefill_length=512,decode_tokens=128):
    model.eval()
    ids=(torch.arange(prefill_length,device='cuda')[None]%997)+50
    mask=torch.ones_like(ids)
    torch.cuda.reset_peak_memory_stats()
    for _ in range(4):
        logits,cache=forward(model,ids,mask)
        del logits,cache
    prefill=[]
    for _ in range(5):
        torch.cuda.synchronize();start=time.perf_counter()
        logits,cache=forward(model,ids,mask)
        torch.cuda.synchronize();prefill.append(time.perf_counter()-start)
        del logits,cache
    # Decode warmup uses its own fresh prefill and is discarded.
    logits,cache=forward(model,ids,mask)
    next_token=logits[:,-1].argmax(-1,keepdim=True)
    for t in range(8):
        complete=torch.ones(1,prefill_length+t+1,device='cuda',dtype=torch.long)
        logits,cache=forward(model,next_token,complete,cache)
        next_token=logits[:,-1].argmax(-1,keepdim=True)
    del logits,cache,next_token
    logits,cache=forward(model,ids,mask)
    next_token=logits[:,-1].argmax(-1,keepdim=True)
    del logits
    samples=[]
    for t in range(decode_tokens):
        complete=torch.ones(1,prefill_length+t+1,device='cuda',dtype=torch.long)
        torch.cuda.synchronize();start=time.perf_counter()
        logits,cache=forward(model,next_token,complete,cache)
        torch.cuda.synchronize();samples.append((time.perf_counter()-start)*1000)
        next_token=logits[:,-1].argmax(-1,keepdim=True)
        del logits
    ordered=sorted(samples)
    assert len(samples)==128 and all(math.isfinite(x) and x>0 for x in samples)
    return {'batch':1,'prefill_length':prefill_length,'prefill_passes':5,
        'prefill_tokens_per_second':prefill_length/statistics.mean(prefill),
        'prefill_seconds':prefill,'decode_tokens':decode_tokens,
        'decode_p50_ms':statistics.median(samples),'decode_p95_ms':ordered[int(.95*(len(ordered)-1))],
        'decode_samples_ms':samples,'peak_cuda_bytes':torch.cuda.max_memory_allocated(),
        'adapter_persistent_state':state_bytes(cache),
        'state_note':'FP32 adapter M+c;counts separately;backbone KV is excluded'}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--arm',choices=('Base',*PARAMETERS),required=True)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if profile_guard(['--output',str(out)]):
        raise SystemExit(3)
    if args.arm!='Base':
        if args.checkpoint is None or not all((args.checkpoint/name).is_file() for name in ('config.json','adapter.pt')):
            raise ValueError('non-Base timing requires an exported checkpoint')
    model=make_model(args.arm,checkpoint=args.checkpoint)
    values={args.arm:measure(model)}
    if isinstance(model,LoRAModel):
        merged=merge_lora(model)
        model._targets.clear()
        del model
        gc.collect();torch.cuda.empty_cache()
        values[args.arm+'-merged']=measure(merged)
    write_json(out/'latency.json',values)
    write_json(out/'manifest.json',{'status':'complete','contract':'E-0019@1',
        'spec_digest':os.environ['SR_SPEC_DIGEST'],'metrics':{'arm':args.arm,'hardware':torch.cuda.get_device_name(),
        'decode_tokens':128},'artifacts':[{'path':str(out/'latency.json'),'role':'latency'}],
        'validator':{'name':'rb_prefill_decode','passed':True}})


if __name__=='__main__':
    main()
