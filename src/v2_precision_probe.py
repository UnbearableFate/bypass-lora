"""MSG-0135 follow-up: true FP32 calibration control and saved BF16 length audit."""
import argparse
import json
import math
import os
from pathlib import Path
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from v2_base_diag import read_calibration
from v2_data import MODEL, REVISION
from v2_eval import evaluate, prompt_tokens


def mean(xs):
    return sum(xs) / len(xs)


def correlation(xs, ys):
    mx, my = mean(xs), mean(ys)
    numerator = sum((x-mx)*(y-my) for x,y in zip(xs,ys))
    denominator = math.sqrt(sum((x-mx)**2 for x in xs)*sum((y-my)**2 for y in ys))
    return numerator / denominator if denominator else None


def comparison(one, eight):
    assert [(r['episode_id'],r['line']) for r in one] == [(r['episode_id'],r['line']) for r in eight]
    return {line: {'examples':len(a),
        'top1_agreement':mean([x['prediction']==y['prediction'] for x,y in zip(a,b)]),
        'nll_mean_abs_difference':mean([abs(x['nll']-y['nll']) for x,y in zip(a,b)]),
        'nll_max_abs_difference':max(abs(x['nll']-y['nll']) for x,y in zip(a,b))}
        for line in ('M','F')
        for a,b in [([r for r in one if r['line']==line],[r for r in eight if r['line']==line])]}


def read_predictions(path):
    return [json.loads(s) for s in path.read_text().splitlines()]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--calibration',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    start=time.monotonic()
    args.output.mkdir(parents=True,exist_ok=True)
    rows=read_calibration(args.calibration)
    tok=AutoTokenizer.from_pretrained(MODEL,revision=REVISION,local_files_only=True)
    bf={batch:read_predictions(args.source/f'original-batch{batch}'/'predictions.jsonl') for batch in (1,8)}
    relationships={}
    length_rows=[]
    for line in ('M','F'):
        one=[r for r in bf[1] if r['line']==line]
        eight=[r for r in bf[8] if r['line']==line]
        assert [r['episode_id'] for r in one] == [r['episode_id'] for r in eight] == [r['episode_id'] for r in rows]
        lengths=[len(prompt_tokens(tok,r,line,split='calibration')) for r in rows]
        pads=[max(lengths[i//8*8:i//8*8+8])-lengths[i] for i in range(len(rows))]
        diffs=[abs(a['nll']-b['nll']) for a,b in zip(one,eight)]
        relationships[line]={'pearson_pad_vs_abs_delta':correlation(pads,diffs),
                            'pearson_length_vs_abs_delta':correlation(lengths,diffs)}
        for axis,values in (('padding_tokens',pads),('prompt_tokens',lengths)):
            relationships[line][axis]={str(v):{'examples':values.count(v),
                'mean_abs_delta_nll':mean([d for x,d in zip(values,diffs) if x==v])} for v in sorted(set(values))}
        length_rows += [{'episode_id':r['episode_id'],'line':line,'prompt_tokens':length,'padding_tokens_batch8':pad,
                         'abs_delta_nll':diff} for r,length,pad,diff in zip(rows,lengths,pads,diffs)]
    (args.output/'bf16-lengths.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in length_rows))
    torch.set_num_threads(4)
    torch.manual_seed(173)
    model=AutoModelForCausalLM.from_pretrained(MODEL,revision=REVISION,local_files_only=True,
                                              torch_dtype=torch.float32,attn_implementation='eager').cuda().eval()
    model.requires_grad_(False)
    assert all(p.dtype == torch.float32 for p in model.parameters())
    summaries, predictions={},{}
    for batch in (1,8):
        out=args.output/f'fp32-batch{batch}'
        out.mkdir(exist_ok=True)
        summaries[str(batch)]=evaluate(model,tok,rows[:256],out,batch_size=batch,split='calibration')
        predictions[batch]=read_predictions(out/'predictions.jsonl')
    ids={r['episode_id'] for r in rows[:256]}
    report={'authorization':'coordinator MSG-0135; follow-up after E-0007@2 gate failure',
            'fp32_rows':256,'fp32_checks':comparison(predictions[1],predictions[8]),
            'bf16_same_prefix_checks':comparison([r for r in bf[1] if r['episode_id'] in ids],
                                                 [r for r in bf[8] if r['episode_id'] in ids]),
            'fp32_summaries':summaries,'bf16_full_length_relationship':relationships,
            'model':MODEL,'revision':REVISION,'dtype':'float32','torch':torch.__version__,
            'cuda':torch.version.cuda,'float32_matmul_precision':torch.get_float32_matmul_precision(),
            'cuda_matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,
            'elapsed_s':time.monotonic()-start}
    (args.output/'precision.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    metrics={f'{line}.{k}':v for line,c in report['fp32_checks'].items() for k,v in c.items()}
    artifacts=['precision.json','bf16-lengths.jsonl','fp32-batch1/predictions.jsonl','fp32-batch8/predictions.jsonl']
    (args.output/'manifest.json').write_text(json.dumps({'status':'complete','spec_digest':os.environ['SR_SPEC_DIGEST'],
        'metrics':metrics,'artifacts':[{'path':str((args.output/f).resolve()),'role':f} for f in artifacts],
        'validator':{'name':'fp32_parameter_dtype_and_prediction_identity','passed':True}},indent=2))
    print(json.dumps(report))


if __name__ == '__main__':
    main()
