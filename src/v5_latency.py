"""Descriptive V5 latency on 256 fixed dev candidates; no held-out data access."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

import torch

from calibrate import MODEL, REVISION
from transformers import AutoTokenizer
from v1_data import encode_candidates
from v1_lora import LoRALinear
from v1_run import batch_tensors, prepare
from v5_train import ARMS, make_model, write_json


def validate_checkpoint_map(checkpoints):
    """Reject missing adapters before any model is constructed."""
    if not isinstance(checkpoints, dict) or set(checkpoints) != set(ARMS):
        raise ValueError('checkpoint map must name exactly the three V5 arms')
    result = {}
    for arm, value in checkpoints.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f'{arm}: checkpoint path must be a nonempty string')
        path = Path(value).resolve()
        for name in ('config.json', 'adapter.pt'):
            if not (path/name).is_file():
                raise ValueError(f'{arm}: missing checkpoint file {path/name}')
        result[arm] = str(path)
    return result


@torch.no_grad()
def merge_lora(model):
    """Fold FP32 BA into frozen weights with one final base-dtype rounding."""
    for (index, path), adapter in model._targets.items():
        parent = model.model.model.layers[index]
        parts = path.split('.')
        for part in parts[:-1]:
            parent = getattr(parent, part)
        assert getattr(parent, parts[-1]) is adapter
        weight = adapter.base.weight
        weight.copy_((weight.float() + (adapter.B @ adapter.A)*adapter.scale).to(weight.dtype))
        setattr(parent, parts[-1], adapter.base)
    assert not any(isinstance(module, LoRALinear) for module in model.model.modules())
    return model.model


@torch.no_grad()
def measure(model, inputs):
    model.eval()
    for value in inputs[:16]:
        model(**value, use_cache=False)
    torch.cuda.synchronize()
    samples = []
    # Three passes over exactly the same 256 candidates. Include CPU dispatch
    # and branch hooks by timing synchronized forward calls, excluding encoding.
    for _ in range(3):
        for value in inputs:
            torch.cuda.synchronize()
            start = time.perf_counter()
            output = model(**value, use_cache=False)
            torch.cuda.synchronize()
            samples.append((time.perf_counter()-start)*1000)
            del output
    ordered = sorted(samples)
    return {'count':len(samples),'mean_ms':statistics.mean(samples),
            'median_ms':statistics.median(samples),'p95_ms':ordered[int(.95*(len(ordered)-1))],
            'samples_ms':samples}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoints',type=Path,required=True,
                        help='JSON object mapping each V5 arm to its seed-1 adapter directory')
    parser.add_argument('--data-dir',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True,exist_ok=True)
    metrics, passed, start = {}, False, time.monotonic()
    try:
        checkpoints = validate_checkpoint_map(json.loads(args.checkpoints.read_text()))
        tok = AutoTokenizer.from_pretrained(MODEL,revision=REVISION,local_files_only=True)
        tok.pad_token_id = tok.eos_token_id
        splits = prepare(args.data_dir)
        encoded, identities = [], []
        for task in ('piqa','boolq'):
            for row in splits['splits'][task]['dev'][:64]:
                for index,candidate in enumerate(encode_candidates(tok,row,max_length=512)):
                    encoded.append(candidate)
                    identities.append([task,row['id'],index,candidate['input_ids']])
        assert len(encoded) == 256
        inputs = [{k:v for k,v in batch_tensors([e],tok.pad_token_id).items() if k!='labels'} for e in encoded]
        metrics['inputs.sha256'] = hashlib.sha256(json.dumps(identities).encode()).hexdigest()
        metrics['inputs.count'] = 256
        metrics['hardware'] = torch.cuda.get_device_name()
        metrics['torch'] = torch.__version__
        metrics['cuda'] = torch.version.cuda
        measurements = {}
        for arm in ('base',*ARMS):
            model = make_model(arm,1, None if arm=='base' else checkpoints[arm])
            measurements[arm] = measure(model,inputs)
            if arm == 'V5-L':
                with torch.no_grad():
                    before = model(**inputs[0],use_cache=False).logits.float()
                merged = merge_lora(model)
                with torch.no_grad():
                    after = merged(**inputs[0],use_cache=False).logits.float()
                metrics['merge.first_input_logit_max_abs'] = (before-after).abs().max().item()
                measurements['V5-L-merged'] = measure(merged,inputs)
                del merged,before,after
            del model
            gc.collect()
            torch.cuda.empty_cache()
        write_json(out/'latency.json',{'metrics':metrics,'measurements':measurements,
                   'rule':'batch1, first64 dev rows per task, both candidates, 16 warmups, 3 timed passes; synchronized host forward walltime',
                   'merge_note':'BF16 merged weights incur one rounding; timing only, unmerged checkpoint used for task scoring'})
        for arm,values in measurements.items():
            metrics.update({f'{arm}.{k}':v for k,v in values.items() if k!='samples_ms'})
        passed = True
    finally:
        metrics['elapsed_s'] = time.monotonic()-start
        write_json(out/'manifest.json',{'status':'complete' if passed else 'partial',
                   'spec_digest':os.environ['SR_SPEC_DIGEST'],'metrics':metrics,
                   'artifacts':[{'path':str(out/'latency.json'),'role':'latency'}],
                   'validator':{'name':'v5_fixed_dev_latency','passed':passed}})


if __name__ == '__main__':
    main()
