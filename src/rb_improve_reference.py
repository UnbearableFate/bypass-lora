"""Read-only diagnostic replay of E-0021/E-0028 anchors on E-0029 train IDs."""
import argparse
import gc
import os
from pathlib import Path
import time

import torch
from transformers import AutoTokenizer

from calibrate import MODEL, REVISION
from rb1_artifacts import reference, validate_formal_run
from rb_improve_init import calibration_examples
from rb_improve_read import diagnose
from rb_train import load_data, make_model
from v5_train import write_json


def run(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    tick = time.monotonic()
    _, refs, adapter = validate_formal_run(args.training_run, args.arm, args.training_source)
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    tok.pad_token_id = tok.eos_token_id
    splits, excluded = load_data(args.data_dir, args.split_manifest, tok)
    examples, sample = calibration_examples(splits, tok, excluded)
    value = {'arm': args.arm, 'sample_sha256': sample['examples_sha256'],
        'training_inputs': refs, 'hardware': torch.cuda.get_device_name(), 'job_id': os.environ['PBS_JOBID']}
    for stage, checkpoint in (('step0', None), ('final', adapter)):
        model = make_model(args.arm, checkpoint=checkpoint)
        value[stage] = diagnose(model, examples[:64], tok)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    write_json(out/'read-diagnostics.json', value)
    write_json(out/'manifest.json', {'status': 'complete', 'contract': 'E-0029@1',
        'spec_digest': os.environ['SR_SPEC_DIGEST'], 'arm': args.arm, 'stage': 'reference-read',
        'metrics': {'elapsed_s': time.monotonic()-tick, 'sample_sha256': sample['examples_sha256']},
        'artifacts': [{'path': str(out/'read-diagnostics.json'), 'role': 'read_diagnostics'}],
        'validator': {'name': 'e0029_reference_read', 'passed': True}})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm', choices=('RB1-09', 'X-01'), required=True)
    p.add_argument('--training-run', type=Path, required=True)
    p.add_argument('--training-source')
    p.add_argument('--split-manifest', type=Path, required=True)
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    run(p.parse_args())
