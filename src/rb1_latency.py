"""E-0021 RB1 latency driver for validated formal checkpoints."""
import argparse
import gc
import json
import math
import os
from pathlib import Path
import time

import torch

from calibrate import MODEL, REVISION
from mig_profile_guard import main as profile_guard
from rb_latency import measure
from rb1_artifacts import (finite_tree, reference, validate_formal_run,
                           verified_latency)
from rb_train import make_model
from v1_lora import LoRAModel
from v5_latency import merge_lora
from v5_train import write_json


TIMED_ARMS = ('RB1-01', 'RB1-02', 'RB1-04', 'RB1-08', 'RB1-07')
CONTRACT = 'E-0021@1'


def validate_formal_checkpoint(run_dir, arm):
    """Compatibility name retained for existing latency caller/tests."""
    return validate_formal_run(run_dir, arm)[2]


def validate_checkpoint_map(checkpoints):
    """Validate each requested formal run before any model is loaded."""
    if not isinstance(checkpoints, dict) or not set(TIMED_ARMS) <= set(checkpoints):
        raise ValueError(f'checkpoint map must contain {", ".join(TIMED_ARMS)}')
    if set(checkpoints) - set(TIMED_ARMS):
        raise ValueError('checkpoint map contains an unrequested arm')
    return {arm: validate_formal_checkpoint(checkpoints[arm], arm) for arm in TIMED_ARMS}


def json_safe(value):
    """Retain diagnostic raw values while keeping partial artifacts valid JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if isinstance(value, dict):
        return {key: json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(child) for child in value]
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoints', type=Path, required=True,
                        help='JSON object mapping each requested arm to its formal run directory')
    parser.add_argument('--include-base', action='store_true', help='also time frozen Base')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    passed, measurements, spec_digest = False, {}, None
    start = time.monotonic()
    try:
        checkpoint_runs = json.loads(args.checkpoints.read_text())
        if not isinstance(checkpoint_runs, dict) or not set(TIMED_ARMS) <= set(checkpoint_runs) or set(checkpoint_runs) - set(TIMED_ARMS):
            raise ValueError(f'checkpoint map must name exactly the requested arms: {", ".join(TIMED_ARMS)}')
        source_commit = os.environ.get('SR_CODE_COMMIT')
        if not source_commit:
            raise ValueError('SR_CODE_COMMIT must bind timing output to the submitted source snapshot')
        spec_digest = os.environ.get('SR_SPEC_DIGEST')
        if not spec_digest:
            raise ValueError('SR_SPEC_DIGEST must bind timing output to E-0021')
        formal = {arm: validate_formal_run(checkpoint_runs[arm], arm) for arm in TIMED_ARMS}
        manifests = {arm: value[0] for arm, value in formal.items()}
        inventories = {arm: value[1] for arm, value in formal.items()}
        adapter_dirs = {arm: value[2] for arm, value in formal.items()}
        # The live profile check is intentionally before make_model/load_base.
        if profile_guard(['--output', str(out)]) != 0:
            raise RuntimeError('MIG profile guard rejected allocation; requires 2g.24gb')
        refs = {}
        arms = (*TIMED_ARMS, 'Base') if args.include_base else TIMED_ARMS
        for arm in arms:
            model = make_model(arm, phase='R1', seed=0,
                               checkpoint=None if arm == 'Base' else adapter_dirs[arm])
            metrics = measure(model)
            measurements[arm] = metrics
            verified_latency(metrics)
            refs[arm] = {'run_dir': None if arm == 'Base' else str(Path(checkpoint_runs[arm]).resolve()),
                         'adapter_dir': None if arm == 'Base' else str(adapter_dirs[arm]),
                         'artifact_inventory': None if arm == 'Base' else inventories[arm]}
            finite_tree(metrics, f'{arm}.metrics')
            if arm == 'RB1-07':
                merged = merge_lora(model)
                model._targets.clear()
                del model
                gc.collect()
                torch.cuda.empty_cache()
                merged_metrics = measure(merged)
                measurements['RB1-07-merged'] = merged_metrics
                verified_latency(merged_metrics)
                finite_tree(merged_metrics, 'RB1-07-merged.metrics')
                refs['RB1-07-merged'] = {'derived_from': 'RB1-07',
                                         'merge_implementation': 'v5_latency.merge_lora'}
                del merged
            else:
                del model
            gc.collect()
            torch.cuda.empty_cache()
        metadata = {
            'contract': CONTRACT, 'arms': list(arms), 'measurements': measurements,
            'loaded_refs': refs,
            'training_manifests': manifests,
            'environment': {'source_commit': source_commit,
                            'environment_lock_sha256': reference(Path(__file__).resolve().parents[1]/'uv.lock')['sha256'],
                            'model': MODEL, 'model_revision': REVISION,
                            'tokenizer_revision': REVISION,
                            'torch': torch.__version__, 'cuda': torch.version.cuda,
                            'hardware': torch.cuda.get_device_name()},
            'measurement_contract': {'batch': 1, 'prefill_length': 512, 'prefill_passes': 5,
                                     'prefill_warmup_passes': 4, 'decode_tokens': 128,
                                     'decode_p50_p95': True,
                                     'merged_state_note': 'RB1-07 adapter state bytes exclude backbone KV'}
        }
        finite_tree(metadata, 'latency')
        write_json(out/'latency.json', metadata)
        passed = True
    finally:
        if not passed:
            # Preserve whatever raw timing samples were collected for diagnosis.
            write_json(out/'latency.json', json_safe({'contract': CONTRACT, 'measurements': measurements,
                                             'status': 'partial'})
                       )
        elapsed = time.monotonic() - start
        write_json(out/'manifest.json', {'status': 'complete' if passed else 'partial',
                   'contract': CONTRACT, 'spec_digest': spec_digest or os.environ.get('SR_SPEC_DIGEST'),
                   'metrics': {'arms': list(measurements), 'elapsed_s': elapsed,
                               'hardware': torch.cuda.get_device_name() if torch.cuda.is_available() else None},
                   'artifacts': [{'path': str(out/'latency.json'), 'role': 'latency'}],
                   'validator': {'name': 'rb1_latency', 'passed': passed}})


if __name__ == '__main__':
    main()
