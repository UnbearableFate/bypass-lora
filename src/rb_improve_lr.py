"""E-0029 up-only LoRA calibration: manifests only, no dev access."""
import hashlib
import json
import math
from pathlib import Path

from rb1_artifacts import read, reference


GRID = (1e-4, 3e-4, 1e-3)


def candidate_grid(manifests):
    from rb_eta_select import ref as step_reference, validate_calibration
    records, refs, artifacts, common, paired_membership = {}, [], [], None, None
    for name in manifests:
        value = read(name)
        metrics = value.get('metrics', {})
        expected = {'contract': 'E-0029@1', 'arm': 'L-56', 'family': 'RBc-UP',
                    'stage': 'calibration', 'phase': 'V0', 'seed': 0, 'status': 'complete'}
        if any(value.get(key) != field for key, field in expected.items()) or value.get('validator', {}).get('passed') is not True:
            raise ValueError('invalid up-only LoRA calibration manifest')
        lr = metrics.get('learning_rate')
        if lr not in (*GRID, 3e-5, 3e-3) or lr in records:
            raise ValueError('duplicate or unregistered LR')
        identity = {key: metrics.get(key) for key in ('source.commit', 'environment.lock_sha256',
                    'split_manifest.sha256', 'plan.order_sha256', 'plan.steps', 'plan.input_tokens',
                    'init.parameters.sha256')}
        if any(identity[key] in (None, 'unknown') for key in identity) or common is not None and common != identity:
            raise ValueError('calibration code/environment/split/order/budget mismatch')
        common = identity
        if any(key.startswith('dev.') for key in metrics) or metrics.get('dev_read') is not False:
            raise ValueError('LR calibration may not read dev')
        config = read(Path(name).parent/'config.json')
        if (config.get('smoke_steps') is not None or config.get('eval_limit') is not None or
                any(config.get(key) != field for key, field in
                    {'arm': 'L-56', 'stage': 'calibration', 'phase': 'V0', 'seed': 0, 'lr': lr}.items())):
            raise ValueError('partial calibration candidate')
        failed = metrics.get('failed')
        nll = metrics.get('calibration.nll')
        inventory = {'config.json': reference(Path(name).parent/'config.json'),
                     'steps.jsonl': step_reference(Path(name).parent/'steps.jsonl')}
        if failed is True:
            reason = metrics.get('failure_reason')
            if (metrics.get('failure_kind') not in ('training_nonfinite', 'evaluation_nonfinite') or
                    not isinstance(reason, str) or not reason.strip() or nll is not None):
                raise ValueError('only typed nonfinite calibration candidates may be excluded')
            records[float(lr)] = None
            refs.append(reference(name))
            artifacts.append(inventory)
            continue
        if failed is not False:
            raise ValueError('candidate requires an explicit failed flag')
        predictions = Path(name).parent/'predictions.json'
        rows = read(predictions)
        validate_calibration(rows, metrics)
        membership = [(row['task'], row['id'], row['gold'], row['gold_loss_tokens']) for row in rows]
        membership_hash = hashlib.sha256(json.dumps(membership).encode()).hexdigest()
        if paired_membership is not None and membership_hash != paired_membership:
            raise ValueError('calibration membership differs between finite candidates')
        paired_membership = membership_hash
        if type(nll) not in (float, int) or not math.isfinite(nll) or nll <= 0:
            raise ValueError('finite positive calibration NLL required')
        if (metrics.get('train.input_tokens') != identity['plan.input_tokens'] or
                metrics.get('train.steps') != identity['plan.steps'] or metrics['train.input_tokens'] < 200000 or
                any(metrics.get(f'calibration.{task}.examples') != 2048 for task in ('piqa', 'boolq'))):
            raise ValueError('incomplete calibration tokens/steps/evaluation')
        before, tokens, count = 0, 0, 0
        with (Path(name).parent/'steps.jsonl').open() as handle:
            for count, line in enumerate(handle, 1):
                row = json.loads(line)
                if row['step'] != count or type(row['input_tokens']) is not int or not 0 < row['input_tokens'] <= 1024:
                    raise ValueError('invalid calibration step accounting')
                before, tokens = tokens, tokens + row['input_tokens']
        if not before < 200000 <= tokens or tokens != metrics['train.input_tokens'] or count != metrics['train.steps']:
            raise ValueError('calibration did not stop at the first full step reaching200k')
        records[float(lr)] = float(nll)
        refs.append(reference(name))
        inventory['predictions.json'] = reference(predictions)
        artifacts.append(inventory)
    if common is not None:
        common = dict(common, calibration_membership_sha256=paired_membership)
    return records, refs, common, artifacts


def endpoint(records):
    if set(records) != set(GRID):
        raise ValueError('initial LR grid is incomplete')
    finite = {lr: nll for lr, nll in records.items() if nll is not None}
    if not finite:
        raise ValueError('initial LR grid has no finite candidate')
    best = min(finite, key=lambda lr: (finite[lr], lr))
    return 3e-5 if best == GRID[0] else 3e-3 if best == GRID[-1] else None


def lock(manifests, final=True):
    records, refs, common, artifacts = candidate_grid(manifests)
    extension = endpoint({lr: records[lr] for lr in GRID if lr in records})
    expected = set(GRID) | ({extension} if extension is not None and final else set())
    if set(records) != expected:
        raise ValueError('LR endpoint extension missing or extraneous')
    finite = {lr: nll for lr, nll in records.items() if nll is not None}
    minimum = min(finite.values())
    choice = min(lr for lr, nll in finite.items() if (nll-minimum)/minimum <= .01)
    return {'experiment': 'E-0029@1', 'family': 'RBc-UP', 'representative': 'L-56',
        'locked_lrs': {'RBc-UP': choice} if final else {}, 'required_extension': extension,
        'boundary_minimum': extension is not None and records.get(extension) == minimum,
        'candidates': {f'{lr:g}': nll for lr, nll in sorted(records.items())},
        'inputs': sorted(refs, key=lambda row: row['path']), 'identity': common,
        'candidate_artifacts': sorted(artifacts, key=lambda row: row['config.json']['path']),
        'split_manifest_sha256': common['split_manifest.sha256'],
        'plan_order_sha256': common['plan.order_sha256'], 'dev_read': False}


def verify_lock(path):
    value = read(path)
    for ref in value.get('inputs', []):
        if reference(ref['path']) != ref:
            raise ValueError('calibration manifest changed after LR locking')
    if value != lock([ref['path'] for ref in value['inputs']]):
        raise ValueError('up-only LR lock does not reconstruct')
    return value
