"""Strict artifact validation shared by the RB1 measurement and analysis tools."""
import hashlib
import json
import math
import re
import statistics
from pathlib import Path

from rb_train import FAMILIES, LORA_RANKS, PARAMETERS, WIDTH_ARMS, contract


CONTRACT = 'E-0021@1'
LOCK_CONTRACT = 'E-0020@1'
TOKEN_BUDGET = 2_000_000
ACCEPTED_LR_LOCK_SHA256 = '38b8ee24af3ded8aee398a4cc3dfaa5a1dfece4737a6cde34642fbff7bf1012d'
ACCEPTED_TRAINING_SOURCE_COMMIT = '7bf091d4af24c2a9fdc666a5f885c31817b97174'


def read(path):
    return json.loads(Path(path).read_text())


def reference(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    if not data:
        raise ValueError(f'empty artifact: {path}')
    return {'path': str(path), 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}


def finite_tree(value, path='value'):
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f'non-finite value at {path}')
        return
    if isinstance(value, dict):
        for key, child in value.items():
            finite_tree(child, f'{path}.{key}')
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            finite_tree(child, f'{path}[{index}]')
        return
    raise TypeError(f'unsupported value at {path}: {type(value).__name__}')


def _require_int(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')


def _require_finite(value, name, *, positive=False, nonnegative=False):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f'{name} must be finite')
    if positive and value <= 0:
        raise ValueError(f'{name} must be positive')
    if nonnegative and value < 0:
        raise ValueError(f'{name} must be nonnegative')


def verify_lr_lock(path, expected_sha256=ACCEPTED_LR_LOCK_SHA256):
    """Verify T-0024's accepted lock and reconstruct it from pinned inputs."""
    from rb_train import lock_lrs

    path = Path(path).resolve()
    data = read(path)
    actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError('LR lock SHA256 differs from the accepted T-0024 lock')
    inputs = data.get('inputs')
    if not isinstance(inputs, list) or not inputs:
        raise ValueError('LR lock has no calibration input references')
    for index, ref in enumerate(inputs):
        if not isinstance(ref, dict) or set(ref) != {'path', 'sha256', 'bytes'}:
            raise ValueError(f'LR lock input reference {index} has an unexpected schema')
        artifact = Path(ref['path']).resolve()
        if not artifact.is_file():
            raise ValueError(f'LR lock input artifact is missing: {artifact}')
        actual_ref = reference(artifact)
        if ref['sha256'] != actual_ref['sha256'] or ref['bytes'] != actual_ref['bytes']:
            raise ValueError(f'LR lock input hash/size mismatch: {artifact}')
    try:
        rebuilt = lock_lrs([ref['path'] for ref in inputs])
    except (AssertionError, KeyError, TypeError, ValueError) as exc:
        raise ValueError('LR lock cannot be reconstructed from calibration inputs') from exc
    order_hashes = {read(ref['path']).get('metrics', {}).get('plan.order_sha256') for ref in inputs}
    if len(order_hashes) != 1 or None in order_hashes:
        raise ValueError('calibration inputs do not share one plan-order hash')
    order_hash = next(iter(order_hashes))
    if data.get('plan_order_sha256') != order_hash:
        raise ValueError('LR lock plan-order hash differs from calibration inputs')
    rebuilt['plan_order_sha256'] = order_hash
    if json.loads(json.dumps(rebuilt)) != data:
        raise ValueError('LR lock differs from reconstructed calibration choices')
    return data


def validate_formal_run(run_dir, arm, source_commit=None):
    """Validate a final RB1/E-0026 producer run; return manifest, refs, adapter path.

    RB1 arms are pinned to the accepted E-0021 training source. E-0026 width
    arms have no accepted source yet, so the caller must name the reviewed one.
    """
    if arm not in PARAMETERS:
        raise ValueError(f'unknown RB1 arm {arm}')
    if arm in WIDTH_ARMS:
        if not isinstance(source_commit, str) or re.fullmatch('[0-9a-f]{40}', source_commit) is None:
            raise ValueError(f'{arm}: E-0026 validation requires the reviewed training source commit')
        expected_source = source_commit
    elif source_commit is not None:
        raise ValueError(f'{arm}: RB1 arms are pinned to the accepted E-0021 training source')
    else:
        expected_source = ACCEPTED_TRAINING_SOURCE_COMMIT
    run_contract = contract(arm, 'train')
    root = Path(run_dir).resolve()
    required = ('manifest.json', 'config.json', 'checkpoint.json', 'steps.jsonl')
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise ValueError(f'{arm}: missing formal-run files: {", ".join(missing)}')
    manifest = read(root / 'manifest.json')
    config = read(root / 'config.json')
    checkpoint = read(root / 'checkpoint.json')
    expected = {'arm': arm, 'phase': 'R1', 'seed': 0, 'stage': 'train'}
    if manifest.get('status') != 'complete' or manifest.get('contract') != run_contract:
        raise ValueError(f'{arm}: requires a complete {run_contract} formal training manifest')
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError(f'{arm}: manifest arm/phase/seed/stage mismatch')
    if manifest.get('family') != FAMILIES[arm] or manifest.get('validator', {}).get('passed') is not True:
        raise ValueError(f'{arm}: family/validator does not match formal training')
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError(f'{arm}: config.json arm/phase/seed/stage mismatch')
    if config.get('eval_limit') is not None or config.get('smoke_steps') is not None:
        raise ValueError(f'{arm}: partial evaluation/smoke budget is not a formal run')
    if config.get('three_epochs') is not False:
        raise ValueError(f'{arm}: explicit three-epoch override is not a formal first-stop run')

    metrics = manifest.get('metrics')
    if not isinstance(metrics, dict):
        raise ValueError(f'{arm}: manifest metrics are missing')
    if metrics.get('failed') is not False or metrics.get('reload.identical') is not True:
        raise ValueError(f'{arm}: failed or non-identical reload training artifact')
    from calibrate import REVISION
    if metrics.get('model_revision') != REVISION or metrics.get('tokenizer_revision') != REVISION:
        raise ValueError(f'{arm}: model/tokenizer revision differs from pinned Base')
    if metrics.get('environment.lock_sha256') != reference(Path(__file__).resolve().parents[1]/'uv.lock')['sha256']:
        raise ValueError(f'{arm}: training environment lock differs from analysis environment')
    for key, length in (('source.commit', 40), ('plan.order_sha256', 64)):
        if not isinstance(metrics.get(key), str) or re.fullmatch('[0-9a-f]{'+str(length)+'}', metrics[key]) is None:
            raise ValueError(f'{arm}: missing or invalid {key}')
    if metrics['source.commit'] != expected_source:
        raise ValueError(f'{arm}: source commit differs from covered training release')
    expected_params = PARAMETERS[arm]
    for key in ('params.stored', 'params.trainable', 'params.with_grad'):
        if metrics.get(key) != expected_params:
            raise ValueError(f'{arm}: {key} differs from registered arm parameter count')

    learning_rate = config.get('lr')
    _require_finite(learning_rate, f'{arm} config.lr', positive=True)
    if metrics.get('learning_rate') != learning_rate:
        raise ValueError(f'{arm}: manifest learning rate differs from run config')
    lock_value = config.get('lr_lock')
    split_value = config.get('split_manifest')
    if not isinstance(lock_value, str) or not lock_value or not isinstance(split_value, str) or not split_value:
        raise ValueError(f'{arm}: formal LR lock and split manifest paths are required')
    lock_path, split_path = Path(lock_value).resolve(), Path(split_value).resolve()
    if not lock_path.is_file() or not split_path.is_file():
        raise ValueError(f'{arm}: LR lock or split manifest file is missing')
    lock = verify_lr_lock(lock_path)
    lock_hash = reference(lock_path)['sha256']
    split_hash = reference(split_path)['sha256']
    if lock.get('experiment') != LOCK_CONTRACT or lock.get('locked_lrs', {}).get(FAMILIES[arm]) != learning_rate:
        raise ValueError(f'{arm}: LR lock family/value mismatch')
    if lock.get('split_manifest_sha256') != split_hash:
        raise ValueError(f'{arm}: LR lock split hash differs from actual split manifest')
    if metrics.get('lr_lock.sha256') != lock_hash:
        raise ValueError(f'{arm}: actual LR lock hash differs from training manifest')
    if metrics.get('split_manifest.sha256') != split_hash:
        raise ValueError(f'{arm}: actual split manifest hash differs from training manifest')

    count_keys = (('plan.steps', 'train.steps'),
                  ('plan.input_tokens', 'train.input_tokens'),
                  ('plan.examples', 'train.examples'))
    for plan_key, train_key in count_keys:
        _require_int(metrics.get(plan_key), f'{arm} {plan_key}', minimum=1)
        _require_int(metrics.get(train_key), f'{arm} {train_key}', minimum=1)
        if metrics[plan_key] != metrics[train_key]:
            raise ValueError(f'{arm}: {plan_key} differs from {train_key}')

    actual_tokens, token_before_last, max_lr, row_count = 0, 0, 0., 0
    from v1_run import lr_factor
    with (root / 'steps.jsonl').open() as handle:
        for index, line in enumerate(handle, 1):
            row_count = index
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f'{arm}: invalid steps.jsonl row {index}') from exc
            if not isinstance(row, dict) or type(row.get('step')) is not int or row['step'] != index:
                raise ValueError(f'{arm}: steps.jsonl step sequence is not contiguous')
            for key in ('example_mean_nll', 'preclip_norm', 'lr', 'seconds'):
                _require_finite(row.get(key), f'{arm} step {index} {key}',
                                positive=key == 'seconds',
                                nonnegative=key in ('example_mean_nll', 'preclip_norm', 'lr'))
            _require_int(row.get('input_tokens'), f'{arm} step {index} input_tokens', minimum=1)
            if row['input_tokens'] > 1024:
                raise ValueError(f'{arm}: step exceeds batch2/max_length512 token bound')
            if not math.isclose(row['lr'], learning_rate*lr_factor(index-1, metrics['train.steps']), rel_tol=1e-12, abs_tol=1e-15):
                raise ValueError(f'{arm}: step LR differs from registered warmup/cosine schedule')
            token_before_last = actual_tokens
            actual_tokens += row['input_tokens']
            max_lr = max(max_lr, row['lr'])
    if row_count != metrics['train.steps']:
        raise ValueError(f'{arm}: steps.jsonl row count differs from train.steps')
    if actual_tokens != metrics['train.input_tokens'] or actual_tokens != metrics['plan.input_tokens']:
        raise ValueError(f'{arm}: step token sum differs from plan/train input tokens')
    if max_lr > learning_rate or max_lr <= 0:
        raise ValueError(f'{arm}: recorded step rates do not match positive locked LR scale')
    if actual_tokens >= TOKEN_BUDGET:
        if token_before_last >= TOKEN_BUDGET:
            raise ValueError(f'{arm}: training continued after reaching the 2M-token stop')
        if metrics['train.examples'] != 2*row_count:
            raise ValueError(f'{arm}: token-threshold stop requires a complete batch2 step')
    elif (metrics.get('train.last_epoch') != 2 or
          metrics.get('plan.epoch_end_steps', {}).get('3') != row_count):
        raise ValueError(f'{arm}: below-budget run did not complete all three epochs')

    if not isinstance(checkpoint, dict) or set(checkpoint) != {'config.json', 'adapter.pt'}:
        raise ValueError(f'{arm}: checkpoint.json must bind config.json and adapter.pt')
    paths = {}
    for name, record in checkpoint.items():
        if not isinstance(record, dict) or set(('path', 'sha256', 'bytes')) - set(record):
            raise ValueError(f'{arm}: malformed checkpoint inventory entry for {name}')
        path = Path(record['path']).resolve()
        if not path.is_file() or path.name != name or root not in path.parents:
            raise ValueError(f'{arm}: checkpoint path escapes or is missing from formal run')
        ref = reference(path)
        if record['sha256'] != ref['sha256'] or record['bytes'] != ref['bytes']:
            raise ValueError(f'{arm}: checkpoint inventory hash/size mismatch for {name}')
        paths[name] = path
    if paths['config.json'].parent != paths['adapter.pt'].parent:
        raise ValueError(f'{arm}: config and adapter checkpoint files are split')
    adapter_dir = paths['adapter.pt'].parent
    model_config = read(adapter_dir / 'config.json')
    if arm in LORA_RANKS:
        from v1_lora import LoRAConfig
        expected_config = json.loads(json.dumps(LoRAConfig(layers=(6, 13, 20, 27), rank=LORA_RANKS[arm],
                                                            alpha=LORA_RANKS[arm]).__dict__))
    else:
        from rb_train import adapter_config
        expected_config = json.loads(json.dumps(adapter_config(arm).__dict__))
    if model_config != expected_config:
        raise ValueError(f'{arm}: adapter config differs from registered arm')

    inventory = {
        'manifest': reference(root / 'manifest.json'),
        'run_config': reference(root / 'config.json'),
        'checkpoint_json': reference(root / 'checkpoint.json'),
        'steps': reference(root / 'steps.jsonl'),
        'lr_lock': reference(lock_path), 'split_manifest': reference(split_path),
        'adapter_config': reference(paths['config.json']),
        'adapter_weights': reference(paths['adapter.pt']),
    }
    finite_tree(metrics, f'{arm}.manifest.metrics')
    return manifest, inventory, adapter_dir


def verified_latency(row):
    """Validate all raw timing samples and their reported summaries."""
    if row.get('batch') != 1 or row.get('prefill_length') != 512 or row.get('prefill_passes') != 5:
        raise ValueError('latency must use batch1, prefill512, five measured passes')
    if row.get('decode_tokens') != 128 or len(row.get('decode_samples_ms', [])) != 128:
        raise ValueError('latency must contain 128 decode samples')
    if len(row.get('prefill_seconds', [])) != 5:
        raise ValueError('latency must contain five prefill samples')
    raw = row['decode_samples_ms'] + row['prefill_seconds']
    for index, value in enumerate(raw):
        _require_finite(value, f'latency raw sample {index}', positive=True)
    ordered = sorted(row['decode_samples_ms'])
    if row.get('decode_p50_ms') != statistics.median(ordered):
        raise ValueError('decode p50 does not match raw samples')
    if row.get('decode_p95_ms') != ordered[int(.95 * (len(ordered) - 1))]:
        raise ValueError('decode p95 does not match raw samples')
    expected_rate = 512 / statistics.mean(row['prefill_seconds'])
    if not math.isclose(row.get('prefill_tokens_per_second', math.nan), expected_rate, rel_tol=1e-12):
        raise ValueError('prefill throughput does not match raw samples')
    state = row.get('adapter_persistent_state', {}).get('M_c_bytes')
    if state not in (0, 16896):
        raise ValueError('unexpected persistent adapter state byte count')
    if row.get('peak_cuda_bytes', 0) <= 0:
        raise ValueError('peak CUDA allocation must be positive')
    return row
