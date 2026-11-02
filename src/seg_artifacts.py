"""Authenticate E-0025/E-0027 final artifacts before intervention or analysis."""
import json
import math
import re
from pathlib import Path

from calibrate import REVISION
from rb1_artifacts import read, reference, finite_tree, verify_lr_lock
from seg_data import STEPS, EXAMPLES, ORDER_SHA256_BY_SEED
from seg_model import PARAMETERS, FAMILIES, KEEP, config_for, SEED_ARMS, contract
from v1_run import lr_factor


def verify_ref(ref):
    if reference(ref['path']) != ref:
        raise ValueError('segmented artifact hash/size binding changed')
    return Path(ref['path'])


def validate_rows(rows, expected=None):
    keys = [(r['task'], r['id']) for r in rows]
    if len(keys) != len(set(keys)) or len(rows) != 2000:
        raise ValueError('segmented final predictions need 2000 unique task IDs')
    for task in ('piqa', 'boolq'):
        if sum(r['task'] == task for r in rows) != 1000:
            raise ValueError('segmented final dev needs 1000 per task')
    if any(r['split'] != 'dev' or type(r['gold']) is not int or type(r['prediction']) is not int or
           r['gold'] not in (0, 1) or r['prediction'] not in (0, 1) or
           len(r['scores']) != 2 or any(not math.isfinite(v) for v in r['scores']) or
           not math.isfinite(r['gold_nll_sum']) or r['gold_nll_sum'] < 0 or
           type(r['gold_loss_tokens']) is not int or r['gold_loss_tokens'] <= 0 or
           r['prediction'] != (0 if r['scores'][0] >= r['scores'][1] else 1) for r in rows):
        raise ValueError('invalid segmented candidate prediction row')
    if expected is not None and [(r['task'], r['id'], r['gold']) for r in rows] != expected:
        raise ValueError('segmented paired task IDs/order/gold differ')
    return rows


def prediction_rows(path, expected=None):
    return validate_rows(read(path), expected)


def validate(root, arm, source_commit, seed=0):
    if arm not in SEED_ARMS.get(seed, ()) or ORDER_SHA256_BY_SEED.get(seed) is None:
        raise ValueError('segmented seed/arm is not registered')
    root = Path(root).resolve()
    manifest = read(root/'manifest.json'); m = manifest['metrics']; cfg = read(root/'config.json')
    finite_tree(manifest)
    stage = 'base' if arm == 'SEG-00' else 'train'
    if (manifest.get('contract') != contract(seed) or manifest.get('status') != 'complete' or
        manifest.get('validator', {}).get('passed') is not True or
        any(manifest.get(k) != v for k, v in {'arm': arm, 'seed': seed, 'phase': 'SEG', 'stage': stage}.items())):
        raise ValueError('not a complete final segmented arm')
    if (cfg.get('arm') != arm or cfg.get('stage') != stage or cfg.get('seed', 0) != seed or
            m.get('source.commit') != source_commit):
        raise ValueError('segmented arm/config/source binding mismatch')
    if seed and (m.get('seed') != seed or re.fullmatch('[0-9a-f]{64}', str(m.get('initial.adapter_sha256'))) is None):
        raise ValueError('new-seed run lacks its seed or initial-parameter digest')
    artifacts = manifest.get('artifacts', [])
    for ref in artifacts:
        if Path(ref['path']).resolve().parent != root:
            raise ValueError('segmented artifact inventory points outside its run')
        if reference(ref['path'])['sha256'] != ref.get('sha256') or reference(ref['path'])['bytes'] != ref.get('bytes'):
            raise ValueError('segmented producer artifact hash/size changed')
    mandatory = {'config.json', 'predictions.json', 'seams-train.json', 'excluded-train.json', 'evaluation-audit.json'}
    if stage == 'train':
        mandatory |= {'checkpoint.json', 'steps.jsonl', *(f'dev-step-{math.ceil(STEPS*f)}.json' for f in (.25, .5, .75))}
        if seed in (3, 4, 5):
            mandatory.add('initial-predictions.json')
            if arm == 'SEG-05':
                mandatory.add('iv-off-dev.json')
    if not mandatory <= {Path(ref['path']).name for ref in artifacts}:
        raise ValueError('segmented producer artifact inventory incomplete')
    if m.get('model_revision') != REVISION or m.get('tokenizer_revision') != REVISION:
        raise ValueError('segmented model/tokenizer revision changed')
    if m.get('environment.lock_sha256') != reference(Path(__file__).resolve().parents[1]/'uv.lock')['sha256']:
        raise ValueError('segmented environment lock differs')
    lock = verify_lr_lock(cfg['lr_lock'])
    if m.get('lr_lock.sha256') != reference(cfg['lr_lock'])['sha256'] or m.get('split_manifest.sha256') != lock['split_manifest_sha256']:
        raise ValueError('segmented LR/split pins differ')
    if reference(cfg['split_manifest'])['sha256'] != lock['split_manifest_sha256']:
        raise ValueError('segmented split file changed')
    if m.get('plan.order_sha256') != ORDER_SHA256_BY_SEED[seed] or m.get('plan.steps') != STEPS or m.get('plan.examples') != EXAMPLES:
        raise ValueError('segmented RB1 sample plan changed')
    if m.get('params.stored') != PARAMETERS[arm] or m.get('params.trainable') != PARAMETERS[arm]:
        raise ValueError('segmented parameter budget differs')
    rows = prediction_rows(root/'predictions.json')
    for task in ('piqa', 'boolq'):
        selected = [r for r in rows if r['task'] == task]
        acc = sum(r['prediction'] == r['gold'] for r in selected)/len(selected)
        nll = sum(r['gold_nll_sum'] for r in selected)/sum(r['gold_loss_tokens'] for r in selected)
        if m.get(f'dev.{task}.accuracy') != acc or m.get(f'dev.{task}.nll') != nll or m.get(f'dev.{task}.examples') != 1000:
            raise ValueError('segmented raw predictions disagree with manifest metrics')
    refs = {'manifest': reference(root/'manifest.json'), 'predictions': reference(root/'predictions.json'),
            'config': reference(root/'config.json')}
    if stage == 'train':
        if (m.get('reload.identical') is not True or m.get('failed') is not False or
                m.get('train.steps') != STEPS or m.get('train.examples') != EXAMPLES or
                m.get('params.with_grad') != PARAMETERS[arm] or
                m.get('learning_rate') != lock['locked_lrs'][FAMILIES[arm]]):
            raise ValueError('segmented final budget/reload/LR check failed')
        checkpoint = read(root/'checkpoint.json')
        if set(checkpoint) != {'config.json', 'adapter.pt'}:
            raise ValueError('segmented checkpoint inventory incomplete')
        for name, ref in checkpoint.items():
            if Path(ref['path']).resolve() != root/'adapter'/name:
                raise ValueError('segmented checkpoint inventory points to another run')
            verify_ref(ref)
        refs['checkpoint'] = reference(root/'checkpoint.json')
        from dataclasses import asdict
        from v1_lora import LoRAConfig
        expected_cfg = LoRAConfig(layers=(6, 13, 20, 27), rank=7, alpha=7) if arm == 'SEG-04' else config_for(arm)
        if read(root/'adapter'/'config.json') != json.loads(json.dumps(asdict(expected_cfg))):
            raise ValueError('segmented checkpoint config differs from registered arm')
        count, tokens = 0, 0
        with (root/'steps.jsonl').open() as f:
            for count, line in enumerate(f, 1):
                step = json.loads(line)
                finite_tree(step)
                if step['step'] != count or step['lr'] != m['learning_rate']*lr_factor(count-1, STEPS):
                    raise ValueError('segmented optimizer steps or schedule changed')
                if type(step['input_tokens']) is not int or step['input_tokens'] <= 0:
                    raise ValueError('segmented forward-token accounting invalid')
                tokens += step['input_tokens']
        if count != STEPS or tokens != m.get('train.input_tokens') or tokens != m.get('train.segment1_tokens')+m.get('train.segment2_tokens'):
            raise ValueError('segmented step/token ledger is incomplete')
        accounting = m.get('runtime.accounting', {})
        expected_phases = {'segment1', 'segment2'} if arm in KEEP else set() if arm == 'SEG-04' else {'segment2'}
        if set(accounting) != expected_phases:
            raise ValueError('segmented executed phase accounting differs')
        for phase, layers in accounting.items():
            if set(layers) != {'6', '13', '20', '27'}:
                raise ValueError('segmented four-layer accounting incomplete')
            valid = m['train.'+phase+'_tokens']
            expected_reads = 0 if arm in ('SEG-01', 'SEG-07') else valid
            for c in layers.values():
                if (c.get('attention') != EXAMPLES or c.get('ffn') != EXAMPLES or
                    c.get('valid_tokens') != valid or c.get('reads') != expected_reads or c.get('writes') != expected_reads):
                    raise ValueError('segmented write/read/backbone counts differ')
        if arm not in KEEP and m.get('train.segment1_tokens') != 0:
            raise ValueError('no-channel arm executed segment 1 during training')
        if arm in KEEP:
            rank = 128 if arm == 'SEG-05' else 32
            if m.get('state.M_c_bytes') != 4*(rank*rank+rank)*4 or m.get('state.counts_bytes') != 32:
                raise ValueError('segmented carry-state byte budget differs')
        for frac in (.25, .5, .75):
            step = math.ceil(STEPS*frac)
            curve = read(root/f'dev-step-{step}.json')
            if curve.get('step') != step:
                raise ValueError('segmented curve step differs')
            validate_rows(curve['predictions'], [(r['task'], r['id'], r['gold']) for r in rows])
        if arm in KEEP:
            if m.get('initial.identical_to_Base_seg') is not True or read(root/'initial-base-predictions.json') != read(root/'initial-predictions.json'):
                raise ValueError('segmented zero-initialization identity failed')
        if seed in (3, 4, 5):
            validate_rows(read(root/'initial-predictions.json'), [(r['task'], r['id'], r['gold']) for r in rows])
            if arm == 'SEG-05' and (m.get('off.identical_to_Base_seg') is not True or
                    read(root/'iv-off-dev.json')['predictions'] != read(root/'initial-base-predictions.json')):
                raise ValueError('E-0030 dev off/Base same-allocation identity failed')
    return manifest, refs, root/'adapter'
