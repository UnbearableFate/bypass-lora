from dataclasses import asdict
import json
from pathlib import Path

import pytest

from rb1_artifacts import reference
import seg_artifacts as a
from seg_data import STEPS, EXAMPLES, ORDER_SHA256
from seg_model import PARAMETERS, FAMILIES, KEEP, config_for
from v1_run import lr_factor

SOURCE = 'b'*40


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value)+'\n')


def predictions(correct=600):
    return [{'task': task, 'id': i, 'split': 'dev', 'gold': 0, 'prediction': int(i >= correct),
             'scores': [-2., -3. if i < correct else -1.], 'gold_nll_sum': 2., 'gold_loss_tokens': 1}
            for task in ('piqa', 'boolq') for i in range(1000)]


def formal(root, arm, monkeypatch):
    root.mkdir(parents=True, exist_ok=True)
    write(root/'lock.json', {'fixture': True}); write(root/'split.json', {'fixture': True})
    lock = {'split_manifest_sha256': reference(root/'split.json')['sha256'],
            'locked_lrs': {'RBc-T': .003, 'RBc-R': .003, 'RBc-L': .0003}}
    monkeypatch.setattr(a, 'verify_lr_lock', lambda p: lock)
    p = predictions(); train = arm != 'SEG-00'; stage = 'train' if train else 'base'
    write(root/'config.json', {'arm': arm, 'stage': stage, 'lr_lock': str(root/'lock.json'), 'split_manifest': str(root/'split.json')})
    write(root/'predictions.json', p)
    for name in ('seams-train.json', 'excluded-train.json', 'evaluation-audit.json'): write(root/name, [])
    m = {'source.commit': SOURCE, 'model_revision': a.REVISION, 'tokenizer_revision': a.REVISION,
         'environment.lock_sha256': reference(Path(a.__file__).resolve().parents[1]/'uv.lock')['sha256'],
         'lr_lock.sha256': reference(root/'lock.json')['sha256'], 'split_manifest.sha256': lock['split_manifest_sha256'],
         'plan.order_sha256': ORDER_SHA256, 'plan.steps': STEPS, 'plan.examples': EXAMPLES,
         'params.stored': PARAMETERS[arm], 'params.trainable': PARAMETERS[arm]}
    for task in ('piqa', 'boolq'):
        m.update({f'dev.{task}.accuracy': .6, f'dev.{task}.nll': 2., f'dev.{task}.examples': 1000})
    if train:
        lr = lock['locked_lrs'][FAMILIES[arm]]
        m.update({'reload.identical': True, 'failed': False, 'train.steps': STEPS, 'train.examples': EXAMPLES,
                  'params.with_grad': PARAMETERS[arm], 'learning_rate': lr,
                  'train.segment1_tokens': EXAMPLES if arm in KEEP else 0,
                  'train.segment2_tokens': EXAMPLES,
                  'train.input_tokens': EXAMPLES*(2 if arm in KEEP else 1)})
        phases = ('segment1', 'segment2') if arm in KEEP else () if arm == 'SEG-04' else ('segment2',)
        m['runtime.accounting'] = {phase: {str(i): {'attention': EXAMPLES, 'ffn': EXAMPLES,
            'valid_tokens': EXAMPLES, 'reads': 0 if arm in ('SEG-01', 'SEG-07') else EXAMPLES,
            'writes': 0 if arm in ('SEG-01', 'SEG-07') else EXAMPLES} for i in (6, 13, 20, 27)} for phase in phases}
        if arm in KEEP:
            rank = 128 if arm == 'SEG-05' else 32
            m.update({'state.M_c_bytes': 4*(rank*rank+rank)*4, 'state.counts_bytes': 32,
                      'initial.identical_to_Base_seg': True})
            write(root/'initial-predictions.json', p); write(root/'initial-base-predictions.json', p)
        from v1_lora import LoRAConfig
        cfg = LoRAConfig(layers=(6, 13, 20, 27), rank=7, alpha=7) if arm == 'SEG-04' else config_for(arm)
        write(root/'adapter'/'config.json', asdict(cfg)); (root/'adapter'/'adapter.pt').write_bytes(b'synthetic checkpoint')
        write(root/'checkpoint.json', {name: reference(root/'adapter'/name) for name in ('config.json', 'adapter.pt')})
        with (root/'steps.jsonl').open('w') as f:
            for step in range(1, STEPS+1):
                f.write(json.dumps({'step': step, 'lr': lr*lr_factor(step-1, STEPS), 'input_tokens': 4 if arm in KEEP else 2})+'\n')
        import math
        for frac in (.25, .5, .75):
            step = math.ceil(STEPS*frac)
            write(root/f'dev-step-{step}.json', {'step': step, 'predictions': p})
    manifest = {'status': 'complete', 'contract': 'E-0025@1', 'arm': arm, 'seed': 0,
                'phase': 'SEG', 'stage': stage, 'metrics': m, 'validator': {'passed': True}}
    seal(root, manifest)
    return manifest


def seal(root, manifest):
    manifest['artifacts'] = [dict(reference(p), role=p.name) for p in root.iterdir() if p.is_file() and p.name != 'manifest.json']
    write(root/'manifest.json', manifest)


@pytest.mark.parametrize('arm', ['SEG-00', 'SEG-01', 'SEG-02', 'SEG-03', 'SEG-04', 'SEG-05', 'SEG-06', 'SEG-07'])
def test_complete_synthetic_formal_artifact_acceptance(tmp_path, monkeypatch, arm):
    root = tmp_path/arm; formal(root, arm, monkeypatch)
    m, refs, _ = a.validate(root, arm, SOURCE)
    assert m['arm'] == arm and refs['manifest']['sha256'] == reference(root/'manifest.json')['sha256']


@pytest.mark.parametrize('failure,match', [('source', 'source'), ('partial', 'complete'),
    ('short', 'budget'), ('reload', 'reload'), ('checkpoint', 'another run'),
    ('curve', 'paired'), ('lr', 'LR'), ('forward', 'phase accounting'), ('metric', 'raw predictions')])
def test_rejects_realistic_provenance_and_completeness_failures(tmp_path, monkeypatch, failure, match):
    root = tmp_path/'SEG-02'; m = formal(root, 'SEG-02', monkeypatch)
    if failure == 'source': m['metrics']['source.commit'] = 'c'*40
    elif failure == 'partial': m['status'] = 'partial'
    elif failure == 'short': m['metrics']['train.steps'] -= 1
    elif failure == 'reload': m['metrics']['reload.identical'] = False
    elif failure == 'lr': m['metrics']['learning_rate'] = .001
    elif failure == 'forward': m['metrics']['runtime.accounting'].pop('segment1')
    elif failure == 'metric': m['metrics']['dev.piqa.accuracy'] = .7
    elif failure == 'checkpoint':
        cp = a.read(root/'checkpoint.json'); cp['adapter.pt']['path'] = str(tmp_path/'other'/'adapter.pt')
        write(root/'checkpoint.json', cp)
    elif failure == 'curve':
        curve = a.read(root/'dev-step-3279.json'); curve['predictions'][0]['id'] = 'foreign'
        write(root/'dev-step-3279.json', curve)
    seal(root, m)
    with pytest.raises(ValueError, match=match): a.validate(root, 'SEG-02', SOURCE)


def test_changed_file_and_foreign_arm_checkpoint_are_not_accepted(tmp_path, monkeypatch):
    root = tmp_path/'SEG-02'; formal(root, 'SEG-02', monkeypatch)
    p = a.read(root/'predictions.json'); p[0]['prediction'] = 1; write(root/'predictions.json', p)
    with pytest.raises(ValueError, match='hash/size'): a.validate(root, 'SEG-02', SOURCE)
    with pytest.raises(ValueError, match='complete final'): a.validate(root, 'SEG-01', SOURCE)
