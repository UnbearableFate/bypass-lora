import json

import pytest

from rb_improve_lr import lock, verify_lock


def candidate(root, lr, nll=1., source='a'*40):
    out = root/f'{lr:g}'
    out.mkdir()
    metrics = {'learning_rate': lr, 'source.commit': source, 'environment.lock_sha256': 'b'*64,
        'split_manifest.sha256': 'c'*64, 'plan.order_sha256': 'd'*64, 'plan.steps': 391,
        'plan.input_tokens': 200000, 'train.input_tokens': 200000, 'train.steps': 391,
        'init.parameters.sha256': 'e'*64, 'failed': False, 'dev_read': False, 'calibration.nll': nll}
    rows = []
    for task in ('piqa', 'boolq'):
        for i in range(2048):
            rows.append({'task': task, 'split': 'calibration', 'id': i, 'gold': 0, 'prediction': 0,
                'scores': [-nll, -nll-1], 'gold_nll_sum': nll, 'gold_loss_tokens': 1})
        metrics.update({f'calibration.{task}.nll': nll, f'calibration.{task}.accuracy': 1.,
            f'calibration.{task}.examples': 2048, f'calibration.{task}.loss_tokens': 2048})
    (out/'predictions.json').write_text(json.dumps(rows))
    (out/'config.json').write_text(json.dumps({'arm': 'L-56', 'stage': 'calibration', 'phase': 'V0',
        'seed': 0, 'lr': lr, 'smoke_steps': None, 'eval_limit': None}))
    (out/'steps.jsonl').write_text(''.join(json.dumps({'step': i+1, 'input_tokens': 512 if i < 390 else 320})+'\n'
        for i in range(391)))
    path = out/'manifest.json'
    path.write_text(json.dumps({'status': 'complete', 'contract': 'E-0029@1', 'arm': 'L-56',
        'family': 'RBc-UP', 'stage': 'calibration', 'phase': 'V0', 'seed': 0,
        'validator': {'passed': True}, 'metrics': metrics}))
    return path


def test_lr_lock_selects_smallest_LR_within_one_percent_and_rebuilds(tmp_path):
    paths = [candidate(tmp_path, 1e-4, 1.009), candidate(tmp_path, 3e-4, 1.), candidate(tmp_path, 1e-3, 1.02)]
    value = lock(paths)
    assert value['locked_lrs'] == {'RBc-UP': 1e-4}
    path = tmp_path/'lr-lock.json'
    path.write_text(json.dumps(value))
    assert verify_lock(path) == value
    predictions = paths[0].parent/'predictions.json'
    predictions.write_text(predictions.read_text()+' ')
    with pytest.raises(ValueError, match='reconstruct'):
        verify_lock(path)


def test_lr_endpoint_requires_exactly_one_extension(tmp_path):
    paths = [candidate(tmp_path, 1e-4, .9), candidate(tmp_path, 3e-4, 1.), candidate(tmp_path, 1e-3, 1.1)]
    assert lock(paths, final=False)['required_extension'] == 3e-5
    with pytest.raises(ValueError, match='extension'):
        lock(paths)
    paths.append(candidate(tmp_path, 3e-5, .91))
    assert lock(paths)['locked_lrs'] == {'RBc-UP': 1e-4}


def test_lr_candidates_reject_changed_source_and_spurious_dev_metrics(tmp_path):
    paths = [candidate(tmp_path, 1e-4), candidate(tmp_path, 3e-4, source='f'*40), candidate(tmp_path, 1e-3)]
    with pytest.raises(ValueError, match='mismatch'):
        lock(paths)
    value = json.loads(paths[1].read_text())
    value['metrics']['source.commit'] = 'a'*40
    value['metrics']['dev.piqa.accuracy'] = .8
    paths[1].write_text(json.dumps(value))
    with pytest.raises(ValueError, match='dev'):
        lock(paths)
