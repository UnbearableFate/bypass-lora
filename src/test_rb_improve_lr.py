import json
from types import SimpleNamespace

import pytest

from rb_improve_lr import lock, verify_lock


def candidate(root, lr, nll=1., source='a'*40, failure_kind=None):
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
    if failure_kind is None:
        (out/'predictions.json').write_text(json.dumps(rows))
    else:
        metrics = {key: value for key, value in metrics.items() if not key.startswith('calibration.')}
        metrics.update({'failed': True, 'failure_kind': failure_kind,
            'failure_reason': 'injected nonfinite candidate', 'calibration.nll': None})
        if failure_kind == 'training_nonfinite':
            metrics.update({'train.steps': 3, 'train.input_tokens': 1536})
    (out/'config.json').write_text(json.dumps({'arm': 'L-56', 'stage': 'calibration', 'phase': 'V0',
        'seed': 0, 'lr': lr, 'smoke_steps': None, 'eval_limit': None}))
    (out/'steps.jsonl').write_text(''.join(json.dumps({'step': i+1, 'input_tokens': 512 if i < 390 else 320})+'\n'
        for i in range(metrics['train.steps'])))
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


@pytest.mark.parametrize('failure_kind', ['training_nonfinite', 'evaluation_nonfinite'])
def test_nonfinite_candidate_is_excluded_and_lock_reconstructs(tmp_path, failure_kind):
    paths = [candidate(tmp_path, 1e-4, 1.1), candidate(tmp_path, 3e-4, 1.),
             candidate(tmp_path, 1e-3, failure_kind=failure_kind)]
    value = lock(paths)
    assert value['locked_lrs'] == {'RBc-UP': 3e-4}
    assert value['candidates']['0.001'] is None and value['required_extension'] is None
    saved = tmp_path/'lr-lock.json'
    saved.write_text(json.dumps(value))
    assert verify_lock(saved) == value


def test_failed_low_endpoint_does_not_turn_the_middle_LR_into_a_grid_endpoint(tmp_path):
    paths = [candidate(tmp_path, 1e-4, failure_kind='training_nonfinite'),
             candidate(tmp_path, 3e-4, 1.), candidate(tmp_path, 1e-3, 1.1)]
    assert lock(paths)['required_extension'] is None


def test_first_step_divergence_can_bind_an_empty_step_log(tmp_path):
    paths = [candidate(tmp_path, 1e-4, 1.1), candidate(tmp_path, 3e-4, 1.),
             candidate(tmp_path, 1e-3, failure_kind='training_nonfinite')]
    (paths[-1].parent/'steps.jsonl').write_text('')
    value = lock(paths)
    saved = tmp_path/'lr-lock.json'
    saved.write_text(json.dumps(value))
    assert verify_lock(saved) == value


def test_finite_low_endpoint_still_requires_one_extension_with_an_excluded_candidate(tmp_path):
    paths = [candidate(tmp_path, 1e-4, 1.), candidate(tmp_path, 3e-4, failure_kind='training_nonfinite'),
             candidate(tmp_path, 1e-3, 1.1)]
    assert lock(paths, final=False)['required_extension'] == 3e-5
    paths.append(candidate(tmp_path, 3e-5, failure_kind='evaluation_nonfinite'))
    assert lock(paths)['locked_lrs'] == {'RBc-UP': 1e-4}


def test_all_nonfinite_candidates_and_engineering_failures_stop_selection(tmp_path):
    paths = [candidate(tmp_path, lr, failure_kind='training_nonfinite') for lr in (1e-4, 3e-4, 1e-3)]
    with pytest.raises(ValueError, match='no finite'):
        lock(paths)
    value = json.loads(paths[0].read_text())
    value['metrics']['failure_kind'] = 'OOM'
    paths[0].write_text(json.dumps(value))
    with pytest.raises(ValueError, match='only typed'):
        lock(paths)


@pytest.mark.parametrize('kind', ['training_nonfinite', 'evaluation_nonfinite', 'engineering'])
def test_calibration_producer_excludes_only_typed_nonfinite_failures(tmp_path, monkeypatch, kind):
    import torch
    import rb_improve_train as runner
    from rb_tap_check import initial_state
    from v1_run import EvaluationNonfinite, TrainingDiverged

    class ToyAdapter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Linear(2, 2, bias=False).requires_grad_(False)
            self.new = torch.nn.Parameter(torch.zeros(7))
            self._targets = {(0, 'mlp.up_proj'): None}
        def adapter_parameters(self):
            yield self.new
    model = ToyAdapter()
    monkeypatch.setitem(runner.PARAMETERS, 'L-56', 7)
    monkeypatch.setattr(runner, 'random_reference', lambda *args: initial_state(model)[1])
    monkeypatch.setattr(runner, 'make_model', lambda *args, **kwargs: model)
    monkeypatch.setattr(runner.AutoTokenizer, 'from_pretrained', lambda *args, **kwargs:
        SimpleNamespace(eos_token_id=0, pad_token_id=0))
    monkeypatch.setattr(runner, 'load_data', lambda *args: ({}, []))
    monkeypatch.setattr(runner, 'calibration_examples', lambda *args:
        ([], {'examples_sha256': 'f'*64, 'input_tokens': 0}))
    example = {'task': 'piqa', 'id': 1, 'epoch': 0, 'epoch_end': False, 'input_ids': [1, 2]}
    monkeypatch.setattr(runner, 'plan_batches', lambda *args: [[example, example]])
    def injected_train(*args, **kwargs):
        if kind == 'training_nonfinite':
            raise TrainingDiverged('injected nonfinite loss', {'train.input_tokens': 2})
        if kind == 'engineering':
            raise RuntimeError('injected OOM')
        return {'params.with_grad': 7}
    def injected_eval(*args, **kwargs):
        raise EvaluationNonfinite('injected nonfinite candidate likelihood')
    monkeypatch.setattr(runner, 'train', injected_train)
    monkeypatch.setattr(runner, 'evaluate', injected_eval)
    monkeypatch.setenv('SR_CODE_COMMIT', 'a'*40)
    monkeypatch.setenv('SR_SPEC_DIGEST', 'blake2b:'+'b'*64)
    split = tmp_path/'split.json'
    split.write_text('{}')
    args = SimpleNamespace(output=tmp_path/'run', stage='calibration', arm='L-56', lr=.001,
        smoke_steps=None, eval_limit=None, data_dir=tmp_path, split_manifest=split)
    if kind == 'engineering':
        with pytest.raises(RuntimeError, match='OOM'):
            runner.run(args)
    else:
        runner.run(args)
    value = json.loads((args.output/'manifest.json').read_text())
    assert value['metrics']['failed'] is True and value['metrics']['dev_read'] is False
    if kind == 'engineering':
        assert value['status'] == 'partial' and value['validator']['passed'] is False
    else:
        assert value['status'] == 'complete' and value['validator']['passed'] is True
        assert value['metrics']['failure_kind'] == kind
        assert value['metrics']['calibration.nll'] is None
