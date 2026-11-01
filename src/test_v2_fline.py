"""F-line regression checks, executed only in PBS."""
import copy
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from v2_data import MODEL, REVISION, identity, make_episode, make_name_pools
from v2_eval import prompt_tokens, left_pad
from v2_fline import answer_losses
from v2_schedule import make_schedule, lr_multiplier, select_learning_rate


def test_answer_only_objective():
    logits = torch.randn(2, 4, 11, requires_grad=True)
    class Model:
        def __call__(self, **kwargs):
            assert kwargs['logits_to_keep'] == 1 and not kwargs['use_cache']
            return SimpleNamespace(logits=logits)
    targets = torch.tensor([3, 7])
    losses = answer_losses(Model(), {}, targets)
    torch.testing.assert_close(losses, F.cross_entropy(logits[:, -1], targets, reduction='none'))
    losses.mean().backward()
    assert logits.grad[:, :-1].count_nonzero() == 0
    assert logits.grad[:, -1].count_nonzero() > 0


def test_explicit_train_split():
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    pools = make_name_pools(tok)
    row = make_episode(tok, pools, identity('train', 0))
    assert prompt_tokens(tok, row, 'F', split='train') == row['support_input_ids'] + row['question_input_ids']
    with pytest.raises(ValueError):
        prompt_tokens(tok, row, 'F')
    row = copy.deepcopy(row)
    row['identity']['split'] = 'confirmation'
    for split in ('train', 'calibration', 'dev', 'confirmation'):
        with pytest.raises(ValueError):
            prompt_tokens(tok, row, 'F', split=split)


def test_schedule_complete_step_and_exhaustion():
    rows = [dict(episode_id=f'row{i}', support_input_ids=[1] * (i + 1), question_input_ids=[2]) for i in range(16)]
    schedule = make_schedule(rows, 101)
    assert all(len(b) == 8 for b in schedule['batches'])
    assert schedule['tokens'] >= 101
    previous = sum(len(rows[i]['support_input_ids']) + 1 for batch in schedule['batches'][:-1] for i in batch)
    assert previous < 101
    assert schedule == make_schedule(rows, 101)
    full = make_schedule(rows, 100000)
    assert full['steps'] == 6
    assert full['order_sha256'] == schedule['order_sha256']
    for epoch in range(3):
        assert sorted(full['order'][epoch*16:(epoch+1)*16]) == list(range(16))
    assert lr_multiplier(0, 100, 5) == .2
    assert lr_multiplier(4, 100, 5) == 1.
    assert lr_multiplier(99, 100, 5) == 0.


def test_lr_finite_and_tie_selection():
    rows = [{'lr': .001, 'nll': 2.}, {'lr': .0001, 'nll': 2.009},
            {'lr': .00001, 'nll': 1., 'finite_loss': False}]
    assert select_learning_rate(rows)['lr'] == .0001
    with pytest.raises(ValueError):
        select_learning_rate([{'lr': .1, 'nll': float('nan')}])


@pytest.mark.parametrize('stage,nonfinite,smoke,scientific', [
    ('reload_evaluation', float('nan'), False, False),
    ('reload_evaluation', float('inf'), False, False),
    ('original_evaluation', float('nan'), False, True),
    ('training', float('inf'), False, True),
    ('training', float('nan'), True, False),
    ('original_evaluation', float('inf'), True, False),
    ('reload_mismatch', None, False, False),
    ('finite', None, True, False),
])
def test_failure_manifest_boundaries(monkeypatch, tmp_path, stage, nonfinite, smoke, scientific):
    import json
    import sys
    import v2_fline as module
    monkeypatch.setenv('SR_SPEC_DIGEST', 'test-spec')
    argv = ['v2_fline', '--data-dir', str(tmp_path), '--output', str(tmp_path / 'out'), '--lr', '.0003']
    if smoke:
        argv += ['--smoke']
    monkeypatch.setattr(sys, 'argv', argv)
    monkeypatch.setattr(module, 'load_data', lambda *a: [{'query': {'answer_token_id': 1}}] * 32)
    monkeypatch.setattr(module, 'make_schedule', lambda *a: {'batches': [[0]*8]*4, 'warmup_steps': 1})
    monkeypatch.setattr(module.AutoTokenizer, 'from_pretrained', lambda *a, **kw: None)
    class Model:
        def export_adapter(self, path):
            pass
    monkeypatch.setattr(module, 'make_model', lambda *a: Model())
    for name in ('reset_peak_memory_stats', 'empty_cache'):
        monkeypatch.setattr(module.torch.cuda, name, lambda: None)
    monkeypatch.setattr(module.torch.cuda, 'max_memory_allocated', lambda: 0)
    def fail():
        module.score_logits(torch.tensor([[0., nonfinite]]), torch.tensor([1]))
    def train(*args):
        if stage == 'training':
            fail()
        return {'train.steps': 4, 'train.tokens': 2681, 'train.seconds': 1.}
    monkeypatch.setattr(module, 'train', train)
    def evaluate(model, tok, rows, split, constant, output):
        reload = 'reloaded' in output.name
        if (stage == 'reload_evaluation' and reload) or (stage == 'original_evaluation' and not reload):
            fail()
        record = {'prediction': 2 if stage == 'reload_mismatch' and reload else 1}
        return {'all.em': .5, 'all.prior_em': .1, 'all.nll': 1.}, [record]
    monkeypatch.setattr(module, 'evaluate', evaluate)
    if scientific or stage == 'finite':
        module.main()
    else:
        with pytest.raises((ValueError, AssertionError)):
            module.main()
    manifest = json.loads((tmp_path / 'out' / 'manifest.json').read_text())
    metrics = manifest['metrics']
    expected_pass = scientific or stage == 'finite'
    assert manifest['validator']['passed'] is expected_pass
    assert manifest['status'] == ('complete' if expected_pass else 'partial')
    assert metrics.get('failed_lr', False) is scientific
    if stage != 'finite':
        assert metrics['failure.stage'] == ('reload_evaluation' if stage == 'reload_mismatch' else stage)
    if stage.startswith('reload'):
        assert metrics['train.steps'] == 4 and metrics['train.tokens'] == 2681
        assert 'error' in metrics
    if stage == 'finite':
        assert metrics['reload.identical']


def test_endpoint_extension_and_calibration_only_selection():
    from v2_fline_select import select
    def report(lr, nll):
        return {'lr': lr, 'smoke': False, 'tokens_budget': 200000, 'eval_batch': 16,
                'metrics': {'failed_lr': False, 'train.steps': 300, 'train.tokens': 200170,
                            'calibration.all.examples': 2048, 'reload.identical': True,
                            'calibration.all.nll': nll, 'dev.all.nll': -100.}}
    rows = [report(.0001, 3.), report(.0003, 2.), report(.001, 1.)]
    pending = select(rows)
    assert pending['extension_required'] and pending['extension_lr'] == .003
    final = select(rows, report(.003, .5))
    assert final['selection_final'] and not final['extension_required'] and final['selected_lr'] == .003
    rows[0]['metrics']['calibration.all.nll'] = .5
    assert select(rows)['extension_lr'] == .0001 / 3
    rows[0]['smoke'] = True
    with pytest.raises(ValueError):
        select(rows)
