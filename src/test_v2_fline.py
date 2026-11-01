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
