"""CPU unit tests for the shared V1 training mechanics."""
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

import v1_run
from v1_data import collate_encoded


class TokenLogits(nn.Module):
    """Make each position's logits a deterministic function of its input token."""
    def forward(self, input_ids, attention_mask=None, use_cache=False):
        logits = F.one_hot(input_ids, num_classes=7).float() * 2
        return SimpleNamespace(logits=logits)


def tensor_batch(encoded):
    values = collate_encoded(encoded, pad_token_id=0)
    return {key: torch.tensor(value, dtype=torch.long) for key, value in values.items()}


def test_token_losses_shifts_completion_labels_and_ignores_masked_positions():
    model = TokenLogits()
    batch = {
        'input_ids': torch.tensor([[0, 1, 2, 3], [4, 5, 6, 0]]),
        'attention_mask': torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]]),
        # The first label is deliberately supervised but must be excluded because
        # there is no preceding causal logit. The last padded label is ignored.
        'labels': torch.tensor([[4, -100, 2, 3], [6, 5, -100, -100]]),
    }
    losses, valid = v1_run.token_losses(model, batch)
    assert valid.tolist() == [[False, True, True], [True, False, False]]
    expected_logits = F.one_hot(torch.tensor([1, 2, 4]), num_classes=7).float() * 2
    expected_labels = torch.tensor([2, 3, 5])
    torch.testing.assert_close(losses, F.cross_entropy(expected_logits, expected_labels, reduction='none'))


class TinyCausalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(7, 5)
        self.projection = nn.Linear(5, 7)

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        return SimpleNamespace(logits=self.projection(self.embedding(input_ids)))


def test_microbatch_accumulation_matches_full_supervised_token_mean():
    examples = [
        {'input_ids': [1, 2, 3], 'labels': [-100, -100, 3]},
        {'input_ids': [2, 4, 5, 6, 1], 'labels': [-100, -100, 5, 6, 1]},
    ]
    assert [sum(label != -100 for label in e['labels'][1:]) for e in examples] == [1, 3]
    seed = 193
    micro = TinyCausalModel()
    torch.manual_seed(seed)
    full = TinyCausalModel()
    full.load_state_dict(micro.state_dict())
    denominator = 4
    accumulated = 0.
    for example in examples:
        loss, _ = v1_run.token_losses(micro, tensor_batch([example]))
        accumulated += loss.sum().item()
        (loss.sum() / denominator).backward()

    all_loss, valid = v1_run.token_losses(full, tensor_batch(examples))
    full_mean = all_loss.sum() / valid.sum()
    full_mean.backward()
    assert abs(accumulated / denominator - full_mean.item()) < 1e-7
    for micro_param, full_param in zip(micro.parameters(), full.parameters()):
        torch.testing.assert_close(micro_param.grad, full_param.grad, rtol=1e-6, atol=1e-7)


def test_lr_factor_warmup_and_cosine_endpoints():
    total = 80  # five-percent warmup is exactly four optimizer steps
    factors = [v1_run.lr_factor(step, total) for step in range(total)]
    assert factors[:4] == [0.25, 0.5, 0.75, 1.0]
    assert factors[4] < factors[3]
    assert all(left >= right for left, right in zip(factors[4:], factors[5:]))
    assert factors[-1] == 0
    assert v1_run.lr_factor(0, 1) == 1


def test_plan_batches_preserves_order_across_epochs_and_stops_at_full_batch(monkeypatch):
    rows_by_epoch = {
        0: [{'task': 'piqa', 'id': name} for name in ('a', 'b', 'c')],
        1: [{'task': 'boolq', 'id': name} for name in ('a', 'b', 'c')],
    }
    monkeypatch.setattr(v1_run, 'build_training_stream', lambda splits, epoch: rows_by_epoch[epoch])

    def encode(tok, row):
        return {'task': row['task'], 'id': row['id'], 'input_ids': [1], 'labels': [-100, 2]}

    monkeypatch.setattr(v1_run, 'encode_training_example', encode)
    batches = v1_run.plan_batches({}, tok=None, budget=4, batch_size=2, max_epochs=3)
    assert [[(e['id'], e['epoch']) for e in batch] for batch in batches] == [
        [('a', 0), ('b', 0)], [('c', 0), ('a', 1)]]
