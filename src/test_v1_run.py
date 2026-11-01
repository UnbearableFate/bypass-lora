"""CPU unit tests for the shared V1 training mechanics."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

import v1_run
import v1_smoke
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


def test_microbatch_accumulation_matches_equal_example_mean():
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
    accumulated = 0.
    for example in examples:
        loss, _ = v1_run.token_losses(micro, tensor_batch([example]))
        accumulated += loss.mean().item()
        v1_run.example_mean_objective(loss, len(examples)).backward()

    all_loss, valid = v1_run.token_losses(full, tensor_batch(examples))
    full_mean = (all_loss[:1].mean() + all_loss[1:].mean()) / 2
    full_mean.backward()
    assert abs(accumulated / 2 - full_mean.item()) < 1e-7
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


class NonfiniteCausalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.))

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        logits = self.weight * torch.full((*input_ids.shape, 7), float('nan'))
        return SimpleNamespace(logits=logits)


def test_train_records_consumption_and_skips_optimizer_step_on_nonfinite_loss(
        monkeypatch, tmp_path):
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    monkeypatch.setattr(v1_run, 'batch_tensors', lambda encoded, pad_id: tensor_batch(encoded))
    optimizer_steps = []
    original_step = torch.optim.AdamW.step

    def count_steps(optimizer, *args, **kwargs):
        optimizer_steps.append(True)
        return original_step(optimizer, *args, **kwargs)

    monkeypatch.setattr(torch.optim.AdamW, 'step', count_steps)
    examples = [
        {'input_ids': [1, 2, 3], 'labels': [-100, -100, 3], 'epoch': 0},
        {'input_ids': [2, 3], 'labels': [-100, 3], 'epoch': 0},
    ]
    with pytest.raises(v1_run.TrainingDiverged, match='nonfinite loss') as exc_info:
        v1_run.train(NonfiniteCausalModel(), [examples], SimpleNamespace(pad_token_id=0),
                     lr=1e-3, output=tmp_path)

    consumed = {'train.input_tokens': 3, 'train.loss_tokens': 1,
                'train.padded_tokens': 0, 'train.examples': 1}
    assert exc_info.value.metrics == consumed
    assert json.loads((tmp_path / 'usage.json').read_text()) == consumed
    assert optimizer_steps == []


def test_main_records_training_divergence_as_completed_and_accepts_next_run(
        monkeypatch, tmp_path):
    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 0

    monkeypatch.setattr(v1_run.AutoTokenizer, 'from_pretrained', lambda *a, **k: Tokenizer())
    monkeypatch.setattr(v1_run, 'prepare', lambda data_dir: {'metadata': {'split_ids': {}}})
    monkeypatch.setattr(v1_run, 'plan_batches', lambda *a, **k: [[
        {'task': 'piqa', 'id': 'row-1', 'epoch': 0, 'input_ids': [1, 2],
         'labels': [-100, 2]}]])
    monkeypatch.setattr(v1_run, 'make_model', lambda *a, **k: nn.Linear(1, 1))
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    monkeypatch.setenv('SR_SPEC_DIGEST', 'unit-test-digest')

    def diverge(*args, **kwargs):
        raise v1_run.TrainingDiverged('nonfinite gradient norm', {
            'train.input_tokens': 2, 'train.loss_tokens': 1,
            'train.padded_tokens': 0, 'train.examples': 1})

    monkeypatch.setattr(v1_run, 'train', diverge)
    manifests = []
    for index, lr in enumerate((1e-3, 5e-4)):
        output = tmp_path / f'run-{index}'
        monkeypatch.setattr(sys, 'argv', [
            'v1_run.py', '--arm', 'lora', '--data-dir', str(tmp_path),
            '--output', str(output), '--smoke-steps', '1', '--lr', str(lr)])
        v1_run.main()
        manifest = json.loads((output / 'manifest.json').read_text())
        manifests.append(manifest)

    assert len(manifests) == 2
    for manifest in manifests:
        assert manifest['status'] == 'complete'
        assert manifest['validator']['passed'] is True
        assert manifest['metrics']['train.diverged'] is True
        assert manifest['metrics']['train.failure_reason'] == 'nonfinite gradient norm'
        assert manifest['metrics']['train.token_overshoot'] == 0


@pytest.mark.parametrize('bad_loss', [float('nan'), float('inf')])
def test_evaluate_rejects_nonfinite_nongold_candidate_losses(monkeypatch, bad_loss):
    class Tokenizer:
        pad_token_id = 0

    monkeypatch.setattr(v1_run, 'encode_candidates',
                        lambda tok, row: [{'which': 0}, {'which': 1}])
    monkeypatch.setattr(v1_run, 'batch_tensors', lambda candidates, pad_id: candidates[0]['which'])

    def loss_for_candidate(model, candidate):
        value = 1.25 if candidate == 0 else bad_loss
        return torch.tensor([value]), torch.tensor([True])

    monkeypatch.setattr(v1_run, 'token_losses', loss_for_candidate)
    splits = {'splits': {
        'piqa': {'calibration': [{'id': 'nongold-bad', 'label': 0}], 'dev': []},
        'boolq': {'calibration': [], 'dev': []},
    }}
    with pytest.raises(v1_run.EvaluationNonfinite, match='calibration/piqa/nongold-bad'):
        v1_run.evaluate(TinyCausalModel(), splits, Tokenizer())


class ExportableModel:
    def export_adapter(self, directory):
        Path(directory).mkdir(parents=True, exist_ok=True)


class MainTokenizer:
    pad_token_id = 0
    eos_token_id = 0


def patch_main_inputs(monkeypatch, tmp_path):
    monkeypatch.setattr(v1_run.AutoTokenizer, 'from_pretrained', lambda *a, **k: MainTokenizer())
    monkeypatch.setattr(v1_run, 'prepare', lambda data_dir: {'metadata': {'split_ids': {}}})
    monkeypatch.setattr(v1_run, 'plan_batches', lambda *a, **k: [[
        {'task': 'piqa', 'id': 'row-1', 'epoch': 0, 'input_ids': [1, 2],
         'labels': [-100, 2]}]])
    monkeypatch.setattr(v1_run, 'make_model', lambda *a, **k: ExportableModel())
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: None)
    monkeypatch.setenv('SR_SPEC_DIGEST', 'unit-test-digest')


def set_main_argv(monkeypatch, tmp_path, arm, output_name):
    output = tmp_path / output_name
    monkeypatch.setattr(sys, 'argv', [
        'v1_run.py', '--arm', arm, '--data-dir', str(tmp_path), '--output', str(output),
        '--smoke-steps', '1', '--lr', '1e-3'])
    return output


def test_main_adapted_evaluation_nonfinite_completes_failed_lr_and_allows_next_candidate(
        monkeypatch, tmp_path):
    patch_main_inputs(monkeypatch, tmp_path)

    def finite_train(model, batches, tok, lr, output):
        usage = {'train.input_tokens': 81, 'train.loss_tokens': 17,
                 'train.padded_tokens': 0, 'train.examples': 2}
        (output / 'usage.json').write_text(json.dumps(usage) + '\n')
        result = dict(usage)
        result.update({'train.steps': 1, 'train.seconds': 0.1,
                       'train.last_epoch': 0, 'train.peak_cuda_bytes': 0,
                       'train.tokens_per_second': 810.})
        return result

    monkeypatch.setattr(v1_run, 'train', finite_train)
    monkeypatch.setattr(v1_run, 'evaluate', lambda *a, **k: (_ for _ in ()).throw(
        v1_run.EvaluationNonfinite('calibration/piqa/nongold: nonfinite candidate loss')))
    results = []
    for index in range(2):
        output = set_main_argv(monkeypatch, tmp_path, 'lora', f'eval-fail-{index}')
        v1_run.main()
        assert json.loads((output / 'usage.json').read_text())['train.input_tokens'] == 81
        manifest = json.loads((output / 'manifest.json').read_text())
        results.append(manifest)

    assert len(results) == 2
    for manifest in results:
        metrics = manifest['metrics']
        assert manifest['status'] == 'complete' and manifest['validator']['passed'] is True
        assert metrics['train.input_tokens'] == 81 and metrics['train.loss_tokens'] == 17
        assert metrics['train.diverged'] is True and metrics['run.failed_lr'] is True
        assert metrics['failure.stage'] == 'evaluation'
        assert metrics['reload.identical'] is False
        assert 'reload.seconds' not in metrics


def test_main_base_evaluation_nonfinite_remains_partial_failure(monkeypatch, tmp_path):
    patch_main_inputs(monkeypatch, tmp_path)
    monkeypatch.setattr(v1_run, 'evaluate', lambda *a, **k: (_ for _ in ()).throw(
        v1_run.EvaluationNonfinite('calibration/piqa/base: nonfinite candidate loss')))
    output = set_main_argv(monkeypatch, tmp_path, 'base', 'base-eval-fail')
    with pytest.raises(v1_run.EvaluationNonfinite, match='nonfinite candidate loss'):
        v1_run.main()
    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest['status'] == 'partial' and manifest['validator']['passed'] is False
    assert manifest['metrics']['failure.stage'] == 'evaluation'
    assert 'train.diverged' not in manifest['metrics']


def test_main_finite_checkpoint_mismatch_remains_partial_assertion(monkeypatch, tmp_path):
    patch_main_inputs(monkeypatch, tmp_path)
    monkeypatch.setattr(v1_run, 'train', lambda *a, **k: {
        'train.input_tokens': 2, 'train.loss_tokens': 1, 'train.examples': 1,
        'train.steps': 1, 'train.seconds': .1, 'train.last_epoch': 0,
        'train.peak_cuda_bytes': 0, 'train.tokens_per_second': 20.})
    calls = []

    def finite_but_mismatching(*args, **kwargs):
        calls.append(True)
        return {'calibration.piqa.nll': 1.0}, [{'prediction': len(calls)}]

    monkeypatch.setattr(v1_run, 'evaluate', finite_but_mismatching)
    output = set_main_argv(monkeypatch, tmp_path, 'lora', 'reload-mismatch')
    with pytest.raises(AssertionError, match='checkpoint/repeated Base evaluation'):
        v1_run.main()
    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest['status'] == 'partial' and manifest['validator']['passed'] is False
    assert manifest['metrics']['reload.identical'] is False
    assert 'train.diverged' not in manifest['metrics']


def finite_smoke_manifest():
    metrics = {'train.steps': 5, 'reload.identical': True}
    for split in ('calibration', 'dev'):
        metrics[f'{split}.nll'] = 1.0
        for task in ('piqa', 'boolq'):
            metrics[f'{split}.{task}.nll'] = 1.0
            metrics[f'{split}.{task}.accuracy'] = .5
    return {'status': 'complete', 'validator': {'passed': True}, 'metrics': metrics}


def test_valid_smoke_requires_complete_finite_evaluation_and_exact_reload():
    manifest = finite_smoke_manifest()
    assert v1_smoke.valid_smoke(manifest, steps=5)

    invalid_manifests = []
    diverged = finite_smoke_manifest()
    diverged['metrics']['train.diverged'] = True
    invalid_manifests.append(diverged)
    for key, value in (('reload.identical', None), ('reload.identical', False),
                       ('train.steps', None), ('train.steps', 4),
                       ('calibration.piqa.nll', float('nan')),
                       ('dev.boolq.accuracy', float('inf'))):
        invalid = finite_smoke_manifest()
        if value is None:
            invalid['metrics'].pop(key)
        else:
            invalid['metrics'][key] = value
        invalid_manifests.append(invalid)

    assert all(not v1_smoke.valid_smoke(candidate, steps=5)
               for candidate in invalid_manifests)
