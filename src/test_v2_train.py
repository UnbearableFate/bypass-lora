"""V2 schedule, evaluation composition and checkpoint contract regressions.

Execute these tests in the registered GPU allocation, including the CPU cases.
"""
import hashlib
import json
import math
from pathlib import Path
import random
from types import SimpleNamespace

import pytest
import torch
from transformers import AutoTokenizer

from calibrate import MODEL, REVISION, load_base
from v1_data import derive_seed
from v2_memory import CONTROLS, collate, config, evaluate_batch
from v2_run import fixture
from v2_schedule import load_data, lr_multiplier, make_schedule, select_learning_rate
from v2_train import adapted, evaluate


def schedule_rows():
    return [{'episode_id': f'episode-{i}', 'support_input_ids': [1]*(i+1),
             'question_input_ids': [2]*(i % 3+1), 'query': {'answer_token_id': 99}}
            for i in range(16)]


def test_schedule_three_seeded_epochs_hash_and_first_complete_budget_crossing():
    rows = schedule_rows()
    expected = []
    for epoch in range(3):
        indices = list(range(len(rows)))
        random.Random(derive_seed(20261029, 'concat_loop_v2', 'V2',
                                 f'train-order-{epoch}')).shuffle(indices)
        expected.extend(indices)
    costs = [len(rows[i]['support_input_ids'])+len(rows[i]['question_input_ids'])
             for i in expected]
    digest = hashlib.sha256(''.join(rows[i]['episode_id']+'\n' for i in expected).encode()).hexdigest()
    # Stop at equality, then exercise one-token-over-boundary overshoot.
    for budget, steps in ((1, 1), (sum(costs[:8]), 1), (sum(costs[:8])+1, 2),
                          (sum(costs)+1, 6)):
        plan = make_schedule(rows, budget)
        assert plan == make_schedule(rows, budget)
        assert plan['order'] == expected
        assert plan['order_sha256'] == digest
        assert plan['batches'] == [expected[i:i+8] for i in range(0, steps*8, 8)]
        assert plan['steps'] == steps
        assert plan['tokens'] == sum(costs[:steps*8])
        assert plan['warmup_steps'] == math.ceil(.05*steps)
        assert plan['tokens'] >= budget or steps == 6
        assert sum(costs[:(steps-1)*8]) < budget
    assert all(sorted(expected[e*16:(e+1)*16]) == list(range(16)) for e in range(3))
    assert expected[:16] != expected[16:32]


@pytest.mark.parametrize('rows,budget', [([], 1), (schedule_rows()[:7], 1),
                                       (schedule_rows(), 0)])
def test_schedule_rejects_incomplete_or_empty_steps(rows, budget):
    with pytest.raises(ValueError):
        make_schedule(rows, budget)


def test_learning_rate_warmup_cosine_endpoints_and_selection_failures():
    factors = [lr_multiplier(step, 80, 4) for step in range(80)]
    assert factors[:4] == [.25, .5, .75, 1.]
    assert factors[-1] == 0
    assert all(a > b for a, b in zip(factors[3:], factors[4:]))
    assert lr_multiplier(0, 1, 1) == 1
    for args in ((-1, 80, 4), (80, 80, 4), (0, 80, 0), (0, 80, 81)):
        with pytest.raises(ValueError):
            lr_multiplier(*args)
    candidates = [{'lr': .003, 'nll': 100.}, {'lr': .001, 'nll': 100.49},
                  {'lr': .0003, 'nll': 100.5},
                  {'lr': .0001, 'nll': 1., 'finite_loss': False},
                  {'lr': .00001, 'nll': float('nan')}]
    assert select_learning_rate(candidates)['lr'] == .001
    assert select_learning_rate(list(reversed(candidates)))['lr'] == .001
    assert select_learning_rate([{'lr': .003, 'nll': 0.}, {'lr': .001, 'nll': 0.}])['lr'] == .001
    with pytest.raises(ValueError, match='all learning rates failed'):
        select_learning_rate([{'lr': .001, 'nll': float('inf')},
                              {'lr': .003, 'nll': 1., 'finite_loss': False}])


def test_forbidden_splits_rejected_before_reading_or_evaluating(monkeypatch, tmp_path):
    def forbidden_read(*args, **kwargs):
        raise AssertionError('forbidden split was opened')
    monkeypatch.setattr(Path, 'read_bytes', forbidden_read)
    with pytest.raises(ValueError, match='authorized'):
        load_data(tmp_path/'confirmation.jsonl', 'confirmation')
    with pytest.raises(ValueError, match='dev/calibration'):
        evaluate(None, [{}, {}], 0, tmp_path, 'confirmation')
    with pytest.raises(ValueError, match='split mismatch'):
        evaluate(None, [{'identity': {'split': 'train'}}]*2, 0, tmp_path, 'dev')


def test_formal_evaluation_uses_global_donors_across_batch_16_boundary(monkeypatch, tmp_path):
    import v2_train
    rows = [{'episode_id': f'dev-{i}', 'identity': {'split': 'dev'},
             'query': {'answer_token_id': i % 2, 'answer_text': 'Alice',
                       'head': 'Alice' if i == 0 else 'Bob', 'relation': 'is',
                       'target_fact_index': i % 3},
             'axes': {'support_order': 'natural'}, '_subject_token_id': 1}
            for i in range(17)]
    calls = []

    def fake_collate(selected, pad_id, device):
        return SimpleNamespace(rows=selected,
                               labels=torch.tensor([r['query']['answer_token_id'] for r in selected]),
                               question_mask=torch.ones(len(selected), 2, dtype=torch.bool))

    def fake_evaluate(model, batch, mode, donor_batch=None):
        assert mode == 'shuffled'
        calls.append(([r['episode_id'] for r in batch.rows],
                      [r['episode_id'] for r in donor_batch.rows]))
        logits = torch.tensor([[2., -1.]]).repeat(len(batch.rows), 1)
        return {'read.raw_norm': 2., 'address.entropy': .5,
                'read.normalized_error': 99., 'read.target_norm': 99.,
                'read.direction_error': 99., 'read.denominator.p50': 3.}, logits

    monkeypatch.setattr(v2_train, 'collate', fake_collate)
    monkeypatch.setattr(v2_train, 'evaluate_batch', fake_evaluate)
    metrics, predictions = evaluate(torch.nn.Linear(1, 1), rows, 0, tmp_path,
                                    'dev', controls=('shuffled',), prior_token=0)
    assert [len(recipients) for recipients, _ in calls] == [16, 1]
    assert calls[0][1] == ['dev-16']+[f'dev-{i}' for i in range(15)]
    assert calls[1] == (['dev-16'], ['dev-15'])
    result = metrics['shuffled']
    assert predictions['shuffled'] == [0]*17
    assert result['count'] == 17
    assert result['em'] == result['prior_em'] == 9/17
    expected_losses = torch.nn.functional.cross_entropy(
        torch.tensor([[2., -1.]]).repeat(17, 1), torch.tensor([i % 2 for i in range(17)]),
        reduction='none')
    assert result['nll'] == pytest.approx(float(expected_losses.mean()), abs=1e-6)
    assert result['answer_ne_subject']['count'] == 16
    assert sum(group['count'] for group in result['strata']['target_fact_position'].values()) == 17
    assert result['diagnostics']['read.raw_norm'] == 2.
    assert result['diagnostics']['address.target_available'] is False
    for key in ('read.normalized_error', 'read.direction_error', 'read.target_norm'):
        assert result['diagnostics'][key] == 99.
    records = [json.loads(line) for line in (tmp_path/'dev.shuffled.predictions.jsonl').read_text().splitlines()]
    assert [record['episode_id'] for record in records] == [r['episode_id'] for r in rows]


def test_training_uneven_microbatches_match_batch_eight_mean(monkeypatch, tmp_path):
    import v2_train

    class TinyTrainModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Identity()
            self.side = torch.nn.Linear(2, 3, bias=False)
            with torch.no_grad():
                self.side.weight.copy_(torch.tensor([[.1, .2], [-.2, .3], [.4, -.1]]))

        def adapter_parameters(self):
            return self.side.parameters()

        def adapters(self):
            return {14: SimpleNamespace(side=self.side)}

    rows = [{'features': [i/7, (i % 3)/2], 'label': i % 3, 'tokens': i+2}
            for i in range(8)]
    schedule = {'steps': 4, 'warmup_steps': 1,
                'batches': [list(range(8)), list(reversed(range(8)))]*2,
                'tokens': 4*sum(row['tokens'] for row in rows)}

    def fake_collate(selected, pad_id, device):
        return SimpleNamespace(features=torch.tensor([r['features'] for r in selected]),
                               labels=torch.tensor([r['label'] for r in selected]),
                               write_mask=torch.ones(len(selected), 2, dtype=torch.bool),
                               input_tokens=sum(r['tokens'] for r in selected))

    def fake_forward(model, batch):
        logits = model.side(batch.features)
        loss = torch.nn.functional.cross_entropy(logits, batch.labels)
        return (loss, logits,
                {14: {'side': {'reads': 0, 'writes': int(batch.write_mask.sum())}}},
                {14: {'side': {'reads': len(batch.labels), 'writes': 0}}})

    evaluations = []

    def fake_eval(model, dev, pad_id, output, split, controls, prior_token):
        evaluations.append((split, controls))
        return {}, {}

    monkeypatch.setattr(v2_train, 'collate', fake_collate)
    monkeypatch.setattr(v2_train, 'train_forward', fake_forward)
    monkeypatch.setattr(v2_train, 'evaluate', fake_eval)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    monkeypatch.setattr(torch.cuda, 'reset_peak_memory_stats', lambda: None)
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda: 0)
    states, histories = [], []
    for microbatch_size in (8, 3, 1):
        model = TinyTrainModel()
        output = tmp_path/str(microbatch_size)
        output.mkdir()
        result = v2_train.train(model, rows, schedule, 0, .01, microbatch_size,
                                output, dev=[], prior_token=None)
        assert result['steps'] == 4 and result['tokens'] == schedule['tokens']
        states.append(model.side.weight.detach().clone())
        histories.append([json.loads(line) for line in (output/'train.jsonl').read_text().splitlines()])
    for state in states[1:]:
        torch.testing.assert_close(state, states[0], atol=1e-7, rtol=1e-6)
    for history in histories[1:]:
        for actual, expected in zip(history, histories[0]):
            assert actual['tokens'] == expected['tokens']
            assert actual['lr'] == expected['lr']
            assert actual['loss'] == pytest.approx(expected['loss'], abs=1e-6)
            assert actual['gradient_norm'] == pytest.approx(expected['gradient_norm'], abs=1e-6)
    assert evaluations == [('dev', ('normal', 'zero', 'shuffled'))]*9


@pytest.fixture(scope='module')
def base():
    return load_base()


@pytest.fixture(scope='module')
def varied_rows():
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    episodes = fixture(tok, n_facts=2) + fixture(tok, n_facts=3)[:1]
    for i, row in enumerate(episodes):
        row['episode_id'] = f'composition-{i}'
        row['question_input_ids'] = tok.encode('Please '*i, add_special_tokens=False) + row['question_input_ids']
        row['query']['target_fact_index'] = i % len(row['facts'])
    assert len({len(r['support_input_ids']) for r in episodes}) > 1
    assert len({len(r['question_input_ids']) for r in episodes}) == len(episodes)
    return episodes, tok.pad_token_id


def nonzero_fusion(model):
    with torch.no_grad():
        for parameter in model.adapters()[14].side.fusion.parameters():
            parameter.normal_(std=.03)


@pytest.mark.parametrize('backend', ['kernel_matrix', 'soft_slot'])
@torch.no_grad()
def test_evaluation_composition_with_fixed_donors_and_nonzero_fusion(base, varied_rows, backend,
                                                                  record_property):
    rows, pad_id = varied_rows
    tolerance = json.loads((Path(__file__).parent/'tolerances.json').read_text())
    donors = [rows[-1], rows[0], rows[1]]
    with adapted(base, cfg=config('qknorm_pre_rope', backend, 'F_out')) as model:
        model.eval()
        nonzero_fusion(model)
        batch = collate(rows, pad_id)
        for mode in CONTROLS:
            kwargs = {'donor_batch': collate(donors, pad_id)} if mode == 'shuffled' else {}
            details, together = evaluate_batch(model, batch, mode, **kwargs)
            separate = []
            for index, row in enumerate(rows):
                kwargs = {'donor_batch': collate([donors[index]], pad_id)} if mode == 'shuffled' else {}
                _, logits = evaluate_batch(model, collate([row], pad_id), mode, **kwargs)
                separate.append(logits)
            separate = torch.cat(separate)
            error = float((together-separate).abs().max())
            record_property(f'{backend}.{mode}.composition_max_abs_error', error)
            torch.testing.assert_close(together, separate, atol=tolerance['bf16_atol'],
                                       rtol=tolerance['bf16_rtol'])
            assert math.isfinite(details['nll'])
            if mode == 'shuffled':
                assert details['address.target_available'] is False
                assert 'address.target_weight' not in details
            if mode == 'normal':
                normal = together
            elif mode == 'zero':
                assert not torch.equal(normal, together), 'fixture must exercise memory in logits'
        with pytest.raises(ValueError, match='differ'):
            evaluate_batch(model, batch, 'shuffled', donor_batch=batch)
        with pytest.raises(ValueError, match='shuffled mode'):
            evaluate_batch(model, batch, 'normal', donor_batch=collate(donors, pad_id))
        model.train()
        with pytest.raises(ValueError, match='eval'):
            evaluate_batch(model, batch, 'oracle_value')


@pytest.mark.parametrize('fusion', ['F_out', 'C_up', 'C_dual'])
@pytest.mark.parametrize('backend', ['kernel_matrix', 'soft_slot'])
@torch.no_grad()
def test_saved_adapter_reload_preserves_all_tensors_config_and_predictions(base, varied_rows,
                                                                        tmp_path, fusion, backend):
    rows, pad_id = varied_rows
    batch = collate(rows, pad_id)
    cfg = config('qknorm_pre_rope', backend, fusion)
    checkpoint = tmp_path/'adapter'
    with adapted(base, cfg=cfg) as model:
        model.eval()
        nonzero_fusion(model)
        _, before = evaluate_batch(model, batch)
        tensors = {name: value.detach().clone()
                   for name, value in model.adapters()[14].side.state_dict().items()}
        model.export_adapter(checkpoint)
    with adapted(base, checkpoint=checkpoint) as reloaded:
        reloaded.eval()
        assert reloaded.config == cfg
        assert all(p.dtype == torch.float32 for p in reloaded.adapter_parameters())
        after_tensors = reloaded.adapters()[14].side.state_dict()
        assert tensors.keys() == after_tensors.keys()
        assert all(torch.equal(value, after_tensors[name]) for name, value in tensors.items())
        _, after = evaluate_batch(reloaded, batch)
        assert torch.equal(before, after)
        assert torch.equal(before.argmax(-1), after.argmax(-1))
