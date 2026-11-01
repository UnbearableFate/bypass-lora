"""E-0015@3 diagnostic controls, isolated encoding and probes. Execute in a GPU allocation."""
import json
import math
from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer

from bypass_core import LatentReader
from calibrate import MODEL, REVISION, load_base
from v2_data import RELATIONS, encode, visible_delimiters
from v2_memory import (CONTROLS, SWEEP_CONTROLS, Snapshot, build_support, collate, config, evaluate_batch,
                       isolated_snapshot, isolated_writes, memory_from_slots, observe_writes, parse_control,
                       read_vectors)
from v2_schedule import load_data
from v2_train import adapted, evaluate
from v2r_data import load_variant
import v2r_diagnostics as diag

E16_DATA = Path('/work/xg24i002/x10041/bypass-lora/runs/ATT-0039-1/data')
VARIANTS = Path('/work/xg24i002/x10041/bypass-lora/runs/ATT-0112-1')


# ---- read substitutions on hand tensors --------------------------------------------

def hand_snapshot():
    torch.manual_seed(0)
    keys = torch.rand(2, 3, 4) + .1
    values = torch.randn(2, 3, 4)
    valid = torch.tensor([[True, True, True], [True, True, False]])
    state = memory_from_slots(keys, values, valid, 'soft_slot')
    return Snapshot({14: state}, keys, values, valid)


def test_read_substitutions_replace_only_declared_positions():
    snap = hand_snapshot()
    reader = LatentReader(4)
    q = torch.randn(2, 3, 4)
    target, last = torch.tensor([2, 1]), torch.tensor([2, 1])
    row = torch.arange(2)
    normal, weights, *_ = read_vectors(q, snap, reader, 'soft_slot', target)
    at_last = torch.zeros(2, 3, dtype=torch.bool)
    at_last[row, last] = True
    oracle = snap.values[row, target]
    for mode in ('last_only', 'last_only_shuffled', 'oracle_last', 'oracle_last_only'):
        raw, w, *_ = read_vectors(q, snap, reader, 'soft_slot', target, mode, last)
        assert torch.equal(w, weights)
        expected_last = normal[at_last] if mode.startswith('last_only') else oracle
        expected_other = normal[~at_last] if mode == 'oracle_last' else torch.zeros_like(normal[~at_last])
        assert torch.equal(raw[at_last], expected_last) and torch.equal(raw[~at_last], expected_other)
        with pytest.raises(ValueError, match='last question position'):
            read_vectors(q, snap, reader, 'soft_slot', target, mode)
    uniform, *_ = read_vectors(q, snap, reader, 'soft_slot', target, 'uniform')
    torch.testing.assert_close(uniform[0], snap.values[0].mean(0).expand(3, 4))
    torch.testing.assert_close(uniform[1], snap.values[1, :2].mean(0).expand(3, 4))
    wrong, *_ = read_vectors(q, snap, reader, 'soft_slot', target, 'wrong_oracle')
    # (target+1) mod E over each row's valid facts: row 0 (E=3) 2->0, row 1 (E=2) 1->0.
    assert torch.equal(wrong[0], snap.values[0, 0].expand(3, 4)) and torch.equal(wrong[1], snap.values[1, 0].expand(3, 4))
    for mode in ('iso_oracle_value', 'oracle_value'):
        raw, *_ = read_vectors(q, snap, reader, 'soft_slot', target, mode)
        assert torch.equal(raw, oracle[:, None].expand(2, 3, 4))
    single = Snapshot(snap.memory, snap.keys[:, :1], snap.values[:, :1], snap.valid[:, :1])
    with pytest.raises(ValueError, match='two facts'):
        read_vectors(q, single, reader, 'soft_slot', torch.zeros(2, dtype=torch.long), 'wrong_oracle')


def test_parse_control_names():
    assert parse_control('normal') == ('normal', None, None)
    assert parse_control('nonfinal@a0.1') == ('normal', .1, 'nonfinal')
    assert parse_control('oracle_value@a0.3') == ('oracle_value', .3, 'all')
    assert len(SWEEP_CONTROLS) == 6
    with pytest.raises(ValueError, match='unknown'):
        parse_control('normal@a0.5')


def test_isolated_snapshot_keys_values_and_slot_order():
    snap = hand_snapshot()
    iso = {'keys': snap.keys.flip(1), 'values': snap.values + 1, 'valid': snap.valid}
    for mode, keys in (('iso_kv', iso['keys']), ('iso_v', snap.keys), ('iso_oracle_value', snap.keys),
                       ('iso_wrong_oracle', snap.keys)):
        out = isolated_snapshot(snap, iso, mode)
        assert torch.equal(out.keys, keys) and torch.equal(out.values, iso['values'])
        expected = memory_from_slots(keys, iso['values'], snap.valid, 'soft_slot')
        assert torch.equal(out.memory[14].slot_keys, expected.slot_keys)
        assert torch.equal(out.memory[14].slot_values, expected.slot_values)
    with pytest.raises(ValueError, match='match'):
        isolated_snapshot(snap, {**iso, 'valid': ~snap.valid}, 'iso_kv')


# ---- probes -------------------------------------------------------------------------

def test_probe_fit_test_baselines_and_unseen_labels():
    torch.manual_seed(0)
    centers = torch.randn(3, 16) * 5
    train_y = torch.tensor([10, 20, 30] * 40)
    train_x = centers[train_y // 10 - 1] + torch.randn(120, 16) * .1
    test_y = torch.tensor([10, 20, 30, 40] * 5)  # 40 never appears in the fit set
    test_x = torch.cat((centers, torch.zeros(1, 16)))[test_y // 10 - 1] + torch.randn(20, 16) * .1
    positions = torch.tensor([0, 1, 4, 7] * 5)
    result = diag.probe(train_x, train_y, test_x, test_y, positions, projection=True)
    assert result['accuracy'] == .75 and result['unseen_label_count'] == 5
    assert result['by_position']['0']['accuracy'] == 1 and result['by_position']['>=4']['n'] == 10
    assert result['baseline_answer_prior']['accuracy'] == .25
    assert result['baseline_label_shuffle']['accuracy'] < .75
    assert 'baseline_random_projection_128' in result


def test_magnitudes_by_fact_position():
    x = torch.zeros(4, 3)
    x[0, 2] = -5.  # position 0: one massive coordinate
    x[1:, 0] = torch.tensor([1., 2., 3.])
    part = {'values': x, 'features': x * 2}
    m = diag.magnitudes({'context': part, 'isolated': part, 'positions': torch.tensor([0, 1, 3, 4])})
    assert m['values.context']['0'] == {'n': 1, 'l2_mean': 5., 'l2_median': 5., 'max_abs_mean': 5., 'max_abs_median': 5.,
                                        'max_abs_max': 5., 'argmax_dimension_mode': 2, 'argmax_dimension_share': 1.}
    assert m['features.isolated']['1-3']['n'] == 2 and m['features.isolated']['1-3']['l2_mean'] == 3.
    assert m['values.context']['>=4']['argmax_dimension_mode'] == 0


# ---- real model ---------------------------------------------------------------------

@pytest.fixture(scope='module')
def base():
    return load_base()


@pytest.fixture(scope='module')
def tok():
    return AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)


NAMES = ['Alex', 'Betty', 'Jeffrey', 'Nathan', 'Caroline', 'Nicole', 'Simon', 'George', 'Thomas', 'Richard']


def make_row(tok, index, facts, target, prefix=0):
    """A row in the V2/E-0014 generator format (make_episode), for tests only."""
    facts = [dict(f) for f in facts]
    support = ''.join(diag.fact_text(f) for f in facts)
    ids, positions = visible_delimiters(tok, support)
    for fact, position in zip(facts, positions):
        fact.update(position)
    query = facts[target]
    question = encode(tok, 'Please ' * prefix + f"Who does {query['head']} {RELATIONS[query['relation']]}?\nAnswer:")
    return {'episode_id': f'e0015-{index}', 'support_text': support, 'support_input_ids': ids, 'facts': facts,
            'question_input_ids': question, 'identity': {'split': 'calibration'},
            'axes': {'support_order': 'natural'},
            'query': {'head': query['head'], 'relation': query['relation'], 'target_fact_index': target,
                      'answer_text': ' ' + query['tail'], 'answer_token_id': encode(tok, ' ' + query['tail'])[0]}}


def rows_with(tok, n_facts):
    relations = list(RELATIONS)
    rows = []
    for i in range(4):
        facts = [{'head': NAMES[(i + 2 * j) % 10], 'relation': relations[(i + j) % 4],
                  'tail': NAMES[(i + 2 * j + 1) % 10]} for j in range(n_facts)]
        rows.append(make_row(tok, i, facts, (i + 1) % n_facts, prefix=i))
    diag.annotate(tok, rows, n_facts)
    return rows


def randomize_exits(model, std=.03):
    with torch.no_grad():
        for parameter in model.adapters()[14].side.fusion.parameters():
            parameter.normal_(std=std)


def fusion_inputs(model, batch, mode, **kwargs):
    seen = {}
    side = model.adapters()[14].side
    handle = side.fusion.register_forward_pre_hook(lambda module, args: seen.update(z=args[2].clone()))
    try:
        details, logits = evaluate_batch(model, batch, mode, **kwargs)
    finally:
        handle.remove()
    return seen['z'], details, logits


@torch.no_grad()
def test_controls_replace_and_scale_only_declared_positions(base, tok):
    rows = rows_with(tok, 3)
    batch = collate(rows, tok.pad_token_id)
    donors = collate(rows[-1:] + rows[:-1], tok.pad_token_id)
    last = batch.question_mask.long().sum(-1) - 1
    at_last = torch.zeros_like(batch.question_mask)
    at_last[torch.arange(len(rows)), last] = True
    others = batch.question_mask & ~at_last
    assert others.any()
    with adapted(base, cfg=config('direct_hidden', 'soft_slot', 'F_out')) as model:
        model.eval()
        randomize_exits(model)
        z = {mode: fusion_inputs(model, batch, mode)[0] for mode in ('normal', 'oracle_value', 'wrong_oracle',
                                                                     'last_only', 'oracle_last', 'oracle_last_only',
                                                                     'uniform') + SWEEP_CONTROLS}
        z['shuffled'] = fusion_inputs(model, batch, 'shuffled', donor_batch=donors)[0]
        z['last_only_shuffled'], details, _ = fusion_inputs(model, batch, 'last_only_shuffled', donor_batch=donors)
        assert details['address.target_available'] is False
        zero = torch.zeros_like(z['normal'][others])
        assert torch.equal(z['last_only'][at_last], z['normal'][at_last]) and torch.equal(z['last_only'][others], zero)
        assert torch.equal(z['last_only_shuffled'][at_last], z['shuffled'][at_last])
        assert torch.equal(z['last_only_shuffled'][others], zero)
        assert torch.equal(z['oracle_last'][at_last], z['oracle_value'][at_last])
        assert torch.equal(z['oracle_last'][others], z['normal'][others])
        assert torch.equal(z['oracle_last_only'][at_last], z['oracle_value'][at_last])
        assert torch.equal(z['oracle_last_only'][others], zero)
        assert not torch.equal(z['wrong_oracle'][at_last], z['oracle_value'][at_last])
        for alpha in (.1, .3):
            assert torch.equal(z[f'normal@a{alpha}'], z['normal'] * alpha)
            assert torch.equal(z[f'oracle_value@a{alpha}'], z['oracle_value'] * alpha)
            assert torch.equal(z[f'nonfinal@a{alpha}'][at_last], z['normal'][at_last])
            assert torch.equal(z[f'nonfinal@a{alpha}'][others], z['normal'][others] * alpha)
        padding = ~batch.question_mask
        assert all(not z[m][padding].any() for m in z)
        for mode in ('wrong_oracle', 'uniform', 'oracle_last', 'last_only', 'nonfinal@a0.1', 'oracle_value@a0.3'):
            details, _ = evaluate_batch(model, batch, mode)
            assert details['read.reconstruction_max_abs'] < 1e-4
        details, _ = evaluate_batch(model, batch, 'wrong_oracle')
        snapshot, _ = build_support(model, batch, observe=True)
        target = torch.tensor([r['query']['target_fact_index'] for r in rows])
        wrong = snapshot.values[torch.arange(4), ((target + 1) % 3).cuda()].cpu()
        assert torch.equal(details['_read_vectors'], wrong)


@torch.no_grad()
def test_isolated_encoding_matches_e1_format_and_slot_order(base, tok):
    rows = rows_with(tok, 4)
    single = rows_with(tok, 1)
    with adapted(base, cfg=config('direct_hidden', 'soft_slot', 'F_out')) as model:
        model.eval()
        randomize_exits(model)
        batch = collate(rows, tok.pad_token_id)
        snapshot, _ = build_support(model, batch, observe=True)
        context = observe_writes(model, batch.support_ids, batch.support_mask, batch.write_mask)
        assert torch.equal(context['keys'], snapshot.keys) and torch.equal(context['values'], snapshot.values)
        iso = isolated_writes(model, rows, tok.pad_token_id, 'cuda')
        assert torch.equal(iso['valid'], snapshot.valid)
        # Slot j of row i is fact j encoded alone, as an E=1 support row of that fact.
        # Batched (padded) vs. single-row BF16 forwards differ slightly: compare by the
        # relative vector error, against the much larger contextual shift.
        def relative(a, b):
            return ((a - b).norm(dim=-1) / b.norm(dim=-1)).max().item()
        for i, row in enumerate(rows):
            for j, fact in enumerate(row['facts']):
                alone = make_row(tok, 99, [fact], 0)
                assert alone['support_input_ids'] == row['_isolated'][j]['ids']
                ab = collate([alone], tok.pad_token_id)
                w = observe_writes(model, ab.support_ids, ab.support_mask, ab.write_mask)
                assert relative(iso['values'][i, j], w['values'][0, 0]) < .03
                assert relative(iso['features'][i, j].float(), w['features'][0, 0].float()) < .03
        # Fact 0 has no preceding context: isolated and contextual writes agree.
        assert relative(iso['values'][:, 0], context['values'][:, 0]) < .03
        shift = ((iso['values'][:, 1:] - context['values'][:, 1:]).norm(dim=-1)
                 / context['values'][:, 1:].norm(dim=-1)).min().item()
        print('isolated-vs-context relative value shift, facts 1..3 (min):', shift)
        assert shift > .1
        # Reordering the support reorders the isolated slots.
        reordered = [make_row(tok, i, list(reversed([{k: f[k] for k in ('head', 'relation', 'tail')}
                                                     for f in r['facts']])), 3 - r['query']['target_fact_index'])
                     for i, r in enumerate(rows)]
        diag.annotate(tok, reordered, 4)
        flipped = isolated_writes(model, reordered, tok.pad_token_id, 'cuda')
        assert relative(flipped['values'], iso['values'].flip(1)) < .03
        # E=1: the isolated encoding is the support itself, so iso controls reproduce the
        # contextual ones (the oracle exactly, the read up to the slot-state rebuild).
        b1 = collate(single, tok.pad_token_id)
        _, oracle = evaluate_batch(model, b1, 'oracle_value')
        _, iso_oracle = evaluate_batch(model, b1, 'iso_oracle_value')
        assert torch.equal(oracle, iso_oracle)
        _, normal = evaluate_batch(model, b1, 'normal')
        for mode in ('iso_kv', 'iso_v'):
            _, logits = evaluate_batch(model, b1, mode)
            torch.testing.assert_close(logits, normal, atol=.5, rtol=0)
        for mode in ('iso_kv', 'iso_v', 'iso_oracle_value', 'iso_wrong_oracle'):
            details, logits = evaluate_batch(model, batch, mode)
            assert math.isfinite(details['nll']) and details['read.reconstruction_max_abs'] < 1e-4
        with pytest.raises(ValueError, match='one question per support owner'):
            evaluate_batch(model, collate(rows, tok.pad_token_id, owners=[1, 0, 2, 3]), 'iso_kv')


@torch.no_grad()
def test_evaluate_captures_last_increments_and_keeps_controls(base, tok, tmp_path):
    rows = rows_with(tok, 3)
    with adapted(base, cfg=config('direct_hidden', 'soft_slot', 'F_out')) as model:
        model.eval()
        randomize_exits(model)
        capture = {}
        metrics, _ = evaluate(model, rows, tok.pad_token_id, tmp_path, 'calibration',
                                        ('normal', 'oracle_value', 'last_only_shuffled', 'nonfinal@a0.1'),
                                        capture=capture, batch_size=3)
        assert set(capture) == set(metrics) and capture['normal'].shape == (4, 1024)
        details, _ = evaluate_batch(model, collate(rows[:3], tok.pad_token_id), 'normal')
        assert torch.equal(capture['normal'][:3], details['_last_fusion'])
        assert (tmp_path / 'calibration.nonfinal@a0.1.predictions.jsonl').exists()
        assert metrics['last_only_shuffled']['diagnostics']['address.target_available'] is False
        with pytest.raises(ValueError, match='unknown'):
            evaluate(model, rows, tok.pad_token_id, tmp_path / 'bad', 'calibration', ('normal@a0.2',))


def test_released_rows_isolated_tokens_follow_e1_format(tok):
    checks = {}
    for facts, rows in ((1, load_variant(VARIANTS / 'e1', 1, 'calibration')[1][:256]),
                        (4, load_variant(VARIANTS / 'e4', 4, 'calibration')[1][:256]),
                        (16, load_data(E16_DATA / 'calibration.jsonl', 'calibration')[:64])):
        checks[facts] = diag.annotate(tok, rows, facts)
        for row in rows:
            for item, fact in zip(row['_isolated'], row['facts']):
                assert item['ids'] == encode(tok, diag.fact_text(fact))
                assert item['delimiter'] == len(item['ids']) - 1
    assert checks[1]['context_fact_tokens_differ_from_isolated'] == 0
    print(json.dumps(checks))


# ---- rule evaluation (RV-0042) ------------------------------------------------------

def synthetic_summaries(tmp_path, edit):
    from types import SimpleNamespace
    modes = diag.CONTROLS_MULTI + SWEEP_CONTROLS
    probe_names = ('Pv.context', 'Pv.isolated', 'Pf.context', 'Pf.isolated', 'Pz.normal', 'Pz.oracle_value')
    tmp_path.mkdir(parents=True, exist_ok=True)
    paths = []
    for label in diag.CHECKPOINTS:
        splits = {}
        for facts in diag.FACT_COUNTS:
            controls = {m: {'em': .05, 'nll': 5., 'prior_em': .02,
                            'target_position': {'0': {'em': .05}, '>=1': {'em': .05}}} for m in modes}
            probes = {n: {'accuracy': .5, 'baseline_answer_prior': {'accuracy': .02},
                          'baseline_label_shuffle': {'accuracy': .02}, 'unseen_label_count': 0, 'by_position': {}}
                      for n in probe_names}
            splits[f'e{facts}'] = {'controls': controls, 'probes': probes, 'feature_sha256': {'x': 'same'},
                                   'isolated_minus_context_value_max_abs': 0., 'magnitudes': {}}
        summary = {'checkpoint': label, 'limit': 0, 'splits': splits, 'wall_seconds': 1.,
                   'reload': {'e16': {'normal': {'same_episode_order': True, 'prediction_mismatches': 0}}},
                   'tokenization': {}}
        edit(label, summary['splits'])
        path = tmp_path / f'{label}.json'
        path.write_text(json.dumps(summary))
        paths.append(f'{label}={path}')
    out = tmp_path / 'analysis.json'
    diag.analyze(SimpleNamespace(summary=paths, output=out))
    return json.loads(out.read_text())['rules']


def test_rule4_is_em_drop_without_nll_and_rule6_covers_r2_r5_at_e4_and_e16(tmp_path):
    def rule4(label, splits):
        for split in ('e1', 'e4', 'e16'):
            splits[split]['probes']['Pf.context']['accuracy'] = .95
        if label == 'R1':
            splits['e1']['controls']['oracle_value'].update(em=.85, nll=3.)
            # E=4/16 oracle_value 35+ pp lower but with LOWER NLL: rule 4 still holds.
            for split in ('e4', 'e16'):
                splits[split]['controls']['oracle_value'].update(em=.4, nll=2.)
    assert synthetic_summaries(tmp_path / 'a', rule4)['4']['holds'] is True
    for label, split in (('R2', 'e16'), ('R5', 'e4'), ('R2', 'e4'), ('R5', 'e16')):
        def rule6(name, splits, label=label, split=split):
            if name == label:
                splits[split]['controls']['iso_v'].update(em=.2, nll=4.)
        rules = synthetic_summaries(tmp_path / f'{label}{split}', rule6)
        assert rules['6']['holds'] is True, (label, split)
        assert {(v['checkpoint'], v['split']) for v in rules['6']['values']} == {
            ('R2', 'e4'), ('R2', 'e16'), ('R5', 'e4'), ('R5', 'e16')}
    assert synthetic_summaries(tmp_path / 'none', lambda *a: None)['8']['holds'] is True
