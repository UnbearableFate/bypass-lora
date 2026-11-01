"""E-0023 rule 4 repair (D-0035): successor labels, the training forward and the pilot rule. Core
tests run anywhere; model tests need a GPU allocation and use the D5 seed-0 adapter as a stand-in
start (T2 train/dev only)."""
from fractions import Fraction

import pytest
import torch

import v3_loop as loop
import v3_succ as succ
import v3_train as v3
from test_v3_train import D5, base, dev, gpu_batch, tok  # noqa: F401


def fake_row(episode, path, n=16):
    return {'episode_id': episode, 'facts': [{}]*n, 'path': {'fact_indices': list(path)}, 'axes': {'hop': len(path)}}


# ---- labels ---------------------------------------------------------------------------------------

def test_successor_labels_follow_the_path_from_step_2():
    assert succ.successor_labels(fake_row('a', [7]), 'correct') == []
    assert succ.successor_labels(fake_row('a', [7, 3]), 'correct') == [(1, 3)]
    assert succ.successor_labels(fake_row('a', [7, 3, 12]), 'correct') == [(1, 3), (2, 12)]
    assert succ.successor_labels(fake_row('a', [7, 3, 12]), 'correct', steps=2) == [(1, 3)]
    with pytest.raises(ValueError):
        succ.successor_labels(fake_row('a', [7, 3]), 'shuffled')


def test_permuted_labels_are_a_fixed_per_episode_derangement():
    row = fake_row('ep-1', [7, 3, 12])
    correct, permuted = succ.successor_labels(row, 'correct'), succ.successor_labels(row, 'permuted')
    assert [s for s, _ in permuted] == [s for s, _ in correct]  # same count and positions
    pi = succ.derangement('ep-1', 16)
    assert sorted(pi) == list(range(16)) and all(pi[i] != i for i in range(16))
    assert permuted == [(s, pi[slot]) for s, slot in correct]
    assert succ.derangement('ep-1', 16) == pi  # seeded
    assert succ.derangement('ep-2', 16) != pi  # per episode
    assert succ.successor_labels(fake_row('ep-1', [5]), 'permuted') == []
    with pytest.raises(ValueError):
        succ.derangement('x', 1)


def test_label_counts_per_hop():
    rows = [fake_row(f'e{i}', list(range(1 + i % 3))) for i in range(6)]
    plan = {'batches': [[0, 1, 2], [3, 4, 5]]}
    counts = succ.label_counts(rows, plan, 'correct')
    assert counts == {'total': 6, 'by_hop': {'1': 0, '2': 2, '3': 4}, 'episodes': 6}
    assert succ.label_counts(rows, plan, 'permuted') == counts


# ---- D-0035 rule ----------------------------------------------------------------------------------

def summary(hop2, hop3, weight, hop1=50, n=100):
    return {'hops': {'hop1': {'count': n, 'correct_count': hop1}, 'hop2': {'count': n, 'correct_count': hop2},
                     'hop3': {'count': n, 'correct_count': hop3}}, 'hop2_step2_fact2_exact': weight}


@pytest.mark.parametrize('correct, permuted, i, ii', [
    (summary(13, 13, Fraction(1, 8)), summary(10, 10, Fraction(1, 9)), True, True),    # exactly +3 pp, exactly 1/8
    (summary(13, 12, Fraction(1, 4)), summary(10, 10, Fraction(1, 9)), False, True),   # +2.5 pp
    (summary(20, 20, Fraction(1, 9)), summary(10, 10, Fraction(1, 10)), True, False),  # correct below 1/8
    (summary(20, 20, Fraction(1, 4)), summary(10, 10, Fraction(1, 8)), True, False),   # permuted exactly 1/8
])
def test_decide_boundaries(correct, permuted, i, ii):
    result = succ.decide(correct, permuted)
    assert (result['i_primary_em']['passed'], result['ii_hop2_step2_fact2']['passed']) == (i, ii)
    assert result['correct_beats_permuted'] == (i and ii)
    assert result['i_primary_em']['difference_exact'] == str(succ.primary(correct['hops'])-succ.primary(permuted['hops']))


def test_pilot_summary_reads_step2_fact2_on_hop2():
    normal = [{'hop': 1, 'correct': True, 'path_weight_last': [[.9]]*4},
              {'hop': 2, 'correct': False, 'path_weight_last': [[.5, .1], [.2, .25], [0, 0], [0, 0]]},
              {'hop': 2, 'correct': True, 'path_weight_last': [[.5, .1], [.2, .5], [0, 0], [0, 0]]},
              {'hop': 3, 'correct': False, 'path_weight_last': [[.5, .1, 0]]*4}]
    result = succ.pilot_summary(normal)
    assert result['hop2_step2_fact2_exact'] == Fraction(3, 8)
    assert result['hops']['hop2'] == {'count': 2, 'correct_count': 1}
    with pytest.raises(ValueError):
        succ.pilot_summary(normal[:3])


# ---- model (GPU) ----------------------------------------------------------------------------------

@pytest.fixture(scope='module')
def train_rows(tok):
    rows = loop.load_t2(tok, 'train')[1]
    return [r for h in (1, 2, 3) for r in [x for x in rows if x['axes']['hop'] == h][:4]]


def test_pilot_schedule_is_the_stage_a_prefix(tok):
    rows = loop.load_t2(tok, 'train')[1]
    pilot, full = v3.v3_schedule(rows, 'train', steps=succ.PILOT_STEPS), v3.v3_schedule(rows, 'train')
    assert pilot['batches'] == full['batches'][:succ.PILOT_STEPS]
    assert pilot['warmup_steps'] == 100 and pilot['steps'] == 2000
    counts = succ.label_counts(rows, pilot, 'correct')
    assert counts['episodes'] == 16000 and counts['total'] == counts['by_hop']['2'] + counts['by_hop']['3'] > 0


def test_forward_without_labels_is_train_forward(base, tok, train_rows):
    from functools import partial
    from v2_memory import isolated_support
    from v2r16_memory import train_forward
    batch = gpu_batch(tok, [r for r in train_rows if r['axes']['hop'] == 1])
    with v3.arm_model(base, succ.ARM, D5/'adapter') as model, torch.no_grad():
        tally = []
        loss, logits, _, _ = succ.succ_forward(model, batch, labels='correct', tally=tally)
        reference, reference_logits, _, _ = partial(train_forward, support_fn=isolated_support)(model, batch)
    assert torch.equal(logits, reference_logits) and float(loss) == float(reference)
    assert tally[0]['labels'] == 0 and tally[0]['successor'] == 0.


def test_successor_ce_uses_the_actual_read_weights(base, tok, train_rows):
    from v2_memory import isolated_support
    from v3_loop import _snapshots
    rows = [r for r in train_rows if r['axes']['hop'] > 1]
    batch = gpu_batch(tok, rows)
    with v3.arm_model(base, succ.ARM, D5/'adapter') as model:
        tally = []
        loss, logits, _, _ = succ.succ_forward(model, batch, labels='correct', tally=tally)
        assert tally[0]['labels'] == sum(len(r['path']['fact_indices'])-1 for r in rows)
        assert tally[0]['replay_max_relative'] <= loop.RECONSTRUCTION_TOLERANCE
        model.eval()
        with torch.no_grad():  # expected CE from the evaluator's own per-step weights at the answer position
            snapshot = _snapshots(model, batch, 'normal', None)
            _, seen, _ = v3._capture(model, batch, snapshot)
            _, trajectory = v3.replay(model.adapters()[14].side, seen['q'], seen['k'], seen['v'], snapshot, 4)
            row = torch.arange(len(rows), device='cuda')
            last = batch.question_mask.long().sum(-1)-1
            terms = [-torch.log(trajectory[s]['weights'][i, last[i], slot])
                     for i, r in enumerate(rows) for s, slot in succ.successor_labels(r, 'correct')]
            assert abs(tally[0]['successor'] - float(torch.stack(terms).mean())) < 1e-4
            answer = torch.nn.functional.cross_entropy(logits, batch.labels)
            assert abs(float(loss) - float(answer) - tally[0]['successor']) < 1e-5
        model.train()
        model.model.eval()
        side = model.adapters()[14].side
        for p in model.adapter_parameters():
            p.requires_grad_(True)
        model.zero_grad(set_to_none=True)
        tally.clear()
        loss, _, _, _ = succ.succ_forward(model, batch, labels='correct', tally=tally)
        loss.backward()
        with_ce = side.reader.Rq.grad.clone()
        model.zero_grad(set_to_none=True)
        answer_only = torch.nn.functional.cross_entropy(succ.succ_forward(model, batch, labels='correct', tally=[])[1],
                                                        batch.labels)
        answer_only.backward()
        assert not torch.allclose(with_ce, side.reader.Rq.grad)  # the successor CE reaches Rq
        tally.clear()
        with torch.no_grad():
            succ.succ_forward(model, batch, labels='permuted', tally=tally)
            succ.succ_forward(model, batch, labels='correct', tally=tally)
        assert tally[0]['labels'] == tally[1]['labels'] and tally[0]['successor'] != tally[1]['successor']


def test_succ_forward_rejects_a_tokenwise_arm(base, tok, train_rows):
    batch = gpu_batch(tok, train_rows[4:8])
    with v3.arm_model(base, 'V3-06', D5/'adapter') as model, torch.no_grad(), pytest.raises(ValueError):
        succ.succ_forward(model, batch, labels='correct', tally=[])


# ---- comparison on synthetic pilot runs -----------------------------------------------------------

def synthetic_pilot(directory, labels, hop_correct, weight, start):
    import json
    from v2r16_train import file_record
    (directory/'eval-s4').mkdir(parents=True)
    normal = []
    for h, correct in zip((1, 2, 3), hop_correct):
        for i in range(4):
            steps = [[.1]*h for _ in range(4)]
            steps[1] = [.1, weight][:h] if h > 1 else [.1]
            normal.append({'episode_id': f'{h}-{i}', 'hop': h, 'correct': i < correct, 'path_weight_last': steps})
    (directory/'eval-s4'/'dev.normal.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in normal))
    shared = {'schedule': {'steps': 2000}, 'config': {'steps': 4}, 'variant': {}, 'data': {'dev_rows': 12},
              'successor_labels': {'total': 10}}
    metrics = dict(shared, start=start, pilot=v3._jsonable(succ.pilot_summary(normal)))
    (directory/'metrics.json').write_text(json.dumps(metrics))
    artifacts = [{'path': str(p), **file_record(p)} for p in (directory/'metrics.json', directory/'eval-s4'/'dev.normal.jsonl')]
    (directory/'manifest.json').write_text(json.dumps({
        'status': 'complete', 'validator': {'passed': True}, 'contract': v3.CONTRACT, 'command': 'succ',
        'labels': labels, 'smoke_only': False, 'eligible_as_evidence': True, 'artifacts': artifacts}))
    return directory


def test_compare_validates_both_pilots_and_applies_d0035(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    start = {'formal_start': True, 'run': '/synthetic/start'}
    monkeypatch.setattr(v3, 'start_provenance', lambda *a, **k: dict(start))
    good = synthetic_pilot(tmp_path/'c', 'correct', (2, 2, 2), .25, start)
    bad = synthetic_pilot(tmp_path/'p', 'permuted', (2, 1, 1), .0625, start)
    args = SimpleNamespace(start_run=None, start_sha256=None, ready_run=None, correct_run=good, permuted_run=bad)
    result = {}
    succ.run_compare(args, tmp_path, result)
    assert result['d0035']['correct_beats_permuted'] is True
    assert result['d0035']['i_primary_em']['difference_pp'] == 25  # (50+50)/2 - (25+25)/2
    with pytest.raises(ValueError):  # swapped roles
        succ.run_compare(SimpleNamespace(**dict(vars(args), correct_run=bad, permuted_run=good)), tmp_path, {})
    other = synthetic_pilot(tmp_path/'o', 'permuted', (2, 1, 1), .0625, dict(start, run='/other'))
    with pytest.raises(ValueError):  # another start
        succ.run_compare(SimpleNamespace(**dict(vars(args), permuted_run=other)), tmp_path, {})
    (bad/'eval-s4'/'dev.normal.jsonl').write_text('')
    with pytest.raises(ValueError):  # a changed artifact
        succ.run_compare(args, tmp_path, {})
    metrics = json.loads((good/'metrics.json').read_text())
    assert metrics['pilot']['hop2_step2_fact2_exact'] == .25
