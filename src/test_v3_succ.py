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


# ---- formal admission and comparison (synthetic trusted reference) --------------------------------

import json  # noqa: E402
from types import SimpleNamespace  # noqa: E402

SYN_START = {'formal_start': True, 'run': '/synthetic/start', 'readiness': {'run': '/synthetic/ready'}}


def syn_reference(per_hop=4):
    dev = [(f'{h}-{i}', h, 100*h + i) for h in (1, 2, 3) for i in range(per_hop)]
    labels = {'total': 30, 'by_hop': {'1': 0, '2': 10, '3': 20}, 'episodes': 16000}
    return {'metadata_sha256': 'm'*64, 'train_sha256': 't'*64, 'dev_sha256': 'd'*64, 'order_sha256': 'o'*64,
            'tokens': 12345, 'warmup_steps': 100, 'labels': {'correct': labels, 'permuted': dict(labels)},
            'config': {'steps': 4, 'rank': 128, 'branch': 'memory'}, 'variant': {'fixed_query': False},
            'dev': dev}


def write_pilot(directory, labels, hop_correct, weight, reference, start=SYN_START, metrics_change=None,
                record_change=None, manifest_change=None):
    """A pilot run consistent with the reference; the *_change hooks corrupt one part."""
    from v2r16_train import file_record
    (directory/'eval-s4').mkdir(parents=True)
    normal = []
    for episode, hop, target in reference['dev']:
        index = int(episode.split('-')[1])
        steps = [[.1]*hop for _ in range(4)]
        if hop > 1:
            steps[1][1] = weight
        correct = index < hop_correct[hop-1]
        normal.append({'episode_id': episode, 'hop': hop, 'target': target, 'prediction': target if correct else 0,
                       'correct': correct, 'steps': 4, 'path_weight_last': steps})
    if record_change:
        record_change(normal)
    (directory/'eval-s4'/'dev.normal.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in normal))
    hops = {f'hop{h}': {'count': sum(r['hop'] == h for r in normal),
                        'correct_count': sum(r['correct'] for r in normal if r['hop'] == h)} for h in (1, 2, 3)}
    metrics = {'experiment': v3.CONTRACT, 'command': 'succ', 'labels': labels, 'arm': succ.ARM, 'coefficient': 1.,
               'decision': 'D-0035', 'smoke_only': False, 'config': reference['config'], 'variant': reference['variant'],
               'start': start, 'schedule': {'steps': 2000, 'stage': 'train', 'order_sha256': reference['order_sha256'],
                                             'tokens': reference['tokens'], 'warmup_steps': 100},
               'train': {'steps': 2000, 'tokens': reference['tokens']},
               'successor_labels': reference['labels'][labels], 'successor_loss': {'labels_seen': 30},
               'data': {'metadata_sha256': 'm'*64, 'train_sha256': 't'*64, 'dev_sha256': 'd'*64,
                        'dev_rows': len(reference['dev'])},
               'checkpoint_reload': {'full_records_identical': True, 'steps': 4},
               'evaluation': {'normal': hops}}
    try:
        metrics['pilot'] = v3._jsonable(succ.pilot_summary(normal))
    except (ValueError, IndexError, TypeError, ZeroDivisionError):
        metrics['pilot'] = None
    metrics = json.loads(json.dumps(metrics))  # no shared dicts with the reference or the start
    if metrics_change:
        metrics_change(metrics)
    (directory/'metrics.json').write_text(json.dumps(metrics))
    artifacts = [{'path': str(p.resolve()), **file_record(p)}
                 for p in (directory/'metrics.json', directory/'eval-s4'/'dev.normal.jsonl')]
    manifest = {'status': 'complete', 'validator': {'passed': True}, 'experiment': v3.EXPERIMENT, 'contract': v3.CONTRACT,
                'command': 'succ', 'labels': labels, 'arm': f'{succ.ARM}-succ-{labels}', 'smoke_only': False,
                'eligible_as_evidence': True, 'artifacts': artifacts}
    if manifest_change:
        manifest_change(manifest)
    (directory/'manifest.json').write_text(json.dumps(manifest))
    return directory


def test_compare_admits_valid_pilots_and_applies_d0035(tmp_path, monkeypatch):
    reference = syn_reference()
    monkeypatch.setattr(v3, 'start_provenance', lambda *a, **k: dict(SYN_START))
    monkeypatch.setattr(succ, 'pilot_reference', lambda: reference)
    good = write_pilot(tmp_path/'c', 'correct', (2, 2, 2), .25, reference)
    bad = write_pilot(tmp_path/'p', 'permuted', (2, 1, 1), .0625, reference)
    args = SimpleNamespace(start_run=None, start_sha256=None, ready_run=None, correct_run=good, permuted_run=bad)
    result = {}
    succ.run_compare(args, tmp_path, result)
    assert result['d0035']['correct_beats_permuted'] is True
    assert result['d0035']['i_primary_em']['difference_pp'] == 25  # (50+50)/2 - (25+25)/2
    with pytest.raises(ValueError):  # swapped roles
        succ.run_compare(SimpleNamespace(**dict(vars(args), correct_run=bad, permuted_run=good)), tmp_path, {})


def _drop(manifest, name):
    manifest['artifacts'] = [a for a in manifest['artifacts'] if not a['path'].endswith(name)]


def _set(path, value):
    def change(metrics):
        node = metrics
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value
    return change


def _shift_hash(manifest):
    manifest['artifacts'][0]['bytes'] += 1


INVALID = {
    'metrics_unbound': {'manifest_change': lambda m: _drop(m, 'metrics.json')},
    'records_unbound': {'manifest_change': lambda m: _drop(m, 'dev.normal.jsonl')},
    'no_artifacts': {'manifest_change': lambda m: m.update(artifacts=[])},
    'duplicate_binding': {'manifest_change': lambda m: m['artifacts'].append(dict(m['artifacts'][1]))},
    'wrong_hash': {'manifest_change': _shift_hash},
    'smoke': {'manifest_change': lambda m: m.update(smoke_only=True)},
    'wrong_arm': {'manifest_change': lambda m: m.update(arm='V3-02-succ-correct')},
    'wrong_coefficient': {'metrics_change': _set(('coefficient',), .5)},
    'wrong_decision': {'metrics_change': _set(('decision',), 'D-0034')},
    'wrong_config': {'metrics_change': _set(('config', 'steps'), 1)},
    'wrong_variant': {'metrics_change': _set(('variant', 'fixed_query'), True)},
    'wrong_start': {'metrics_change': _set(('start', 'run'), '/other')},
    'wrong_order': {'metrics_change': _set(('schedule', 'order_sha256'), 'x'*64)},
    'one_step_schedule': {'metrics_change': lambda m: (m['schedule'].update(steps=1), m['train'].update(steps=1))},
    'smoke_schedule': {'metrics_change': _set(('schedule', 'smoke_steps'), 4)},
    'incomplete_training': {'metrics_change': _set(('train', 'steps'), 1999)},
    'wrong_tokens': {'metrics_change': _set(('train', 'tokens'), 1)},
    'wrong_label_count': {'metrics_change': _set(('successor_labels', 'total'), 29)},
    'wrong_labels_seen': {'metrics_change': _set(('successor_loss', 'labels_seen'), 29)},
    'wrong_data': {'metrics_change': _set(('data', 'dev_sha256'), 'x'*64)},
    'missing_reload': {'metrics_change': lambda m: m.pop('checkpoint_reload')},
    'reload_differs': {'metrics_change': _set(('checkpoint_reload', 'full_records_identical'), False)},
    'stored_summary_differs': {'metrics_change': _set(('evaluation', 'normal', 'hop2', 'correct_count'), 3)},
    'invented_row': {'record_change': lambda rs: rs.__setitem__(0, dict(rs[0], episode_id='new'))},
    'missing_row': {'record_change': lambda rs: rs.pop()},
    'reordered_rows': {'record_change': lambda rs: rs.reverse()},
    'wrong_target': {'record_change': lambda rs: rs.__setitem__(5, dict(rs[5], target=-1))},
    'inconsistent_correct': {'record_change': lambda rs: rs.__setitem__(5, dict(rs[5], correct=not rs[5]['correct']))},
    'wrong_steps': {'record_change': lambda rs: rs.__setitem__(5, dict(rs[5], steps=2))},
    'nan_weight': {'record_change': lambda rs: rs.__setitem__(5, dict(rs[5], path_weight_last=[[.1, float('nan')]]*4))},
    'short_weights': {'record_change': lambda rs: rs.__setitem__(5, dict(rs[5], path_weight_last=[[.1, .1]]*3))},
    'missing_weights': {'record_change': lambda rs: rs.__setitem__(5, dict(rs[5], path_weight_last=None))},
}


@pytest.mark.parametrize('case', sorted(INVALID))
def test_admission_rejects_each_invalid_pilot(tmp_path, case):
    reference = syn_reference()
    write_pilot(tmp_path/'ok', 'correct', (2, 2, 2), .25, reference)
    assert succ.validate_pilot(tmp_path/'ok', 'correct', SYN_START, reference)[0]['hop2_step2_fact2_exact'] == Fraction(1, 4)
    run = write_pilot(tmp_path/'bad', 'correct', (2, 2, 2), .25, reference, **INVALID[case])
    with pytest.raises((ValueError, KeyError, IndexError, TypeError)):
        succ.validate_pilot(run, 'correct', SYN_START, reference)


def test_admission_rejects_a_shared_wrong_specification(tmp_path, monkeypatch):
    """Both pilots agree with each other on a one-step S1 run: each fails on its own."""
    reference = syn_reference()
    monkeypatch.setattr(v3, 'start_provenance', lambda *a, **k: dict(SYN_START))
    monkeypatch.setattr(succ, 'pilot_reference', lambda: reference)
    wrong = lambda m: (m['schedule'].update(steps=1), m['train'].update(steps=1), m['config'].update(steps=1))  # noqa: E731
    good = write_pilot(tmp_path/'c', 'correct', (2, 2, 2), .25, reference, metrics_change=wrong)
    bad = write_pilot(tmp_path/'p', 'permuted', (2, 1, 1), .0625, reference, metrics_change=wrong)
    with pytest.raises(ValueError):
        succ.run_compare(SimpleNamespace(start_run=None, start_sha256=None, ready_run=None,
                                         correct_run=good, permuted_run=bad), tmp_path, {})


def test_pilot_reference_and_real_smokes(tok):
    """The trusted reference is the released 2,000-step prefix; the ATT-0254-1 smokes are not formal."""
    reference = succ.pilot_reference()
    rows = loop.load_t2(tok, 'train')[1]
    plan = v3.v3_schedule(rows, 'train', steps=succ.PILOT_STEPS)
    assert (reference['order_sha256'], reference['tokens']) == (plan['order_sha256'], plan['tokens'])
    assert reference['labels']['correct'] == succ.label_counts(rows, plan, 'correct') == reference['labels']['permuted']
    assert reference['config']['steps'] == 4 and len(reference['dev']) == 1000
    start = v3.start_provenance(loop.RUNS/'ATT-0196-1'/'run', 'f70ea08e5f378012975e08bd5bac0083023fef78659174ba1a1d2a9977395417',
                                ready_run=loop.RUNS/'ATT-0227-1'/'run')
    for labels in succ.LABELS:
        with pytest.raises(ValueError):
            succ.validate_pilot(loop.RUNS/'ATT-0254-1'/f'smoke-{labels}', labels, start, reference)
