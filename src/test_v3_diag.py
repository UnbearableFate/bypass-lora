"""E-0023 diagnostics and the stage-A rule. Core tests run anywhere; model tests need a GPU
allocation and use the D5 seed-0 adapter as a stand-in common start (T2 dev/train only)."""
from types import SimpleNamespace

import pytest
import torch

import v3_diag as diag
import v3_loop as loop
import v3_train as v3
from test_v3_train import D5, SYN_START, base, dev, gpu_batch, synthetic_arm, synthetic_reference, tok  # noqa: F401


# ---- stage A rule ---------------------------------------------------------------------------------

def main(**arms):
    """arm=(hop1, hop2, hop3) correct counts out of 100 each."""
    return {arm.replace('_', '-'): {f'hop{h}': {'count': 100, 'correct_count': c} for h, c in zip((1, 2, 3), hops)}
            for arm, hops in arms.items()}


@pytest.mark.parametrize('arms, case, star', [
    # exactly +3 pp and hop1 exactly -1 pp: A1 (thresholds include equality)
    ({'V3_01': (50, 10, 10), 'V3_02': (49, 13, 13), 'V3_03': (50, 13, 13)}, 'A1', 'V3-02'),
    ({'V3_01': (50, 10, 10), 'V3_02': (48, 13, 13), 'V3_03': (50, 13, 13)}, 'A2', 'V3-02'),
    ({'V3_01': (50, 10, 10), 'V3_02': (50, 12, 13), 'V3_03': (50, 12, 12)}, 'A3', 'V3-02'),
    ({'V3_01': (50, 10, 10), 'V3_02': (50, 10, 10), 'V3_03': (50, 9, 10)}, 'A4', 'V3-02'),
    # V3-03 exactly 0.5 pp above V3-02 is S_M*; its gain is +2.5 pp: A3
    ({'V3_01': (50, 10, 10), 'V3_02': (50, 12, 12), 'V3_03': (50, 12, 13)}, 'A3', 'V3-03'),
    # V3-03 exactly 0.5 pp above V3-02 with a gain of exactly +3 pp: A1 through V3-03
    ({'V3_01': (50, 10, 10), 'V3_02': (50, 12, 13), 'V3_03': (50, 13, 13)}, 'A1', 'V3-03'),
])
def test_stage_a_cases_and_boundaries(arms, case, star):
    result = diag.stage_a(main(**arms))
    assert (result['case'], result['S_M*']) == (case, star), result


def test_stage_a_needs_exactly_the_three_arms():
    with pytest.raises(ValueError):
        diag.stage_a(main(V3_01=(1, 1, 1), V3_02=(1, 1, 1)))


def test_analyze_stage_a_on_validated_records(tmp_path):
    reference = synthetic_reference()
    runs = [synthetic_arm(tmp_path/a, a, c, reference)
            for a, c in (('V3-01', (2, 1, 1)), ('V3-02', (2, 2, 1)), ('V3-03', (2, 3, 3)))]
    result = diag.analyze_stage_a(runs, reference, SYN_START)
    assert result['stage_a']['S_M*'] == 'V3-03' and result['stage_a']['case'] == 'A1'
    assert result['stage_a']['primary_exact']['V3-03'] == '75/1'
    assert set(result['bootstrap']['intervals_pp']) == {'V3-02 - V3-01', 'V3-03 - V3-01', 'V3-03 - V3-02'}
    with pytest.raises(ValueError):
        diag.analyze_stage_a(runs[:2] + runs[:1], reference, SYN_START)
    with pytest.raises(ValueError):  # the analysis start must carry the passed R record
        diag.analyze_stage_a(runs, reference, dict(SYN_START, readiness=None))
    other = synthetic_arm(tmp_path/'V3-04', 'V3-04', (1, 1, 1), reference)
    with pytest.raises(ValueError):
        diag.analyze_stage_a(runs[:2] + [other], reference, SYN_START)


# ---- saved-record diagnostics ---------------------------------------------------------------------

def weight_records():
    # hop 2, S = 2: step 1 on (fact1, fact2), step 2 on (fact1, fact2)
    return [{'hop': 2, 'path_weight_last': [[.6, .1], [.2, .5]]},
            {'hop': 2, 'path_weight_last': [[.4, .1], [.2, .3]]},
            {'hop': 1, 'path_weight_last': [[.9], [.8]]}]


def test_path_weight_table_and_diagonal():
    table = diag.path_weight_table(weight_records(), 2)
    assert table['hop2']['table'] == [[pytest.approx(.5), pytest.approx(.1)], [pytest.approx(.2), pytest.approx(.4)]]
    assert table['hop2']['diagonal'] == [pytest.approx(.5), pytest.approx(.4)]
    assert table['hop1']['diagonal'] == [pytest.approx(.9)]  # min(S, hop) entries
    second = diag.secondary_mechanism(table, 2)
    assert second['hop2_step2_fact2']['value'] == pytest.approx(.4) and 'hop3_step3_fact3' not in second
    with pytest.raises(ValueError):
        diag.path_weight_table(weight_records(), 4)
    with pytest.raises(ValueError):
        diag.path_weight_table([{'hop': 2, 'path_weight_last': None}], 2)


def test_frequency_strata_median_split():
    name = lambda n: {'query': {'answer_text': f' {n}'}}
    train = [name('A')]*5 + [name('B')]*3 + [name('C')]*1
    rows = [dict(name(n), episode_id=f'e{i}') for i, n in enumerate('AABC')]
    records = [{'episode_id': f'e{i}', 'hop': 1 + i % 3, 'correct': i < 2} for i in range(4)]
    result = diag.frequency_strata(records, rows, train)
    assert result['median_train_answer_count'] == 4  # counts 5, 5, 3, 1
    assert result['strata']['high']['all']['count'] == 2 and result['strata']['high']['all']['em'] == 1
    assert result['strata']['low']['all']['count'] == 2 and result['strata']['low']['all']['em'] == 0
    assert result['strata']['high']['all']['train_count_range'] == [5, 5]
    with pytest.raises(ValueError):
        diag.frequency_strata([{'episode_id': 'x', 'hop': 1, 'correct': True}], rows, train)


# ---- value oracles (GPU) --------------------------------------------------------------------------

def fake_snapshot(b=3, n=5, r=128, device='cuda'):
    g = torch.Generator().manual_seed(0)
    return SimpleNamespace(values=torch.randn(b, n, r, generator=g).to(device), valid=torch.ones(b, n, dtype=torch.bool, device=device))


@pytest.mark.parametrize('arm', ('V3-01', 'V3-03', 'V3-08', 'V3-09', 'V3-10', 'V3-05'))
def test_oracle_replay_closed_forms(base, arm):
    with v3.arm_model(base, arm, D5/'adapter') as model, torch.no_grad():
        side = model.adapters()[14].side
        snapshot = fake_snapshot()
        qbar = torch.randn(3, 4, 128, generator=torch.Generator().manual_seed(1)).cuda()
        fact = torch.tensor([0, 2, 4], device='cuda')
        steps = v3.ARMS[arm]['steps']
        z = diag.oracle_replay(side, qbar, snapshot, steps, fact)
        read = (snapshot.values[torch.arange(3), fact]*side.reader.read_gain)[:, None]
        eta = side.reader.eta
        if arm == 'V3-08':
            expected = sum(eta*(read @ side.reader.step_matrices(s)[1]) for s in range(steps)).expand_as(qbar)
        elif arm == 'V3-09':  # anchored: every step restarts from z0
            expected = (eta*(read @ side.reader.Ro)).expand_as(qbar)
        elif arm == 'V3-10':  # full readout keeps z0
            expected = qbar + steps*eta*(read @ side.reader.Ro)
        else:
            expected = (steps*eta*(read @ side.reader.Ro)).expand_as(qbar)
        assert torch.allclose(z, expected, atol=1e-4, rtol=1e-4), arm
    with v3.arm_model(base, 'V3-06', D5/'adapter') as model, pytest.raises(ValueError):
        diag.oracle_replay(model.adapters()[14].side, qbar, snapshot, 4, fact)
    snapshot.valid[0, 0] = False
    with v3.arm_model(base, 'V3-01', D5/'adapter') as model, pytest.raises(ValueError):
        diag.oracle_replay(model.adapters()[14].side, qbar, snapshot, 1, fact)


def test_first_edge_oracle_is_the_reviewed_oracle_value_at_s1(base, tok, dev, tmp_path):
    from v2_memory import isolated_support
    from v2r16_memory import evaluate
    with v3.arm_model(base, 'V3-01', D5/'adapter') as model:
        oracles = diag.evaluate_oracles(model, dev, tok.pad_token_id, 1)
        _, reviewed = evaluate(model, dev, tok.pad_token_id, tmp_path/'reviewed', 'dev', ('oracle_value',), None,
                               support_fn=isolated_support)
    assert diag.compare_first_edge_to_reviewed(oracles['first_edge_value'], reviewed['oracle_value'])['identical']
    by_id = {r['episode_id']: r for r in dev}
    for name, pick in (('first_edge_value', 0), ('end_edge_value', -1)):
        assert [r['fact'] for r in oracles[name]] == [by_id[r['episode_id']]['path']['fact_indices'][pick]
                                                       for r in oracles[name]]
    first, end = oracles['first_edge_value'], oracles['end_edge_value']
    assert all(a == b for a, b in zip(first, end) if a['hop'] == 1)  # hop 1: the same fact
    assert all(a['fact'] != b['fact'] for a, b in zip(first, end) if a['hop'] > 1)
    summary = diag.summarize_by_hop(end)
    assert {k: v['count'] for k, v in summary.items()} == {'hop1': 8, 'hop2': 8, 'hop3': 8}


def test_oracles_on_a_looped_arm_change_with_the_fact(base, tok, dev):
    with v3.arm_model(base, 'V3-03', D5/'adapter') as model:
        oracles = diag.evaluate_oracles(model, dev, tok.pad_token_id, 4)
        again = diag.evaluate_oracles(model, dev, tok.pad_token_id, 4)
    assert again == oracles
    first, end = oracles['first_edge_value'], oracles['end_edge_value']
    assert any(a['nll'] != b['nll'] for a, b in zip(first, end) if a['hop'] > 1)
    assert all(r['steps'] == 4 for r in first)
