"""E-0023 rule 4 frozen-transfer diagnostic and the D-0034 mapping. Core tests run anywhere; model
tests need a GPU allocation and use the D5 seed-0 adapter as a stand-in start (T2 dev only)."""
from fractions import Fraction

import pytest
import torch

import v3_loop as loop
import v3_train as v3
import v3_transfer as tr
from test_v3_train import D5, base, dev, gpu_batch, tok  # noqa: F401


def fake_row(i, path_entities=('A', 'B', 'C'), facts=None):
    facts = facts or [{'head': 'A', 'relation': 'r', 'tail': 'B'}, {'head': 'B', 'relation': 'r', 'tail': 'C'},
                      {'head': 'D', 'relation': 's', 'tail': 'E'}, {'head': 'B', 'relation': 'r', 'tail': 'X'},
                      {'head': 'F', 'relation': 'r', 'tail': 'G'}]
    return {'episode_id': f'e{i}', 'facts': facts, 'query': {'relation': 'r'},
            'path': {'entities': list(path_entities), 'fact_indices': [0, 1]}}


def test_wrong_fact_is_same_relation_off_path_with_a_non_path_head():
    assert tr.wrong_fact(fake_row(0)) == 4  # fact 3 has a path head (B); fact 2 another relation
    row = fake_row(0, facts=fake_row(0)['facts'][:4])
    assert tr.wrong_fact(row) is None


def test_donor_bridge_is_the_nearest_earlier_bridge_absent_from_the_episode():
    rows = [fake_row(0), fake_row(1, ('P', 'Q', 'R')), fake_row(2, ('S', 'F', 'T')), fake_row(3, ('U', 'V', 'W'))]
    assert tr.donor_bridge(rows, 3) == 'Q'  # row 2's bridge F occurs in row 3's facts; row 1's Q does not
    assert tr.donor_bridge(rows, 0) == 'V'  # cyclic
    same = [fake_row(i) for i in range(3)]
    with pytest.raises(ValueError):
        tr.donor_bridge(same, 0)


def rec(fact2_step1, fact2_step2=0., hop=2, correct=False):
    w1, w2 = [0.]*16, [0.]*16
    w1[5], w2[5] = fact2_step1, fact2_step2
    return {'hop': hop, 'path': [4, 5] if hop == 2 else [4, 5, 6], 'correct': correct, 'nll': 1.,
            'own_fact': None, 'weights_last': [w1, w2]}


def scenario(end=12, zero=0, shuffled=2, n=100, bridge=.125, wrong=.1, donor=.1, native=.1):
    records = {'native': [rec(0., native)], 'bridge': [rec(bridge)], 'wrong_bridge': [rec(wrong)],
               'donor_bridge': [rec(donor)], 'forced_first': [rec(0., .9)]}
    oracles = {'end_edge_value': [{'correct': i < end} for i in range(n)],
               'first_edge_value': [{'correct': False} for _ in range(n)]}
    controls = {'zero': [{'correct': i < zero} for i in range(n)], 'shuffled': [{'correct': i < shuffled} for i in range(n)]}
    return tr.mapping(records, oracles, controls)


def test_d0034_boundaries_and_correspondence():
    # gap exactly 10 pp, bridge exactly 1/8, controls and native below: (a)(b)(c) -> successor CE
    result = scenario()
    c = result['criteria']
    assert c['a_end_value_decodes']['gap_exact'] == '10' and all(v['passed'] for v in c.values())
    assert result['repair'].startswith('successor-slot CE')
    assert scenario(end=11)['repair'].startswith('none: (a) fails')
    assert scenario(shuffled=3)['repair'].startswith('none: (a) fails')  # max(zero, shuffled)
    assert scenario(bridge=.1249)['repair'].startswith('linear-start')
    assert scenario(wrong=.125)['repair'].startswith('linear-start')  # a control at 1/8 is not below it
    assert scenario(donor=.2)['repair'].startswith('linear-start')
    assert scenario(native=.125)['repair'].startswith('none: (a)(b) hold but (c) fails')
    assert scenario()['criteria']['b_bridge_addresses_next_edge']['bridge_step1_fact2'] == Fraction(.125)


def test_summary_reports_every_step_on_every_path_fact():
    records = {'native': [rec(.2, .3), rec(.4, .1, hop=3)]}
    for r in records['native']:
        r['weights_last'][0][4] = .5
    summary = tr.summarize_reads(records, 2)['native']
    assert summary['hops23']['count'] == 2 and len(summary['hops23']['path_weight_last'][0]) == 2
    assert summary['hop2']['path_weight_last'] == [[.5, .2], [0., .3]]
    assert summary['hop3']['path_weight_last'][0] == [.5, .4, 0.]


# ---- GPU ------------------------------------------------------------------------------------------

def test_rewritten_questions_keep_the_label_and_template(tok, dev):
    from v2_data import encode
    rows = [r for r in dev if r['axes']['hop'] > 1]
    reads = tr.read_rows(tok, rows)
    for original, bridge, donor in zip(rows, reads['bridge'], reads['donor_bridge']):
        b = original['path']['entities'][1]
        assert bridge['question_text'] == original['question_text'].replace(f"from {original['query']['head']},", f'from {b},')
        assert bridge['question_input_ids'] == encode(tok, bridge['question_text'])
        assert bridge['query'] == original['query'] and bridge['facts'] == original['facts']
        name = donor['question_text'].split('from ')[1].split(',')[0]
        assert name not in {x for f in original['facts'] for x in (f['head'], f['tail'])}
    for w in reads['wrong_bridge']:
        assert w['question_text'].startswith(f"Starting from {w['facts'][w['_own_fact']]['head']},")


def test_native_read_is_the_arm_forward_and_forced_read_matches_its_closed_form(base, tok, dev):
    rows = [r for r in dev if r['axes']['hop'] > 1]
    with v3.arm_model(base, 'V3-02', D5/'adapter') as model:
        model.eval()
        records = tr.frozen_reads(model, {'native': rows, 'forced_first': rows}, tok.pad_token_id, 2)
        arm, _ = v3.evaluate_arm(model, rows, tok.pad_token_id, 2, ('normal',))
        assert [(r['prediction'], r['nll']) for r in records['native']] == \
               [(r['prediction'], r['nll']) for r in arm['normal']]
        # forced: step-2 weights come from z1 = q + eta (v[fact1] g) Ro
        batch = gpu_batch(tok, rows[:4])
        side = model.adapters()[14].side
        with torch.no_grad(), v3.eval_steps(model, 2):
            snapshot = loop._snapshots(model, batch, 'normal', None)
            _, seen, _ = v3._capture(model, batch, snapshot)
            row = torch.arange(4, device='cuda')
            first = torch.tensor([r['path']['fact_indices'][0] for r in rows[:4]], device='cuda')
            z1 = seen['q'] + side.reader.eta*((snapshot.values[row, first][:, None]*side.reader.read_gain) @ side.reader.Ro)
            query = v3.elu_plus_one(v3.rmsnorm_no_affine(z1, side.reader.eps_norm) @ side.reader.Rq)
            _, weights = v3.slot_read(query, snapshot)
            last = batch.question_mask.long().sum(-1)-1
            expected = weights[row, last]
        got = torch.tensor([r['weights_last'][1] for r in records['forced_first'][:4]], device='cuda')
        assert torch.allclose(got, expected, atol=1e-5)
        assert any(a['nll'] != b['nll'] for a, b in zip(records['native'], records['forced_first']))
        with pytest.raises(ValueError):
            tr.frozen_batch(model, batch, 1, forced=True)
    with v3.arm_model(base, 'V3-06', D5/'adapter') as model, pytest.raises(ValueError):
        tr.frozen_batch(model, gpu_batch(tok, rows[:2]), 4)
