"""E-0023 readiness rule R on synthetic records (CPU)."""
import json

import pytest

import v3_ready as ready

MODES = ('normal', 'zero', 'shuffled')


def rows(per_hop=40):
    out = []
    for hop in (1, 2, 3):
        for i in range(per_hop):
            out.append({'episode_id': f'h{hop}e{i}', 'axes': {'hop': hop},
                        'path': {'fact_indices': [i % 16, (i+1) % 16, (i+2) % 16][:hop]}})
    return out


def records(data, correct=None, weight=.5, argmax=None, nll=None):
    """normal: `correct` hop-1 correct examples; controls none. argmax(i) gives the hop-1 last argmax."""
    correct = 10 if correct is None else correct
    nll = nll or {'normal': 2., 'zero': 8., 'shuffled': 7.}
    out = {}
    for mode in MODES:
        group = []
        for k, r in enumerate(data):
            i = int(r['episode_id'].split('e')[1])
            first = r['path']['fact_indices'][0]
            group.append({'episode_id': r['episode_id'], 'correct': mode == 'normal' and r['axes']['hop'] == 1 and i < correct,
                          'nll': nll[mode], 'target_fact_position': first, 'question_tokens': 23,
                          'last_target_weight': weight if mode == 'normal' else 0., 'all_target_weight': .05,
                          'last_argmax': (argmax(i) if argmax and r['axes']['hop'] == 1 else first)})
        out[mode] = group
    return out


def test_passes_at_every_boundary():
    data = rows()
    # 4/40 = 1/10 margin, weight exactly 1/8, top slot 10/40 = 1/4: all inclusive thresholds hold.
    result = ready.readiness(records(data, correct=4, weight=.125, argmax=lambda i: 0 if i < 10 else 1+i % 15), data)
    c = result['criteria']
    assert c['hop1_em_margin']['fraction'] == '1/10' and c['hop1_em_margin']['passed']
    assert c['answer_position_first_fact_weight']['passed']
    assert c['hop1_answer_position_top_slot_share']['fraction'] == '1/4'
    assert c['hop1_answer_position_top_slot_share']['passed'] and result['passed']
    json.dumps(ready._descriptive({m: {h: [r for r in g if r['episode_id'].startswith(f'h{h}')] for h in (1, 2, 3)}
                                   for m, g in records(data).items()}))


@pytest.mark.parametrize('kwargs, failing', [
    ({'correct': 3}, 'hop1_em_margin'),
    ({'nll': {'normal': 7., 'zero': 8., 'shuffled': 7.}}, 'hop1_nll'),
    ({'weight': .1249999}, 'answer_position_first_fact_weight'),
    ({'argmax': lambda i: 0 if i < 11 else 1+i % 15}, 'hop1_answer_position_top_slot_share'),
])
def test_each_criterion_fails_alone(kwargs, failing):
    data = rows()
    kwargs.setdefault('argmax', lambda i: i % 16)
    result = ready.readiness(records(data, **kwargs), data)
    assert not result['passed']
    assert [k for k, v in result['criteria'].items() if not v['passed']] == [failing]


def test_one_hop_below_threshold_fails_rule_three():
    data = rows()
    recs = records(data)
    for r in recs['normal']:
        if r['episode_id'].startswith('h3'):
            r['last_target_weight'] = .1
    c = ready.readiness(recs, data)['criteria']['answer_position_first_fact_weight']
    assert not c['passed'] and c['by_hop']['hop1']['passed'] and not c['by_hop']['hop3']['passed']


def test_rejects_mismatched_records_and_targets():
    data = rows()
    recs = records(data)
    recs['zero'] = recs['zero'][1:]
    with pytest.raises(ValueError, match='do not match'):
        ready.readiness(recs, data)
    recs = records(data)
    recs['normal'][0]['target_fact_position'] += 1
    with pytest.raises(ValueError, match='first path fact'):
        ready.readiness(recs, data)
