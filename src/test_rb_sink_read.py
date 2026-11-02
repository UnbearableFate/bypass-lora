"""E-0024 aggregation: question-level shares, first-position vectors, cosine and rules."""
import pytest
import torch

from rb_sink_read import Accumulator, interpret, mean_pairwise_cosine


def test_mean_pairwise_cosine_on_constructed_vectors():
    assert mean_pairwise_cosine(torch.ones(5, 3)) == pytest.approx(1)
    assert mean_pairwise_cosine(torch.eye(3)) == pytest.approx(0)
    assert mean_pairwise_cosine(torch.tensor([[1., 0.], [-2., 0.]])) == pytest.approx(-1)
    # Pairs (a,b)=1, (a,c)=0, (b,c)=0: the diagonal is excluded from the mean.
    assert mean_pairwise_cosine(torch.tensor([[1., 0.], [3., 0.], [0., 2.]])) == pytest.approx(1/3)
    with pytest.raises(ValueError):
        mean_pairwise_cosine(torch.tensor([[1., 0.], [0., 0.]]))


def test_question_means_differ_from_position_weighted_and_first_position_is_sampled():
    # Row 0 answers at positions 1-3 (share0 1,1,1); row 1 answers at position 2 only (share0 0).
    labels = torch.tensor([[-100, -100, 5, 6, 7], [-100, -100, -100, 8, -100]])
    mask = torch.ones(2, 5, dtype=torch.long)
    share = torch.tensor([[0., 1., 1., 1., 0.], [0., 0., 0., 0., 0.]])
    read = torch.arange(40.).reshape(2, 5, 4)
    values = -read
    record = {'step': 1, 'read_vector': read, **{name: share for name in
              ('share0', 'share0_2', 'argmax0', 'uniform0', 'uniform0_2')}}
    acc = Accumulator({'piqa': [11]})
    acc.add('piqa', [10, 11], {6: {'history': {'values': values}, 'read_metrics': [record]}},
            mask, labels, 'iv_window8')
    assert acc.rows[('piqa', 6, 1)]['share0'] == [1., 0.]   # question mean 0.5, position-weighted 0.75
    store = acc.vectors[('piqa', 6, 1)]
    assert store['ids'] == [11] and store['first_position'] == [2]
    assert torch.equal(store['read'][0], read[1, 2].double())
    assert torch.equal(store['value'][0], values[1, 2].double())
    # Every answer position is below 8, so none can violate the window rule.
    assert acc.window_short_positions == 4 and acc.window_violation == 0


def _arm(cells):
    return {'concentrated': [key for key, _, _ in cells],
            'conditions': {'normal': {'answer_question_level': [
                {'task': key[0], 'layer': key[1], 'step': key[2], 'question_means': {'share0': s},
                 'constant_read': {'C_read': c}} for key, s, c in cells]}}}


@pytest.mark.parametrize('values,reading', [
    ([(.51, .91), (.9, .99)], 'sink'),
    ([(.51, .91), (.9, .9)], 'mixed'),       # C must exceed 0.9
    ([(.19, .99), (.0, .1)], 'content'),
    ([(.19, .99), (.2, .1)], 'mixed'),       # share0 must be below 0.2
])
def test_preregistered_reading_thresholds(values, reading):
    arms = {'RB1-02': _arm([(('piqa', 27, 1), *values[0])]),
            'RB1-03': _arm([(('boolq', 27, 1), *values[1])])}
    assert interpret(arms)['reading'] == reading


def _reference(nll, n_eff):
    return ({'predictions': [{'task': 'piqa', 'id': 1, 'gold_nll_sum': 2.0}],
             'read_diagnostics': [{'task': 'piqa', 'layer': 6, 'step': 1, 'scope': 'answer_prediction',
                                   'count': 3, 'empty_reads': 0, 'nonempty_means': {'n_eff': 1.5}}]},
            [{'task': 'piqa', 'layer': 6, 'step': 1, 'scope': 'answer_prediction', 'count': 3,
              'empty_reads': 0, 'nonempty_means': {'n_eff': n_eff}}],
            {('piqa', 1): nll})


@pytest.mark.parametrize('nll,n_eff,passed', [
    (2.0, 1.5, True), (2.0+5e-5, 1.5, True), (2.001, 1.5, False),
    (2.0, 1.5*(1+1e-5), False), (float('nan'), 1.5, False), (2.0, float('inf'), False)])
def test_reference_gate_rejects_excess_or_nonfinite_differences(nll, n_eff, passed):
    from rb_sink_read import compare_reference
    reference, summaries, values = _reference(nll, n_eff)
    assert compare_reference(reference, 'normal', summaries, values)['passed'] is passed


def _report_arms(strict_gap=0., late=0., reference_ok=True):
    arms = {}
    for arm in ('RB1-02', 'RB1-03', 'RB1-08', 'RB1-09'):
        cell = lambda share: [{'task': 'piqa', 'layer': 27, 'step': 1,
                               'question_means': {'share0': share}, 'constant_read': {'C_read': .95}}]
        conditions = {c: {'answer_question_level': cell(.6 + (strict_gap if c == 'iv_strict' else 0)),
                          'reference_check': {'passed': reference_ok}, 'elapsed_s': 1.,
                          'window8_share0_max_late': late if c == 'iv_window8' else None,
                          'window8_answer_positions_before_8': 0 if c == 'iv_window8' else None}
                      for c in ('normal', 'iv_strict', 'iv_window8')}
        arms[arm] = {'arm': arm, 'concentrated': [('piqa', 27, 1)], 'position0_token': {},
                     'conditions': conditions}
    return arms


@pytest.mark.parametrize('kwargs,reading', [
    ({}, 'sink'), ({'strict_gap': .02}, 'implementation_check_failed'),
    ({'late': 1e-3}, 'implementation_check_failed'), ({'reference_ok': False}, 'implementation_check_failed')])
def test_failed_consistency_or_reference_checks_suppress_the_reading(kwargs, reading):
    from rb_sink_read import build_report
    report = build_report(_report_arms(**kwargs))
    assert report['interpretation']['reading'] == reading
    assert report['checks_passed'] is (reading == 'sink')


@pytest.mark.parametrize('arms,checks,expected', [(4, True, True), (3, True, False), (4, False, False)])
def test_manifest_passes_only_with_all_arms_and_passing_summary(tmp_path, monkeypatch, arms, checks, expected):
    import json
    from argparse import Namespace
    from rb_sink_read import ARMS, write_manifest
    monkeypatch.setenv('SR_SPEC_DIGEST', 'blake2b:test')
    for arm in ARMS[:arms]:
        (tmp_path/arm).mkdir()
        (tmp_path/arm/'sink.json').write_text('{}')
    reading = 'sink' if checks else 'implementation_check_failed'
    (tmp_path/'summary.json').write_text(json.dumps({'checks_passed': checks,
                                                     'interpretation': {'reading': reading}}))
    manifest = write_manifest(Namespace(output=tmp_path, elapsed_s=1.))
    assert manifest['validator']['passed'] is expected
    assert manifest['status'] == ('complete' if expected else 'partial')
    assert json.loads((tmp_path/'manifest.json').read_text()) == manifest


@pytest.fixture(scope='module')
def rbbase():
    from calibrate import load_base
    return load_base()


@pytest.mark.parametrize('arm', ['RB1-03', 'RB1-09'])
@pytest.mark.parametrize('kind', ['normal', 'iv_strict', 'iv_window8'])
def test_gold_only_forwards_match_the_e0021_path(rbbase, arm, kind):
    """Batch-one gold forwards pass the production gate against the reviewed path."""
    from transformers import AutoTokenizer
    from calibrate import MODEL, REVISION
    from rb1_intervene import evaluate_history
    from rb_sink_read import compare_reference, gold_index, run_condition
    from rb_train import adapter_config
    from test_model import nonzero_exits
    from test_rb_model import attached
    from v1_data import encode_candidates
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    tok.pad_token_id = tok.eos_token_id
    rows = {'piqa': [{'task': 'piqa', 'id': i, 'goal': 'Move the object ' + 'very '*(3*i) + 'carefully.',
                      'solutions': ['Lift it ' + 'slowly '*i + 'up.', 'Drop the table.'], 'label': i % 2}
                     for i in range(5)],
            'boolq': [{'task': 'boolq', 'id': i, 'passage': 'A red object is on the table. '*(1+2*i),
                       'question': 'Is there an object on the table?', 'label': i % 2 == 0} for i in range(5)]}
    splits = {'splits': {task: {'dev': value} for task, value in rows.items()}}
    encoded = {task: [encode_candidates(tok, row)[gold_index(row)] for row in value]
               for task, value in rows.items()}
    with attached(rbbase, adapter_config(arm)) as model, torch.no_grad():
        model.eval()
        nonzero_exits(model)
        _, predictions, reads = evaluate_history(model, splits, tok, None if kind == 'normal' else kind)
        summaries, acc, nll = run_condition(model, rows, encoded, kind, {'piqa': [0, 3], 'boolq': [1, 4]},
                                            batch_size=1, pad_id=tok.pad_token_id)
    check = compare_reference({'predictions': predictions, 'read_diagnostics': reads}, kind, summaries, nll)
    assert check['passed'], check
    # Batch-one gold forwards repeat the reference computation exactly.
    assert check['gold_nll_max_abs_diff'] == 0, check
    assert max(check['read_summary_max_rel_diff'].values()) == 0, check
    assert {(r['task'], r['step']) for r in acc.finish()} == {
        (t, s) for t in ('piqa', 'boolq') for s in ((1, 2) if arm == 'RB1-03' else (1,))}
    if kind == 'iv_window8':
        assert acc.window_violation == 0
