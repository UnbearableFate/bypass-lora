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


@pytest.fixture(scope='module')
def rbbase():
    from calibrate import load_base
    return load_base()


@pytest.mark.parametrize('arm', ['RB1-03', 'RB1-09'])
@pytest.mark.parametrize('kind', ['normal', 'iv_strict', 'iv_window8'])
def test_batched_gold_forwards_match_batch_one_e0021_path(rbbase, arm, kind):
    """Right-padded batches reproduce the reviewed batch-one reads and gold NLL."""
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
                                            batch_size=4, pad_id=tok.pad_token_id)
    check = compare_reference({'predictions': predictions, 'read_diagnostics': reads}, kind, summaries, nll)
    assert check['gold_nll_max_abs_diff'] < 5e-2
    assert max(check['read_summary_max_rel_diff'].values()) < 1e-2
    assert {(r['task'], r['step']) for r in acc.finish()} == {
        (t, s) for t in ('piqa', 'boolq') for s in ((1, 2) if arm == 'RB1-03' else (1,))}
    if kind == 'iv_window8':
        assert acc.window_violation == 0
