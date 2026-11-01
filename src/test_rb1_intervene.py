"""Same-allocation model checks of V1 scoring and frozen history diagnostics."""
import pytest
import torch
from transformers import AutoTokenizer

from calibrate import load_base, MODEL, REVISION
from rb1_intervene import evaluate_history, original_backbone, MODES
from rb_train import adapter_config
from test_rb_model import attached
from test_model import nonzero_exits
from v1_run import evaluate


@pytest.fixture(scope='module')
def fixture_data():
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    tok.pad_token_id = tok.eos_token_id
    rows = {'piqa': [{'task': 'piqa', 'id': i, 'goal': f'Move object {i} onto the table.',
                       'solutions': ['Lift the object.', 'Drop the table.'], 'label': i % 2} for i in range(2)],
            'boolq': [{'task': 'boolq', 'id': i, 'passage': f'A red object {i} is on the table.',
                       'question': 'Is there an object on the table?', 'label': i == 0} for i in range(2)]}
    return tok, {'splits': {task: {'dev': value} for task, value in rows.items()}}


@pytest.fixture(scope='module')
def rbbase():
    return load_base()


@pytest.mark.parametrize('arm', ['RB1-01', 'RB1-02', 'RB1-04', 'RB1-08', 'RB1-09'])
def test_v1_scorer_identity_off_base_and_native_first_last_reads(rbbase, fixture_data, arm):
    tok, splits = fixture_data
    with attached(rbbase, adapter_config(arm)) as model, torch.no_grad():
        model.eval()
        nonzero_exits(model)
        direct_metrics, direct = evaluate(model, splits, tok, split_names=('dev',))
        normal_metrics, normal, diagnostics = evaluate_history(model, splits, tok)
        assert direct == normal and direct_metrics == normal_metrics
        if arm != 'RB1-01':
            assert {row['layer'] for row in diagnostics} == {6, 13, 20, 27}
            assert {row['step'] for row in diagnostics} == ({1, 4} if arm == 'RB1-04' else {1})
            assert all(0 <= row['means']['relative_n_eff'] <= 1.000001 for row in diagnostics)
        with original_backbone(model) as base:
            base_metrics, base_rows = evaluate(base, splits, tok, split_names=('dev',))
        off_metrics, off, _ = evaluate_history(model, splits, tok, 'iv_off')
        assert off == base_rows and off_metrics == base_metrics
        # All wrappers and their state protocol are restored even after Base scoring.
        assert set(model.adapters()) == {6, 13, 20, 27}
        for kind in (('iv_swap',) if arm == 'RB1-01' else MODES[1:]):
            metrics, changed, reads = evaluate_history(model, splits, tok, kind)
            assert len(changed) == 4
            assert all(torch.isfinite(torch.tensor(row['scores'])).all() for row in changed)
            if arm == 'RB1-01':
                assert changed == normal and metrics == normal_metrics
            else:
                assert reads and {row['step'] for row in reads} == ({1, 4} if arm == 'RB1-04' else {1})


def test_original_backbone_restores_wrappers_on_failure(rbbase):
    with attached(rbbase, adapter_config('RB1-02')) as model:
        adapted = model.adapters()
        with pytest.raises(RuntimeError, match='synthetic failure'):
            with original_backbone(model):
                raise RuntimeError('synthetic failure')
        assert all(model.model.model.layers[i] is layer for i, layer in adapted.items())
