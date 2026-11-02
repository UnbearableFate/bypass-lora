"""A2 access cannot precede the locked dev gate; synthetic rows only."""
from types import SimpleNamespace

import pytest

import seg_a2_eval as runner
from test_seg_artifacts import predictions


def test_same_slice_requires_row_identity_and_other_slice_class_limit():
    expected = predictions()
    actual = [dict(r) for r in expected]
    assert runner.dev_gate(actual, expected, '2g', '2g')['passed']
    actual[0]['scores'] = [s+1 for s in actual[0]['scores']]
    assert not runner.dev_gate(actual, expected, '2g', '2g')['passed']
    assert runner.dev_gate(actual, expected, '1g', '2g')['passed']
    for i in range(10):
        actual[i].update(prediction=1-expected[i]['prediction'], scores=list(reversed(expected[i]['scores'])))
    assert runner.dev_gate(actual, expected, '1g', '2g')['passed']
    actual[10].update(prediction=1-expected[10]['prediction'], scores=list(reversed(expected[10]['scores'])))
    assert not runner.dev_gate(actual, expected, '1g', '2g')['passed']


def test_failed_lock_or_missing_unseal_never_opens_a2(monkeypatch, tmp_path):
    monkeypatch.setenv('SR_SPEC_DIGEST', 'synthetic')
    monkeypatch.setattr(runner, 'read_a2', lambda *a, **k: pytest.fail('A2 opened'))
    monkeypatch.setattr(runner, 'verify_lock', lambda *a: (_ for _ in ()).throw(ValueError('lock rejected')))
    args = SimpleNamespace(output=tmp_path, unseal=False, lock=tmp_path/'lock', lock_sha256='0'*64, seed=3)
    with pytest.raises(ValueError, match='unseal'):
        runner.run(args)
    args.unseal = True
    with pytest.raises(ValueError, match='lock rejected'):
        runner.run(args)
