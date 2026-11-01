"""Crash recovery and process-lifetime unseal ownership regressions."""
import json
import multiprocessing
import os
from pathlib import Path

import pytest

import v5_confirm as confirm
from v5_analysis import analyze


def _hold_claim(lock, lock_path, output, acquired, release, reached_reader):
    with confirm.claim_unseal(lock, lock_path, output, resume=Path(lock['unseal_receipt']).exists()):
        acquired.set()
        # Reaching this line is the synthetic semantic-reader boundary.
        reached_reader.set()
        release.wait(20)


def _claim_once(lock, lock_path, output, acquired):
    with confirm.claim_unseal(lock, lock_path, output, resume=True):
        acquired.set()


def _compete(lock, lock_path, output, rejected, reached_reader):
    try:
        with confirm.claim_unseal(lock, lock_path, output, resume=True):
            reached_reader.set()
    except confirm.LockError:
        rejected.set()


def _receipt_fixture(tmp_path):
    lock_path = tmp_path / 'lock.json'
    lock_path.write_text('{"synthetic":"locked"}\n')
    output = tmp_path / 'attempt-output'
    lock = {'unseal_receipt': str(tmp_path / 'receipt.json'), 'code_commit': 'a' * 40,
            'checkpoints': [{'arm': arm, 'seed': seed} for arm in ('V5-C', 'V5-F', 'V5-L')
                            for seed in (1, 2, 3)]}
    return lock, lock_path, output


def _synthetic_predictions():
    rows = []
    for task, n in (('piqa', 1838), ('boolq', 3270)):
        for ident in range(n):
            rows.append({'task': task, 'split': 'validation', 'id': f'{task}-{ident}',
                         'gold': ident % 2, 'prediction': ident % 2, 'scores': [0., -1.],
                         'gold_nll_sum': 1., 'gold_loss_tokens': 1})
    return rows


def _publish_all(lock, lock_path, output):
    output.mkdir(parents=True, exist_ok=True)
    digest = confirm.sha256_file(lock_path)
    rows = _synthetic_predictions()
    audit = {task: {'raw_examples': n, 'examples': n, 'excluded_ids': [],
                    'excluded': [], 'excluded_fraction': 0., 'protocol_deviation': False}
             for task, n in (('piqa', 1838), ('boolq', 3270))}
    specs = [('Base', None, 2.0)] + [(arm, seed, 2.0 + .1 * seed)
                                      for arm in ('V5-C', 'V5-F', 'V5-L')
                                      for seed in (1, 2, 3)]
    for arm, seed, wiki_nll in specs:
        name = 'Base' if seed is None else f'{arm}-s{seed}'
        model_output = output / name
        model_output.mkdir(parents=True, exist_ok=True)
        metrics = {'metrics': {'wikitext.nll': wiki_nll}, 'validation': audit,
                   'wikitext': {'input_tokens': 2048, 'blocks': 2}}
        confirm.publish_predictions(model_output, arm, seed, metrics, rows, digest)


def _fail_on_atomic_target(monkeypatch, target_name, *, after_replace=False):
    original = confirm._atomic_write_bytes
    seen = {'count': 0}

    def injected(path, data):
        path = Path(path)
        if path.name == target_name and not after_replace:
            raise OSError(f'injected failure before {target_name}')
        result = original(path, data)
        if path.name == target_name:
            seen['count'] += 1
            if after_replace:
                raise OSError(f'injected failure after {target_name} replace')
        return result

    monkeypatch.setattr(confirm, '_atomic_write_bytes', injected)
    return seen


def test_claim_lifetime_blocks_second_process_and_incomplete_retry_succeeds(tmp_path):
    lock, lock_path, output = _receipt_fixture(tmp_path)
    ctx = multiprocessing.get_context('fork')
    first_acquired, rejected = ctx.Event(), ctx.Event()
    release, first_reader, second_reader = ctx.Event(), ctx.Event(), ctx.Event()
    first = ctx.Process(target=_hold_claim,
                        args=(lock, lock_path, output, first_acquired, release, first_reader))
    first.start()
    assert first_acquired.wait(5)
    assert first_reader.is_set()

    second = ctx.Process(target=_compete,
                         args=(lock, lock_path, tmp_path / 'second-output', rejected, second_reader))
    second.start()
    assert rejected.wait(5), 'the competing process did not reject the active owner'
    assert not second_reader.is_set(), 'a concurrent resume reached semantic evaluation'
    second.join(5)
    release.set()
    first.join(5)
    assert first.exitcode == 0 and second.exitcode == 0

    # An incomplete started receipt becomes retryable once its owner exits.
    retry_acquired = ctx.Event()
    retry = ctx.Process(target=_claim_once, args=(lock, lock_path, tmp_path / 'retry-output', retry_acquired))
    retry.start()
    assert retry_acquired.wait(5)
    retry.join(5)
    assert retry.exitcode == 0


def test_crash_after_tenth_prediction_publication_is_terminal_and_recoverable(tmp_path, monkeypatch):
    lock, lock_path, output = _receipt_fixture(tmp_path)
    with confirm.claim_unseal(lock, lock_path, output) as receipt:
        assert receipt['status'] == 'started'
    original = confirm._atomic_write_bytes
    count = {'predictions': 0}

    def crash_after_tenth(path, data):
        result = original(path, data)
        if Path(path).name == 'predictions.jsonl':
            count['predictions'] += 1
            if count['predictions'] == 10:
                raise OSError('simulated crash after final prediction publication')
        return result

    monkeypatch.setattr(confirm, '_atomic_write_bytes', crash_after_tenth)
    with pytest.raises(OSError, match='after final prediction'):
        _publish_all(lock, lock_path, output)
    assert count['predictions'] == 10
    monkeypatch.setattr(confirm, '_atomic_write_bytes', original)

    with pytest.raises(confirm.LockError, match='predictions already complete'):
        with confirm.claim_unseal(lock, lock_path, output, resume=True):
            pytest.fail('complete predictions must never trigger a second evaluation')
    input_path = output / 'analysis-input.json'
    assert input_path.exists()
    report = analyze(input_path, 'validation', repetitions=8)
    assert report['counts_by_task'] == {'piqa': 1838, 'boolq': 3270}
    receipt_after = confirm.read_json(lock['unseal_receipt'])
    assert receipt_after['status'] == 'predictions_complete'


def test_receipt_replace_failure_preserves_valid_started_state_and_retry_reconciles(
        tmp_path, monkeypatch):
    lock, lock_path, output = _receipt_fixture(tmp_path)
    with confirm.claim_unseal(lock, lock_path, output) as receipt:
        assert receipt['status'] == 'started'
    _publish_all(lock, lock_path, output)
    receipt_path = Path(lock['unseal_receipt'])
    prior = confirm.read_json(receipt_path)
    original_replace = os.replace
    failed = {'once': False}

    def fail_receipt_once(source, destination):
        if Path(destination) == receipt_path and not failed['once']:
            failed['once'] = True
            raise OSError('simulated receipt replace failure')
        return original_replace(source, destination)

    monkeypatch.setattr(confirm.os, 'replace', fail_receipt_once)
    with pytest.raises(OSError, match='receipt replace'):
        confirm.recover_analysis_input(lock, prior)
    assert confirm.read_json(receipt_path) == prior
    assert (output / 'analysis-input.json').exists()

    monkeypatch.setattr(confirm.os, 'replace', original_replace)
    recovered = confirm.recover_analysis_input(lock, confirm.read_json(receipt_path))
    assert Path(recovered) == output / 'analysis-input.json'
    assert confirm.read_json(receipt_path)['status'] == 'predictions_complete'


def test_failure_before_analysis_manifest_keeps_bundles_and_resume_repairs(tmp_path, monkeypatch):
    lock, lock_path, output = _receipt_fixture(tmp_path)
    with confirm.claim_unseal(lock, lock_path, output) as receipt:
        assert receipt['status'] == 'started'
    _publish_all(lock, lock_path, output)
    prior = confirm.read_json(lock['unseal_receipt'])
    _fail_on_atomic_target(monkeypatch, 'analysis-input.json')
    with pytest.raises(OSError, match='analysis-input.json'):
        confirm.recover_analysis_input(lock, prior)
    receipt_path = Path(lock['unseal_receipt'])
    assert confirm.read_json(receipt_path)['status'] == 'started'
    assert len(list(output.glob('*/prediction-bundle.json'))) == 10
    assert len(list(output.glob('*/predictions.jsonl'))) == 10

    monkeypatch.undo()
    with pytest.raises(confirm.LockError, match='predictions already complete'):
        with confirm.claim_unseal(lock, lock_path, output, resume=True):
            pytest.fail('resume must repair analysis metadata then refuse evaluation')
    input_path = output / 'analysis-input.json'
    assert input_path.exists()
    report = analyze(input_path, 'validation', repetitions=8)
    assert report['primary']['decision'] in ('supported', 'challenged', 'inconclusive')
