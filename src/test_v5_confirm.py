"""Synthetic safeguards; these tests never access the actual sealed data."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

import v5_confirm as confirm
from v1_data import encode_candidates


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, (dict, list)):
        path.write_text(json.dumps(value))
    else:
        path.write_text(value)
    return confirm.file_reference(path)


@pytest.fixture
def locked(tmp_path, monkeypatch):
    root = Path(confirm.__file__).resolve().parents[1]
    commit = 'a' * 40
    monkeypatch.setenv('SR_CODE_COMMIT', commit)
    data, sealed = {}, {}
    for name in confirm.SEALED:
        reference = put(tmp_path / 'opaque' / name, f'opaque synthetic {name}')
        reference.update(revision='b' * 40, source_url=f'https://example.invalid/{"b" * 40}/{name}')
        if name == 'wikitext':
            reference.update(config='wikitext-103-raw-v1', split='test')
        data[name] = reference
        sealed[name] = (reference['revision'], reference['sha256'])
    monkeypatch.setattr(confirm, 'SEALED', sealed)
    checkpoints = []
    lr_lock = put(tmp_path / 'lr-lock.json', {'contract': confirm.EXPERIMENT,
                  'locked_lrs': {arm: .001 for arm in confirm.PARAMETERS}})
    for arm, parameters in confirm.PARAMETERS.items():
        for seed in (1, 2, 3):
            run_directory = tmp_path / f'{arm}-{seed}'
            directory = run_directory / 'adapter'
            config = {'model_revision': confirm.REVISION,
                      'layers': list(range(28)) if arm == 'V5-L' else [6, 13, 20, 27],
                      'rank': {'V5-C': 32, 'V5-F': 64, 'V5-L': 1}[arm]}
            if arm == 'V5-L':
                config['alpha'] = 1.
            else:
                config.update(branch='tokenwise_silu', fusion='C_up' if arm == 'V5-C' else 'F_out')
            files = {'config.json': put(directory / 'config.json', config),
                     'adapter.pt': put(directory / 'adapter.pt', f'opaque weights {arm} {seed}')}
            inventory = put(run_directory / 'checkpoint.json', files)
            manifest = put(run_directory / 'manifest.json', {'status': 'complete', 'arm': arm, 'seed': seed,
                'contract': confirm.EXPERIMENT, 'phase': 'train', 'validator': {'passed': True},
                'artifacts': [{'path': inventory['path'], 'role': 'checkpoint'}],
                'metrics': {'reload.identical': True, 'params.trainable': parameters, 'learning_rate': .001,
                            'train.examples': 58329, 'train.steps': 29165, 'lr_lock.sha256': lr_lock['sha256']}})
            checkpoints.append({'arm': arm, 'seed': seed, 'path': str(directory),
                'expected_parameters': parameters, 'learning_rate': .001,
                'files': files, 'training_manifest': manifest, 'checkpoint_inventory': inventory})
    contract = put(tmp_path / 'contract.md', '---\nexperiment: E-0013\n---\nSynthetic contract fixture.\n')
    monkeypatch.setattr(confirm, 'CONTRACT_SHA256', contract['sha256'])
    lock = {'schema_version': 1, 'experiment': confirm.EXPERIMENT, 'code_commit': commit,
        'code_files': {p: confirm.sha256_file(root / p) for p in confirm.REQUIRED_CODE},
        'model': confirm.MODEL, 'model_revision': confirm.REVISION, 'tokenizer_revision': confirm.REVISION,
        'max_length': 512, 'wikitext_block_length': 1024,
        'contract': contract,
        'learning_rate_lock': lr_lock,
        'prerequisites': {'code_review': {'id': 'RV-9999', 'passed': True,
            'answer': put(tmp_path / 'review.md', 'Synthetic passing review.')},
            'dev_rehearsal': put(tmp_path / 'rehearsal.json', {'split': 'dev'})},
        'unseal_receipt': str(tmp_path / 'unseal.json'), 'checkpoints': checkpoints, 'data': data}
    path = tmp_path / 'lock.json'
    path.write_text(json.dumps(lock))
    return path, lock


def test_complete_lock_checks_without_parsing_validation(locked, monkeypatch):
    path, expected = locked
    monkeypatch.setattr(confirm, 'read_validation', lambda *a: pytest.fail('validation parsed'))
    assert confirm.verify_lock(path) == expected


def test_missing_lock_and_explicit_unseal_guard_never_parse_validation(tmp_path, monkeypatch):
    monkeypatch.setattr(confirm, 'read_validation', lambda *a: pytest.fail('validation parsed'))
    with pytest.raises(confirm.LockError, match='lock file is required'):
        confirm.run_confirmation(None, tmp_path, unseal=True)
    with pytest.raises(confirm.LockError, match='cannot read JSON'):
        confirm.run_confirmation(tmp_path / 'absent.json', tmp_path, unseal=True)
    with pytest.raises(confirm.LockError, match='explicit --unseal'):
        confirm.run_confirmation(None, tmp_path)


@pytest.mark.parametrize('mutation', ['field', 'checkpoint', 'source', 'data', 'reload', 'rate', 'review'])
def test_bad_lock_refuses_before_validation_content(locked, monkeypatch, mutation):
    path, lock = locked
    monkeypatch.setattr(confirm, 'read_validation', lambda *a: pytest.fail('validation parsed'))
    if mutation == 'field':
        lock.pop('code_files')
    elif mutation == 'checkpoint':
        entry = lock['checkpoints'][-1]['files']['adapter.pt']
        Path(entry['path']).write_text('x' * entry['bytes'])
    elif mutation == 'source':
        lock['code_files']['src/v5_confirm.py'] = '0' * 64
    elif mutation == 'data':
        entry = lock['data']['piqa']
        Path(entry['path']).write_text('x' * entry['bytes'])
    elif mutation == 'reload':
        ref = lock['checkpoints'][-1]['training_manifest']
        obj = json.loads(Path(ref['path']).read_text())
        obj['metrics']['reload.identical'] = False
        lock['checkpoints'][-1]['training_manifest'] = put(Path(ref['path']), obj)
    elif mutation == 'rate':
        lock['checkpoints'][1]['learning_rate'] = .003
    else:
        lock['prerequisites']['code_review']['passed'] = False
    path.write_text(json.dumps(lock))
    with pytest.raises(confirm.LockError):
        confirm.run_confirmation(path, path.parent / 'out', unseal=True)
    assert not Path(lock['unseal_receipt']).exists()


def test_unseal_receipt_retries_require_same_lock(locked):
    path, lock = locked
    with confirm.claim_unseal(lock, path, path.parent / 'first'):
        pass
    with pytest.raises(confirm.LockError, match='already claimed'):
        with confirm.claim_unseal(lock, path, path.parent / 'second'):
            pytest.fail('duplicate claim')
    with confirm.claim_unseal(lock, path, path.parent / 'retry', resume=True):
        pass
    path.write_text(path.read_text() + '\n')
    with pytest.raises(confirm.LockError, match='identical lock'):
        with confirm.claim_unseal(lock, path, path.parent / 'bad-retry', resume=True):
            pytest.fail('changed lock admitted')


def test_completed_predictions_cannot_be_reopened(locked):
    path, lock = locked
    with confirm.claim_unseal(lock, path, path.parent / 'first'):
        pass
    receipt = json.loads(Path(lock['unseal_receipt']).read_text())
    receipt['status'] = 'predictions_complete'
    Path(lock['unseal_receipt']).write_text(json.dumps(receipt))
    with pytest.raises(confirm.LockError, match='predictions already complete'):
        with confirm.claim_unseal(lock, path, path.parent / 'retry', resume=True):
            pytest.fail('completed predictions admitted')


@pytest.mark.parametrize('mutation', ['seed-swap', 'inventory-bytes', 'smoke', 'incomplete', 'contract',
                                     'validator', 'missing-phase', 'lr-lock'])
def test_final_training_provenance_failures_precede_receipt_and_reader(locked, monkeypatch, mutation):
    path, lock = locked
    monkeypatch.setattr(confirm, 'read_validation', lambda *a: pytest.fail('validation parsed'))
    first, second = lock['checkpoints'][:2]
    if mutation == 'seed-swap':
        for key in ('path', 'files'):
            first[key], second[key] = second[key], first[key]
    elif mutation == 'inventory-bytes':
        ref = first['checkpoint_inventory']
        value = json.loads(Path(ref['path']).read_text())
        value['adapter.pt']['sha256'] = '0' * 64
        first['checkpoint_inventory'] = put(Path(ref['path']), value)
    else:
        ref = first['training_manifest']
        value = json.loads(Path(ref['path']).read_text())
        if mutation == 'smoke':
            value['phase'] = 'smoke'
            value['metrics']['train.steps'] = 5
            value['metrics']['plan.steps'] = 29165
        elif mutation == 'incomplete':
            value['metrics']['train.examples'] = 58328
        elif mutation == 'contract':
            value['contract'] = 'E-0013@2'
        elif mutation == 'validator':
            value['validator']['passed'] = False
        elif mutation == 'missing-phase':
            value.pop('phase')
        else:
            value['metrics']['lr_lock.sha256'] = '0' * 64
        first['training_manifest'] = put(Path(ref['path']), value)
    path.write_text(json.dumps(lock))
    with pytest.raises(confirm.LockError):
        confirm.run_confirmation(path, path.parent / 'forbidden', unseal=True)
    assert not Path(lock['unseal_receipt']).exists()


class ToyTokenizer:
    pad_token_id = 0
    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [(ord(character) % 6) + 1 for character in text]


class ToyModel(nn.Module):
    def forward(self, input_ids, attention_mask=None, use_cache=False):
        return SimpleNamespace(logits=F.one_hot(input_ids, num_classes=8).float() * 2)


def examples():
    return {'piqa': [{'task': 'piqa', 'id': 0, 'goal': 'goal', 'solutions': ['a', 'abc'], 'label': 1}],
            'boolq': [{'task': 'boolq', 'id': i, 'question': 'q?', 'passage': 'passage', 'label': answer}
                      for i, answer in enumerate((True, False))]}


def test_scoring_predictions_and_metrics_exactly_match_v1_evaluate(monkeypatch):
    import v1_run
    from v1_data import collate_encoded
    def cpu_batch(encoded, pad):
        return {key: torch.tensor(value, dtype=torch.long)
                for key, value in collate_encoded(encoded, pad).items()}
    monkeypatch.setattr(v1_run, 'batch_tensors', cpu_batch)
    rows, tok = examples(), ToyTokenizer()
    encoded, audit = confirm.encode_validation(rows, tok)
    expected = v1_run.evaluate(ToyModel(), {'splits': {t: {'validation': r} for t, r in rows.items()}},
                               tok, split_names=('validation',))
    observed = confirm.evaluate_candidates(ToyModel(), encoded, tok, device='cpu')
    assert observed == expected
    assert [row['gold'] for row in observed[1] if row['task'] == 'boolq'] == [0, 1]
    assert all(not value['excluded_ids'] for value in audit.values())


def test_exclusions_are_shared_and_only_known_encoding_errors_are_skipped(monkeypatch):
    rows = examples()
    rows['piqa'].append({'task': 'piqa', 'id': 1, 'goal': 'g', 'solutions': ['a' * 512, 'b'], 'label': 0})
    encoded, audit = confirm.encode_validation(rows, ToyTokenizer())
    assert [row['id'] for row, _ in encoded['piqa']] == [0]
    assert audit['piqa']['excluded_ids'] == [1]
    assert audit['piqa']['raw_examples'] == 2 and audit['piqa']['examples'] == 1
    assert audit['piqa']['protocol_deviation'] is True
    monkeypatch.setattr(confirm, 'encode_candidates', lambda *a, **kw: (_ for _ in ()).throw(RuntimeError('broken tokenizer')))
    with pytest.raises(RuntimeError, match='broken tokenizer'):
        confirm.encode_validation(rows, ToyTokenizer())


@pytest.mark.parametrize('task,bad_count', [('piqa', 1837), ('boolq', 3269)])
def test_validation_row_counts_are_checked_before_encoding(monkeypatch, task, bad_count):
    monkeypatch.setattr(confirm, 'read_piqa_train', lambda *a: [{}] * (bad_count if task == 'piqa' else 1838))
    monkeypatch.setattr(confirm, 'read_boolq_train', lambda *a: [{}] * (bad_count if task == 'boolq' else 3270))
    lock = {'data': {k: {'path': '/synthetic'} for k in ('piqa', 'piqa_labels', 'boolq')}}
    with pytest.raises(confirm.LockError, match=f'{task}: validation row count mismatch'):
        confirm.read_validation(lock)


def test_wikitext_joins_raw_rows_and_discards_tail_without_special_tokens():
    class IdentityTokenizer:
        def encode(self, text, add_special_tokens):
            assert text == 'ab\ncdef\nxy' and add_special_tokens is False
            return list(range(len(text)))
    blocks, metadata = confirm.wikitext_blocks(['ab\n', '', 'cdef\n', 'xy'], IdentityTokenizer(), block_length=4)
    assert blocks == [list(range(4)), list(range(4, 8))]
    assert metadata == {'input_tokens': 10, 'blocks': 2, 'discarded_tail_tokens': 2,
                        'predicted_tokens': 6, 'block_length': 4}


def test_wikitext_nll_predicts_only_within_each_block():
    blocks = [[1, 2, 3, 4], [5, 6, 1, 2]]
    result = confirm.evaluate_wikitext(ToyModel(), blocks, device='cpu')
    losses = []
    for block in blocks:
        logits = F.one_hot(torch.tensor(block[:-1]), num_classes=8).float() * 2
        losses.extend(F.cross_entropy(logits, torch.tensor(block[1:]), reduction='none').tolist())
    assert result['wikitext.loss_tokens'] == 6
    assert result['wikitext.nll'] == pytest.approx(sum(losses) / 6)


def test_base_wikitext_command_does_not_verify_or_read_validation(locked, monkeypatch, tmp_path):
    _, lock = locked
    seal = tmp_path / 'seal.json'
    seal.write_text(json.dumps({'files': {'wikitext': lock['data']['wikitext']}}))
    monkeypatch.setattr(confirm, 'verify_lock', lambda *a, **kw: pytest.fail('confirmation lock accessed'))
    monkeypatch.setattr(confirm, 'read_validation', lambda *a: pytest.fail('validation parsed'))
    monkeypatch.setattr(confirm, 'load_tokenizer', lambda: ToyTokenizer())
    monkeypatch.setattr(confirm, 'read_wikitext_blocks', lambda *a: ([[1, 2]], {'blocks': 1}))
    monkeypatch.setattr(confirm, 'load_model', lambda: ToyModel())
    monkeypatch.setattr(confirm, 'evaluate_wikitext', lambda *a: {'wikitext.nll': 2.0})
    assert confirm.run_wikitext_base(seal, tmp_path / 'wiki')['wikitext.nll'] == 2.0


def test_confirmation_orchestration_writes_ten_paired_results_and_closes_receipt(monkeypatch, tmp_path):
    """Exercise output/analysis integration using synthetic rows, never sealed files."""
    rows = {}
    for task in confirm.TASKS:
        prototype = examples()[task][0]
        rows[task] = [{**prototype, 'id': i} for i in range(confirm.EXPECTED_ROWS[task])]
    lock_path = tmp_path / 'synthetic-lock.json'
    lock_path.write_text('synthetic lock for orchestration')
    lock = {'checkpoints': [{'arm': arm, 'seed': seed, 'files': {}}
                           for arm in confirm.PARAMETERS for seed in (1, 2, 3)],
            'data': {'wikitext': {'path': '/synthetic-wiki-only'}},
            'code_commit': 'a' * 40, 'unseal_receipt': str(tmp_path / 'receipt.json')}
    order = []
    monkeypatch.setenv('SR_SPEC_DIGEST', 'synthetic-spec')
    monkeypatch.setattr(confirm, 'verify_lock', lambda path: order.append('verify') or lock)
    def synthetic_reader(current):
        assert Path(current['unseal_receipt']).exists()
        order.append('read-synthetic')
        return rows
    monkeypatch.setattr(confirm, 'read_validation', synthetic_reader)
    monkeypatch.setattr(confirm, 'load_tokenizer', lambda: ToyTokenizer())
    monkeypatch.setattr(confirm, 'load_model', lambda entry: ToyModel())
    monkeypatch.setattr(confirm, 'read_wikitext_blocks', lambda *a: ([[1, 2, 3]], {'blocks': 1}))
    monkeypatch.setattr(confirm, 'candidate_losses', lambda *a: torch.ones(1))
    output = tmp_path / 'results'
    confirm.run_confirmation(lock_path, output, unseal=True)
    assert order == ['verify', 'read-synthetic']
    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest['status'] == 'complete' and manifest['validator']['passed'] is True
    analysis = json.loads((output / 'analysis' / 'analysis.json').read_text())
    assert analysis['counts_by_task'] == {'piqa': 1838, 'boolq': 3270}
    assert analysis['bootstrap']['repetitions'] == 10000
    assert set(analysis['wikitext']['delta_nll_vs_base']) == set(confirm.PARAMETERS)
    inputs = json.loads((output / 'analysis-input.json').read_text())
    assert len(inputs['entries']) == 10
    for entry in inputs['entries']:
        assert len(Path(entry['predictions']).read_text().splitlines()) == 5108
    assert json.loads(Path(lock['unseal_receipt']).read_text())['status'] == 'complete'
    with pytest.raises(confirm.LockError, match='predictions already complete'):
        confirm.run_confirmation(lock_path, tmp_path / 'forbidden-reopen', unseal=True, resume=True)
