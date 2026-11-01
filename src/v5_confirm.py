"""Locked E-0013 confirmation; importing this module never opens task data.

``wikitext-base`` is the only command authorized by T-0014. ``evaluate`` is
reserved for the coordinator's later unsealing task and needs an explicit lock.
See research/notes/T-0014-lock-schema.md for the lock format.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time

from v1_data import (_CandidateEncodingError, collate_encoded, encode_candidates,
                     read_boolq_train, read_piqa_train)

EXPERIMENT = 'E-0013@3'
CONTRACT_SHA256 = 'a25529bc011fd8b639378a8aa3318b0e4cde835ba3489cdf51fa6da2ba4b60e7'
MODEL = 'Qwen/Qwen3-0.6B-Base'
REVISION = 'da87bfb608c14b7cf20ba1ce41287e8de496c0cd'
TASKS = ('piqa', 'boolq')
EXPECTED_ROWS = {'piqa': 1838, 'boolq': 3270}
PARAMETERS = {'V5-C': 524288, 'V5-F': 524288, 'V5-L': 630784}
REQUIRED_CODE = ('src/v5_confirm.py', 'src/v5_analysis.py', 'src/v1_data.py',
                 'src/bypass_model.py', 'src/bypass_core.py', 'src/v1_lora.py',
                 'src/calibrate.py', 'pyproject.toml', 'uv.lock')
SEALED = {
    'piqa': ('21edab439af693b961be2f069e8690a88d3b4e37',
             '93503cc97c679e459b065c3d13e848282e44b2a25213985bed3e5d458abef72d'),
    'piqa_labels': ('21edab439af693b961be2f069e8690a88d3b4e37',
                    'b4192dc3a2a0363d9d60ccf79800cbbe2f32ebb17726efdde6970e0b8131bceb'),
    'boolq': ('35b264d03638db9f4ce671b711558bf7ff0f80d5',
              '52355d11524b4b874a9b9dcc278feb10f672d52c4f4eff9872e695ede59820f8'),
    'wikitext': ('b08601e04326c79dfdd32d625aee71d232d685c3',
                 '5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91'),
}


class LockError(ValueError):
    """A missing or inconsistent prerequisite; validation must remain unread."""


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise LockError(f'cannot read JSON metadata {path}: {exc}') from exc


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def file_reference(path):
    path = Path(path).resolve()
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha256_file(path)}


def require(condition, message):
    if not condition:
        raise LockError(message)


def verify_file(reference, name):
    require(isinstance(reference, dict), f'{name}: file reference required')
    require(isinstance(reference.get('path'), str) and Path(reference['path']).is_absolute(),
            f'{name}: absolute path required')
    require(isinstance(reference.get('sha256'), str) and
            re.fullmatch('[0-9a-f]{64}', reference['sha256']), f'{name}: SHA256 required')
    require(type(reference.get('bytes')) is int and reference['bytes'] > 0,
            f'{name}: positive byte count required')
    path = Path(reference['path'])
    try:
        require(path.stat().st_size == reference['bytes'], f'{name}: byte count mismatch')
        require(sha256_file(path) == reference['sha256'], f'{name}: SHA256 mismatch')
    except OSError as exc:
        raise LockError(f'{name}: unavailable file {path}') from exc
    return path


def validate_data_reference(reference, name):
    require(isinstance(reference, dict), f'{name}: data reference missing')
    revision, digest = SEALED[name]
    require(reference.get('revision') == revision and reference.get('sha256') == digest,
            f'{name}: differs from sealed source')
    require(isinstance(reference.get('source_url'), str) and revision in reference['source_url'],
            f'{name}: revision-pinned source URL missing')
    if name == 'wikitext':
        require(reference.get('config') == 'wikitext-103-raw-v1' and
                reference.get('split') == 'test', 'WikiText must use raw-103 test')


def verify_adapter_config(config, arm):
    require(config.get('model_revision') == REVISION, f'{arm}: model revision mismatch')
    layers = list(range(28)) if arm == 'V5-L' else [6, 13, 20, 27]
    require(config.get('layers') == layers, f'{arm}: layer mismatch')
    rank = {'V5-C': 32, 'V5-F': 64, 'V5-L': 1}[arm]
    require(config.get('rank') == rank, f'{arm}: rank mismatch')
    if arm == 'V5-L':
        require(config.get('alpha') == 1, 'V5-L: alpha must equal rank 1')
    else:
        require(config.get('branch') == 'tokenwise_silu', f'{arm}: branch mismatch')
        require(config.get('fusion') == ('C_up' if arm == 'V5-C' else 'F_out'),
                f'{arm}: fusion mismatch')


def verify_lock(lock_path, *, code_root=None, actual_commit=None):
    """Check every prerequisite and hash, without parsing any validation data.

    Opaque SHA256 reads are the sole access to sealed data in this function.
    All configuration structure is checked before those hashes are computed.
    """
    require(lock_path is not None, 'lock file is required')
    lock = read_json(lock_path)
    require(isinstance(lock, dict) and lock.get('schema_version') == 1, 'lock schema mismatch')
    require(lock.get('experiment') == EXPERIMENT, 'contract revision mismatch')
    commit = lock.get('code_commit')
    require(isinstance(commit, str) and re.fullmatch('[0-9a-f]{40}', commit),
            'full code commit is required')
    root = Path(code_root or Path(__file__).resolve().parents[1]).resolve()
    running_commit = actual_commit or os.environ.get('SR_CODE_COMMIT')
    if running_commit is None and root.parent.name == 'snapshots':
        running_commit = root.name
    require(running_commit == commit, 'executing code commit differs from lock')
    require(lock.get('model') == MODEL and lock.get('model_revision') == REVISION and
            lock.get('tokenizer_revision') == REVISION, 'model/tokenizer revision mismatch')
    require(lock.get('max_length') == 512 and lock.get('wikitext_block_length') == 1024,
            'scoring/block protocol mismatch')
    code_files = lock.get('code_files')
    require(isinstance(code_files, dict) and set(code_files) == set(REQUIRED_CODE),
            'complete source/dependency hash inventory required')
    for relative, digest in code_files.items():
        require(isinstance(digest, str) and re.fullmatch('[0-9a-f]{64}', digest),
                f'{relative}: SHA256 required')
        require((root / relative).resolve().is_relative_to(root), 'source path escapes snapshot')
        require(sha256_file(root / relative) == digest, f'{relative}: source SHA256 mismatch')
    contract = verify_file(lock.get('contract'), 'contract')
    require(lock['contract']['sha256'] == CONTRACT_SHA256, 'contract is not sealed E-0013@3')
    require(f'experiment: E-0013\n' in contract.read_text(), 'wrong contract document')
    prerequisites = lock.get('prerequisites')
    require(isinstance(prerequisites, dict), 'prerequisites required')
    review = prerequisites.get('code_review', {})
    require(isinstance(review, dict) and re.fullmatch(r'RV-\d+', str(review.get('id', ''))) and
            review.get('passed') is True, 'passing code review required')
    verify_file(review.get('answer'), 'review answer')
    rehearsal = read_json(verify_file(prerequisites.get('dev_rehearsal'), 'dev rehearsal'))
    require(rehearsal.get('split') == 'dev', 'dev rehearsal report required')
    receipt = lock.get('unseal_receipt')
    require(isinstance(receipt, str) and Path(receipt).is_absolute(),
            'absolute shared unseal_receipt required')
    checkpoints = lock.get('checkpoints')
    require(isinstance(checkpoints, list) and len(checkpoints) == 9, 'nine checkpoints required')
    found, directories, rates = set(), set(), {}
    for entry in checkpoints:
        require(isinstance(entry, dict), 'checkpoint entry must be an object')
        arm, seed = entry.get('arm'), entry.get('seed')
        require(arm in PARAMETERS and type(seed) is int and seed in (1, 2, 3),
                'unexpected checkpoint arm/seed')
        require((arm, seed) not in found, 'duplicate checkpoint arm/seed')
        found.add((arm, seed))
        require(entry.get('expected_parameters') == PARAMETERS[arm], f'{arm}: wrong parameter count')
        require(isinstance(entry.get('path'), str) and Path(entry['path']).is_absolute(),
                f'{arm}/{seed}: absolute checkpoint directory required')
        directory = str(Path(entry['path']).resolve())
        require(directory not in directories, 'checkpoint directory reused across seeds/arms')
        directories.add(directory)
        rate = entry.get('learning_rate')
        require(type(rate) in (int, float) and math.isfinite(rate) and rate > 0,
                f'{arm}/{seed}: locked learning rate required')
        require(arm not in rates or rates[arm] == rate, f'{arm}: learning rate differs across seeds')
        rates[arm] = rate
        files = entry.get('files')
        require(isinstance(files, dict) and set(files) == {'config.json', 'adapter.pt'},
                f'{arm}/{seed}: checkpoint file inventory required')
        for name, reference in files.items():
            path = verify_file(reference, f'{arm}/{seed}/{name}')
            require(path.resolve() == (Path(entry['path']) / name).resolve(),
                    f'{arm}/{seed}: checkpoint inventory path mismatch')
        verify_adapter_config(read_json(files['config.json']['path']), arm)
        training = read_json(verify_file(entry.get('training_manifest'), f'{arm}/{seed} training manifest'))
        metrics = training.get('metrics', {})
        require(training.get('status') == 'complete' and metrics.get('reload.identical') is True,
                f'{arm}/{seed}: final training/reload incomplete')
        require(training.get('arm') == arm and training.get('seed') == seed,
                f'{arm}/{seed}: training manifest arm/seed mismatch')
        require(metrics.get('params.trainable') == PARAMETERS[arm] and
                metrics.get('learning_rate') == rate, f'{arm}/{seed}: training metadata mismatch')
    require(found == {(arm, seed) for arm in PARAMETERS for seed in (1, 2, 3)},
            'checkpoint grid incomplete')
    data = lock.get('data')
    require(isinstance(data, dict) and set(data) == set(SEALED), 'all four sealed data references required')
    for name, reference in data.items():
        validate_data_reference(reference, name)
    # This is deliberately last: malformed locks cannot reach the validation files.
    for name, reference in data.items():
        verify_file(reference, name)
    return lock


def read_validation(lock):
    """Only called after verify_lock and the one-time unseal receipt."""
    data = lock['data']
    rows = {'piqa': read_piqa_train(data['piqa']['path'], data['piqa_labels']['path']),
            'boolq': read_boolq_train(data['boolq']['path'])}
    for task in TASKS:
        require(len(rows[task]) == EXPECTED_ROWS[task], f'{task}: validation row count mismatch')
    return rows


def encode_validation(rows, tokenizer):
    """One shared encoding/exclusion pass, before any arm is scored."""
    encoded, audit = {}, {}
    for task in TASKS:
        kept, excluded = [], []
        for row in rows[task]:
            try:
                candidates = encode_candidates(tokenizer, row, max_length=512)
            except _CandidateEncodingError as exc:
                excluded.append({'id': row['id'], 'reason': str(exc),
                                 'candidate_lengths': list(exc.candidate_lengths)})
            else:
                kept.append((row, candidates))
        require(bool(kept), f'{task}: no encodable validation rows')
        encoded[task] = kept
        audit[task] = {'raw_examples': len(rows[task]), 'examples': len(kept),
                       'excluded_ids': [r['id'] for r in excluded], 'excluded': excluded,
                       'excluded_fraction': len(excluded) / len(rows[task]),
                       'protocol_deviation': len(excluded) / len(rows[task]) > .005}
    return encoded, audit


def candidate_losses(model, candidate, pad_id, device='cuda'):
    import torch
    import torch.nn.functional as functional
    batch = {name: torch.tensor(values, device=device, dtype=torch.long)
             for name, values in collate_encoded([candidate], pad_id).items()}
    labels = batch['labels'][:, 1:]
    valid = labels != -100
    logits = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], use_cache=False).logits
    return functional.cross_entropy(logits[:, :-1][valid].float(), labels[valid], reduction='none')


def evaluate_candidates(model, encoded, tokenizer, *, split='validation', device='cuda'):
    """V1 evaluate semantics: FP32 CE, PIQA mean, BoolQ sum, first-index ties."""
    import torch
    model.eval()
    predictions, metrics = [], {}
    with torch.no_grad():
        for task in TASKS:
            total, count, correct = 0., 0, 0
            for row, candidates in encoded[task]:
                scores, sums, counts = [], [], []
                for candidate in candidates:
                    loss = candidate_losses(model, candidate, tokenizer.pad_token_id, device)
                    if not loss.numel() or not torch.isfinite(loss).all():
                        raise ValueError(f'{split}/{task}/{row["id"]}: nonfinite candidate loss')
                    sums.append(loss.sum().item())
                    counts.append(loss.numel())
                    scores.append(-sums[-1] / (counts[-1] if task == 'piqa' else 1))
                    if not math.isfinite(sums[-1]) or not math.isfinite(scores[-1]):
                        raise ValueError('nonfinite candidate score')
                gold = int(row['label']) if task == 'piqa' else (0 if row['label'] else 1)
                pred = 0 if scores[0] >= scores[1] else 1
                correct += pred == gold
                total += sums[gold]
                count += counts[gold]
                if not math.isfinite(total):
                    raise ValueError('nonfinite accumulated NLL')
                predictions.append({'task': task, 'split': split, 'id': row['id'], 'gold': gold,
                                    'prediction': pred, 'scores': scores, 'gold_nll_sum': sums[gold],
                                    'gold_loss_tokens': counts[gold]})
            metrics[f'{split}.{task}.nll'] = total / count
            metrics[f'{split}.{task}.accuracy'] = correct / len(encoded[task])
            metrics[f'{split}.{task}.examples'] = len(encoded[task])
            metrics[f'{split}.{task}.loss_tokens'] = count
        metrics[f'{split}.nll'] = sum(metrics[f'{split}.{task}.nll'] for task in TASKS) / 2
    return metrics, predictions


def wikitext_blocks(texts, tokenizer, block_length=1024):
    """Join raw lines with no separator, tokenize once, discard the short tail."""
    require(type(block_length) is int and block_length >= 2, 'invalid WikiText block length')
    require(all(isinstance(text, str) for text in texts), 'WikiText rows must be strings')
    tokens = tokenizer.encode(''.join(texts), add_special_tokens=False)
    full = len(tokens) // block_length
    return [tokens[i * block_length:(i + 1) * block_length] for i in range(full)], {
        'input_tokens': len(tokens), 'blocks': full,
        'discarded_tail_tokens': len(tokens) % block_length,
        'predicted_tokens': full * (block_length - 1), 'block_length': block_length}


def read_wikitext_blocks(path, tokenizer):
    import pyarrow.parquet as parquet
    texts = parquet.read_table(path, columns=['text']).column('text').to_pylist()
    blocks, metadata = wikitext_blocks(texts, tokenizer)
    require(bool(blocks), 'WikiText contains no full blocks')
    return blocks, metadata


def evaluate_wikitext(model, blocks, *, device='cuda'):
    import torch
    import torch.nn.functional as functional
    total, count = 0., 0
    model.eval()
    tick = time.monotonic()
    with torch.no_grad():
        for block in blocks:
            inputs = torch.tensor([block], device=device, dtype=torch.long)
            logits = model(input_ids=inputs, attention_mask=torch.ones_like(inputs), use_cache=False).logits
            losses = functional.cross_entropy(logits[0, :-1].float(), inputs[0, 1:], reduction='none')
            if not losses.numel() or not torch.isfinite(losses).all():
                raise ValueError('nonfinite WikiText loss')
            total += losses.double().sum().item()
            count += losses.numel()
    require(count > 0 and math.isfinite(total), 'invalid WikiText NLL')
    return {'wikitext.nll': total / count, 'wikitext.nll_sum': total,
            'wikitext.loss_tokens': count, 'wikitext.seconds': time.monotonic() - tick}


def load_tokenizer():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def load_model(entry=None):
    import torch
    from calibrate import load_base
    base = load_base()
    if entry is None:
        return base
    if entry['arm'] == 'V5-L':
        from v1_lora import LoRAModel
        model = LoRAModel.load_adapter(base, entry['path'])
    else:
        from bypass_model import BranchModel
        model = BranchModel.load_adapter(base, entry['path'])
    parameters = [p for p in model.parameters() if p.requires_grad]
    require(sum(p.numel() for p in parameters) == entry['expected_parameters'], 'parameter count mismatch')
    require(all(p.dtype == torch.float32 and torch.isfinite(p).all().item() for p in parameters),
            'adapter parameters must be finite FP32')
    return model.eval()


def claim_unseal(lock, lock_path, *, resume=False):
    receipt = Path(lock['unseal_receipt'])
    digest = sha256_file(lock_path)
    if receipt.exists():
        previous = read_json(receipt)
        require(resume and previous.get('lock_sha256') == digest,
                'unseal already claimed; retries require --resume with the identical lock')
        require(previous.get('status') == 'started',
                'predictions already complete; evaluation cannot be reopened')
    else:
        require(not resume, 'no unseal receipt to resume')
        receipt.parent.mkdir(parents=True, exist_ok=True)
        with receipt.open('x') as handle:
            json.dump({'experiment': EXPERIMENT, 'lock_sha256': digest,
                       'code_commit': lock['code_commit'], 'status': 'started',
                       'attempt': os.environ.get('SR_ATTEMPT_ID')}, handle)


def run_confirmation(lock_path, output, *, unseal=False, resume=False):
    require(unseal, 'evaluate requires explicit --unseal in the later authorized task')
    lock = verify_lock(lock_path)
    output = Path(output).resolve()
    require(not (output / 'manifest.json').exists(), 'output already contains a result; use a fresh attempt')
    output.mkdir(parents=True, exist_ok=True)
    claim_unseal(lock, lock_path, resume=resume)
    # The first semantic validation read is below, behind the complete lock gate.
    rows = read_validation(lock)
    tokenizer = load_tokenizer()
    encoded, audit = encode_validation(rows, tokenizer)
    blocks, wiki_metadata = read_wikitext_blocks(lock['data']['wikitext']['path'], tokenizer)
    entries, all_metrics = [], {}
    import torch
    for spec in [None, *sorted(lock['checkpoints'], key=lambda r: (r['arm'], r['seed']))]:
        arm, seed = ('Base', None) if spec is None else (spec['arm'], spec['seed'])
        name = 'Base' if seed is None else f'{arm}-s{seed}'
        model_output = output / name
        model_output.mkdir()
        started = time.monotonic()
        if spec is not None:
            for filename, reference in spec['files'].items():
                verify_file(reference, f'{name}/{filename} before loading')
        model = load_model(spec)
        values, predictions = evaluate_candidates(model, encoded, tokenizer)
        values.update(evaluate_wikitext(model, blocks))
        values['elapsed_s'] = time.monotonic() - started
        metrics = {'metrics': values, 'validation': audit, 'wikitext': wiki_metadata}
        write_json(model_output / 'metrics.json', metrics)
        with (model_output / 'predictions.jsonl').open('w') as handle:
            for row in predictions:
                handle.write(json.dumps(row, allow_nan=False) + '\n')
        entries.append({'arm': arm, 'seed': seed,
                        'predictions': str(model_output / 'predictions.jsonl'),
                        'metrics': str(model_output / 'metrics.json')})
        all_metrics.update({f'{name}.{key}': value for key, value in values.items()})
        del model
        gc.collect()
        torch.cuda.empty_cache()
    receipt = read_json(lock['unseal_receipt'])
    receipt['status'] = 'predictions_complete'
    receipt['analysis_input'] = str(output / 'analysis-input.json')
    write_json(lock['unseal_receipt'], receipt)
    analysis_manifest = {'experiment': EXPERIMENT, 'split': 'validation', 'entries': entries,
                         'lock_sha256': sha256_file(lock_path)}
    write_json(output / 'analysis-input.json', analysis_manifest)
    from v5_analysis import analyze, write_outputs
    report = analyze(analysis_manifest, 'validation')
    write_outputs(report, output / 'analysis')
    write_json(output / 'manifest.json', {'status': 'complete',
        'spec_digest': os.environ['SR_SPEC_DIGEST'], 'metrics': all_metrics,
        'artifacts': [{'path': str(p), 'role': p.name} for p in output.rglob('*')
                      if p.is_file() and p.name != 'manifest.json'],
        'validator': {'name': 'v5-locked-confirmation', 'passed': True}})
    receipt['status'] = 'complete'
    write_json(lock['unseal_receipt'], receipt)


def run_wikitext_base(sealed_manifest, output):
    """Dev-time measurement: intentionally never access a validation path."""
    seal = read_json(sealed_manifest)
    reference = seal['files']['wikitext']
    validate_data_reference(reference, 'wikitext')
    path = verify_file(reference, 'wikitext')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    require(not (output / 'metrics.json').exists(), 'WikiText output exists')
    start = time.monotonic()
    tokenizer = load_tokenizer()
    blocks, metadata = read_wikitext_blocks(path, tokenizer)
    tick = time.monotonic()
    model = load_model()
    load_seconds = time.monotonic() - tick
    values = evaluate_wikitext(model, blocks)
    values.update({'load.seconds': load_seconds, 'elapsed_s': time.monotonic() - start})
    write_json(output / 'metrics.json', {'metrics': values, 'wikitext': metadata,
        'data': reference, 'model_revision': REVISION, 'arm': 'Base', 'seed': None})
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    confirm = commands.add_parser('evaluate')
    confirm.add_argument('--lock', type=Path, required=True)
    confirm.add_argument('--output', type=Path, required=True)
    confirm.add_argument('--unseal', action='store_true')
    confirm.add_argument('--resume', action='store_true')
    wiki = commands.add_parser('wikitext-base')
    wiki.add_argument('--sealed-manifest', type=Path, required=True)
    wiki.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'evaluate':
        run_confirmation(args.lock, args.output, unseal=args.unseal, resume=args.resume)
    else:
        run_wikitext_base(args.sealed_manifest, args.output)


if __name__ == '__main__':
    main()
