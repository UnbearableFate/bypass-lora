"""E-0015@3 memory-route diagnostics: evaluation of recorded checkpoints only, no training.

`run` loads one recorded checkpoint and
- checks that its recorded dev predictions are reproduced per episode (dev is used
  for nothing else);
- evaluates the read substitutions, negative controls and (on its training E) the
  magnitude sweep on the E=1, E=4 (ATT-0112-1) and E=16 (V2) calibration splits;
- fits the Pv/Pf/Pz probes and their baselines on the first 4,096 train rows of the
  same E and tests them on calibration.
`analyze` combines the four checkpoint summaries and evaluates E-0015 rules 1-8.
Run in an allocated GPU job; never on a login node. Confirmation is never opened.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time

import torch
from transformers import AutoTokenizer

from calibrate import MODEL, REVISION, load_base
from v2_data import encode, visible_delimiters
from v2_memory import SWEEP_CONTROLS, collate, evaluate_batch, isolated_writes, observe_writes
from v2_schedule import DATA_HASHES, load_data
from v2_train import EVAL_BATCH_SIZE, adapted, evaluate, write_json
from v2r_data import METADATA_SHA256, load_variant

ARM = 'direct_hidden.soft_slot.F_out'
CHECKPOINTS = {'R1': 'ATT-0122-1', 'R2': 'ATT-0123-1', 'R5': 'ATT-0126-1', 'V2-03': 'ATT-0082-1'}
FACT_COUNTS = (1, 4, 16)
# Recorded E-0014 / E-0010 dev evaluation directories (relative to the arm directory).
RECORDED_DEV = {'R1': {16: '.', 1: 'e1'}, 'R2': {16: '.', 4: 'e4'}, 'R5': {16: '.'}, 'V2-03': {16: '.'}}
RELOAD_CONTROLS = ('normal', 'zero', 'shuffled', 'oracle_value')
CONTROLS_E1 = ('normal', 'zero', 'shuffled', 'oracle_value', 'last_only', 'oracle_last', 'oracle_last_only',
               'uniform', 'last_only_shuffled')
CONTROLS_MULTI = CONTROLS_E1 + ('wrong_oracle', 'iso_kv', 'iso_v', 'iso_oracle_value', 'iso_wrong_oracle')
SWEEP_SPLIT = {'R1': 1, 'R2': 4, 'R5': 16}
PZ_CONTROLS = ('normal', 'oracle_value')
PROBE_TRAIN_ROWS = 4096
DIAGNOSTIC_BATCH = 64  # calibration/train diagnostics only; the reload check uses EVAL_BATCH_SIZE
RIDGE = 1.
SHUFFLE_SEED = 20261101
PROJECTION_SEED = 20261102
PROJECTION_RANK = 128
FACT_GROUPS = (('0', 0, 0), ('1-3', 1, 3), ('>=4', 4, 15))


def fact_text(fact):
    return f"{fact['head']} {fact['relation']} {fact['tail']} ;\n"


def annotate(tok, rows, n_facts):
    """Subject token (as v2_train), per-fact tail token and isolated E=1-format support.

    The isolated support of a fact is the E=1 generator's support for that fact alone
    (same text, same visible_delimiters tokenization; its delimiter is the last token).
    Returns how many facts' tokens inside the contextual support differ from it.
    """
    differing = 0
    for row in rows:
        if len(row['facts']) != n_facts:
            raise ValueError('fact count mismatch')
        if row['support_text'] != ''.join(fact_text(f) for f in row['facts']):
            raise ValueError('support text is not the concatenated fact format')
        subject = encode(tok, ' ' + row['query']['head'])
        if len(subject) != 1:
            raise ValueError('query subject is not a single token')
        row['_subject_token_id'] = subject[0]
        tails, isolated, previous = [], [], -1
        for fact in row['facts']:
            tail = encode(tok, ' ' + fact['tail'])
            if len(tail) != 1:
                raise ValueError('tail is not a single token')
            tails.append(tail[0])
            ids, positions = visible_delimiters(tok, fact_text(fact))
            if len(positions) != 1 or positions[0]['delimiter_token_index'] != len(ids) - 1:
                raise ValueError('isolated fact must end at its single delimiter')
            isolated.append({'ids': ids, 'delimiter': positions[0]['delimiter_token_index']})
            differing += row['support_input_ids'][previous + 1:fact['delimiter_token_index'] + 1] != ids
            previous = fact['delimiter_token_index']
        if tails[row['query']['target_fact_index']] != row['query']['answer_token_id']:
            raise ValueError('target tail token differs from the answer token')
        if n_facts == 1 and (isolated[0]['ids'] != row['support_input_ids']
                             or isolated[0]['delimiter'] != row['facts'][0]['delimiter_token_index']):
            raise ValueError('isolated encoding differs from the E=1 data format')
        row['_tail_token_ids'] = tails
        row['_isolated'] = isolated
    return {'rows': len(rows), 'facts': sum(len(r['facts']) for r in rows),
            'context_fact_tokens_differ_from_isolated': differing}


def load_splits(e16_dir, variant_dirs):
    v2_train = load_data(Path(e16_dir) / 'train.jsonl', 'train')
    data = {16: {'dev': load_data(Path(e16_dir) / 'dev.jsonl', 'dev'),
                 'calibration': load_data(Path(e16_dir) / 'calibration.jsonl', 'calibration'),
                 'train': v2_train[:PROBE_TRAIN_ROWS]}}
    # The E=16 prior token comes from the V2 train split, as in v2_train.main.
    priors = {16: Counter(r['query']['answer_token_id'] for r in v2_train).most_common(1)[0][0]}
    for facts in (1, 4):
        metadata, dev = load_variant(variant_dirs[facts], facts, 'dev')
        data[facts] = {'dev': dev, 'calibration': load_variant(variant_dirs[facts], facts, 'calibration')[1],
                       'train': load_variant(variant_dirs[facts], facts, 'train')[1][:PROBE_TRAIN_ROWS]}
        priors[facts] = metadata['prior']['answer_token_id']
    for splits in data.values():
        ids = {s: {r['episode_id'] for r in rows} for s, rows in splits.items()}
        contents = {s: {r['content_sha256'] for r in rows} for s, rows in splits.items()}
        if ids['train'] & ids['calibration'] or contents['train'] & contents['calibration']:
            raise ValueError('probe fit (train) and test (calibration) episodes overlap')
        if len(splits['train']) != PROBE_TRAIN_ROWS:
            raise ValueError('probe train rows missing')
    return data, priors


@torch.no_grad()
def collect_writes(model, rows, pad_id, device):
    """Pv/Pf inputs per written slot in support order, contextual and isolated."""
    out = {kind: {'values': [], 'features': []} for kind in ('context', 'isolated')}
    labels, positions = [], []
    for start in range(0, len(rows), DIAGNOSTIC_BATCH):
        selected = rows[start:start + DIAGNOSTIC_BATCH]
        batch = collate(selected, pad_id, device)
        writes = {'context': observe_writes(model, batch.support_ids, batch.support_mask, batch.write_mask),
                  'isolated': isolated_writes(model, selected, pad_id, device)}
        if not torch.equal(writes['context']['valid'], writes['isolated']['valid']):
            raise ValueError('isolated slots differ from contextual slots')
        valid = writes['context']['valid']
        for kind, w in writes.items():
            out[kind]['values'].append(w['values'][valid].cpu())
            out[kind]['features'].append(w['features'][valid].cpu())
        for row in selected:
            labels.extend(row['_tail_token_ids'])
            positions.extend(range(len(row['facts'])))
    result = {kind: {name: torch.cat(chunks) for name, chunks in parts.items()} for kind, parts in out.items()}
    result['labels'] = torch.tensor(labels)
    result['positions'] = torch.tensor(positions)
    if any(len(result[k][n]) != len(labels) for k in ('context', 'isolated') for n in ('values', 'features')):
        raise ValueError('probe samples are not aligned with slot labels')
    return result


@torch.no_grad()
def collect_increments(model, rows, pad_id, device, mode):
    """Last-position FP32 F_out increments (Pz inputs) for probe-fit rows."""
    chunks = []
    for start in range(0, len(rows), DIAGNOSTIC_BATCH):
        details, _ = evaluate_batch(model, collate(rows[start:start + DIAGNOSTIC_BATCH], pad_id, device), mode)
        chunks.append(details['_last_fusion'])
    return torch.cat(chunks)


def ridge_fit_predict(train_x, train_y, test_x, ridge=RIDGE):
    """Closed-form ridge classifier with bias (as v2_memory.independent_probes), in FP64."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    classes = torch.unique(train_y, sorted=True).to(device)
    y = (train_y.to(device)[:, None] == classes).double()
    ones = lambda n: torch.ones(n, 1, device=device, dtype=torch.float64)
    x = torch.cat((train_x.to(device).double(), ones(len(train_x))), -1)
    w = torch.linalg.solve(x.T @ x + ridge * torch.eye(x.shape[1], device=device, dtype=torch.float64), x.T @ y)
    t = torch.cat((test_x.to(device).double(), ones(len(test_x))), -1)
    return classes[(t @ w).argmax(-1)].cpu(), classes.cpu()


def summarize(predicted, test_y, classes, positions):
    correct = predicted == test_y
    result = {'n': len(correct), 'accuracy': float(correct.float().mean()), 'correct': int(correct.sum()),
              'unseen_label_count': int((~torch.isin(test_y, classes)).sum()), 'by_position': {}}
    for name, low, high in FACT_GROUPS:
        mask = (positions >= low) & (positions <= high)
        if mask.any():
            result['by_position'][name] = {'n': int(mask.sum()), 'accuracy': float(correct[mask].float().mean())}
    return result


def probe(train_x, train_y, test_x, test_y, positions, projection=False):
    """A probe with its baselines on identical samples: answer prior and label shuffle
    (and, for Pf, a fixed random projection to 128 dimensions)."""
    predicted, classes = ridge_fit_predict(train_x, train_y, test_x)
    result = summarize(predicted, test_y, classes, positions)
    prior = Counter(train_y.tolist()).most_common(1)[0][0]
    result['baseline_answer_prior'] = summarize(torch.full_like(test_y, prior), test_y, classes, positions)
    generator = torch.Generator().manual_seed(SHUFFLE_SEED)
    shuffled = train_y[torch.randperm(len(train_y), generator=generator)]
    predicted, classes = ridge_fit_predict(train_x, shuffled, test_x)
    result['baseline_label_shuffle'] = summarize(predicted, test_y, classes, positions)
    if projection:
        generator = torch.Generator().manual_seed(PROJECTION_SEED)
        matrix = torch.randn(train_x.shape[-1], PROJECTION_RANK, generator=generator, dtype=torch.float64)
        matrix /= train_x.shape[-1] ** .5
        predicted, classes = ridge_fit_predict(train_x.double() @ matrix, train_y, test_x.double() @ matrix)
        result['baseline_random_projection_128'] = summarize(predicted, test_y, classes, positions)
    return result


def tensor_digest(x):
    return hashlib.sha256(x.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def target_strata(metrics):
    """EM/NLL for target position 0 and >= 1, with per-position counts (exact aggregation)."""
    strata = metrics['strata']['target_fact_position']
    groups = {'0': [k for k in strata if int(k) == 0], '>=1': [k for k in strata if int(k) >= 1]}
    result = {}
    for name, keys in groups.items():
        count = sum(strata[k]['count'] for k in keys)
        result[name] = None if not count else {
            'count': count, 'em': sum(strata[k]['em'] * strata[k]['count'] for k in keys) / count,
            'nll': sum(strata[k]['nll'] * strata[k]['count'] for k in keys) / count}
    ordered = sorted(strata, key=int)
    result['by_position'] = {k: {'count': strata[k]['count'], 'em': strata[k]['em'], 'nll': strata[k]['nll']}
                             for k in ordered}
    return result


def compare_recorded(recorded_dir, output_dir, split, controls):
    """Per-episode equality of predictions with every recorded control file evaluated here."""
    report = {}
    for mode in controls:
        path = Path(recorded_dir) / f'{split}.{mode}.predictions.jsonl'
        if not path.exists():
            continue
        old = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
        new = [json.loads(x) for x in (Path(output_dir) / path.name).read_text().splitlines() if x.strip()]
        same_ids = [a['episode_id'] for a in old] == [b['episode_id'] for b in new]
        report[mode] = {'recorded': str(path), 'episodes': len(old), 'same_episode_order': same_ids,
                        'prediction_mismatches': sum(a['prediction'] != b['prediction'] for a, b in zip(old, new))
                        if same_ids else None,
                        'nll_max_abs_difference': max(abs(a['nll'] - b['nll']) for a, b in zip(old, new))
                        if same_ids else None}
    if 'normal' not in report:
        raise AssertionError(f'no recorded normal predictions in {recorded_dir}')
    return report


def control_summary(metrics):
    return {mode: {'em': m['em'], 'nll': m['nll'], 'count': m['count'], 'prior_em': m.get('prior_em'),
                   'target_position': target_strata(m),
                   'last_position': {k: m['last_position'].get(k) for k in
                                     ('argmax_target_rate', 'target_weight_mean', 'n_eff_mean')},
                   'address.target_weight': m['diagnostics'].get('address.target_weight'),
                   'read.reconstruction_max_abs': m['diagnostics'].get('read.reconstruction_max_abs')}
            for mode, m in metrics.items()}


def run(args):
    started = time.monotonic()
    label = args.checkpoint
    arm_dir = args.runs / CHECKPOINTS[label] / ARM
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'manifest.json').exists():
        raise SystemExit('output already contains run results')
    passed, failure = False, None
    summary = {'experiment': 'E-0015@3', 'checkpoint': label, 'attempt': CHECKPOINTS[label], 'limit': args.limit}
    try:
        recorded_config = json.loads((arm_dir / 'metrics.json').read_text())['config']
        data, priors = load_splits(args.e16_data, {1: args.e1_data, 4: args.e4_data})
        if args.limit:  # smoke only
            data = {f: {s: rows[:args.limit] for s, rows in splits.items()} for f, splits in data.items()}
        tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        summary['tokenization'] = {f'e{f}': {s: annotate(tok, rows, f) for s, rows in splits.items()}
                                   for f, splits in data.items()}
        base = load_base()
        device = next(base.parameters()).device
        with adapted(base, checkpoint=arm_dir / 'adapter') as model:
            loaded = asdict(model.config)
            if any(loaded.get(k) != v for k, v in recorded_config.items()):
                raise AssertionError('loaded checkpoint configuration differs from the recorded one')
            model.eval()
            summary['config'] = loaded
            # 1. Reload identity on the recorded dev splits (the only use of dev).
            summary['reload'] = {}
            for facts, recorded in RECORDED_DEV[label].items():
                reload_dir = output / 'reload' / f'e{facts}'
                controls = [m for m in RELOAD_CONTROLS
                            if (arm_dir / recorded / f'dev.{m}.predictions.jsonl').exists()]
                evaluate(model, data[facts]['dev'], pad_id, reload_dir, 'dev', controls, priors[facts])
                if not args.limit:
                    report = compare_recorded(arm_dir / recorded, reload_dir, 'dev', controls)
                    summary['reload'][f'e{facts}'] = report
                    write_json(output / 'summary.json', summary)
                    bad = sorted(m for m, r in report.items() if not r['same_episode_order'] or r['prediction_mismatches'])
                    if bad:
                        raise AssertionError(f'e{facts} dev: reloaded predictions differ from records for {bad}')
            # 2. Calibration diagnostics and probes.
            summary['splits'] = {}
            for facts in FACT_COUNTS:
                split_dir = output / f'e{facts}'
                rows = data[facts]
                controls = (CONTROLS_E1 if facts == 1 else CONTROLS_MULTI)
                if SWEEP_SPLIT.get(label) == facts:
                    controls = controls + SWEEP_CONTROLS
                pz_test = {}
                tick = time.monotonic()
                metrics, _ = evaluate(model, rows['calibration'], pad_id, split_dir, 'calibration', controls,
                                      priors[facts], capture=pz_test, batch_size=DIAGNOSTIC_BATCH)
                part = {'controls': control_summary(metrics), 'control_seconds': time.monotonic() - tick}
                tick = time.monotonic()
                writes = {s: collect_writes(model, rows[s], pad_id, device) for s in ('train', 'calibration')}
                fit, test = writes['train'], writes['calibration']
                probes = {}
                for kind in ('context', 'isolated'):
                    probes[f'Pv.{kind}'] = probe(fit[kind]['values'], fit['labels'], test[kind]['values'],
                                                 test['labels'], test['positions'])
                    probes[f'Pf.{kind}'] = probe(fit[kind]['features'], fit['labels'], test[kind]['features'],
                                                 test['labels'], test['positions'], projection=True)
                # Cross-encoding Pv (fit encoding -> test encoding).
                for a, b in (('isolated', 'context'), ('context', 'isolated')):
                    probes[f'Pv.{a}->{b}'] = probe(fit[a]['values'], fit['labels'], test[b]['values'],
                                                   test['labels'], test['positions'])
                fit_labels = torch.tensor([r['query']['answer_token_id'] for r in rows['train']])
                test_labels = torch.tensor([r['query']['answer_token_id'] for r in rows['calibration']])
                test_positions = torch.tensor([r['query']['target_fact_index'] for r in rows['calibration']])
                for mode in PZ_CONTROLS:
                    fit_z = collect_increments(model, rows['train'], pad_id, device, mode)
                    probes[f'Pz.{mode}'] = probe(fit_z, fit_labels, pz_test[mode], test_labels, test_positions)
                part['probes'] = probes
                part['probe_seconds'] = time.monotonic() - tick
                part['feature_sha256'] = {f'{s}.{k}': tensor_digest(writes[s][k]['features'])
                                          for s in writes for k in ('context', 'isolated')}
                part['isolated_minus_context_value_max_abs'] = float(
                    (test['isolated']['values'] - test['context']['values']).abs().max())
                summary['splits'][f'e{facts}'] = part
                write_json(output / 'summary.json', summary)
        passed = True
    except Exception as exc:
        failure = {'type': type(exc).__name__, 'message': str(exc)}
        raise
    finally:
        wall = time.monotonic() - started
        summary['wall_seconds'] = wall
        write_json(output / 'summary.json', summary)
        artifacts = [{'path': str(p), 'role': p.name} for p in output.rglob('*') if p.is_file() and p.name != 'manifest.json']
        write_json(output / 'manifest.json', {
            'status': 'complete' if passed else 'failed', 'spec_digest': os.environ.get('SR_SPEC_DIGEST'),
            'experiment': 'E-0015@3', 'checkpoint': label, 'checkpoint_attempt': CHECKPOINTS[label],
            'adapter': str((arm_dir / 'adapter').resolve()), 'model': MODEL, 'revision': REVISION,
            'data': {'e16': str(Path(args.e16_data).resolve()), 'e16_sha256': DATA_HASHES,
                     'e1': str(Path(args.e1_data).resolve()), 'e4': str(Path(args.e4_data).resolve()),
                     'variant_metadata_sha256': {str(k): METADATA_SHA256[k] for k in (1, 4)}},
            'diagnostic_split': 'calibration', 'probe_fit': f'train[:{PROBE_TRAIN_ROWS}] (file order)',
            'reload_batch_size': EVAL_BATCH_SIZE, 'diagnostic_batch_size': DIAGNOSTIC_BATCH,
            'ridge': RIDGE, 'shuffle_seed': SHUFFLE_SEED, 'projection_seed': PROJECTION_SEED,
            'limit': args.limit, 'torch': torch.__version__, 'cuda': torch.version.cuda,
            'metrics': {'wall_seconds': wall, 'mig_instance_hours': wall / 3600, 'node_hours': wall / 3600 * .25},
            'failure': failure, 'artifacts': artifacts,
            'validator': {'name': 'v2r_diagnostics.run', 'passed': passed}})
    return 0


# ---- analysis ---------------------------------------------------------------------

RULES = {
    1: 'value 内容在、解码失败：R2@E=4 或 R5@E=16 的 Pv ≥ 80%，且 oracle_value EM < 30%。',
    2: 'value 内容丢失：Pv < 50%，而 Pf ≥ 90%（按 checkpoint × split 列出成立处）。',
    3: '非末位混合读取干扰：R2@E=4 或 R5@E=16 上，last_only − normal ≥ 10 pp 或 oracle_last_only − oracle_value ≥ 10 pp，'
       'NLL 同向改善，且 last_only_shuffled 不出现同样的提升（last_only_shuffled − shuffled < 10 pp 或 NLL 不同向改善）。',
    4: '跨上下文迁移失败：R1 的 oracle_value 在 E=4 与 E=16 上各比 E=1 低 ≥ 30 pp 且 NLL 更高，该 split 的 Pf ≥ 90%。',
    5: '上下文迁移确认：R1@E=16 的 iso_oracle_value 在目标第 0 条与第 ≥1 条都 ≥ 70%；上下文 oracle_value 只在第 0 条高'
       '（第 0 条 ≥ 70%，第 ≥1 条 < 70%）；iso_wrong_oracle 接近先验（EM ≤ 答案先验 EM + 5 pp）。',
    6: 'R2@E=4 或 R5@E=16 的 iso_v 比其 normal 高 ≥ 10 pp，且 NLL 同向改善。',
    7: '幅度：R2@E=4 或 R5@E=16 上某个 α < 1 使 EM 比同读取方式的 α=1 高 ≥ 10 pp 且 NLL 同向改善，'
       '同时 R1@E=1 在同一 α、同一方式下 EM ≥ 50%。',
    8: '以上都不成立。',
}
TRAINED_SPLIT = {'R2': 'e4', 'R5': 'e16'}
SWEEP_REFERENCE = {'normal': 'normal', 'oracle_value': 'oracle_value', 'nonfinal': 'normal'}


def gain(control_a, control_b, margin):
    """EM(a) - EM(b) >= margin with NLL improving in the same direction."""
    diff = control_a['em'] - control_b['em']
    return {'em_difference': diff, 'nll_difference': control_a['nll'] - control_b['nll'],
            'holds': diff >= margin and control_a['nll'] < control_b['nll']}


def analyze(args):
    summaries = {}
    for item in args.summary:
        label, _, path = item.partition('=')
        summaries[label] = json.loads(Path(path).read_text())
        if summaries[label].get('checkpoint') != label or summaries[label].get('limit'):
            raise ValueError('summary label mismatch or smoke summary')
    if set(summaries) != set(CHECKPOINTS):
        raise ValueError('analysis needs all four checkpoints')

    def control(label, split, mode):
        return summaries[label]['splits'][split]['controls'][mode]

    def stratum(label, split, mode, name):
        return control(label, split, mode)['target_position'][name]['em']

    def accuracy(label, split, name):
        return summaries[label]['splits'][split]['probes'][name]['accuracy']

    checks = {
        'pf_identical_across_checkpoints': all(
            len({json.dumps(s['splits'][f'e{f}']['feature_sha256'], sort_keys=True) for s in summaries.values()}) == 1
            for f in FACT_COUNTS),
        'reload_identical': all(r['same_episode_order'] and r['prediction_mismatches'] == 0
                                for s in summaries.values() for split in s['reload'].values() for r in split.values()),
        'isolated_equals_context_at_e1': {k: s['splits']['e1']['isolated_minus_context_value_max_abs']
                                          for k, s in summaries.items()},
        'context_fact_tokens_differ_from_isolated': {
            split: summaries['R1']['tokenization'][split] for split in summaries['R1']['tokenization']},
    }
    rules = {}
    values = [{'checkpoint': c, 'split': s, 'Pv': accuracy(c, s, 'Pv.context'),
               'oracle_value': control(c, s, 'oracle_value')['em']} for c, s in TRAINED_SPLIT.items()]
    rules[1] = {'holds': any(v['Pv'] >= .8 and v['oracle_value'] < .3 for v in values), 'values': values}
    cells = [{'checkpoint': c, 'split': f'e{f}', 'Pv': accuracy(c, f'e{f}', 'Pv.context'),
              'Pf': accuracy(c, f'e{f}', 'Pf.context')} for c in CHECKPOINTS for f in FACT_COUNTS]
    where = [x for x in cells if x['Pv'] < .5 and x['Pf'] >= .9]
    rules[2] = {'holds': bool(where), 'where': where, 'values': cells}
    values = []
    for c, s in TRAINED_SPLIT.items():
        a = gain(control(c, s, 'last_only'), control(c, s, 'normal'), .1)
        b = gain(control(c, s, 'oracle_last_only'), control(c, s, 'oracle_value'), .1)
        negative = gain(control(c, s, 'last_only_shuffled'), control(c, s, 'shuffled'), .1)
        values.append({'checkpoint': c, 'split': s, 'last_only_vs_normal': a,
                       'oracle_last_only_vs_oracle_value': b, 'last_only_shuffled_vs_shuffled': negative,
                       'holds': (a['holds'] or b['holds']) and not negative['holds']})
    rules[3] = {'holds': any(v['holds'] for v in values), 'values': values}
    r1 = {s: control('R1', s, 'oracle_value') for s in ('e1', 'e4', 'e16')}
    drops = {s: gain(r1['e1'], r1[s], .3) for s in ('e4', 'e16')}
    pf = {s: accuracy('R1', s, 'Pf.context') for s in ('e4', 'e16')}
    rules[4] = {'holds': all(drops[s]['holds'] and pf[s] >= .9 for s in ('e4', 'e16')),
                'values': {'R1_oracle_value_em': {s: r1[s]['em'] for s in r1}, 'e1_minus_split': drops,
                           'Pf.context': pf}}
    iso = {k: stratum('R1', 'e16', 'iso_oracle_value', k) for k in ('0', '>=1')}
    ctx = {k: stratum('R1', 'e16', 'oracle_value', k) for k in ('0', '>=1')}
    wrong = control('R1', 'e16', 'iso_wrong_oracle')
    near_prior = wrong['em'] <= wrong['prior_em'] + .05
    rules[5] = {'holds': iso['0'] >= .7 and iso['>=1'] >= .7 and ctx['0'] >= .7 and ctx['>=1'] < .7 and near_prior,
                'values': {'iso_oracle_value': iso, 'oracle_value': ctx, 'iso_wrong_oracle_em': wrong['em'],
                           'answer_prior_em': wrong['prior_em']}}
    values = [{'checkpoint': c, 'split': s, **gain(control(c, s, 'iso_v'), control(c, s, 'normal'), .1)}
              for c, s in TRAINED_SPLIT.items()]
    rules[6] = {'holds': any(v['holds'] for v in values), 'values': values}
    values = []
    for c, s in TRAINED_SPLIT.items():
        for read, reference in SWEEP_REFERENCE.items():
            for alpha in (.1, .3):
                mode = f'{read}@a{alpha}'
                g = gain(control(c, s, mode), control(c, s, reference), .1)
                r1_em = control('R1', 'e1', mode)['em']
                values.append({'checkpoint': c, 'split': s, 'control': mode, 'reference': reference, **g,
                               'R1_e1_em': r1_em, 'holds': g['holds'] and r1_em >= .5})
    rules[7] = {'holds': any(v['holds'] for v in values), 'values': values}
    rules[8] = {'holds': not any(rules[k]['holds'] for k in range(1, 8))}
    for k, text in RULES.items():
        rules[k]['rule'] = text
    tables = {label: {split: {
        'controls': {m: {'em': c['em'], 'nll': c['nll'],
                         'em_target_0': (c['target_position']['0'] or {}).get('em'),
                         'em_target_ge1': (c['target_position']['>=1'] or {}).get('em')}
                     for m, c in part['controls'].items()},
        'probes': {n: {'accuracy': p['accuracy'], 'answer_prior': p['baseline_answer_prior']['accuracy'],
                       'label_shuffle': p['baseline_label_shuffle']['accuracy'],
                       **({'random_projection_128': p['baseline_random_projection_128']['accuracy']}
                          if 'baseline_random_projection_128' in p else {}),
                       'unseen': p['unseen_label_count'], 'by_position': p['by_position']}
                   for n, p in part['probes'].items()}}
        for split, part in s['splits'].items()} for label, s in summaries.items()}
    report = {'experiment': 'E-0015@3', 'evidence_level': 'diagnostic', 'checks': checks, 'rules': rules,
              'tables': tables, 'wall_seconds': {k: s['wall_seconds'] for k, s in summaries.items()}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(json.dumps({'checks': checks, 'rules': {k: v['holds'] for k, v in rules.items()}}, indent=2))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    r = sub.add_parser('run')
    r.add_argument('--checkpoint', choices=sorted(CHECKPOINTS), required=True)
    r.add_argument('--runs', type=Path, default=Path('/work/xg24i002/x10041/bypass-lora/runs'))
    r.add_argument('--e16-data', type=Path, required=True)
    r.add_argument('--e1-data', type=Path, required=True)
    r.add_argument('--e4-data', type=Path, required=True)
    r.add_argument('--output', type=Path, required=True)
    r.add_argument('--limit', type=int, default=0, help='smoke only: first N rows of every split, no reload comparison')
    a = sub.add_parser('analyze')
    a.add_argument('--summary', action='append', required=True, metavar='LABEL=PATH')
    a.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'run' and args.limit and args.limit < 2:
        parser.error('--limit must be >= 2')
    return run(args) if args.command == 'run' else analyze(args)


if __name__ == '__main__':
    raise SystemExit(main())
