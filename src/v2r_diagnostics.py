"""E-0015 memory-route diagnostics: evaluation of recorded checkpoints only, no training.

`run` evaluates one recorded checkpoint on E=1, E=4 (ATT-0112-1) and E=16 (V2) dev
with the read substitutions, collects the Pv/Pf/Pz probe inputs on dev and
calibration and fits the probes (calibration) and tests them (dev). It first checks
that every recorded prediction file of the checkpoint is reproduced per episode.
`analyze` combines the checkpoint summaries and evaluates E-0015 rules 1-7.
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
from v2_memory import DIAGNOSTIC_CONTROLS, collate, isolated_writes, observe_writes
from v2_schedule import DATA_HASHES, load_data
from v2_train import adapted, evaluate, write_json
from v2r_data import METADATA_SHA256, load_variant

ARM = 'direct_hidden.soft_slot.F_out'
CHECKPOINTS = {'R1': 'ATT-0122-1', 'R2': 'ATT-0123-1', 'R5': 'ATT-0126-1', 'V2-03': 'ATT-0082-1'}
FACT_COUNTS = (1, 4, 16)
DEV_CONTROLS = ('normal', 'zero', 'shuffled', 'oracle_value') + DIAGNOSTIC_CONTROLS
CALIBRATION_CONTROLS = ('normal', 'oracle_value')  # Pz training inputs
PROBE_BATCH = 16
RIDGE = 1.
# Recorded evaluation directories per split (relative to the arm directory).
RECORDED = {'R1': {16: '.', 1: 'e1'}, 'R2': {16: '.', 4: 'e4'}, 'R5': {16: '.'}, 'V2-03': {16: '.'}}
FACT_GROUPS = (('0', 0, 0), ('1-3', 1, 3), ('>=4', 4, 15))


def fact_text(fact):
    return f"{fact['head']} {fact['relation']} {fact['tail']} ;\n"


def annotate(tok, rows, n_facts):
    """Subject token (as v2_train), per-fact tail token and isolated E=1-format support.

    Returns tokenization checks: isolated fact tokens vs. the same fact's tokens
    inside the contextual support.
    """
    differing = 0
    for row in rows:
        if len(row['facts']) != n_facts:
            raise ValueError('fact count mismatch')
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
            if len(positions) != 1:
                raise ValueError('isolated fact needs exactly one delimiter')
            isolated.append({'ids': ids, 'delimiter': positions[0]['delimiter_token_index']})
            context = row['support_input_ids'][previous + 1:fact['delimiter_token_index'] + 1]
            differing += context != ids[:positions[0]['delimiter_token_index'] + 1]
            previous = fact['delimiter_token_index']
        if tails[row['query']['target_fact_index']] != row['query']['answer_token_id']:
            raise ValueError('target tail token differs from the answer token')
        if row['support_text'] != ''.join(fact_text(f) for f in row['facts']):
            raise ValueError('support text is not the concatenated fact format')
        if n_facts == 1 and (isolated[0]['ids'] != row['support_input_ids']
                             or isolated[0]['delimiter'] != row['facts'][0]['delimiter_token_index']):
            raise ValueError('isolated encoding differs from the E=1 data format')
        row['_tail_token_ids'] = tails
        row['_isolated'] = isolated
    return {'facts': sum(len(r['facts']) for r in rows), 'context_tokens_differ_from_isolated': differing}


def load_splits(e16_dir, variant_dirs):
    data = {16: {s: load_data(Path(e16_dir) / f'{s}.jsonl', s) for s in ('dev', 'calibration')}}
    v2_train = load_data(Path(e16_dir) / 'train.jsonl', 'train')
    # The E=16 prior token comes from the V2 train split, as in v2_train.main.
    priors = {16: Counter(r['query']['answer_token_id'] for r in v2_train).most_common(1)[0][0]}
    for facts in (1, 4):
        metadata, dev = load_variant(variant_dirs[facts], facts, 'dev')
        _, calibration = load_variant(variant_dirs[facts], facts, 'calibration')
        data[facts] = {'dev': dev, 'calibration': calibration}
        priors[facts] = metadata['prior']['answer_token_id']
    for facts, splits in data.items():
        dev_ids = {r['episode_id'] for r in splits['dev']}
        if dev_ids & {r['episode_id'] for r in splits['calibration']}:
            raise ValueError('probe fit and test episodes overlap')
    return data, priors


@torch.no_grad()
def collect_writes(model, rows, pad_id, device):
    """Pv/Pf inputs per written slot in support order, contextual and isolated."""
    out = {kind: {'values': [], 'features': []} for kind in ('context', 'isolated')}
    labels, positions = [], []
    for start in range(0, len(rows), PROBE_BATCH):
        selected = rows[start:start + PROBE_BATCH]
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


def ridge_probe(train_x, train_y, test_x, test_y, ridge=RIDGE):
    """Closed-form ridge classifier with bias (as v2_memory.independent_probes), in FP64.

    Classes are the training labels; a test label outside them is always an error.
    Returns per-test-example correctness and the unseen-label mask.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    classes = torch.unique(train_y, sorted=True).to(device)
    y = (train_y.to(device)[:, None] == classes).double()
    x = torch.cat((train_x.to(device).double(), torch.ones(len(train_x), 1, device=device, dtype=torch.float64)), -1)
    w = torch.linalg.solve(x.T @ x + ridge * torch.eye(x.shape[1], device=device, dtype=torch.float64), x.T @ y)
    t = torch.cat((test_x.to(device).double(), torch.ones(len(test_x), 1, device=device, dtype=torch.float64)), -1)
    predicted = classes[(t @ w).argmax(-1)]
    test_y = test_y.to(device)
    return (predicted == test_y).cpu(), (~torch.isin(test_y, classes)).cpu()


def probe_summary(correct, unseen, positions=None):
    result = {'n': len(correct), 'accuracy': float(correct.float().mean()), 'correct': int(correct.sum()),
              'unseen_label_count': int(unseen.sum())}
    if positions is not None:
        result['by_fact_position'] = {}
        for name, low, high in FACT_GROUPS:
            mask = (positions >= low) & (positions <= high)
            if mask.any():
                result['by_fact_position'][name] = {'n': int(mask.sum()), 'accuracy': float(correct[mask].float().mean())}
    return result


def tensor_digest(x):
    return hashlib.sha256(x.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def write_probes(cal, dev, include_features):
    result = {}
    for kind in ('context', 'isolated'):
        names = (('Pv', 'values'),) + ((('Pf', 'features'),) if include_features else ())
        for probe, name in names:
            correct, unseen = ridge_probe(cal[kind][name], cal['labels'], dev[kind][name], dev['labels'])
            result[f'{probe}.{kind}'] = probe_summary(correct, unseen, dev['positions'])
    return result


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
    result['counts_by_position'] = {k: strata[k]['count'] for k in sorted(strata, key=int)}
    result['em_by_position'] = {k: strata[k]['em'] for k in sorted(strata, key=int)}
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
        mismatches = sum(a['prediction'] != b['prediction'] for a, b in zip(old, new)) if same_ids else None
        report[mode] = {'recorded': str(path), 'episodes': len(old), 'same_episode_order': same_ids,
                        'prediction_mismatches': mismatches,
                        'nll_max_abs_difference': max(abs(a['nll'] - b['nll']) for a, b in zip(old, new))
                        if same_ids else None}
    return report


def run(args):
    started = time.monotonic()
    label = args.checkpoint
    arm_dir = args.runs / CHECKPOINTS[label] / ARM
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'manifest.json').exists():
        raise SystemExit('output already contains run results')
    passed, failure, summary = False, None, {'checkpoint': label, 'attempt': CHECKPOINTS[label]}
    try:
        recorded_config = json.loads((arm_dir / 'metrics.json').read_text())['config']
        data, priors = load_splits(args.e16_data, {1: args.e1_data, 4: args.e4_data})
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
            summary['splits'] = {}
            for facts in FACT_COUNTS:
                split_dir = output / f'e{facts}'
                rows = data[facts]
                part = {'reload': {}}
                pz = {'dev': {}, 'calibration': {}}
                metrics, _ = evaluate(model, rows['dev'], pad_id, split_dir, 'dev', DEV_CONTROLS, priors[facts],
                                      capture=pz['dev'])
                recorded = RECORDED[label].get(facts)
                if recorded is not None:
                    part['reload']['dev'] = compare_recorded(arm_dir / recorded, split_dir, 'dev', DEV_CONTROLS)
                cal_metrics, _ = evaluate(model, rows['calibration'], pad_id, split_dir, 'calibration',
                                          CALIBRATION_CONTROLS, priors[facts], capture=pz['calibration'])
                if recorded is not None and (arm_dir / recorded / 'calibration.normal.predictions.jsonl').exists():
                    part['reload']['calibration'] = compare_recorded(arm_dir / recorded, split_dir, 'calibration',
                                                                     CALIBRATION_CONTROLS)
                write_json(split_dir / 'reload_check.json', part['reload'])
                for split, report in part['reload'].items():
                    bad = {m: r for m, r in report.items() if not r['same_episode_order'] or r['prediction_mismatches']}
                    if bad:
                        raise AssertionError(f'e{facts} {split}: reloaded predictions differ from records: {sorted(bad)}')
                part['controls'] = {mode: {'em': m['em'], 'nll': m['nll'], 'count': m['count'],
                                           'target_position': target_strata(m),
                                           'last_position': {k: m['last_position'].get(k) for k in
                                                             ('argmax_target_rate', 'target_weight_mean', 'n_eff_mean')},
                                           'address.target_weight': m['diagnostics'].get('address.target_weight'),
                                           'read.reconstruction_max_abs': m['diagnostics'].get('read.reconstruction_max_abs')}
                                    for mode, m in metrics.items()}
                part['calibration_controls'] = {mode: {'em': m['em'], 'nll': m['nll'], 'count': m['count']}
                                                for mode, m in cal_metrics.items()}
                writes = {s: collect_writes(model, rows[s], pad_id, device) for s in ('calibration', 'dev')}
                part['probes'] = write_probes(writes['calibration'], writes['dev'], include_features=True)
                part['feature_sha256'] = {f'{s}.{k}': tensor_digest(writes[s][k]['features'])
                                          for s in writes for k in ('context', 'isolated')}
                part['isolated_vs_context_value_max_abs'] = float(
                    (writes['dev']['isolated']['values'] - writes['dev']['context']['values']).abs().max())
                cal_labels = torch.tensor([r['query']['answer_token_id'] for r in rows['calibration']])
                dev_labels = torch.tensor([r['query']['answer_token_id'] for r in rows['dev']])
                dev_positions = torch.tensor([r['query']['target_fact_index'] for r in rows['dev']])
                for mode in CALIBRATION_CONTROLS:
                    correct, unseen = ridge_probe(pz['calibration'][mode], cal_labels, pz['dev'][mode], dev_labels)
                    part['probes'][f'Pz.{mode}'] = probe_summary(correct, unseen, dev_positions)
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
            'experiment': 'E-0015', 'checkpoint': label, 'checkpoint_attempt': CHECKPOINTS[label],
            'adapter': str((arm_dir / 'adapter').resolve()), 'model': MODEL, 'revision': REVISION,
            'data': {'e16': str(Path(args.e16_data).resolve()), 'e16_sha256': DATA_HASHES,
                     'e1': str(Path(args.e1_data).resolve()), 'e4': str(Path(args.e4_data).resolve()),
                     'variant_metadata_sha256': {str(k): METADATA_SHA256[k] for k in (1, 4)}},
            'dev_controls': DEV_CONTROLS, 'calibration_controls': CALIBRATION_CONTROLS, 'ridge': RIDGE,
            'torch': torch.__version__, 'cuda': torch.version.cuda,
            'metrics': {'wall_seconds': wall, 'mig_instance_hours': wall / 3600, 'node_hours': wall / 3600 * .25},
            'failure': failure, 'artifacts': artifacts,
            'validator': {'name': 'v2r_diagnostics.run', 'passed': passed}})
    return 0


# ---- analysis ---------------------------------------------------------------------

RULES = {
    1: 'value 内容在、解码失败：R2@E=4 或 R5@E=16 的 Pv ≥ 80%，且 oracle_value EM < 30%。',
    2: 'value 内容丢失：Pv < 50%，而 Pf ≥ 90%（按 checkpoint × split 列出成立处）。',
    3: '非末位混合读取干扰：R2@E=4 或 R5@E=16 上，last_only − normal ≥ 10 pp，或 oracle_last_only − oracle_value ≥ 10 pp。',
    4: '跨上下文迁移失败：R1 的 oracle_value 在 E=4 与 E=16 上各比 E=1 低 ≥ 30 pp，且该 split 的 Pf ≥ 90%。',
    5: '上下文迁移确认：R1@E=16 的 iso_oracle_value 在目标第 0 条与第 ≥1 条都 ≥ 70%，而上下文 oracle_value 只在第 0 条高（第 0 条 ≥ 70%，第 ≥1 条 < 70%）。',
    6: 'R2 或 R5 的 iso_v 比其 normal 高 ≥ 10 pp（R2@E=4、R5@E=16）。',
    7: '以上都不成立。',
}
TRAINED_SPLIT = {'R2': 'e4', 'R5': 'e16'}


def analyze(args):
    summaries = {}
    for item in args.summary:
        label, _, path = item.partition('=')
        summaries[label] = json.loads(Path(path).read_text())
        if summaries[label].get('checkpoint') != label:
            raise ValueError('summary label mismatch')
    if set(summaries) != set(CHECKPOINTS):
        raise ValueError('analysis needs all four checkpoints')

    def em(label, split, mode, stratum=None):
        control = summaries[label]['splits'][split]['controls'][mode]
        return control['em'] if stratum is None else control['target_position'][stratum]['em']

    def probe(label, split, name):
        return summaries[label]['splits'][split]['probes'][name]['accuracy']

    checks = {'pf_identical_across_checkpoints': all(
        len({json.dumps(s['splits'][f'e{f}']['feature_sha256'], sort_keys=True) for s in summaries.values()}) == 1
        for f in FACT_COUNTS)}
    rules = {}
    rows = []
    for label, split in TRAINED_SPLIT.items():
        rows.append({'checkpoint': label, 'split': split, 'Pv': probe(label, split, 'Pv.context'),
                     'oracle_value': em(label, split, 'oracle_value')})
    rules[1] = {'holds': any(r['Pv'] >= .8 and r['oracle_value'] < .3 for r in rows), 'values': rows}
    cells = [{'checkpoint': c, 'split': s, 'Pv': probe(c, s, 'Pv.context'), 'Pf': probe(c, s, 'Pf.context')}
             for c in CHECKPOINTS for s in (f'e{f}' for f in FACT_COUNTS)]
    where = [x for x in cells if x['Pv'] < .5 and x['Pf'] >= .9]
    rules[2] = {'holds': bool(where), 'where': where, 'values': cells}
    rows = []
    for label, split in TRAINED_SPLIT.items():
        rows.append({'checkpoint': label, 'split': split,
                     'last_only_minus_normal': em(label, split, 'last_only') - em(label, split, 'normal'),
                     'oracle_last_only_minus_oracle_value': em(label, split, 'oracle_last_only') - em(label, split, 'oracle_value')})
    rules[3] = {'holds': any(r['last_only_minus_normal'] >= .1 or r['oracle_last_only_minus_oracle_value'] >= .1 for r in rows),
                'values': rows}
    r1 = {s: em('R1', s, 'oracle_value') for s in ('e1', 'e4', 'e16')}
    pf = {s: probe('R1', s, 'Pf.context') for s in ('e4', 'e16')}
    rules[4] = {'holds': all(r1['e1'] - r1[s] >= .3 and pf[s] >= .9 for s in ('e4', 'e16')),
                'values': {'R1_oracle_value': r1, 'Pf.context': pf}}
    iso = {k: em('R1', 'e16', 'iso_oracle_value', k) for k in ('0', '>=1')}
    ctx = {k: em('R1', 'e16', 'oracle_value', k) for k in ('0', '>=1')}
    rules[5] = {'holds': iso['0'] >= .7 and iso['>=1'] >= .7 and ctx['0'] >= .7 and ctx['>=1'] < .7,
                'values': {'iso_oracle_value': iso, 'oracle_value': ctx}}
    rows = [{'checkpoint': c, 'split': s, 'iso_v_minus_normal': em(c, s, 'iso_v') - em(c, s, 'normal')}
            for c, s in TRAINED_SPLIT.items()]
    rules[6] = {'holds': any(r['iso_v_minus_normal'] >= .1 for r in rows), 'values': rows}
    rules[7] = {'holds': not any(rules[k]['holds'] for k in range(1, 7))}
    for k, text in RULES.items():
        rules[k]['rule'] = text
    tables = {label: {split: {'controls': {m: {'em': c['em'], 'nll': c['nll'],
                                                 'em_target_0': (c['target_position']['0'] or {}).get('em'),
                                                 'em_target_ge1': (c['target_position']['>=1'] or {}).get('em')}
                                             for m, c in part['controls'].items()},
                              'probes': {n: p['accuracy'] for n, p in part['probes'].items()}}
                      for split, part in s['splits'].items()}
              for label, s in summaries.items()}
    report = {'experiment': 'E-0015', 'evidence_level': 'diagnostic', 'checks': checks, 'rules': rules,
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
    a = sub.add_parser('analyze')
    a.add_argument('--summary', action='append', required=True, metavar='LABEL=PATH')
    a.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    return run(args) if args.command == 'run' else analyze(args)


if __name__ == '__main__':
    raise SystemExit(main())
