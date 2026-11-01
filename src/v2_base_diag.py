"""E-0007@2 calibration-only Base diagnostics. Run inside PBS, never train."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import random
import time

import torch
from transformers import AutoTokenizer

from v2_data import (MODEL, REVISION, CANDIDATES, derive_seed, encode, file_digest,
                     visible_delimiters)
from v2_eval import evaluate, left_pad, load_base, prompt_tokens, score_logits

CALIBRATION_SHA = 'c08753365165216b5a9f4c3d629271b2183e1323973886256867e826c1712f18'
SUBSET_SEED = derive_seed(20261029, 'concat_loop_v2', 'V2', 'diag-fact-subset')
FACT_COUNTS = (1, 2, 4, 8, 16)


def read_calibration(path):
    if file_digest(path) != CALIBRATION_SHA:
        raise ValueError('calibration digest mismatch')
    rows = [json.loads(s) for s in Path(path).read_text().splitlines()]
    if len(rows) != 2048 or len({r['episode_id'] for r in rows}) != 2048:
        raise ValueError('calibration size/identity mismatch')
    if any(r['identity']['split'] != 'calibration' for r in rows):
        raise ValueError('only calibration is allowed')
    return rows


def fact_view(tok, row, count):
    """Nested deterministic subsets; preserve source identity and relative order."""
    if count not in FACT_COUNTS or len(row['facts']) != 16:
        raise ValueError('invalid diagnostic fact count')
    target = row['query']['target_fact_index']
    others = [i for i in range(16) if i != target]
    random.Random(derive_seed(SUBSET_SEED, row['episode_id'])).shuffle(others)
    chosen = sorted([target] + others[:count - 1])
    facts = [{k: row['facts'][i][k] for k in ('head', 'relation', 'tail')} for i in chosen]
    # Exact literal template in v2_data.make_episode; no prompt redesign.
    support = ''.join(f"{f['head']} {f['relation']} {f['tail']} ;\n" for f in facts)
    ids, positions = visible_delimiters(tok, support)
    if count == 16 and (support != row['support_text'] or ids != row['support_input_ids']):
        raise ValueError('E16 rerender mismatch')
    for fact, position in zip(facts, positions):
        fact.update(position)
    view = dict(row, support_text=support, support_input_ids=ids, facts=facts,
                query=dict(row['query'], target_fact_index=chosen.index(target)))
    view['diagnostic'] = {'source_fact_indices': chosen, 'n_facts': count,
                          'original_target_fact_index': target}
    prompt_tokens(tok, view, 'F', split='calibration')
    return view


def token_names(tok):
    return {name: ids[0] for name in CANDIDATES
            if len(ids := encode(tok, ' ' + name)) == 1 and tok.decode(ids) == ' ' + name}


def candidate_sets(tok, row):
    facts, query = row['facts'], row['query']
    all_names = {f[k] for f in facts for k in ('head', 'tail')}
    names = {'a': all_names, 'b': all_names - {query['head']},
             'c': {f['tail'] for f in facts if f['relation'] == query['relation']}}
    ids, excluded = {}, {}
    for key, values in names.items():
        valid, invalid = [], []
        for name in sorted(values):
            tokens = encode(tok, ' ' + name)
            if len(tokens) == 1:
                valid.append(tokens[0])
            else:
                invalid.append(name)
        ids[key], excluded[key] = sorted(set(valid)), invalid
    return ids, excluded


def category(pred, row, names):
    q, facts = row['query'], row['facts']
    if pred == q['answer_token_id']:
        return 'correct'
    if pred == names[q['head']]:
        return 'question_subject'
    if pred in {names[f['tail']] for f in facts if f['relation'] == q['relation']}:
        return 'other_tail_same_relation'
    if pred in {names[f['tail']] for f in facts}:
        return 'other_tail_other_relation'
    if pred in {names[f['head']] for f in facts}:
        return 'other_head'
    if pred in set(names.values()):
        return 'name_outside_support'
    return 'non_name_token'


def candidate_score(logits, ids, target):
    # Lowest token ID wins ties, as in full-vocabulary torch.argmax.
    ids = sorted(set(ids))
    pred = ids[int(logits[ids].argmax())] if ids else None
    covered = target in ids
    return {'size': len(ids), 'token_ids': ids, 'prediction': pred,
            'correct': pred == target, 'covered': covered,
            'random_nonempty': 1 / len(ids) if ids else None,
            'random_attainable': 1 / len(ids) if covered else 0.}


def describe(logits, tok, row, names, line, scored=None):
    target = row['query']['answer_token_id']
    if scored is None:
        predictions, losses = score_logits(logits[None], torch.tensor([target], device=logits.device))
        pred, nll = int(predictions[0]), float(losses[0])
    else:
        pred, nll = scored
    probs = logits.float().softmax(-1)
    # Stable token-ID tie break also makes the first top-5 token equal argmax.
    top = torch.argsort(logits, descending=True, stable=True)[:5].tolist()
    ids, excluded = candidate_sets(tok, row)
    head_id = names[row['query']['head']]
    return {'episode_id': row['episode_id'], 'line': line, 'E': len(row['facts']),
            'source_fact_indices': row['diagnostic']['source_fact_indices'],
            'target_fact_index': row['query']['target_fact_index'],
            'original_target_fact_index': row['diagnostic']['original_target_fact_index'],
            'relation': row['query']['relation'], 'support_order': row['axes']['support_order'],
            'answer_frequency_stratum': row['axes']['answer_frequency_stratum'],
            'target': target, 'subject_token_id': head_id, 'answer_is_subject': target == head_id,
            'prediction': pred, 'correct': pred == target, 'subject_copy': pred == head_id,
            'category': category(pred, row, names), 'nll': nll,
            'answer_probability': float(probs[target]), 'subject_probability': float(probs[head_id]),
            'answer_rank': 1 + int((logits > logits[target]).sum()),
            'answer_tie_count': int((logits == logits[target]).sum()),
            'top5': [{'token_id': i, 'text': tok.decode([i]), 'probability': float(probs[i])} for i in top],
            'candidates': {k: candidate_score(logits, v, target) for k, v in ids.items()},
            'excluded_non_single_token_names': excluded}


def average(values):
    return sum(values) / len(values) if values else None


def distribution(values):
    x = torch.tensor(values, dtype=torch.float64)
    return {'mean': float(x.mean()), 'min': float(x.min()), 'max': float(x.max()),
            **{f'p{p}': float(torch.quantile(x, p / 100)) for p in (25, 50, 75, 90, 95, 99)}}


def summarize(records):
    if not records:
        return {'examples': 0}
    nonself = [r for r in records if not r['answer_is_subject']]
    result = {'examples': len(records), 'answer_is_subject_count': len(records) - len(nonself),
              'em': average([r['correct'] for r in records]),
              'nll': average([r['nll'] for r in records]),
              'subject_copy_rate': average([r['subject_copy'] for r in records]),
              'nonself_examples': len(nonself),
              'nonself_em': average([r['correct'] for r in nonself]),
              'nonself_subject_copy_rate': average([r['subject_copy'] for r in nonself]),
              'category_counts': dict(Counter(r['category'] for r in records)),
              'distributions': {k: distribution([r[k] for r in records]) for k in
                                ('answer_rank', 'answer_probability', 'subject_probability')},
              'candidates': {}}
    for key in ('a', 'b', 'c'):
        cs = [r['candidates'][key] for r in records]
        result['candidates'][key] = {
            'accuracy': average([c['correct'] for c in cs]),
            'coverage': average([c['covered'] for c in cs]),
            'empty_count': sum(c['size'] == 0 for c in cs),
            'random_nonempty': average([c['random_nonempty'] for c in cs if c['size']]),
            'random_attainable': average([c['random_attainable'] for c in cs]),
            'conditional_accuracy': average([c['correct'] for c in cs if c['covered']])}
    result['nonself_candidate_a_accuracy'] = average([r['candidates']['a']['correct'] for r in nonself])
    return result


@torch.inference_mode()
def diagnose(model, tok, views, output):
    names = token_names(tok)
    device = next(model.parameters()).device
    model.eval()
    if any(p.requires_grad for p in model.parameters()):
        raise ValueError('Base must be frozen')
    records = []
    with (output / 'diagnostics.jsonl').open('w') as handle:
        for line, count in [('M', 16)] + [('F', e) for e in FACT_COUNTS]:
            rows = views[count]
            for start in range(0, len(rows), 8):
                batch = rows[start:start + 8]
                inputs = left_pad([prompt_tokens(tok, r, line, split='calibration') for r in batch],
                                  tok.pad_token_id, device)
                logits = model(**inputs, use_cache=False, logits_to_keep=1).logits[:, -1, :]
                targets = torch.tensor([r['query']['answer_token_id'] for r in batch], device=device)
                predictions, losses = score_logits(logits, targets)
                # Preserve original device/batch reduction for EM/NLL. CPU
                # vocabulary summaries must not silently replace CUDA CE.
                scored = list(zip(predictions.tolist(), losses.tolist()))
                for row, vector, scores in zip(batch, logits.float().cpu(), scored):
                    record = describe(vector, tok, row, names, line, scored=scores)
                    handle.write(json.dumps(record, allow_nan=False) + '\n')
                    records.append(record)
    return records


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--calibration', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--smoke-limit', type=int, default=0)
    args = p.parse_args()
    tick = time.monotonic()
    rows = read_calibration(args.calibration)
    if args.smoke_limit < 0:
        raise ValueError('negative smoke limit')
    if args.smoke_limit:
        rows = rows[:args.smoke_limit]
    args.output.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    views = {e: [fact_view(tok, r, e) for r in rows] for e in FACT_COUNTS}
    model = load_base()
    original, predictions = {}, {}
    for batch in (1, 8):
        out = args.output / f'original-batch{batch}'
        out.mkdir(exist_ok=True)
        original[str(batch)] = evaluate(model, tok, rows, out, batch_size=batch, split='calibration')
        predictions[batch] = [json.loads(s) for s in (out / 'predictions.jsonl').read_text().splitlines()]
    records = diagnose(model, tok, views, args.output)
    groups = {f'{line}.E{e}': summarize([r for r in records if r['line'] == line and r['E'] == e])
              for line, e in [('M', 16)] + [('F', e) for e in FACT_COUNTS]}
    checks = {}
    for line in ('M', 'F'):
        one = [r for r in predictions[1] if r['line'] == line]
        eight = [r for r in predictions[8] if r['line'] == line]
        diag = [r for r in records if r['line'] == line and r['E'] == 16]
        assert [r['episode_id'] for r in one] == [r['episode_id'] for r in eight] == [r['episode_id'] for r in diag]
        checks[line] = {
            'batch_top1_agreement': average([a['prediction'] == b['prediction'] for a, b in zip(one, eight)]),
            'batch_nll_mean_abs_difference': average([abs(a['nll'] - b['nll']) for a, b in zip(one, eight)]),
            'diagnostic_original_top1_agreement': average([a['prediction'] == b['prediction'] for a, b in zip(eight, diag)]),
            'diagnostic_original_nll_max_abs_difference': max(abs(a['nll'] - b['nll']) for a, b in zip(eight, diag))}
        if checks[line]['diagnostic_original_top1_agreement'] != 1 or checks[line]['diagnostic_original_nll_max_abs_difference'] > 1e-5:
            raise ValueError('diagnostic scoring differs from original batch8 path')
    passed = all(c['batch_top1_agreement'] >= .99 and c['batch_nll_mean_abs_difference'] <= .05 for c in checks.values())
    passed = passed and groups['F.E1']['nll'] < groups['M.E16']['nll']
    decision = ('investigate_implementation' if not passed else 'entity_found_argmax_suppressed'
                if groups['F.E16']['candidates']['b']['accuracy'] >= .5 else 'weak_E16_retrieval')
    f16 = [r for r in records if r['line'] == 'F' and r['E'] == 16]
    strata = {}
    for axis in ('position_bin', 'relation', 'support_order', 'answer_frequency_stratum'):
        key = lambda r: str(r['original_target_fact_index'] // 4) if axis == 'position_bin' else str(r[axis])
        strata[axis] = {v: summarize([r for r in f16 if key(r) == v]) for v in sorted({key(r) for r in f16})}
    report = {'contract': 'E-0007@2', 'calibration_sha256': CALIBRATION_SHA, 'model': MODEL,
              'revision': REVISION, 'torch': torch.__version__, 'cuda': torch.version.cuda,
              'smoke_limit': args.smoke_limit, 'groups': groups, 'strata_F_E16': strata,
              'original_path': original, 'implementation_checks': checks,
              'implementation_checks_passed': passed, 'decision': decision,
              'elapsed_s': time.monotonic() - tick}
    (args.output / 'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    examples = []
    # First five calibration identities, not selected by their predictions.
    for index, e in enumerate((1, 16, 1, 16, 16)):
        if index >= len(rows):
            break
        row = views[e][index]
        record = next(r for r in records if r['episode_id'] == row['episode_id'] and r['line'] == 'F' and r['E'] == e)
        examples.append(f"### Calibration index {row['identity']['index']}, E={e}\n\n```text\n{row['support_text']}{row['question_text']}\n```\n\nAnswer: {row['query']['answer_text']!r}\n\nTop-5: {json.dumps(record['top5'])}\n")
    (args.output / 'examples.md').write_text('\n'.join(examples))
    artifacts = ['metrics.json', 'diagnostics.jsonl', 'examples.md',
                 'original-batch1/predictions.jsonl', 'original-batch8/predictions.jsonl']
    metrics = {f'{k}.{m}': v[m] for k, v in groups.items() for m in ('em', 'nll', 'subject_copy_rate')}
    metrics['implementation_checks_passed'] = passed
    (args.output / 'manifest.json').write_text(json.dumps({'status': 'complete',
        'spec_digest': os.environ['SR_SPEC_DIGEST'], 'metrics': metrics,
        'artifacts': [{'path': str((args.output / f).resolve()), 'role': f} for f in artifacts],
        'validator': {'name': 'v2_base_diag', 'passed': True}}, indent=2))
    print(json.dumps({'groups': groups, 'checks': checks, 'decision': decision, 'elapsed_s': report['elapsed_s']}))


if __name__ == '__main__':
    main()
