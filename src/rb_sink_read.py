"""E-0024: position-0 read share and cross-episode constant-read index on RB1.

No training. Each RB1 arm's final checkpoint (the one E-0021 intervened on,
matched by manifest and checkpoint-inventory SHA256) reads the dev gold
candidate under normal, iv_strict and iv_window8. The forward, intervention and
read-distribution code are the reviewed rb_history/rb_read_diagnostics paths.
Only gold candidates are run, one row per forward exactly as E-0021 encoded
them: right-padded batches change BF16 results even for the plain Base model
(ATT-0270-1). Every row's gold NLL and the read summaries must match the E-0021
run of the same checkpoint within the gates below, or the arm fails.
"""
import argparse
import json
import math
import os
from pathlib import Path
import random
import time

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from calibrate import MODEL, REVISION
from rb_read_diagnostics import ReadSummary
from rb_train import make_model
from v1_data import derive_seed, encode_candidates
from v1_run import batch_tensors, prepare
from v5_train import file_sha256, write_json


ARMS = ('RB1-02', 'RB1-03', 'RB1-08', 'RB1-09')
RESIDUAL = ('RB1-02', 'RB1-03')
CONDITIONS = ('normal', 'iv_strict', 'iv_window8')
TASKS = ('piqa', 'boolq')
SHARES = ('share0', 'share0_2', 'argmax0', 'uniform0', 'uniform0_2')
SAMPLE = 200
# Acceptance gates against the batch-one E-0021 run of the same checkpoint.
# Batch-one forwards repeat E-0021's computation, so only float noise is allowed.
NLL_TOLERANCE = 1e-4
READ_TOLERANCE = 1e-6


def gold_index(row):
    return int(row['label']) if row['task'] == 'piqa' else (0 if row['label'] else 1)


def sample_indices(count, task):
    rng = random.Random(derive_seed(20261101, 'residual_la_v21', 'E-0024', 'constant_read', task))
    return sorted(rng.sample(range(count), min(SAMPLE, count)))


def mean_pairwise_cosine(vectors):
    """Mean cosine over unordered distinct pairs; zero vectors are rejected."""
    x = torch.as_tensor(vectors).double()
    if x.ndim != 2 or x.shape[0] < 2:
        raise ValueError('pairwise cosine needs at least two vectors')
    norms = x.norm(dim=1)
    if not bool((norms > 0).all()):
        raise ValueError('pairwise cosine is undefined for a zero vector')
    unit = x/norms[:, None]
    n = x.shape[0]
    gram = unit @ unit.T
    return float((gram.sum() - gram.diagonal().sum())/(n*(n-1)))


def answer_mask(labels, attention_mask):
    """Positions predicting gold answer tokens (as in ReadSummary)."""
    answer = torch.zeros_like(attention_mask, dtype=torch.bool)
    answer[:, :-1] = labels[:, 1:] != -100
    return answer & attention_mask.bool()


class Accumulator:
    """Question-level shares, constant-read vectors and implementation checks."""

    def __init__(self, sampled):
        self.sampled = sampled
        self.rows = {}
        self.vectors = {}
        self.window_violation = 0.
        self.window_short_positions = 0

    def add(self, task, row_indices, diagnostics, attention_mask, labels, kind):
        answer = answer_mask(labels, attention_mask)
        counts = answer.sum(1)
        if not bool((counts > 0).all()):
            raise ValueError('gold candidate without answer positions')
        first = answer.long().argmax(1)
        late = answer & (torch.arange(answer.shape[1], device=answer.device)[None] >= 8)
        chosen = [b for b, index in enumerate(row_indices) if index in self.sampled[task]]
        for layer, diagnostic in diagnostics.items():
            values = diagnostic['history']['values']
            for record in diagnostic['read_metrics']:
                key = (task, layer, record['step'])
                rows = self.rows.setdefault(key, {name: [] for name in SHARES})
                for name in SHARES:
                    means = (record[name].double()*answer).sum(1)/counts
                    rows[name].extend(means.tolist())
                if kind == 'iv_window8':
                    self.window_short_positions += int((answer & ~late).sum())
                    if bool(late.any()):
                        self.window_violation = max(self.window_violation,
                                                    float(record['share0'][late].abs().max()))
                store = self.vectors.setdefault(key, {'ids': [], 'read': [], 'value': [],
                                                      'first_position': []})
                for b in chosen:
                    p = int(first[b])
                    store['ids'].append(row_indices[b])
                    store['first_position'].append(p)
                    store['read'].append(record['read_vector'][b, p].double().cpu())
                    store['value'].append(values[b, p].double().cpu())

    def finish(self):
        result = []
        for (task, layer, step), rows in sorted(self.rows.items()):
            item = {'task': task, 'layer': layer, 'step': step, 'questions': len(rows['share0']),
                    'question_means': {name: sum(v)/len(v) for name, v in rows.items()}}
            store = self.vectors[(task, layer, step)]
            if sorted(store['ids']) != self.sampled[task]:
                raise ValueError('constant-read sample incomplete')
            item['constant_read'] = {
                'questions': len(store['ids']),
                'C_read': mean_pairwise_cosine(torch.stack(store['read'])),
                'C_current_value': mean_pairwise_cosine(torch.stack(store['value'])),
                'first_position_mean': sum(store['first_position'])/len(store['first_position'])}
            result.append(item)
        return result


@torch.no_grad()
def run_condition(model, rows_by_task, encoded_by_task, kind, sampled, batch_size, pad_id):
    summary, accumulator = ReadSummary(), Accumulator(sampled)
    nll = {}
    options = None if kind == 'normal' else {'kind': kind}
    for task in TASKS:
        encoded = encoded_by_task[task]
        order = sorted(range(len(encoded)), key=lambda i: (len(encoded[i]['input_ids']), i))
        for start in range(0, len(order), batch_size):
            indices = order[start:start+batch_size]
            batch = batch_tensors([encoded[i] for i in indices], pad_id)
            output = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                           use_cache=False, intervention=options, trace_reads=True, capture_history=True)
            labels = batch['labels']
            shifted = labels[:, 1:]
            for b, index in enumerate(indices):
                valid = shifted[b] != -100
                loss = F.cross_entropy(output.logits[b, :-1][valid].float(), shifted[b][valid], reduction='sum')
                if not torch.isfinite(loss):
                    raise ValueError(f'{task}/{rows_by_task[task][index]["id"]}: nonfinite gold NLL')
                nll[(task, rows_by_task[task][index]['id'])] = float(loss)
            summary.add(task, output.diagnostics, batch['attention_mask'], labels)
            accumulator.add(task, indices, output.diagnostics, batch['attention_mask'], labels, kind)
            del output
    return summary.finish(), accumulator, nll


def compare_reference(reference, condition, summaries, nll, nll_tolerance=NLL_TOLERANCE,
                      read_tolerance=READ_TOLERANCE):
    """Differences from the batch-one E-0021 run of the same checkpoint, with acceptance gates."""
    predictions = {(row['task'], row['id']): row['gold_nll_sum'] for row in reference['predictions']}
    if set(predictions) != set(nll):
        raise ValueError(f'{condition}: dev rows differ from E-0021')
    nll_diff = [abs(nll[key]-predictions[key]) for key in nll]
    old = {(r['task'], r['layer'], r['step'], r['scope']): r for r in reference['read_diagnostics']}
    new = {(r['task'], r['layer'], r['step'], r['scope']): r for r in summaries}
    if set(old) != set(new):
        raise ValueError(f'{condition}: read-summary keys differ from E-0021')
    relative, absolute = {}, {}
    for key, value in old.items():
        if value['count'] != new[key]['count'] or value['empty_reads'] != new[key]['empty_reads']:
            raise ValueError(f'{condition}: read-summary counts differ from E-0021 at {key}')
        for name, before in value['nonempty_means'].items():
            after = new[key]['nonempty_means'][name]
            gap = abs(after-before)
            absolute[name] = max(absolute.get(name, 0.), gap)
            relative[name] = max(relative.get(name, 0.), gap/max(abs(before), 1e-12))
    gaps = [*nll_diff, *relative.values()]
    passed = (all(math.isfinite(x) for x in gaps) and max(nll_diff) <= nll_tolerance and
              max(relative.values()) <= read_tolerance)
    return {'gold_nll_max_abs_diff': max(nll_diff), 'gold_nll_mean_abs_diff': sum(nll_diff)/len(nll_diff),
            'read_summary_max_abs_diff': absolute, 'read_summary_max_rel_diff': relative,
            'tolerances': {'gold_nll_abs': nll_tolerance, 'read_summary_rel': read_tolerance},
            'passed': passed}


def run(args):
    from rb1_artifacts import validate_formal_run
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    reference_dir = args.reference.resolve()
    reference_provenance = json.loads((reference_dir/'provenance.json').read_text())
    manifest_sha = file_sha256(Path(args.training_run)/'manifest.json')
    inventory_sha = file_sha256(Path(args.training_run)/'checkpoint.json')
    if (manifest_sha != reference_provenance['training_manifest']['sha256'] or
            inventory_sha != reference_provenance['checkpoint_inventory']['sha256']):
        raise ValueError(f'{args.arm}: checkpoint is not the one E-0021 intervened on')
    _, inventory, adapter = validate_formal_run(args.training_run, args.arm)
    reference = json.loads((reference_dir/'conditions.json').read_text())
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    splits = prepare(args.data_dir)
    rows_by_task = {task: splits['splits'][task]['dev'] for task in TASKS}
    encoded_by_task = {task: [encode_candidates(tokenizer, row)[gold_index(row)] for row in rows]
                       for task, rows in rows_by_task.items()}
    first_tokens = {}
    for task, encoded in encoded_by_task.items():
        ids = sorted({e['input_ids'][0] for e in encoded})
        first_tokens[task] = {'ids': ids, 'identical': len(ids) == 1,
                              'decoded': [tokenizer.decode([i]) for i in ids[:10]]}
    sampled = {task: sample_indices(len(rows), task) for task, rows in rows_by_task.items()}
    model = make_model(args.arm, phase='R1', seed=0, checkpoint=adapter).eval()
    result = {'arm': args.arm, 'position0_token': first_tokens,
              'sample': {task: {'seed_label': ['E-0024', 'constant_read', task], 'size': len(v)}
                         for task, v in sampled.items()},
              'conditions': {}}
    for condition in CONDITIONS:
        tick = time.monotonic()
        summaries, accumulator, nll = run_condition(model, rows_by_task, encoded_by_task, condition,
                                                    sampled, 1, tokenizer.pad_token_id)
        result['conditions'][condition] = {
            'answer_question_level': accumulator.finish(),
            'read_summary': summaries,
            'window8_share0_max_late': accumulator.window_violation if condition == 'iv_window8' else None,
            'window8_answer_positions_before_8': accumulator.window_short_positions if condition == 'iv_window8' else None,
            'reference_check': compare_reference(reference[condition], condition, summaries, nll),
            'elapsed_s': time.monotonic()-tick}
        write_json(out/'sink.json', result)
        if not result['conditions'][condition]['reference_check']['passed']:
            raise ValueError(f'{args.arm}/{condition}: differs from the E-0021 reference beyond tolerance')
    result['provenance'] = {
        'formal_run': inventory, 'reference_run': str(reference_dir),
        'training_manifest_sha256': manifest_sha, 'checkpoint_inventory_sha256': inventory_sha,
        'source_commit': os.environ['SR_CODE_COMMIT'], 'model_revision': REVISION,
        'environment_lock_sha256': file_sha256(Path(__file__).resolve().parents[1]/'uv.lock'),
        'batch_size': 1, 'reference_hardware': reference_provenance.get('hardware'),
        'hardware': torch.cuda.get_device_name()}
    write_json(out/'sink.json', result)
    return result


def concentrated_layers(reference_normal):
    """DOC-0263 concentration: nonempty answer-position n_eff < 2 under normal."""
    return sorted((r['task'], r['layer'], r['step']) for r in reference_normal['read_diagnostics']
                  if r['scope'] == 'answer_prediction' and r['nonempty_means']['n_eff'] < 2)


def interpret(arms):
    """Preregistered E-0024 reading for the residual group (normal condition)."""
    cells = []
    for arm in RESIDUAL:
        normal = {(r['task'], r['layer'], r['step']): r
                  for r in arms[arm]['conditions']['normal']['answer_question_level']}
        for key in arms[arm]['concentrated']:
            row = normal[key]
            cells.append({'arm': arm, 'task': key[0], 'layer': key[1], 'step': key[2],
                          'share0': row['question_means']['share0'],
                          'C_read': row['constant_read']['C_read']})
    if not cells:
        return {'cells': cells, 'reading': 'no concentrated residual layers'}
    if all(c['share0'] > .5 and c['C_read'] > .9 for c in cells):
        reading = 'sink'
    elif all(c['share0'] < .2 for c in cells):
        reading = 'content'
    else:
        reading = 'mixed'
    return {'cells': cells, 'reading': reading}


def consistency(arms):
    checks = {}
    for arm, value in arms.items():
        normal = {(r['task'], r['layer'], r['step']): r['question_means']['share0']
                  for r in value['conditions']['normal']['answer_question_level']}
        strict = {(r['task'], r['layer'], r['step']): r['question_means']['share0']
                  for r in value['conditions']['iv_strict']['answer_question_level']}
        gap = max(abs(normal[k]-strict[k]) for k in normal)
        window = value['conditions']['iv_window8']
        checks[arm] = {'strict_minus_normal_share0_max_abs': gap, 'strict_ok': gap < .01,
                       'window8_share0_max_late': window['window8_share0_max_late'],
                       'window8_answer_positions_before_8': window['window8_answer_positions_before_8'],
                       'window8_ok': window['window8_share0_max_late'] == 0}
    return checks


def summarize(args):
    arms = {}
    for path in args.inputs:
        value = json.loads(Path(path).read_text())
        reference = json.loads((Path(value['provenance']['reference_run'])/'conditions.json').read_text())
        value['concentrated'] = concentrated_layers(reference['normal'])
        arms[value['arm']] = value
    report = build_report(arms)
    write_json(args.output, report)
    if not report['checks_passed']:
        raise SystemExit('E-0024 implementation checks failed; no interpretation recorded')
    return report


def build_report(arms):
    if set(arms) != set(ARMS):
        raise ValueError('summary needs exactly the four E-0024 arms')
    checks = consistency(arms)
    references = {arm: {c: v['conditions'][c]['reference_check']['passed'] for c in CONDITIONS}
                  for arm, v in arms.items()}
    passed = (all(c['strict_ok'] and c['window8_ok'] for c in checks.values()) and
              all(all(v.values()) for v in references.values()))
    report = {'contract': 'E-0024@1', 'arms': {arm: {'concentrated': v['concentrated'],
              'position0_token': v['position0_token'], 'conditions': {
                  c: {k: v['conditions'][c][k] for k in ('answer_question_level', 'reference_check',
                      'window8_share0_max_late', 'window8_answer_positions_before_8', 'elapsed_s')}
                  for c in CONDITIONS}} for arm, v in arms.items()},
              'consistency': checks, 'reference_checks_passed': references, 'checks_passed': passed,
              # E-0024: a failed implementation check is investigated before any reading.
              'interpretation': interpret(arms) if passed else
                  {'reading': 'implementation_check_failed', 'cells': []}}
    return report


def write_manifest(args):
    """Attempt manifest; success needs every arm and a summary whose checks passed."""
    out = args.output
    artifacts = [{'path': str(out/arm/'sink.json'), 'role': f'sink:{arm}'} for arm in ARMS
                 if (out/arm/'sink.json').is_file()]
    summary = json.loads((out/'summary.json').read_text()) if (out/'summary.json').is_file() else None
    if summary is not None:
        artifacts.append({'path': str(out/'summary.json'), 'role': 'summary'})
    passed = (len(artifacts) == len(ARMS)+1 and summary['checks_passed'] is True and
              summary['interpretation']['reading'] != 'implementation_check_failed')
    manifest = {'status': 'complete' if passed else 'partial', 'contract': 'E-0024@1',
                'spec_digest': os.environ['SR_SPEC_DIGEST'], 'phase': 'R1', 'seed': 0,
                'stage': 'diagnostic', 'artifacts': artifacts,
                'metrics': {'elapsed_s': args.elapsed_s,
                            'arms_written': sum(a['role'].startswith('sink:') for a in artifacts),
                            'reading': summary['interpretation']['reading'] if summary else None},
                'validator': {'name': 'rb_sink_read', 'passed': passed}}
    write_json(out/'manifest.json', manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    one = sub.add_parser('arm')
    one.add_argument('--arm', choices=ARMS, required=True)
    one.add_argument('--training-run', type=Path, required=True)
    one.add_argument('--reference', type=Path, required=True, help='E-0021 intervention run of this arm')
    one.add_argument('--data-dir', required=True)
    one.add_argument('--output', type=Path, required=True)
    both = sub.add_parser('summarize')
    both.add_argument('--inputs', nargs='+', required=True)
    both.add_argument('--output', type=Path, required=True)
    final = sub.add_parser('manifest')
    final.add_argument('--output', type=Path, required=True, help='attempt directory')
    final.add_argument('--elapsed-s', type=float, required=True)
    args = parser.parse_args()
    {'arm': run, 'summarize': summarize, 'manifest': write_manifest}[args.command](args)


if __name__ == '__main__':
    main()
