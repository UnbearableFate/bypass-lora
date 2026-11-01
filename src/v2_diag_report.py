"""Verify saved diagnostic aggregates and render tables; PBS analysis only."""
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    report = json.loads((args.source / 'metrics.json').read_text())
    records = [json.loads(s) for s in (args.source / 'diagnostics.jsonl').read_text().splitlines()]
    assert len(records) == 2048 * 6
    assert len({(r['episode_id'], r['line'], r['E']) for r in records}) == len(records)
    lines = ['## Full calibration results (ATT-0050-1)', '',
             'All scores below use batch 8; failed implementation gate limits interpretation.', '',
             '| Line / E | n | EM | NLL | Subject copy | Nonself EM | Nonself copy | Candidate a | Candidate b | Candidate c |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name, s in report['groups'].items():
        rows = [r for r in records if f"{r['line']}.E{r['E']}" == name]
        n = len(rows)
        assert n == s['examples'] == 2048
        assert sum(r['answer_is_subject'] for r in rows) == s['answer_is_subject_count'] == 27
        assert math.isclose(sum(r['prediction'] == r['target'] for r in rows) / n, s['em'], abs_tol=1e-12)
        assert math.isclose(sum(r['nll'] for r in rows) / n, s['nll'], abs_tol=1e-12)
        assert Counter(r['category'] for r in rows) == s['category_counts']
        for k in ('a', 'b', 'c'):
            cs = [r['candidates'][k] for r in rows]
            assert all(c['correct'] == (c['prediction'] == r['target']) for c, r in zip(cs, rows))
            assert math.isclose(sum(c['correct'] for c in cs) / n, s['candidates'][k]['accuracy'], abs_tol=1e-12)
            assert math.isclose(sum(c['covered'] for c in cs) / n, s['candidates'][k]['coverage'], abs_tol=1e-12)
        values = [s['em'], s['nll'], s['subject_copy_rate'], s['nonself_em'], s['nonself_subject_copy_rate']]
        values += [s['candidates'][k]['accuracy'] for k in ('a','b','c')]
        lines.append(f'| {name} | {n} | ' + ' | '.join(f'{x:.6f}' for x in values) + ' |')
    lines += ['', '27/2048 episodes have answer = subject; nonself denominator is 2021.',
              'E1 nonself candidate-a two-choice accuracy: ' + str(report['groups']['F.E1']['nonself_candidate_a_accuracy']),
              'E1 candidate-b is degenerate and is not interpreted.', '',
              '## Candidate denominators and random baselines', '',
              '| Line / E | Set | Accuracy | Coverage | Empty | Random nonempty | Random attainable | Conditional accuracy |',
              '|---|---|---:|---:|---:|---:|---:|---:|']
    for name,s in report['groups'].items():
        for k,c in s['candidates'].items():
            vals = [c[x] for x in ('accuracy','coverage','empty_count','random_nonempty','random_attainable','conditional_accuracy')]
            lines.append(f'| {name} | {k} | ' + ' | '.join(str(v) if v is None else f'{v:.6f}' for v in vals) + ' |')
    lines += ['', '## Top-1 categories and score distributions', '']
    for name in ('M.E16','F.E16'):
        s=report['groups'][name]
        lines += [f'### {name}', '', 'Category counts: ' + json.dumps(s['category_counts']), '',
                  '| Quantity | Mean | Min | P25 | Median | P75 | P90 | P95 | P99 | Max |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for k,d in s['distributions'].items():
            lines.append(f'| {k} | ' + ' | '.join(f'{d[x]:.7g}' for x in ('mean','min','p25','p50','p75','p90','p95','p99','max')) + ' |')
        lines.append('')
    lines += ['## E16 F strata', '', 'Position bins 0/1/2/3 mean indices 0–3/4–7/8–11/12–15.', '',
              '| Axis | Value | n | EM | NLL | Subject copy | Candidate a | Candidate b | Candidate c |',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for axis, groups in report['strata_F_E16'].items():
        assert sum(s['examples'] for s in groups.values()) == 2048
        for k,s in groups.items():
            vals=[s['em'],s['nll'],s['subject_copy_rate']] + [s['candidates'][c]['accuracy'] for c in ('a','b','c')]
            lines.append(f"| {axis} | {k} | {s['examples']} | " + ' | '.join(f'{v:.6f}' for v in vals) + ' |')
    lines += ['', '## Five fixed calibration examples', '', (args.source/'examples.md').read_text()]
    args.output.mkdir(exist_ok=True, parents=True)
    (args.output/'tables.md').write_text('\n'.join(lines)+'\n')
    summary={'checked_records':len(records),'checked_groups':len(report['groups']), 'passed':True}
    (args.output/'validation.json').write_text(json.dumps(summary,indent=2))
    (args.output/'manifest.json').write_text(json.dumps({'status':'complete','spec_digest':os.environ['SR_SPEC_DIGEST'],
        'metrics':summary,'artifacts':[{'path':str((args.output/f).resolve()),'role':f} for f in ('tables.md','validation.json')],
        'validator':{'name':'saved_diagnostic_aggregate_check','passed':True}},indent=2))
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
