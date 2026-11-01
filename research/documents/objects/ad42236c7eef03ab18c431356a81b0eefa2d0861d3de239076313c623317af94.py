"""PBS reporting wrapper around the RV-0008-reviewed formal analysis.

No selection or bootstrap rule is implemented here; v1_analysis owns both.
Additional outputs tabulate existing learning-curve metrics and ledger usage.
"""
import json
import math
import os
from pathlib import Path
import sys
import time

from v1_analysis import analyze, ARM_IDS, _arrays, _point_metrics


ROOT = Path('/work/xg24i002/x10041/bypass-lora')


def read(path):
    return json.loads(path.read_text())


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    paths = {arm:ROOT/'runs'/f'ATT-{25+i:04d}-1' for i,arm in enumerate(ARM_IDS)}
    passed = False
    metrics = {'train.input_tokens':0}
    start = time.monotonic()
    try:
        report = analyze(paths, ROOT/'runs'/'ATT-0024-1')
        (out/'analysis.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        curves, costs = {}, {}
        lines = ['# E-0003 deterministic report', '',
                 'Reviewed analysis defaults: 10000 paired task-stratified replicates, seed20261029, percentile95%. Single-seed screening.', '',
                 '| Arm | PIQA acc % | BoolQ acc % | Macro acc % | PIQA NLL | BoolQ NLL | Macro NLL |',
                 '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
        for arm, data in report['arms'].items():
            m = data['metrics']
            lines.append(f"| {arm} {data['name']} | {100*m['piqa_accuracy']:.2f} | {100*m['boolq_accuracy']:.2f} | {100*m['macro_accuracy']:.2f} | {m['piqa_nll']:.6f} | {m['boolq_nll']:.6f} | {m['macro_nll']:.6f} |")
        lines += ['', 'Selection (verbatim reviewed analysis output):', '', '```json',
                  json.dumps(report['selection'],indent=2), '```', '',
                  '| Paired comparison (left minus right) | Macro accuracy delta pp [95% CI] | Macro NLL delta [95% CI] |',
                  '| --- | ---: | ---: |']
        for pair, values in report['comparisons'].items():
            a,n = values['macro_accuracy'],values['macro_nll']
            lines.append(f"| {pair} | {100*a['difference']:.3f} [{100*a['ci95'][0]:.3f}, {100*a['ci95'][1]:.3f}] | {n['difference']:.6f} [{n['ci95'][0]:.6f}, {n['ci95'][1]:.6f}] |")
        lines += ['', '| Arm | Step | PIQA acc % | BoolQ acc % | Macro acc % | Macro NLL |',
                  '| --- | ---: | ---: | ---: | ---: | ---: |']
        for arm,path in paths.items():
            final = read(path/'predictions.json')
            ids = [(r['task'],r['id'],r['gold']) for r in final]
            curves[arm] = []
            excluded = read(path/'excluded-train.json')
            assert [(r['task'],r['id'],r['candidate_lengths']) for r in excluded] == [('piqa',5528,[537,537])]
            for step in (3270,6539,9808,13077):
                if step == 13077:
                    m = report['arms'][arm]['metrics']
                else:
                    point = read(path/f'dev-step-{step}.json')
                    assert point['step'] == step
                    rows = point['predictions']
                    assert all(r['split']=='dev' for r in rows)
                    assert [(r['task'],r['id'],r['gold']) for r in rows] == ids
                    assert all(math.isfinite(v) for r in rows for v in r['scores'])
                    m = _point_metrics(_arrays(rows))
                    for task in ('piqa','boolq'):
                        for key in ('accuracy','nll'):
                            assert math.isclose(m[f'{task}_{key}'],point['metrics'][f'dev.{task}.{key}'],rel_tol=1e-12,abs_tol=1e-12)
                assert all(math.isfinite(v) for v in m.values())
                curves[arm].append({'step':step,**m})
                lines.append(f"| {arm} | {step} | {100*m['piqa_accuracy']:.2f} | {100*m['boolq_accuracy']:.2f} | {100*m['macro_accuracy']:.2f} | {m['macro_nll']:.6f} |")
            obs = read(ROOT/'research'/'artifacts'/path.name/'ledger-ended.json')
            costs[arm] = {'attempt':path.name,'mig_instance_hours':obs['usage']['quantity'],
                          'node_hours':obs['usage']['quantity']*.25,'source':obs['usage']['source_ref']}
        lines += ['', '| Arm | Total stored | Adapter stored/trainable/with grad | Steady step s | Input tokens/s | MIG-h | Node-h |',
                  '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
        for arm in ARM_IDS:
            m = report['arms'][arm]['manifest_metrics']
            c = costs[arm]
            lines.append(f"| {arm} | {m['params.stored']} | {m['params.adapter_stored']}/{m['params.trainable']}/{m['params.with_grad']} | {m['train.steady_step_seconds']:.6f} | {m['train.tokens_per_second']:.3f} | {c['mig_instance_hours']:.6f} | {c['node_hours']:.8f} |")
        metrics['formal.mig_instance_hours'] = sum(c['mig_instance_hours'] for c in costs.values())
        metrics['formal.node_hours'] = metrics['formal.mig_instance_hours']*.25
        metrics['formal.input_tokens'] = sum(report['arms'][arm]['manifest_metrics']['train.input_tokens'] for arm in ARM_IDS)
        metrics['formal.loss_tokens'] = sum(report['arms'][arm]['manifest_metrics']['train.loss_tokens'] for arm in ARM_IDS)
        metrics['formal.examples'] = sum(report['arms'][arm]['manifest_metrics']['train.examples'] for arm in ARM_IDS)
        metrics['selection.F_star'] = report['selection']['F_star']['selected']
        metrics['selection.C_star'] = report['selection']['C_star']
        metrics['selection.concat_wins'] = report['selection']['concat_wins']
        metrics['selection.C_dual_better_than_C_gate'] = report['selection']['C_dual_better_than_C_gate']
        (out/'curves.json').write_text(json.dumps(curves,indent=2,allow_nan=False)+'\n')
        (out/'costs.json').write_text(json.dumps(costs,indent=2)+'\n')
        (out/'report.md').write_text('\n'.join(lines)+'\n')
        passed = True
    finally:
        metrics['elapsed_s'] = time.monotonic()-start
        (out/'manifest.json').write_text(json.dumps({'status':'complete' if passed else 'partial',
            'spec_digest':os.environ['SR_SPEC_DIGEST'],'metrics':metrics,
            'artifacts':[{'path':str(out/name),'role':name} for name in ('analysis.json','curves.json','costs.json','report.md') if (out/name).exists()],
            'validator':{'name':'reviewed_e3_analysis_and_curve_audit','passed':passed}},indent=2)+'\n')


if __name__ == '__main__':
    main()
