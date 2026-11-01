"""Artifact-bound E-0021 RB1 screening analysis. Run only in PBS.

Input JSON: training_runs={RB1-01: /absolute/run, ... RB1-09: /absolute/run},
base_run=/absolute/ATT-0024-1. Final mode additionally needs intervention_runs
for RB1-01/02/08/09 and selected S_F*; latency_run; observations=[JSON exports].
Selection mode has no scientific status action and exists to dispatch the
pre-registered frozen intervention grid after all nine training runs finish.
"""
import argparse
import json
import math
import os
from pathlib import Path

from rb1_artifacts import (finite_tree, read, reference, validate_formal_run,
                           verified_latency, ACCEPTED_LR_LOCK_SHA256,
                           ACCEPTED_TRAINING_SOURCE_COMMIT)
from rb1_rules import (ARM_IDS, TASKS, analyze_counts, counts_from_rows,
                       paired_interval)
from v1_analysis import _read_run
from v5_train import write_json


MODES = ('iv_off', 'iv_window1', 'iv_window8', 'iv_strict', 'iv_swap', 'iv_permute')
ACCEPTED_BASE_MANIFEST_SHA256 = '076a1a0595de6620508a7e99810a96baf979483d805428465807d40beadde196'


def expected_release(spec):
    expected = spec.get('expected')
    pins = {'source_commit': ACCEPTED_TRAINING_SOURCE_COMMIT,
            'lr_lock_sha256': ACCEPTED_LR_LOCK_SHA256,
            'base_manifest_sha256': ACCEPTED_BASE_MANIFEST_SHA256}
    if expected != pins:
        raise ValueError('expected source commit/LR lock/Base manifest must match the pre-result RB1 release pins')
    return expected


def validate_rows(rows, label, paired=None):
    if not isinstance(rows, list):
        raise ValueError(f'{label}: predictions must be a list')
    rows = [row for row in rows if row.get('split') == 'dev']
    identities = []
    for row in rows:
        if row.get('task') not in TASKS or type(row.get('id')) is not int:
            raise ValueError(f'{label}: unexpected task/id')
        for key in ('gold', 'prediction'):
            if type(row.get(key)) is not int or row[key] not in (0, 1):
                raise ValueError(f'{label}: {key} must be binary')
        scores = row.get('scores')
        if not isinstance(scores, list) or len(scores) != 2 or any(
                type(x) not in (float, int) or not math.isfinite(x) for x in scores):
            raise ValueError(f'{label}: two finite candidate scores required')
        tokens, total = row.get('gold_loss_tokens'), row.get('gold_nll_sum')
        if type(tokens) is not int or tokens <= 0 or type(total) not in (int, float) or not math.isfinite(total) or total < 0:
            raise ValueError(f'{label}: invalid gold answer NLL/token count')
        if row['prediction'] != (0 if scores[0] >= scores[1] else 1):
            raise ValueError(f'{label}: prediction does not follow the V1 tie rule')
        gold_score = -total/(tokens if row['task'] == 'piqa' else 1)
        if not math.isclose(scores[row['gold']], gold_score, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f'{label}: gold score does not match V1 likelihood')
        identities.append((row['task'], row['id'], row['gold'], tokens))
    if len({(t, i) for t, i, _, _ in identities}) != len(identities):
        raise ValueError(f'{label}: duplicate dev identity')
    if any(sum(row['task'] == task for row in rows) != 1000 for task in TASKS):
        raise ValueError(f'{label}: full 1000-row dev per task required')
    if paired is not None and identities != [(r['task'], r['id'], r['gold'], r['gold_loss_tokens']) for r in paired]:
        raise ValueError(f'{label}: paired dev membership/order/gold/token counts differ')
    return rows


def summarize_rows(rows):
    result = {}
    for task in TASKS:
        subset = [row for row in rows if row['task'] == task]
        nll = sum(row['gold_nll_sum'] for row in subset)/sum(row['gold_loss_tokens'] for row in subset)
        result[task] = {'accuracy': sum(row['prediction'] == row['gold'] for row in subset)/len(subset),
                        'nll': nll, 'examples': len(subset),
                        'loss_tokens': sum(row['gold_loss_tokens'] for row in subset)}
    result['macro_accuracy'] = sum(result[t]['accuracy'] for t in TASKS)/2
    result['macro_nll'] = sum(result[t]['nll'] for t in TASKS)/2
    return result


def check_metrics(rows, metrics, label):
    summaries = summarize_rows(rows)
    for task in TASKS:
        for field in ('accuracy', 'nll', 'examples', 'loss_tokens'):
            observed = metrics.get(f'dev.{task}.{field}')
            if observed is None or not math.isclose(observed, summaries[task][field], rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f'{label}: recorded dev {task}/{field} differs from predictions')
    return summaries


def validate_read_summary(records, arm, mode):
    if arm == 'RB1-01' or mode == 'iv_off':
        if records:
            raise ValueError(f'{arm}/{mode}: disabled/T0 read has no KernelMatrix distribution')
        return
    steps = (1, 2) if arm == 'RB1-03' else (1, 4) if arm == 'RB1-04' else (1,)
    expected = {(task, layer, step, scope) for task in TASKS for layer in (6, 13, 20, 27)
                for step in steps for scope in ('all_valid_input', 'answer_prediction')}
    found = set()
    for record in records:
        key = (record['task'], record['layer'], record['step'], record['scope'])
        if key in found or key not in expected:
            raise ValueError(f'{arm}/{mode}: duplicate/unexpected read diagnostic stratum')
        found.add(key)
        if type(record.get('count')) is not int or record['count'] <= 0 or not 0 <= record.get('empty_reads', -1) <= record['count']:
            raise ValueError(f'{arm}/{mode}: invalid read diagnostic denominator')
        means = record.get('means', {})
        if set(means) != {'current_weight', 'n_eff', 'entropy', 'relative_n_eff', 'readable_positions'}:
            raise ValueError(f'{arm}/{mode}: missing normalized read metrics')
        finite_tree(means)
        if any(value < 0 for value in means.values()) or any(means[k] > 1.00001 for k in ('current_weight', 'relative_n_eff')):
            raise ValueError(f'{arm}/{mode}: invalid normalized read metric range')
        if mode == 'iv_strict' and means['current_weight'] != 0:
            raise ValueError(f'{arm}/{mode}: strict read cannot include current-token weight')
        conditional = record.get('nonempty_means', {})
        if set(conditional) != set(means):
            raise ValueError(f'{arm}/{mode}: missing nonempty read means')
        count = record['count']-record['empty_reads']
        if count == 0:
            if any(value is not None for value in conditional.values()):
                raise ValueError(f'{arm}/{mode}: all-empty read has undefined nonempty means')
        else:
            finite_tree(conditional)
            if any(not math.isclose(conditional[name], value*record['count']/count, rel_tol=1e-12, abs_tol=1e-12)
                   for name, value in means.items()):
                raise ValueError(f'{arm}/{mode}: nonempty read denominator is inconsistent')
    if found != expected:
        raise ValueError(f'{arm}/{mode}: incomplete task/layer/first-last read diagnostic grid')


def measured_cost(paths):
    refs, entries = [], {}
    for path in paths:
        refs.append(reference(path))
        for item in read(path):
            usage = item.get('usage')
            if not usage:
                continue
            if item.get('execution_state') != 'ended' or usage.get('basis') != 'measured':
                raise ValueError('cost requires terminal measured scheduler usage')
            key = usage['accounting_ref']
            record = {'attempt': item['attempt_id'], 'accounting_ref': key,
                      'quantity': usage['quantity'], 'unit': usage['unit'],
                      'source_ref': usage['source_ref']}
            if record['quantity'] < 0 or not math.isfinite(record['quantity']):
                raise ValueError('invalid measured usage quantity')
            if record['unit'] != 'mig_instance_hours':
                raise ValueError('RB1 expects measured MIG instance hours')
            record['node_h'] = record['quantity']*.25
            if key in entries and entries[key] != record:
                raise ValueError('conflicting duplicate cost accounting reference')
            entries[key] = record
    return {'sources': refs, 'rows': list(entries.values()),
            'node_h': sum(r['node_h'] for r in entries.values()),
            'conversion': 'node_h = measured MIG instance hours * 0.25; includes supplied attempts only'}


def normal_training_agreement(normal, trained, training_hardware='unknown', intervention_hardware='unknown'):
    if len(normal) != len(trained) or [(r['task'], r['id']) for r in normal] != [(r['task'], r['id']) for r in trained]:
        raise ValueError('agreement report requires paired normal/training rows')
    return {'prediction_mismatches': sum(a['prediction'] != b['prediction'] for a, b in zip(normal, trained)),
            'max_abs_score_difference': max(abs(a-b) for left, right in zip(normal, trained)
                                             for a, b in zip(left['scores'], right['scores'])),
            'macro_accuracy_difference': summarize_rows(normal)['macro_accuracy']-summarize_rows(trained)['macro_accuracy'],
            'training_hardware': training_hardware, 'intervention_hardware': intervention_hardware,
            'scope': 'descriptive agreement of repeated evaluation; hardware is metadata'}


def analyze(spec, stage='final'):
    expected = expected_release(spec)
    if set(spec.get('training_runs', {})) != set(ARM_IDS):
        raise ValueError('all nine registered training arms are required')
    base_root = Path(spec['base_run'])
    if reference(base_root/'manifest.json')['sha256'] != expected['base_manifest_sha256']:
        raise ValueError('Base manifest differs from accepted ATT-0024-1')
    base_rows, base_manifest = _read_run(base_root, 'Base', 'base')
    rows = {'Base': validate_rows(base_rows, 'Base')}
    sources = {'Base': {name: reference(base_root/name) for name in
                       ('manifest.json', 'config.json', 'predictions.json', 'split-map.json', 'data-metadata.json')}}
    training, performance, curves, runtime = {}, {}, {}, {}
    common = None
    for arm in ARM_IDS:
        root = Path(spec['training_runs'][arm])
        manifest, refs, _ = validate_formal_run(root, arm)
        metrics = manifest['metrics']
        if metrics['source.commit'] != expected['source_commit'] or metrics['lr_lock.sha256'] != expected['lr_lock_sha256']:
            raise ValueError(f'{arm}: source commit or LR lock differs from expected release')
        plan = {k: metrics.get(k) for k in ('plan.steps', 'plan.examples', 'plan.input_tokens',
                'plan.order_sha256', 'split_manifest.sha256', 'lr_lock.sha256', 'source.commit',
                'environment.lock_sha256', 'model_revision', 'tokenizer_revision')}
        if not isinstance(plan['plan.order_sha256'], str) or len(plan['plan.order_sha256']) != 64:
            raise ValueError(f'{arm}: missing recorded paired training order digest')
        if common is not None and common != plan:
            raise ValueError(f'{arm}: training sample budget/order/split/LR-lock differs')
        common = plan
        training[arm] = manifest
        rows[arm] = validate_rows(read(root/'predictions.json'), arm, rows['Base'])
        refs['predictions'] = reference(root/'predictions.json')
        performance[arm] = check_metrics(rows[arm], metrics, arm)
        curves[arm] = []
        for fraction in (.25, .5, .75):
            step = math.ceil(metrics['plan.steps']*fraction)
            path = root/f'dev-step-{step}.json'
            value = read(path)
            if value.get('step') != step:
                raise ValueError(f'{arm}: learning-curve step mismatch')
            curve_rows = validate_rows(value['predictions'], f'{arm}@{step}', rows['Base'])
            curves[arm].append({'fraction': fraction, 'step': step,
                                'metrics': check_metrics(curve_rows, value['metrics'], f'{arm}@{step}'),
                                'source': reference(path)})
        sources[arm] = refs
        accounting = metrics.get('runtime.accounting')
        if not isinstance(accounting, dict):
            raise ValueError(f'{arm}: missing original/side operation accounting')
        if arm != 'RB1-07':
            if set(accounting.get('layers', {})) != {'6', '13', '20', '27'}:
                raise ValueError(f'{arm}: accounting must cover all four adapted layers')
            depth = 2 if arm == 'RB1-03' else 4 if arm in ('RB1-04', 'RB1-05', 'RB1-06') else 1
            expected = 0 if arm == 'RB1-01' else metrics['train.input_tokens']
            for counts in accounting['layers'].values():
                if (counts['writes'] != expected or counts['reads'] != expected*depth or
                    counts['valid_tokens'] != metrics['train.input_tokens'] or
                    counts['attention'] != metrics['train.examples'] or counts['ffn'] != metrics['train.examples']):
                    raise ValueError(f'{arm}: invalid per-layer write/read/original call counts')
        runtime[arm] = {k: v for k, v in metrics.items() if k.startswith(('train.', 'plan.', 'params.', 'runtime.', 'intermediate.'))
                        or k in ('learning_rate', 'model_revision', 'tokenizer_revision', 'source.commit', 'environment.lock_sha256')}
        runtime[arm].update(warmup_steps=math.ceil(metrics['plan.steps']*.05), clip=1.0,
                            optimizer='AdamW betas(0.9,0.999) eps1e-8 weight_decay0.01;cosine after5%warmup',
                            training_objective='per-example mean answer CE;effective batch2,microbatch1,dropout0')
        if arm != 'RB1-07':
            depth = 2 if arm == 'RB1-03' else 4 if arm in ('RB1-04', 'RB1-05', 'RB1-06') else 1
            fusion = 196608 if arm == 'RB1-08' else 98304 if arm == 'RB1-09' else 32768
            runtime[arm]['analytic_side_dense_MAC_per_token_per_layer'] = {
                'projection': 131072, 'fusion': fusion,
                'loop_query_and_output': depth*2*32**2,
                'kernel_read_and_denominator': 0 if arm == 'RB1-01' else depth*(32**2+32),
                'kernel_write_outer_product': 0 if arm == 'RB1-01' else 32**2,
                'excludes': 'normalization/ELU/SiLU/additions;backbone;prefix-scan additions;diagnostic tracing'}
    performance['Base'] = check_metrics(rows['Base'], base_manifest['metrics'], 'Base')
    counts = {label: counts_from_rows(value) for label, value in rows.items()}
    preliminary = analyze_counts(counts)
    s_star = preliminary['selection']['S_F_star']
    intervention_tables, read_tables, agreement = {}, {}, {}
    if stage == 'final':
        arms = {'RB1-01', 'RB1-02', 'RB1-08', 'RB1-09', s_star}
        if set(spec.get('intervention_runs', {})) != arms:
            raise ValueError('final analysis requires exactly T0 plus the registered four intervention targets')
        for arm in sorted(arms):
            root = Path(spec['intervention_runs'][arm])
            manifest, provenance, conditions = (read(root/name) for name in
                                               ('manifest.json', 'provenance.json', 'conditions.json'))
            if (manifest.get('status') != 'complete' or manifest.get('validator', {}).get('passed') is not True or
                manifest.get('contract') != 'E-0021@1' or manifest.get('arm') != arm or
                manifest.get('stage') != 'intervention' or manifest.get('seed') != 0):
                raise ValueError(f'{arm}: incomplete or wrong intervention run')
            if provenance.get('training_manifest') != {k: sources[arm]['manifest'][k] for k in ('path', 'sha256')}:
                raise ValueError(f'{arm}: intervention checkpoint is not the supplied training run')
            modes = ('iv_swap',) if arm == 'RB1-01' else MODES
            if set(conditions) != {'normal', 'Base_same_allocation', *modes}:
                raise ValueError(f'{arm}: incomplete intervention condition grid')
            agreement[arm] = normal_training_agreement(conditions['normal']['predictions'], rows[arm],
                training[arm]['metrics'].get('hardware', 'unknown'), provenance.get('hardware', 'unknown'))
            for mode, condition in conditions.items():
                label = f'{arm}/{mode}'
                condition_rows = validate_rows(condition['predictions'], label, rows['Base'])
                performance[label] = check_metrics(condition_rows, condition['metrics'], label)
                if mode != 'Base_same_allocation':
                    validate_read_summary(condition.get('read_diagnostics', []), arm, mode)
                    rows[label] = condition_rows
                    counts[label] = counts_from_rows(condition_rows)
                    read_tables[label] = condition.get('read_diagnostics', [])
                if mode not in ('iv_off', 'Base_same_allocation') and arm != 'RB1-01' and not condition.get('read_diagnostics'):
                    raise ValueError(f'{label}: missing first/last read diagnostics')
            if arm == 'RB1-01':
                if conditions['iv_swap']['predictions'] != conditions['normal']['predictions']:
                    raise ValueError('T0 swap no-op check failed')
            elif conditions['iv_off']['predictions'] != conditions['Base_same_allocation']['predictions']:
                raise ValueError(f'{arm}: iv_off independent same-allocation Base check failed')
            sources[f'{arm}/interventions'] = {name: reference(root/name) for name in
                                              ('manifest.json', 'conditions.json', 'provenance.json')}
    nll = {label: {task: (sum(r['gold_nll_sum'] for r in value if r['task'] == task),
                         sum(r['gold_loss_tokens'] for r in value if r['task'] == task)) for task in TASKS}
           for label, value in rows.items()}
    arrays = {label: {task: [int(r['prediction'] == r['gold']) for r in value if r['task'] == task]
                     for task in TASKS} for label, value in rows.items()}
    report = analyze_counts(counts, nll, bootstrap_rows=arrays if stage == 'final' else None)
    if stage == 'final':
        comparisons = report['bootstrap']['comparisons']
        for arm in sorted(spec['intervention_runs']):
            modes = ('iv_swap',) if arm == 'RB1-01' else MODES
            intervention_tables[arm] = {mode: {metric: paired_interval(comparisons, f'{arm}/{mode}', f'{arm}/normal', metric)
                                               for metric in ('macro_accuracy', 'piqa_accuracy', 'boolq_accuracy')}
                                        for mode in modes}
        root = Path(spec['latency_run'])
        manifest, latency, guard = (read(root/name) for name in ('manifest.json', 'latency.json', 'mig-profile.json'))
        if manifest.get('status') != 'complete' or manifest.get('validator', {}).get('passed') is not True or manifest.get('contract') != 'E-0021@1':
            raise ValueError('incomplete RB1 latency run')
        if guard.get('passed') is not True or guard.get('profile') != '2g.24gb':
            raise ValueError('comparative latency was not measured on required 2g profile')
        expected = {'RB1-01', 'RB1-02', 'RB1-04', 'RB1-08', 'RB1-07', 'RB1-07-merged'}
        if set(latency['measurements']) not in (expected, expected | {'Base'}):
            raise ValueError('latency grid differs from E-0021')
        for label, value in latency['measurements'].items():
            verified_latency(value)
            expected_state = 16896 if label in ('RB1-02', 'RB1-04', 'RB1-08') else 0
            if value['adapter_persistent_state']['M_c_bytes'] != expected_state:
                raise ValueError(f'{label}: persistent M+c bytes differ from registered layout')
            if label in ARM_IDS and latency['loaded_refs'][label]['artifact_inventory']['manifest'] != sources[label]['manifest']:
                raise ValueError(f'{label}: latency checkpoint differs from training input')
        report['latency'] = latency
        sources['latency'] = {name: reference(root/name) for name in ('manifest.json', 'latency.json', 'mig-profile.json')}
        report['cost'] = measured_cost(spec['observations'])
        required_cost = {Path(path).name for path in spec['training_runs'].values()} | {
            Path(path).name for path in spec['intervention_runs'].values()} | {Path(spec['latency_run']).name}
        if not required_cost <= {row['attempt'] for row in report['cost']['rows']}:
            raise ValueError('measured usage is missing for a required training/intervention/latency attempt')
    report.update(contract='E-0021@1', stage=stage, expected=expected, sources=sources, common_training=common,
                  performance=performance, learning_curves=curves, training_runtime=runtime,
                  interventions=intervention_tables, read_diagnostics=read_tables,
                  interventions_normal_vs_training=agreement,
                  decision_authority='Screening facts and pre-registered gates only; coordinator conclusion review precedes any hypothesis status or release.')
    finite_tree(report)
    return report


def markdown(report):
    lines = ['# E-0021 RB1 screening analysis', '', report['decision_authority'], '',
             '| Arm | PIQA accuracy % | BoolQ accuracy % | Macro accuracy % | PIQA NLL | BoolQ NLL |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for label in ('Base', *ARM_IDS):
        row = report['performance'][label]
        lines.append(f'| {label} | {row["piqa"]["accuracy"]*100:.3f} | {row["boolq"]["accuracy"]*100:.3f} | {row["macro_accuracy"]*100:.3f} | {row["piqa"]["nll"]:.6f} | {row["boolq"]["nll"]:.6f} |')
    lines.extend(['', f'S_F*: {report["selection"]["S_F_star"]}; C*_RB: {report["selection"]["C_star_RB"]}.',
                  f'Progression rule: {report["progression"]}.', '', '## Rules 1-11', ''])
    for name, rule in report['rules'].items():
        lines.append(f'- {name}: `{json.dumps(rule, ensure_ascii=False, sort_keys=True)}`')
    if report['stage'] == 'final':
        lines.extend(['', '## Repeated normal evaluation agreement', '',
                      '| Arm | Prediction mismatches | Max absolute candidate-score difference | Normal minus training macro pp | Training hardware | Intervention hardware |',
                      '| --- | ---: | ---: | ---: | --- | --- |'])
        for arm, value in report['interventions_normal_vs_training'].items():
            lines.append(f'| {arm} | {value["prediction_mismatches"]} | {value["max_abs_score_difference"]:.8g} | {value["macro_accuracy_difference"]*100:.3f} | {value["training_hardware"]} | {value["intervention_hardware"]} |')
        lines.extend(['', 'Agreement is descriptive; hardware profiles are metadata under D-0018/D-0019.', ''])
        lines.extend(['', '## Frozen side-history interventions', '',
                      '| Arm | Intervention | PIQA difference pp | BoolQ difference pp | Macro difference pp [95% descriptive interval] |',
                      '| --- | --- | ---: | ---: | ---: |'])
        for arm, modes in report['interventions'].items():
            for mode, results in modes.items():
                macro = results['macro_accuracy']; lo, hi = macro['ci95']
                lines.append(f'| {arm} | {mode} | {results["piqa_accuracy"]["difference"]*100:.3f} | {results["boolq_accuracy"]["difference"]*100:.3f} | {macro["difference"]*100:.3f} [{lo*100:.3f}, {hi*100:.3f}] |')
        lines.extend(['', '## Read distributions at answer-predicting positions', '',
                      'n_eff, entropy and relative n_eff below condition on nonempty reads. Current-token weight retains zero for an empty read. All-valid-input summaries and both denominator conventions are retained in JSON.', '',
                      '| Condition | Task | Layer | Read step | Empty / queries | Current weight | n_eff (nonempty) | Entropy (nonempty) | Relative n_eff (nonempty) |',
                      '| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |'])
        for label, records in report['read_diagnostics'].items():
            for record in records:
                if record['scope'] == 'answer_prediction':
                    conditional = record['nonempty_means']
                    lines.append(f'| {label} | {record["task"]} | {record["layer"]} | {record["step"]} | {record["empty_reads"]} / {record["count"]} | {record["means"]["current_weight"]} | {conditional["n_eff"]} | {conditional["entropy"]} | {conditional["relative_n_eff"]} |')
        lines.extend(['', 'Differences use each intervention run\'s own normal condition. iv_off and Base are independently checked in the same allocation; T0 swap is a no-op.', '',
                      '## Training and persistent side state', '',
                      '| Arm | Stored / trainable / with gradient | Steps | Input / loss / padded tokens | Train seconds | Peak CUDA GiB | M+c bytes |',
                      '| --- | --- | ---: | --- | ---: | ---: | ---: |'])
        for arm, value in report['training_runtime'].items():
            lines.append(f'| {arm} | {value["params.stored"]} / {value["params.trainable"]} / {value["params.with_grad"]} | {value["train.steps"]} | {value["train.input_tokens"]} / {value.get("train.loss_tokens")} / {value.get("train.padded_tokens")} | {value.get("train.seconds", 0):.3f} | {value.get("train.peak_cuda_bytes", 0)/2**30:.3f} | {value["runtime.accounting"]["state_M_c_bytes"]} |')
        lines.extend(['', '## Comparative latency', '',
                      '| Arm | Prefill512 tokens/s | Decode128 p50 ms | p95 ms | Peak CUDA GiB | M+c bytes |',
                      '| --- | ---: | ---: | ---: | ---: | ---: |'])
        for arm, value in report['latency']['measurements'].items():
            lines.append(f'| {arm} | {value["prefill_tokens_per_second"]:.3f} | {value["decode_p50_ms"]:.3f} | {value["decode_p95_ms"]:.3f} | {value["peak_cuda_bytes"]/2**30:.3f} | {value["adapter_persistent_state"]["M_c_bytes"]} |')
        lines.extend(['', '## Measured cost', '', '| Attempt | MIG instance hours | node hours |', '| --- | ---: | ---: |'])
        for row in report['cost']['rows']:
            lines.append(f'| {row["attempt"]} | {row["quantity"]:.6f} | {row["node_h"]:.6f} |')
        lines.extend(['', f'Supplied measured usage: {report["cost"]["node_h"]:.6f} node hours. The current analysis job is accounted after it ends.', ''])
    lines.extend(['', 'Single-seed dev screening; paired task-stratified 95% intervals are descriptive. Selection uses these same dev outcomes; intervals do not account for selection or trained-seed uncertainty. Point-estimate gates are inclusive; C* near-best rule is strictly <1 pp. T0 is S1 only; no same-depth T0 or parameter-efficiency claim.', '',
                  'Intervention differences, read diagnostics, learning curves, parameter/state/operation counts, raw latency and measured cost provenance are retained in analysis.json.', ''])
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--stage', choices=('select', 'final'), default='final')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    passed, metrics, artifacts = False, {}, []
    try:
        report = analyze(read(args.input), args.stage)
        report['input'] = reference(args.input)
        write_json(out/'analysis.json', report)
        (out/'analysis.md').write_text(markdown(report))
        metrics = {'stage': args.stage, 'selection': report['selection'], 'screening_directions': report['screening_directions'],
                   'bootstrap_repeats': 10000 if args.stage == 'final' else 0}
        artifacts = [{'path': str(out/name), 'role': name} for name in ('analysis.json', 'analysis.md')]
        passed = True
    finally:
        write_json(out/'manifest.json', {'status': 'complete' if passed else 'partial', 'contract': 'E-0021@1',
            'spec_digest': os.environ['SR_SPEC_DIGEST'], 'metrics': metrics, 'artifacts': artifacts,
            'validator': {'name': 'rb1_screening_analysis', 'passed': passed}})


if __name__ == '__main__':
    main()
