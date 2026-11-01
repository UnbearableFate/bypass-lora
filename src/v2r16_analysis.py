"""Read-only E-0016 seed-0 screening analysis for D1, D2a, D2b and D3."""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
from v2_factorial_analysis import paired_bootstrap, BOOTSTRAP_REPEATS, BOOTSTRAP_SEED

ARMS = ('D1', 'D2a', 'D2b', 'D3')
STANDARD = ('normal', 'zero', 'shuffled', 'permuted', 'oracle_key', 'oracle_value')
TEACHER = ('tf_normal', 'tf_shuffled', 'tf_wrong', 'zero')
E1 = ('normal', 'zero', 'shuffled')
BOOTSTRAP_NOTE = ('V2 paired episode bootstrap: 10000 draws, RNG 20261029, common episode indices '
                  'across controls within each arm/split. Intervals are descriptive only and never gate.')


class AnalysisError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise AnalysisError(message)


def _finite_number(value, name):
    require(isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value), f'{name} must be a finite number')
    return float(value)


def _integer(value, name):
    require(type(value) is int and value >= 0, f'{name} must be a nonnegative integer')
    return value


def _read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise AnalysisError(f'cannot read JSON {path}: {exc}') from exc


def _read_jsonl(path):
    rows = []
    try:
        with Path(path).open() as handle:
            for line_number, line in enumerate(handle, 1):
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise AnalysisError(f'{path}:{line_number}: invalid JSON: {exc}') from exc
    except OSError as exc:
        raise AnalysisError(f'cannot read JSONL {path}: {exc}') from exc
    return rows


def validate_manifest(manifest, arm):
    """Validate the sealed formal-run contract before consuming run artifacts."""
    require(manifest.get('status') == 'complete', f'{arm}: run is not complete')
    require(manifest.get('validator', {}).get('passed') is True,
            f'{arm}: runner validator did not pass')
    require(manifest.get('experiment') == 'E-0016', f'{arm}: wrong experiment')
    require(manifest.get('contract') == 'E-0016@1', f'{arm}: missing or wrong E-0016@1 contract')
    require(manifest.get('arm') == arm, f'{arm}: manifest arm mismatch')
    require(manifest.get('eligible_as_evidence') is True, f'{arm}: run is not evidence eligible')
    require(not manifest.get('smoke_steps') and not manifest.get('smoke_eval_limit'),
            f'{arm}: smoke output is ineligible for screening')


def _check_point_summary(summary, records, label):
    require(isinstance(summary, dict), f'{label}: missing summary')
    n = len(records)
    require(n > 0, f'{label}: no prediction records')
    require(_integer(summary.get('count'), f'{label}.count') == n,
            f'{label}: summary count differs from predictions')
    correct_count = sum(row['correct'] for row in records)
    require(_integer(summary.get('correct_count'), f'{label}.correct_count') == correct_count,
            f'{label}: summary correct_count differs from predictions')
    em = correct_count / n
    nll = sum(_finite_number(row['nll'], f'{label}.prediction.nll') for row in records) / n
    require(math.isclose(_finite_number(summary.get('em'), f'{label}.em'), em,
                         rel_tol=0.0, abs_tol=1e-12), f'{label}: summary EM differs from predictions')
    require(math.isclose(_finite_number(summary.get('nll'), f'{label}.nll'), nll,
                         rel_tol=0.0, abs_tol=1e-10), f'{label}: summary NLL differs from predictions')
    return {'count': n, 'correct_count': correct_count, 'em': em, 'nll': nll}


def _check_strata(summary, records, label, *, require_later_stratum=True):
    require(all(type(r.get('target_fact_position')) is int for r in records),
            f'{label}: missing or invalid target fact positions')
    expected = {
        'position0': [r for r in records if r.get('target_fact_position') == 0],
        'position1plus': [r for r in records if type(r.get('target_fact_position')) is int
                          and r['target_fact_position'] > 0],
    }
    require(sum(map(len, expected.values())) == len(records),
            f'{label}: predictions have invalid target fact positions')
    strata = summary.get('position_strata')
    require(isinstance(strata, dict), f'{label}: missing position_strata')
    result = {}
    for name, rows in expected.items():
        if not rows:
            require(not require_later_stratum and name == 'position1plus',
                    f'{label}: predictions have no {name} rows')
            require(name not in strata, f'{label}: empty {name} unexpectedly has a summary')
            result[name] = None
            continue
        require(name in strata, f'{label}: summary lacks {name}')
        result[name] = _check_point_summary(strata[name], rows, f'{label}.{name}')
    return result


def load_protocol(run_dir, metrics, arm, protocol, controls, split='dev'):
    """Load one runner protocol, verify aligned predictions, and recompute point metrics."""
    summaries = metrics.get('evaluation', {}).get(protocol)
    require(isinstance(summaries, dict), f'{arm}: missing evaluation protocol {protocol}')
    base = Path(run_dir) / 'evaluation' / protocol
    raw, points, position_strata, prediction_rows = {}, {}, {}, {}
    ids, targets, target_positions = None, None, None
    for control in controls:
        require(control in summaries, f'{arm}/{protocol}: missing summary for {control}')
        rows = _read_jsonl(base / f'{split}.{control}.predictions.jsonl')
        require(rows and all(isinstance(row, dict) for row in rows),
                f'{arm}/{protocol}/{control}: empty or malformed predictions')
        require(len(rows) == 1000, f'{arm}/{protocol}/{control}: expected the full 1000-row dev split')
        seen = [row.get('episode_id') for row in rows]
        require(all(isinstance(value, str) and value for value in seen),
                f'{arm}/{protocol}/{control}: missing episode IDs')
        require(len(set(seen)) == len(seen), f'{arm}/{protocol}/{control}: duplicate episode IDs')
        for index, row in enumerate(rows):
            require(type(row.get('target')) is int and type(row.get('prediction')) is int,
                    f'{arm}/{protocol}/{control}: target/prediction must be integer token IDs')
            require(type(row.get('correct')) is bool and row['correct'] == (row['prediction'] == row['target']),
                    f'{arm}/{protocol}/{control}: inconsistent correctness')
            _finite_number(row.get('nll'), f'{arm}/{protocol}/{control}.nll[{index}]')
        if ids is None:
            ids = seen
            targets = [row['target'] for row in rows]
            target_positions = [row.get('target_fact_position') for row in rows]
        else:
            require(seen == ids, f'{arm}/{protocol}: controls are not aligned by episode ID/order')
            require([row['target'] for row in rows] == targets,
                    f'{arm}/{protocol}: controls have different answer labels')
            require([row.get('target_fact_position') for row in rows] == target_positions,
                    f'{arm}/{protocol}: controls have different target positions')
        points[control] = _check_point_summary(summaries[control], rows, f'{arm}/{protocol}/{control}')
        position_strata[control] = _check_strata(
            summaries[control], rows, f'{arm}/{protocol}/{control}',
            require_later_stratum=split != 'dev' or protocol != 'e1')
        raw[control] = summaries[control]
        prediction_rows[control] = rows
    return {'controls': points, 'summaries': raw, 'position_strata': position_strata,
            'episode_ids': ids, 'targets': targets, 'bootstrap': descriptive_bootstrap(prediction_rows)}


def descriptive_bootstrap(records):
    names = list(records)
    reference = 'tf_normal' if 'tf_normal' in records else 'normal'
    n = len(records[reference])
    require(all(len(records[name]) == n for name in names), 'bootstrap control counts differ')
    columns = [(name, metric) for name in names for metric in ('correct', 'nll')]
    values = np.asarray([[float(records[name][i][metric]) for name, metric in columns] for i in range(n)])
    draws = paired_bootstrap(values)
    index = {key: i for i, key in enumerate(columns)}
    return {f'{reference}-{name}': {
        metric: {'estimate': float((values[:, index[reference, source]]-values[:, index[name, source]]).mean()),
                 'ci95': np.quantile(draws[:, index[reference, source]]-draws[:, index[name, source]], [.025, .975]).tolist()}
        for metric, source in (('em', 'correct'), ('nll', 'nll'))}
        for name in names if name != reference}


def em_margin(normal, zero, shuffled):
    n = normal['count']
    require(n > 0 and zero['count'] == n and shuffled['count'] == n,
            'EM gate requires aligned equal nonzero counts')
    return Fraction(normal['correct_count'] - max(zero['correct_count'], shuffled['correct_count']), n)


def d1_gate(teacher, plateau):
    """Classify the D1 teacher-forced stage-one screening criterion."""
    margin = em_margin(teacher['tf_normal'], teacher['zero'], teacher['tf_shuffled'])
    passed = margin >= Fraction(1, 2)
    if passed:
        status = 'pass'
    elif margin >= Fraction(1, 5):
        status = 'partial'
    elif plateau:
        status = 'fail'
    else:
        status = 'inconclusive'
    return {'status': status, 'passed': passed,
            'margin': float(margin), 'margin_fraction': str(margin),
            'threshold_fraction': '1/2', 'fail_requires_plateau': margin < Fraction(1, 5)}


def h3_gate(controls, all_question_target_weight=None, last_position_target_weight=None,
            *, weight_scope='all_question_positions'):
    """Exact-fraction E-0010 H3 point gate, or D3's registered hard last-only variant."""
    require(weight_scope in ('all_question_positions', 'last_position'), 'unknown H3 address-weight scope')
    chosen_weight = (all_question_target_weight if weight_scope == 'all_question_positions'
                     else last_position_target_weight)
    chosen_weight = _finite_number(chosen_weight, f'{weight_scope} target weight')
    margin = em_margin(controls['normal'], controls['zero'], controls['shuffled'])
    nll_pass = controls['normal']['nll'] < min(controls['zero']['nll'], controls['shuffled']['nll'])
    em_pass = margin >= Fraction(1, 10)
    weight_pass = chosen_weight >= 0.125
    return {'passed': em_pass and nll_pass and weight_pass,
            'protocol': 'E-0010@2 standard H3' if weight_scope == 'all_question_positions'
                        else 'E-0016 D3 last-only hard H3 (distinct protocol)',
            'em_margin': float(margin), 'em_margin_fraction': str(margin),
            'em_threshold_fraction': '1/10', 'em_pass': em_pass,
            'nll_pass': nll_pass, 'target_weight': chosen_weight,
            'target_weight_scope': weight_scope, 'weight_threshold': 0.125,
            'weight_pass': weight_pass,
            'counts': {name: controls[name]['correct_count'] for name in ('normal', 'zero', 'shuffled')},
            'nll': {name: controls[name]['nll'] for name in ('normal', 'zero', 'shuffled')}}


def summarize_curve(rows):
    """Summarize per-step answer CE, 500-step means and final-window OLS trend."""
    require(len(rows) == 6000, 'formal training curve must contain all 6000 steps')
    for expected_step, row in enumerate(rows, 1):
        require(row.get('step') == expected_step, 'training curve steps must be ordered 1..6000')
        _finite_number(row.get('loss'), f'train.loss[{expected_step}]')
        _finite_number(row.get('lr'), f'train.lr[{expected_step}]')
    bins = []
    for start in range(0, 6000, 500):
        chunk = rows[start:start+500]
        bins.append({'steps': [start+1, start+len(chunk)],
                     'mean_answer_ce': sum(float(row['loss']) for row in chunk)/len(chunk)})
    window = rows[5000:6000]
    ys = [float(row['loss']) for row in window]
    xs = list(range(5001, 6001))
    xbar, ybar = sum(xs)/len(xs), sum(ys)/len(ys)
    slope = sum((x-xbar)*(y-ybar) for x, y in zip(xs, ys)) / sum((x-xbar)**2 for x in xs)
    fitted_decline = -999.0 * slope
    lrs = [float(row['lr']) for row in window]
    peak_lr = max(float(row['lr']) for row in rows)
    peak_fraction = max(lrs)/peak_lr if peak_lr else 0.0
    return {'per_500_step_mean_answer_ce': bins,
            'final_1000_step_mean_answer_ce': sum(ys)/len(ys),
            'final_window_ols_slope_per_step': slope,
            'fitted_decline_999_steps': fitted_decline,
            'plateau': fitted_decline < 0.05,
            'final_1000_lr_min': min(lrs), 'final_1000_lr_max': max(lrs),
            'peak_lr': peak_lr, 'final_1000_max_lr_fraction_of_peak': peak_fraction,
            'interpretation': ('The final 1000-step window runs at low learning rate; its fitted decline is a '
                               'plateau rule only and does not establish convergence.')}


def load_run(arm, run_dir):
    run_dir = Path(run_dir).resolve()
    manifest = _read_json(run_dir/'manifest.json')
    validate_manifest(manifest, arm)
    metrics_path = run_dir/'metrics.json'
    metrics = _read_json(metrics_path)
    require(metrics.get('arm') == arm, f'{arm}: metrics arm mismatch')
    require(metrics.get('eligible_as_evidence') is True and metrics.get('smoke_only') is False,
            f'{arm}: metrics mark run as smoke or ineligible')
    require(metrics.get('train', {}).get('steps') == 6000,
            f'{arm}: metrics must record exactly 6000 train steps')
    require(metrics.get('checkpoint_reload', {}).get('full_records_identical') is True
            and metrics['checkpoint_reload'].get('summaries_identical') is True,
            f'{arm}: metrics checkpoint reload mismatch')

    curve_rows = _read_jsonl(run_dir/'train.jsonl')
    curve = summarize_curve(curve_rows)
    protocols = {}
    if arm == 'D1':
        protocols['standard'] = load_protocol(run_dir, metrics, arm, 'standard', STANDARD)
        protocols['teacher'] = load_protocol(run_dir, metrics, arm, 'teacher', TEACHER)
        protocols['e1'] = load_protocol(run_dir, metrics, arm, 'e1', E1, split='dev')
        teacher_gate = d1_gate(protocols['teacher']['controls'], curve['plateau'])
        primary = {'teacher_forced': teacher_gate,
                   'teacher_controls': protocols['teacher']['controls']}
    elif arm in ('D2a', 'D2b'):
        protocols['standard'] = load_protocol(run_dir, metrics, arm, 'standard', STANDARD)
        protocols['e1'] = load_protocol(run_dir, metrics, arm, 'e1', E1, split='dev')
        normal = protocols['standard']['summaries']['normal']
        gate = h3_gate(protocols['standard']['controls'],
                       normal['diagnostics']['address.target_weight'],
                       normal['last_position']['target_weight_mean'])
        primary = {'h3': gate}
    else:
        protocols['hard'] = load_protocol(run_dir, metrics, arm, 'hard', STANDARD)
        protocols['soft'] = load_protocol(run_dir, metrics, arm, 'soft', STANDARD)
        protocols['marginal'] = load_protocol(run_dir, metrics, arm, 'marginal',
                                              ('normal', 'zero', 'shuffled', 'permuted'))
        normal = protocols['hard']['summaries']['normal']
        gate = h3_gate(protocols['hard']['controls'],
                       last_position_target_weight=normal['last_position']['target_weight_mean'],
                       weight_scope='last_position')
        primary = {'hard_last_only_h3': gate}
    input_files = [run_dir/'manifest.json', metrics_path, run_dir/'train.jsonl']
    for protocol, payload in protocols.items():
        file_protocol = protocol
        for control in payload['controls']:
            input_files.append(run_dir/'evaluation'/file_protocol/f'dev.{control}.predictions.jsonl')
    provenance = [{'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                   'bytes': path.stat().st_size} for path in input_files]
    return {'run_directory': str(run_dir), 'arm': arm, 'primary_screening': primary,
            'protocols': protocols, 'training_curve': curve,
            'training': {key: metrics['train'].get(key) for key in
                         ('steps', 'tokens', 'training_seconds', 'stable_step_seconds',
                          'parameter_names_with_nonzero_gradient')},
            'timing': {'wall_seconds': metrics.get('wall_seconds'),
                       'mig_instance_hours': metrics.get('mig_instance_hours'),
                       'node_hours': metrics.get('node_hours'),
                       'evaluation_seconds': metrics.get('evaluation_seconds'),
                       'checkpoint_save_seconds': metrics.get('checkpoint_save_seconds'),
                       'checkpoint_reload': metrics.get('checkpoint_reload')},
            'supervision_note': ('D1 stage one uses target_fact_index to teacher-force the target slot; '
                                 'its result is extra-supervision screening only.' if arm == 'D1' else
                                 'This arm starts from the supervised D1 adapter; its result is conditional '
                                 'on that teacher-forced value prewarm.' if arm in ('D2a', 'D2b') else
                                 'D3 uses the registered last-question-position protocol and no target fact index.'),
            'single_seed_note': 'Seed 0 only; screening signal, not replication or confirmation.',
            'input_provenance': provenance}


def analyze(arm_paths):
    require(bool(arm_paths), 'at least one --arm is required')
    require(len(arm_paths) == len(set(arm_paths)), 'duplicate arms are not allowed')
    require(set(arm_paths) <= set(ARMS), 'unsupported E-0016 arm')
    arms = {arm: load_run(arm, arm_paths[arm]) for arm in ARMS if arm in arm_paths}
    missing = [arm for arm in ARMS if arm not in arms]
    return {'experiment': 'E-0016', 'contract': 'E-0016@1',
            'scope': 'seed-0 per-arm screening only; no hypothesis conclusion',
            'completeness': {'required_arms': list(ARMS), 'included_arms': list(arms),
                             'missing_arms': missing, 'complete': not missing},
            'arms': arms, 'bootstrap': {'generated': True, 'repetitions': BOOTSTRAP_REPEATS,
                                      'seed': BOOTSTRAP_SEED, 'note': BOOTSTRAP_NOTE},
            'spec_digest': os.environ.get('SR_SPEC_DIGEST')}


def markdown(report):
    lines = ['# E-0016 per-arm screening', '',
             f"Coverage: {', '.join(report['completeness']['included_arms'])}; "
             f"missing: {', '.join(report['completeness']['missing_arms']) or 'none'}.",
             'No cross-arm hypothesis conclusion is made.', '',
             '| arm | protocol | primary result | EM margin | NLL gate | address weight |',
             '|---|---|---|---:|---|---:|']
    for arm, item in report['arms'].items():
        result = item['primary_screening']
        if arm == 'D1':
            gate = result['teacher_forced']
            lines.append(f"| {arm} | teacher-forced stage one | {gate['status']} | "
                         f"{gate['margin']*100:+.1f} pp | n/a | n/a |")
        else:
            key = 'hard_last_only_h3' if arm == 'D3' else 'h3'
            gate = result[key]
            weight = gate['target_weight']
            lines.append(f"| {arm} | {gate['protocol']} | {'pass' if gate['passed'] else 'not passed'} | "
                         f"{gate['em_margin']*100:+.1f} pp | {'pass' if gate['nll_pass'] else 'fail'} | {weight:.4f} |")
    for arm, item in report['arms'].items():
        lines.extend(['', f'## {arm}', '', item['supervision_note'], item['single_seed_note'],
                      '', '### Controls and position strata', '',
                      '| protocol | control | correct / count | EM | NLL | all-position target weight | last-position target weight | n_eff | argmax slot counts | position 0 (correct/count, EM, NLL) | position 1+ (correct/count, EM, NLL) |',
                      '|---|---|---:|---:|---:|---:|---:|---:|---|---|---|'])
        for protocol, data in item['protocols'].items():
            for control, point in data['controls'].items():
                summary = data['summaries'][control]
                diag = summary.get('diagnostics', {})
                last = summary.get('last_position', {})
                all_weight = diag.get('address.target_weight')
                last_weight = last.get('target_weight_mean')
                neff = diag.get('address.n_eff')
                counts = last.get('argmax_slot_counts')
                strata = data['position_strata'][control]
                p0, p1 = strata['position0'], strata['position1plus']
                later = 'n/a' if p1 is None else f"{p1['correct_count']}/{p1['count']}, {p1['em']:.4f}, {p1['nll']:.6f}"
                lines.append(f"| {protocol} | {control} | {point['correct_count']} / {point['count']} | "
                             f"{point['em']:.4f} | {point['nll']:.6f} | "
                             f"{_cell(all_weight, '.6f')} | {_cell(last_weight, '.6f')} | "
                             f"{_cell(neff, '.4f')} | {counts} | "
                             f"{p0['correct_count']}/{p0['count']}, {p0['em']:.4f}, {p0['nll']:.6f} | "
                             f"{later} |")
        curve = item['training_curve']
        lines.extend(['', '### Training and timing', '',
                      f"Training steps: {item['training']['steps']}; final 1000-step mean answer CE: "
                      f"{curve['final_1000_step_mean_answer_ce']:.6f}; fitted decline over 999 steps: "
                      f"{curve['fitted_decline_999_steps']:.6f}; plateau: {curve['plateau']}.",
                      f"Final-window LR: {curve['final_1000_lr_min']:.6g}–{curve['final_1000_lr_max']:.6g}; "
                      f"maximum is {curve['final_1000_max_lr_fraction_of_peak']*100:.2f}% of peak "
                      f"({'below ~8% of peak' if curve['final_1000_max_lr_fraction_of_peak'] <= 0.08 else 'above ~8% of peak'}). "
                      f"{curve['interpretation']}",
                      f"Timing: wall {item['timing']['wall_seconds']} s; MIG "
                      f"{item['timing']['mig_instance_hours']} MIG-h; node "
                      f"{item['timing']['node_hours']} node-h.", '',
                      'Per-500-step mean answer CE:', ''])
        lines.extend(f"- Steps {bucket['steps'][0]}–{bucket['steps'][1]}: {bucket['mean_answer_ce']:.6f}"
                     for bucket in curve['per_500_step_mean_answer_ce'])
    lines.extend(['', BOOTSTRAP_NOTE, ''])
    return '\n'.join(lines)


def _file_artifact(path, role):
    data = Path(path).read_bytes()
    return {'path': str(Path(path).resolve()), 'role': role,
            'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}


def _cell(value, format_spec):
    return '' if value is None else format(value, format_spec)


def write_report(report, output):
    output = Path(output).resolve()
    require(not any(output.iterdir()) if output.exists() else True,
            'analysis output directory must be empty')
    output.mkdir(parents=True, exist_ok=True)
    json_path, md_path = output/'report.json', output/'report.md'
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    md_path.write_text(markdown(report))
    artifacts = [_file_artifact(json_path, 'report.json'), _file_artifact(md_path, 'report.md')]
    manifest = {'status': 'complete', 'experiment': 'E-0016', 'contract': 'E-0016@1',
                'spec_digest': report.get('spec_digest'),
                'metrics': {'arms_included': report['completeness']['included_arms'],
                            'missing_arms': report['completeness']['missing_arms'],
                            'complete': report['completeness']['complete']},
                'artifacts': artifacts,
                'validator': {'name': 'v2r16_analysis', 'passed': True}}
    (output/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', action='append', required=True, metavar='ARM=DIR')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    arm_paths = {}
    input_paths = []
    for item in args.arm:
        name, sep, path = item.partition('=')
        require(sep and name in ARMS and path, f'invalid --arm {item!r}; expected ARM=DIR')
        require(name not in arm_paths, f'duplicate --arm {name}')
        arm_paths[name] = Path(path).resolve()
        input_paths.append(arm_paths[name])
    output = args.output.resolve()
    require(not any(output == path or path in output.parents for path in input_paths),
            'analysis output must be outside every input run')
    report = analyze(arm_paths)
    write_report(report, output)
    print(f"included={','.join(report['completeness']['included_arms'])}; "
          f"complete={str(report['completeness']['complete']).lower()}")


if __name__ == '__main__':
    main()
