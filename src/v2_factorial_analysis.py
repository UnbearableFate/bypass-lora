"""Read-only E-0010 analysis; run in PBS, never open confirmation data.

Decisions use final dev point estimates. Bootstrap intervals, calibration,
curves and oracle diagnostics are descriptive and never enter selection.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import itertools
import json
import math
import os
from pathlib import Path

import numpy as np


ARM_SPECS = {
    'V2-01': ('direct_hidden', 'kernel_matrix', 'F_out'),
    'V2-02': ('direct_hidden', 'kernel_matrix', 'C_up'),
    'V2-03': ('direct_hidden', 'soft_slot', 'F_out'),
    'V2-04': ('direct_hidden', 'soft_slot', 'C_up'),
    'V2-05': ('qknorm_pre_rope', 'kernel_matrix', 'F_out'),
    'V2-06': ('qknorm_pre_rope', 'kernel_matrix', 'C_up'),
    'V2-07': ('qknorm_pre_rope', 'soft_slot', 'F_out'),
    'V2-08': ('qknorm_pre_rope', 'soft_slot', 'C_up'),
    'V2-09': ('qknorm_pre_rope', 'kernel_matrix', 'C_dual'),
    'V2-10': ('qknorm_pre_rope', 'soft_slot', 'C_dual'),
}
CONTROLS = ('normal', 'zero', 'shuffled', 'permuted', 'oracle_key', 'oracle_value')
COUNTS = {'dev': 1000, 'calibration': 2048}
DATA_HASHES = {
    'train': 'a2acb5451bc8aade1c274ebe225b4411bb3063e81bf57ba73c35101d9ac807d2',
    'calibration': 'c08753365165216b5a9f4c3d629271b2183e1323973886256867e826c1712f18',
    'dev': 'c6583cf8f7cd9d5652ed57093432b278b0600c611076fac2fe6c6b7d9e7715f4',
}
REVISION = 'da87bfb608c14b7cf20ba1ce41287e8de496c0cd'
ORDER = '0d4b55ed8655d91cf0d0532bdac1820a518b5c5730ef70e48f264da6b4bc9e79'
SCHEDULE = dict(steps=6000, tokens=4000470, warmup_steps=300, budget=4000000,
                order_sha256=ORDER)
BOOTSTRAP_REPEATS, BOOTSTRAP_SEED = 10000, 20261029
# Verbatim decision and interpretation rules from E-0010@2, Design.
RULES = {
    'qualification': '合格：EM_normal − max(EM_zero, EM_shuffled) ≥ 10 pp，且 NLL_normal < min(NLL_zero, NLL_shuffled)。',
    'nondegenerate': '读取非退化：该 arm 在 normal memory 下对目标事实的平均地址权重 ≥ 2/16 = 0.125（均匀分布的两倍）。',
    'positive_control': '可用正对照：E-0008 通过放行门槛（开始条件已保证）。',
    'H-0003': 'supported（screening）：至少一个 arm 合格且读取非退化。challenged（screening）：没有 arm 合格。inconclusive：有 arm 合格，但它们的读取都退化；此时先用 oracle 与诊断定位。',
    'probe': 'C_dual 探针（V2-09、V2-10）计入合格判定，但不进入下面 H-0004 的四格比较。',
    'H-0004': '在一格中，soft_slot 同时有更高的 normal EM、更高的平均目标权重与更低的平均 n_eff，记该格 SoftmaxSlot 更好；kernel_matrix 的 normal EM 更高，记该格 KernelMatrix 更好。≥ 3/4 格 SoftmaxSlot 更好 -> supported（screening）；≥ 3/4 格 KernelMatrix 更好 -> challenged（screening）；其余 inconclusive。同时报告归一化读误差。',
    'V3': '在合格且读取非退化的 arm 中，取 EM_normal − max(EM_zero, EM_shuffled) 最大者；与它相差不到 1 pp 的 arm 中，取可训练参数最少者；仍相同则取 dev NLL_normal 较低者。',
    'oracle': 'value oracle 高而 normal 低 -> 寻址瓶颈；两者都低 -> 写入值或融合瓶颈',
}
IDENTITY = ('episode_id', 'target', 'relation', 'target_fact_position',
            'support_order', 'answer_ne_subject')


class AnalysisError(ValueError):
    """Artifacts cannot support the locked comparison."""


def require(condition, message):
    if not condition:
        raise AnalysisError(message)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def read(path, provenance, *, lines=False):
    path = Path(path)
    try:
        payload = path.read_bytes()
        value = ([json.loads(line) for line in payload.splitlines() if line.strip()]
                 if lines else json.loads(payload))
    except (OSError, ValueError) as exc:
        raise AnalysisError(f'cannot read {path}: {exc}') from exc
    provenance[str(path.resolve())] = hashlib.sha256(payload).hexdigest()
    return value


def identity(rows):
    return [tuple(r[k] for k in IDENTITY) for r in rows]


def point(rows):
    n = len(rows)
    return dict(count=n, correct_count=sum(r['correct'] for r in rows),
                em=sum(r['correct'] for r in rows)/n if n else None,
                nll=sum(r['nll'] for r in rows)/n if n else None)


def check_summary(summary, rows, label):
    expected = point(rows)
    for key in ('count', 'em', 'nll'):
        actual, wanted = summary.get(key), expected[key]
        require(actual == wanted or (finite(actual) and finite(wanted)
                and abs(actual-wanted) <= 1e-10), f'{label}: summary {key} mismatch')


def check_records(rows, summary, split, label):
    require(len(rows) == COUNTS[split], f'{label}: incomplete {split}')
    require(len({r.get('episode_id') for r in rows}) == len(rows), f'{label}: duplicate episode')
    for r in rows:
        require(all(k in r for k in IDENTITY), f'{label}: missing episode metadata')
        require(isinstance(r['episode_id'], str) and len(r['episode_id']) == 64,
                f'{label}: invalid episode identity')
        require(type(r.get('prediction')) is int and r['prediction'] >= 0
                and type(r['target']) is int and r['target'] >= 0, f'{label}: invalid token')
        require(type(r.get('correct')) is bool and r['correct'] == (r['prediction'] == r['target']),
                f'{label}: incorrect correctness flag')
        require(finite(r.get('nll')) and r['nll'] >= 0, f'{label}: nonfinite/negative NLL')
        require(type(r['answer_ne_subject']) is bool and isinstance(r['relation'], str)
                and type(r['target_fact_position']) is int and 0 <= r['target_fact_position'] < 16
                and r['support_order'] in ('natural', 'shuffled'), f'{label}: invalid strata')
    check_summary(summary, rows, label)
    check_summary(summary['answer_ne_subject'], [r for r in rows if r['answer_ne_subject']], label+' nonself')
    for axis in ('relation', 'target_fact_position', 'support_order'):
        names = {str(r[axis]) for r in rows}
        require(set(summary['strata'][axis]) == names, f'{label}: strata membership mismatch')
        for name in names:
            check_summary(summary['strata'][axis][name], [r for r in rows if str(r[axis]) == name],
                          label+' '+axis+' '+name)
    diagnostics = summary.get('diagnostics', {})
    require(all(not isinstance(v, (int, float)) or isinstance(v, bool) or finite(v)
                for v in diagnostics.values()), f'{label}: nonfinite diagnostic')


def load_run(path, arm):
    """Read only declared train/dev/calibration outputs, never data or checkpoints."""
    path, provenance = Path(path).resolve(), {}
    manifest = read(path/'manifest.json', provenance)
    require(manifest.get('status') == 'complete' and manifest.get('validator', {}).get('passed') is True
            and manifest.get('failure') is None, f'{arm}: run incomplete or failed')
    expected_manifest = dict(model='Qwen/Qwen3-0.6B-Base', revision=REVISION, seed=0,
                             batch_size=8, evaluation_batch_size=16, token_budget=4000000,
                             smoke_steps=0, smoke_eval_limit=0, data_sha256=DATA_HASHES,
                             torch='2.10.0+cu128', cuda='12.8')
    require(all(manifest.get(k) == v for k, v in expected_manifest.items()),
            f'{arm}: wrong formal manifest/model/data/environment')
    require(manifest.get('microbatch_size') in range(1, 9), f'{arm}: invalid microbatch')
    name = '.'.join(ARM_SPECS[arm])
    results = read(path/'metrics.json', provenance)
    require(set(results) == {name}, f'{arm}: wrong arm or multiple arms')
    result, directory = results[name], path/name
    require(read(directory/'metrics.json', provenance) == result, f'{arm}: nested metrics mismatch')
    tap, backend, fusion = ARM_SPECS[arm]
    expected_config = dict(layers=[14], rank=128, branch='memory', fusion=fusion,
        tap=tap, backend=backend, steps=1, mode='inclusive', update='accumulate',
        readout='increment', eta=.25, read_gain=1., eps_norm=1e-6, eps_read=1e-6,
        write_scope='visible_fact_delimiter', slot_capacity=16, model_revision=REVISION)
    require(result.get('config') == expected_config, f'{arm}: wrong adapter configuration')
    require(result.get('lr') == (.01 if backend == 'kernel_matrix' else .003), f'{arm}: wrong released LR')
    require(result.get('initialization_max_abs') == 0., f'{arm}: zero-exit initialization failed')
    require(all(result['schedule'].get(k) == v for k, v in SCHEDULE.items()), f'{arm}: wrong schedule')
    require(read(path/'schedule.json', provenance) == result['schedule'], f'{arm}: schedule file mismatch')
    train = result['train']
    require((train.get('steps'), train.get('tokens')) == (6000, 4000470), f'{arm}: partial training')
    params = [train.get(k) for k in ('stored_parameters', 'trainable_parameters', 'parameters_with_nonzero_gradient')]
    require(all(type(p) is int and p > 0 for p in params) and params[0] >= params[1] >= params[2],
            f'{arm}: invalid parameter counts')
    for key in ('training_seconds', 'stable_step_seconds', 'tokens_per_second', 'peak_allocated_bytes',
                'gradient_norm_mean', 'gradient_norm_max', 'clip_fraction'):
        require(finite(train.get(key)) and train[key] >= 0, f'{arm}: nonfinite training statistic')
    history = read(directory/'train.jsonl', provenance, lines=True)
    require(len(history) == 6000, f'{arm}: incomplete training history')
    previous_tokens = 0
    for step, h in enumerate(history, 1):
        require(h.get('step') == step and type(h.get('tokens')) is int and h['tokens'] > previous_tokens,
                f'{arm}: training history order/count mismatch')
        require(all(finite(h.get(k)) and h[k] >= 0 for k in ('loss', 'lr', 'gradient_norm', 'seconds')),
                f'{arm}: nonfinite training history')
        previous_tokens = h['tokens']
    require(previous_tokens == 4000470 and history[-1]['lr'] == 0., f'{arm}: final training step mismatch')
    require(set(result['evaluation']) == set(COUNTS), f'{arm}: missing final split')
    records = {}
    for split in COUNTS:
        summaries = result['evaluation'][split]
        require(set(summaries) == set(CONTROLS), f'{arm}: missing final controls')
        require(read(directory/f'{split}.metrics.json', provenance) == summaries,
                f'{arm}: split metrics mismatch')
        records[split] = {}
        for mode in CONTROLS:
            rows = read(directory/f'{split}.{mode}.predictions.jsonl', provenance, lines=True)
            check_records(rows, summaries[mode], split, f'{arm} {split} {mode}')
            records[split][mode] = rows
            if mode != 'normal':
                require(identity(rows) == identity(records[split]['normal']), f'{arm}: control episode pairing mismatch')
        diag = summaries['normal']['diagnostics']
        require(finite(diag.get('address.target_weight')) and 0 <= diag['address.target_weight'] <= 1
                and finite(diag.get('address.n_eff')) and 0 < diag['address.n_eff'] <= 16.0001
                and finite(diag.get('read.normalized_error')) and diag['read.normalized_error'] >= 0,
                f'{arm}: missing/invalid normal read diagnostics')
        require(summaries['shuffled']['diagnostics'].get('address.target_available') is False
                and 'address.target_weight' not in summaries['shuffled']['diagnostics'],
                f'{arm}: shuffled recipient target is undefined')
    require(result['checkpoint_reload'].get('predictions_identical') is True, f'{arm}: reload failed')
    reload_rows = read(directory/'reload/dev.normal.predictions.jsonl', provenance, lines=True)
    check_records(reload_rows, result['checkpoint_reload']['dev_normal'], 'dev', arm+' reload')
    require(identity(reload_rows) == identity(records['dev']['normal'])
            and [r['prediction'] for r in reload_rows] == [r['prediction'] for r in records['dev']['normal']],
            f'{arm}: reload predictions mismatch')
    curves = train['learning_curves']
    require(set(curves) == {'25', '50', '75'}, f'{arm}: missing learning-curve milestone')
    for milestone, summaries in curves.items():
        require(set(summaries) == {'normal', 'zero', 'shuffled'}, f'{arm}: invalid curve controls')
        require(read(directory/f'curve-{milestone}/dev.metrics.json', provenance) == summaries,
                f'{arm}: curve metrics mismatch')
        for mode, summary in summaries.items():
            rows = read(directory/f'curve-{milestone}/dev.{mode}.predictions.jsonl', provenance, lines=True)
            check_records(rows, summary, 'dev', f'{arm} curve {milestone} {mode}')
            require(identity(rows) == identity(records['dev']['normal']), f'{arm}: curve episode pairing mismatch')
    return dict(result=result, records=records, source=dict(attempt=path.name, path=str(path),
                spec_digest=manifest.get('spec_digest'), sha256=provenance))


def check_alignment(loaded):
    reference = loaded['V2-01']['records']
    for arm, run in loaded.items():
        for split in COUNTS:
            require(identity(run['records'][split]['normal']) == identity(reference[split]['normal']),
                    f'{arm}: cross-arm {split} episode pairing mismatch')
    dev_ids = {r['episode_id'] for r in reference['dev']['normal']}
    require(not dev_ids.intersection(r['episode_id'] for r in reference['calibration']['normal']),
            'dev and calibration episode identities overlap')


def locked_decisions(arms):
    """All thresholds are literal; exact count fractions avoid EM boundary drift."""
    qualified, eligible, inputs = [], [], {}
    for arm in ARM_SPECS:
        d = arms[arm]
        normal, zero, shuffled = (d['controls'][m] for m in ('normal', 'zero', 'shuffled'))
        n = normal['count']
        require(n == 1000 and zero['count'] == n and shuffled['count'] == n, 'decisions require full dev')
        margin = Fraction(normal['correct_count']-max(zero['correct_count'], shuffled['correct_count']), n)
        em_pass, nll_pass = margin >= Fraction(1, 10), normal['nll'] < min(zero['nll'], shuffled['nll'])
        nondegenerate = d['target_weight'] >= .125
        passed = em_pass and nll_pass
        if passed:
            qualified.append(arm)
            if nondegenerate:
                eligible.append(arm)
        inputs[arm] = dict(normal_em=normal['em'], zero_em=zero['em'], shuffled_em=shuffled['em'],
            normal_nll=normal['nll'], zero_nll=zero['nll'], shuffled_nll=shuffled['nll'],
            em_margin=float(margin), em_margin_fraction=str(margin), em_pass=em_pass,
            nll_pass=nll_pass, target_weight=d['target_weight'], nondegenerate=nondegenerate, qualified=passed)
    status = 'supported' if eligible else ('inconclusive' if qualified else 'challenged')
    cells = []
    for kernel, soft in (('V2-01', 'V2-03'), ('V2-02', 'V2-04'), ('V2-05', 'V2-07'), ('V2-06', 'V2-08')):
        k, s = arms[kernel], arms[soft]
        soft_better = (s['controls']['normal']['correct_count'] > k['controls']['normal']['correct_count']
                       and s['target_weight'] > k['target_weight'] and s['n_eff'] < k['n_eff'])
        kernel_better = k['controls']['normal']['correct_count'] > s['controls']['normal']['correct_count']
        cells.append(dict(kernel=kernel, soft=soft, inputs={a: dict(normal_em=arms[a]['controls']['normal']['em'],
            target_weight=arms[a]['target_weight'], n_eff=arms[a]['n_eff'],
            normalized_read_error=arms[a]['normalized_read_error']) for a in (kernel, soft)},
            classification='SoftmaxSlot' if soft_better else ('KernelMatrix' if kernel_better else 'inconclusive')))
    sw = sum(c['classification'] == 'SoftmaxSlot' for c in cells)
    kw = sum(c['classification'] == 'KernelMatrix' for c in cells)
    selection = dict(selected_arm=None, eligible=eligible, near_best=[], finalists=[])
    if eligible:
        best = max(Fraction(inputs[a]['em_margin_fraction']) for a in eligible)
        near = [a for a in eligible if best-Fraction(inputs[a]['em_margin_fraction']) < Fraction(1, 100)]
        min_params = min(arms[a]['trainable_parameters'] for a in near)
        smallest = [a for a in near if arms[a]['trainable_parameters'] == min_params]
        min_nll = min(arms[a]['controls']['normal']['nll'] for a in smallest)
        finalists = [a for a in smallest if arms[a]['controls']['normal']['nll'] == min_nll]
        selection.update(best_margin=float(best), near_best=near, minimum_trainable_parameters=min_params,
                         minimum_nll=min_nll, finalists=finalists,
                         selected_arm=finalists[0] if len(finalists) == 1 else None,
                         unresolved_exact_tie=len(finalists) > 1,
                         candidate_inputs={a: dict(margin=inputs[a]['em_margin'],
                             trainable_parameters=arms[a]['trainable_parameters'],
                             nll=arms[a]['controls']['normal']['nll']) for a in eligible})
    return {'H-0003': dict(status=status, level='screening', qualified=qualified, eligible=eligible,
                           inputs=inputs, positive_control='E-0010@2 Release: E-0008 gate passed, ATT-0069-1'),
            'H-0004': dict(status='supported' if sw >= 3 else ('challenged' if kw >= 3 else 'inconclusive'),
                           level='screening', soft_wins=sw, kernel_wins=kw, cells=cells), 'V3': selection}


def paired_bootstrap(values, repeats=BOOTSTRAP_REPEATS, seed=BOOTSTRAP_SEED, chunk_size=64):
    """Shared iid episode indices for every column, with bounded working memory."""
    values = np.asarray(values, dtype=np.float64)
    require(values.ndim == 2 and len(values) > 0 and np.isfinite(values).all(), 'invalid bootstrap matrix')
    require(repeats > 0 and chunk_size > 0, 'positive repeats and chunk size required')
    rng, n = np.random.default_rng(seed), len(values)
    result = np.empty((repeats, values.shape[1]), dtype=np.float64)
    for start in range(0, repeats, chunk_size):
        size = min(chunk_size, repeats-start)
        indices = rng.integers(0, n, size=(size, n))
        counts = np.zeros((size, n), dtype=np.float64)
        np.add.at(counts, (np.arange(size)[:, None], indices), 1.)
        result[start:start+size] = counts @ values / n
    return result


def effect_weights():
    """Conventional 2^3 effects = 2/8 sum(product(signs)*response), also for interactions."""
    factors = ('feature', 'reader', 'fusion')
    signs = {a: (1 if spec[0] == 'qknorm_pre_rope' else -1,
                  1 if spec[1] == 'soft_slot' else -1, 1 if spec[2] == 'C_up' else -1)
             for a, spec in ARM_SPECS.items() if spec[2] != 'C_dual'}
    return {' x '.join(factors[i] for i in subset):
                {a: math.prod(s[i] for i in subset)/4 for a, s in signs.items()}
            for order in (1, 2, 3) for subset in itertools.combinations(range(3), order)}


def descriptive(loaded, split, *, repeats=BOOTSTRAP_REPEATS):
    names, columns = [], []
    for arm in ARM_SPECS:
        for mode in CONTROLS:
            rows = loaded[arm]['records'][split][mode]
            for metric in ('em', 'nll'):
                names.append((arm, mode, metric))
                columns.append([r['correct' if metric == 'em' else 'nll'] for r in rows])
    values = np.asarray(columns, dtype=np.float64).T
    points, boots = values.mean(0), paired_bootstrap(values, repeats=repeats)
    lookup = {name: i for i, name in enumerate(names)}

    def estimate(terms):
        weights = np.zeros(len(names))
        for name, weight in terms.items():
            weights[lookup[name]] = weight
        return dict(estimate=float(points @ weights),
                    ci95=np.quantile(boots @ weights, [.025, .975]).tolist())

    comparisons = {}
    for arm in ARM_SPECS:
        comparisons[arm] = {mode: {metric: estimate({(arm, 'normal', metric): 1, (arm, mode, metric): -1})
                                  for metric in ('em', 'nll')} for mode in CONTROLS if mode != 'normal'}
    factorial = {}
    for effect, weights in effect_weights().items():
        factorial[effect] = {}
        for outcome in ('normal', 'normal-zero', 'normal-shuffled', 'normal-permuted'):
            factorial[effect][outcome] = {}
            for metric in ('em', 'nll'):
                terms = {(a, 'normal', metric): w for a, w in weights.items()}
                if outcome != 'normal':
                    terms.update({(a, outcome.split('-')[1], metric): -w for a, w in weights.items()})
                factorial[effect][outcome][metric] = estimate(terms)
    dual = {f'{probe} - {comparator}': {metric: estimate({(probe, 'normal', metric): 1,
             (comparator, 'normal', metric): -1}) for metric in ('em', 'nll')}
            for probe, comparators in (('V2-09', ('V2-05', 'V2-06')), ('V2-10', ('V2-07', 'V2-08')))
            for comparator in comparators}
    reader = {f'{soft} - {kernel}': {metric: estimate({(soft, 'normal', metric): 1,
              (kernel, 'normal', metric): -1}) for metric in ('em', 'nll')}
              for kernel, soft in (('V2-01', 'V2-03'), ('V2-02', 'V2-04'),
                                   ('V2-05', 'V2-07'), ('V2-06', 'V2-08'))}
    strata = {}
    for arm in ARM_SPECS:
        strata[arm] = {}
        for mode in CONTROLS:
            rows = loaded[arm]['records'][split][mode]
            strata[arm][mode] = {'answer_ne_subject': point([r for r in rows if r['answer_ne_subject']])}
            for axis in ('relation', 'target_fact_position', 'support_order'):
                strata[arm][mode][axis] = {name: point([r for r in rows if str(r[axis]) == name])
                                          for name in sorted({str(r[axis]) for r in rows})}
    return dict(normal_minus_control=comparisons, factorial=factorial, C_dual_minus_comparator=dual,
                soft_minus_kernel=reader, strata=strata,
                bootstrap=dict(unit='episode', pairing='same episode draws for all arms and controls within split',
                    repeats=repeats, seed=BOOTSTRAP_SEED, interval='percentile 2.5/97.5; numpy linear quantiles'),
                units=dict(em='fraction; multiply by 100 for percentage points', nll='natural logarithm'),
                factorial_definition='2/8 * sum(product of contrast signs * response); interactions use this same normalization',
                factorial_positive_levels=dict(feature='qknorm_pre_rope', reader='soft_slot', fusion='C_up'),
                strata_uncertainty='strata/subset tables are descriptive point estimates; intervals above use all episodes')


def analyze(paths):
    require(set(paths) == set(ARM_SPECS), 'exactly V2-01 through V2-10 required')
    require(len({Path(p).resolve() for p in paths.values()}) == 10, 'duplicate run directory')
    loaded = {arm: load_run(paths[arm], arm) for arm in ARM_SPECS}
    check_alignment(loaded)
    arms = {}
    for arm, run in loaded.items():
        r, dev = run['result'], run['records']['dev']
        d, train = r['evaluation']['dev']['normal']['diagnostics'], r['train']
        arms[arm] = dict(config=r['config'], lr=r['lr'],
            controls={m: point(dev[m]) for m in CONTROLS},
            calibration={m: point(run['records']['calibration'][m]) for m in CONTROLS},
            target_weight=d['address.target_weight'], n_eff=d['address.n_eff'],
            normalized_read_error=d['read.normalized_error'],
            read_diagnostics={s: {m: r['evaluation'][s][m]['diagnostics'] for m in CONTROLS} for s in COUNTS},
            **{k: train[k] for k in ('stored_parameters', 'trainable_parameters', 'parameters_with_nonzero_gradient',
                'steps', 'tokens', 'tokens_per_second', 'peak_allocated_bytes', 'gradient_norm_mean',
                'gradient_norm_max', 'clip_fraction')},
            app_wall_seconds=r['wall_seconds'], app_node_hours=r['wall_seconds']/3600*.25,
            learning_curves={p: {m: {k: v[k] for k in ('count', 'em', 'nll')}
                                for m, v in c.items()} for p, c in train['learning_curves'].items()},
            priors={s: r['evaluation'][s]['normal']['prior_em'] for s in COUNTS}, source=run['source'])
    return dict(contract='E-0010@2', scope='M-line seed 0 screening; final dev decisions; no confirmation read',
                rules=RULES, decisions=locked_decisions(arms), arms=arms,
                descriptive={s: descriptive(loaded, s) for s in COUNTS},
                oracle_interpretation=dict(rule=RULES['oracle'],
                    limitation='No numerical high/low threshold is sealed. Report oracle EM/NLL, errors and paired differences; coordinator interprets them without adding a decision gate.'),
                diagnostics_weighting='valid question tokens, as emitted by reviewed v2_train; EM/NLL and bootstrap use episodes',
                cost_limitation='Application node-hours exclude PBS overhead. Actual budget accounting comes from the Miyabi ledger.')


def markdown(report):
    lines = ['# E-0010 V2 M-line factorial', '', report['scope'], '',
             '| Arm | EM normal / zero / shuffled / permuted (%) | NLL normal / zero / shuffled / permuted | Target weight | n_eff |',
             '| --- | --- | --- | --- | --- |']
    for arm, a in report['arms'].items():
        modes = ('normal', 'zero', 'shuffled', 'permuted')
        em = ' / '.join(f"{100*a['controls'][m]['em']:.2f}" for m in modes)
        nll = ' / '.join(f"{a['controls'][m]['nll']:.6f}" for m in modes)
        lines.append(f"| {arm} | {em} | {nll} | {a['target_weight']:.6f} | {a['n_eff']:.4f} |")
    d = report['decisions']
    lines.extend(['', f"H-0003: {d['H-0003']['status']} (screening). Qualified: {d['H-0003']['qualified']}; nondegenerate eligible: {d['H-0003']['eligible']}.",
                  f"H-0004: {d['H-0004']['status']} (screening); SoftmaxSlot wins {d['H-0004']['soft_wins']}/4, KernelMatrix wins {d['H-0004']['kernel_wins']}/4.",
                  f"V3 selection: {d['V3']['selected_arm']}; finalists: {d['V3']['finalists']}.", '',
                  'Full rule inputs, paired intervals, oracle results, strata, curves, parameter counts and source hashes are in factorial-report.json.',
                  report['oracle_interpretation']['limitation'], report['cost_limitation'], ''])
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='append', required=True, metavar='V2-ID=/absolute/run/path')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    paths = {}
    for item in args.run:
        arm, separator, path = item.partition('=')
        require(separator and arm in ARM_SPECS and arm not in paths and Path(path).is_absolute(),
                'each --run must name a distinct V2 arm and absolute path')
        paths[arm] = Path(path)
    output = args.output.resolve()
    require(not any(output == p.resolve() or p.resolve() in output.parents for p in paths.values()),
            'analysis output must be outside every input run')
    require(not (output/'manifest.json').exists(), 'analysis output already has a manifest')
    report = analyze(paths)
    output.mkdir(parents=True, exist_ok=True)
    target = output/'factorial-report.json'
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    (output/'factorial-report.md').write_text(markdown(report))
    metrics = {'arms_completed': 10, 'H3.qualified': len(report['decisions']['H-0003']['qualified']),
               'H3.eligible': len(report['decisions']['H-0003']['eligible']),
               'H4.soft_wins': report['decisions']['H-0004']['soft_wins'],
               'H4.kernel_wins': report['decisions']['H-0004']['kernel_wins']}
    (output/'manifest.json').write_text(json.dumps(dict(status='complete', spec_digest=os.environ['SR_SPEC_DIGEST'],
        metrics=metrics, artifacts=[dict(path=str(p), role=p.name) for p in (target, output/'factorial-report.md')],
        validator=dict(name='E-0010_locked_rules_and_paired_artifacts', passed=True)), indent=2)+'\n')
    print(markdown(report), flush=True)


if __name__ == '__main__':
    main()
