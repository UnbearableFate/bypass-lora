"""Validate RB0 timing artifacts and combine diagnostic cost tables;PBS only."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics


MAIN_PARAMS={**{f'RB1-{i:02}':663552 for i in range(1,7)},'RB1-07':630784,
             'RB1-08':1318912,'RB1-09':925696,'RB1-02-reference':663552}
EXTRA_PARAMS={'Cdual32-S4':1318912,'Cup32-S4':925696,'T0-R32-S4':663552,
    'T0-Cdual32-S4':1318912,'T0-Cup32-S4':925696,'R44-S1':916608,'R44-S4':916608,
    'R64-S1':1343488,'R64-S4':1343488,'LoRA28-r1':630784,'LoRA28-r2':1261568}


def reference(path):
    path=Path(path).resolve();data=path.read_bytes()
    assert data
    return {'path':str(path),'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def positive(value):
    assert type(value) in (int,float) and math.isfinite(value) and value>0,value


def verified_timings(root,values,parameters):
    assert set(values)==set(parameters)
    token_sequences=[]
    for label,row in values.items():
        assert row['diagnostic_only'] is True and row['warmup_steps']==20 and row['stable_steps']==100
        assert row['params.stored']==row['params.trainable']==row['params.with_grad']==parameters[label]
        for key in ('step_seconds','tokens_per_second','load_seconds','save_seconds','sample_eval_seconds','peak_cuda_bytes'):
            positive(row[key])
        steps=[json.loads(line) for line in (root/label/'steps.jsonl').read_text().splitlines()]
        assert len(steps)==120 and [s['step'] for s in steps]==list(range(1,121))
        assert all(math.isfinite(s['seconds']) and s['seconds']>0 for s in steps)
        seconds=sum(s['seconds'] for s in steps[20:])
        assert math.isclose(seconds/100,row['step_seconds'],rel_tol=1e-12)
        assert math.isclose(sum(s['input_tokens'] for s in steps[20:])/seconds,row['tokens_per_second'],rel_tol=1e-12)
        tokens=[s['input_tokens'] for s in steps]
        token_sequences.append(tokens)
        accounting=row['runtime.accounting']
        for counts in accounting['layers'].values():
            assert counts['attention']==counts['ffn']==240
            assert counts['valid_tokens']==sum(tokens)
            if label.startswith('T0-') or label=='RB1-01':
                assert counts['writes']==counts['reads']==0
            else:
                depth=row.get('model_config',{}).get('steps',
                    2 if label=='RB1-03' else 4 if label in ('RB1-04','RB1-05','RB1-06') else 1)
                assert counts['writes']==sum(tokens) and counts['reads']==sum(tokens)*depth
    assert all(t==token_sequences[0] for t in token_sequences)
    return token_sequences[0]


def verified_latency(row):
    assert row['batch']==1 and row['prefill_length']==512 and row['prefill_passes']==5
    assert row['decode_tokens']==128 and len(row['decode_samples_ms'])==128
    assert len(row['prefill_seconds'])==5
    for value in row['decode_samples_ms']+row['prefill_seconds']:positive(value)
    ordered=sorted(row['decode_samples_ms'])
    assert row['decode_p50_ms']==statistics.median(ordered)
    assert row['decode_p95_ms']==ordered[int(.95*(len(ordered)-1))]
    assert math.isclose(row['prefill_tokens_per_second'],512/statistics.mean(row['prefill_seconds']),rel_tol=1e-12)
    assert row['adapter_persistent_state']['M_c_bytes'] in (0,16896)
    return {k:v for k,v in row.items() if k not in ('decode_samples_ms','prefill_seconds')}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--jobs',type=Path,required=True)
    p.add_argument('--probe',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    passed=False;metrics={};artifacts=[]
    try:
        jobs=read(args.jobs);assert len(jobs['runs'])==2
        locations={r['attempt']:r for r in jobs['runs']}
        roots={a:Path(r['output']) for a,r in locations.items()}
        for attempt,root in roots.items():
            manifest=read(root/'manifest.json')
            assert manifest['status']=='complete' and manifest['validator']['passed'] is True
            assert manifest['spec_digest']==locations[attempt]['spec_digest']
            guard=read(root/'mig-profile.json')
            assert guard['passed'] is True and guard['profile']=='2g.24gb'
            assert guard['uuid'] and guard['uuid']==guard['cuda_visible_devices']
        main_root=roots['ATT-0189-1']/'benchmark';extra_root=roots['ATT-0191-1']
        primary,extra=read(main_root/'timings.json'),read(extra_root/'timings.json')
        assert verified_timings(main_root,primary,MAIN_PARAMS)==verified_timings(extra_root,extra,EXTRA_PARAMS)
        schedule=read(main_root/'schedules.json');other=read(extra_root/'schedules.json')
        assert schedule['RB1']==other['R1'] and schedule['RB5']==other['RB5']
        assert schedule['RB1']['plan.steps']==13115 and schedule['RB5']['plan.steps']==26165
        probe=read(args.probe/'manifest.json')
        assert probe['status']=='complete' and probe['metrics']['checks']['checks.cases']==397
        assert probe['metrics']['checks']['checks.failed']==0
        # Timing and test snapshots must have identical numerical/training code.
        snapshots=Path('/work/xg24i002/x10041/bypass-lora/snapshots')
        tested=snapshots/'70a49714777960922fee7d03410faeaffe7acabf'
        numerical=('src/bypass_core.py','src/bypass_model.py','src/rb_train.py','src/rb_data.py',
                   'src/rb_history.py','src/v1_run.py','src/v1_data.py','src/v1_lora.py',
                   'src/calibrate.py','src/tolerances.json','pyproject.toml','uv.lock')
        source_hashes={name:reference(tested/name)['sha256'] for name in numerical}
        for run in locations.values():
            for name,digest in source_hashes.items():
                assert reference(snapshots/run['code_commit']/name)['sha256']==digest
        anchor=primary['RB1-02'];positive(anchor['full_eval_seconds'])
        scale=anchor['full_eval_seconds']/anchor['sample_eval_seconds']
        rows=[]
        for label,value in {**primary,**extra}.items():
            job='ATT-0189-1' if label in primary else 'ATT-0191-1'
            evaluation=value['sample_eval_seconds']*scale
            estimates={}
            for stage,passes in (('RBc',2),('RB1',5),('RB5',5)):
                per_pass=evaluation*(2.048 if stage=='RBc' else 1)
                seconds=2*value['load_seconds']+schedule[stage]['plan.steps']*value['step_seconds']+passes*per_pass+value['save_seconds']
                estimates[stage]={'seconds':seconds,'node_h':seconds/14400,'eval_passes':passes,
                                  'estimated_eval_seconds_per_pass':per_pass}
            rows.append({'case':label,'attempt':job,'step_seconds':value['step_seconds'],
                'tokens_per_second':value['tokens_per_second'],'peak_cuda_bytes':value['peak_cuda_bytes'],
                'parameters':value['params.trainable'],'load_seconds':value['load_seconds'],
                'save_seconds':value['save_seconds'],'sample_eval_seconds':value['sample_eval_seconds'],
                'adapter_state_M_c_bytes':value['runtime.accounting']['state_M_c_bytes'],
                'estimates':estimates})
        row_map={r['case']:r for r in rows}
        rb1=sum(row_map[f'RB1-{i:02}']['estimates']['RB1']['node_h'] for i in range(1,10))
        original=read(main_root/'cost-estimates.json')
        assert math.isclose(rb1,original['RB1_training_monitor_reload_node_h'],rel_tol=1e-12)
        latency={'Base':verified_latency(read(main_root/'latency-base.json'))}
        for label,row in primary.items():
            if 'latency' in row:latency[label]=verified_latency(row['latency'])
            if 'latency_merged' in row:latency[label+'-merged']=verified_latency(row['latency_merged'])
        assert len(latency)==7
        for label,row in latency.items():
            expected=16896 if label in ('RB1-02','RB1-04','RB1-08') else 0
            assert row['adapter_persistent_state']['M_c_bytes']==expected
        prefix_max=max(r['step_seconds'] for r in rows if r['case']!='RB1-02-reference' and not r['case'].startswith('LoRA28-'))
        family_cases={'RBc-L':'RB1-07','RBc-T':'RB1-01','RBc-R':'RB1-02','RBc-C':'RB1-08'}
        family_costs={f:row_map[c]['estimates']['RBc']['node_h'] for f,c in family_cases.items()}
        sources=[reference(root/'manifest.json') for root in roots.values()]+[
            reference(main_root/'timings.json'),reference(extra_root/'timings.json'),
            reference(main_root/'cost-estimates.json'),reference(main_root/'schedules.json'),
            reference(main_root/'latency-base.json'),reference(args.probe/'checks'/'checks.xml')]
        summary={'experiment':'E-0019@1','scope':'RB0 engineering/cost diagnostics only',
            'sources':sources,'source_hashes_unchanged_since397check_snapshot':source_hashes,
            'schedules':schedule,'cases':rows,'latency':latency,'full_dev_seconds':anchor['full_eval_seconds'],
            'RB1_training_monitor_reload_node_h':rb1,'max_nonreference_RB1_or_side_proxy_step_seconds':prefix_max,
            'profile_required':prefix_max>.4,'RB1_core_estimate_exceeds4':rb1>4,
            'RBc_family_node_h_per_candidate':family_costs,'RBc_12candidate_node_h':sum(family_costs.values())*3,
            'RBc_16candidate_node_h':sum(family_costs.values())*4,
            'actual_T0022_through_timing_MIG_seconds':1755,'actual_T0022_through_timing_node_h':1755/14400,
            'timing_sample_rule':'20warmup+100stable optimizer steps,effectivebatch2/micro1;pairedfirst120R1batches',
            'estimate_rule':'two loads+plannedsteps*measuredstep+evalpasses*scaleddev+save;RBc2passes4096calibration/2000dev;RB1/RB5fivepasses',
            'limits':['Estimates omit future frozen interventions,bootstrap/analysis and data planning overhead;they are not billed usage.',
                'Only R_res S1 full dev was timed;other evals scale32rows/task against that full/sample ratio;RBc also scales row count,not measured calibration lengths.',
                'RB5 matrix/monitoring remains conditional;S2 unmeasured layouts need labeled interpolation;S1concatT0 and mixed-rankLoRA are not measured here.',
                'Latency uses synthetic512tokens and128fixed decoder steps;mergedLoRA timing only,unmerged adapters for task scoring.',
                'D0026/D0027 preserve0.50002;reference-matched BF16shape-change exceedances remain diagnostics,decisiveFP32 and same-shapeBF16 gates pass.'],
            'formal_RB_training_started':False,'confirmation_evaluated':False}
        write(out/'cost-summary.json',summary)
        lines=['# RB0 measured timing and conditional costs','',
            '| Case | Trainable/stored/with-grad | Step s | Tokens/s | Peak CUDA GiB | RB5 3epochs node_h |',
            '| --- | ---: | ---: | ---: | ---: | ---: |']
        for r in rows:
            lines.append(f'| {r["case"]} | {r["parameters"]} | {r["step_seconds"]:.6f} | {r["tokens_per_second"]:.1f} | {r["peak_cuda_bytes"]/2**30:.3f} | {r["estimates"]["RB5"]["node_h"]:.6f} |')
        lines+=['','RB1 arms:training,three intermediate dev passes,final dev and fresh reload evaluation.','',
                '| Arm | Estimated node_h |','| --- | ---: |']
        for i in range(1,10):
            label=f'RB1-{i:02}';lines.append(f'| {label} | {row_map[label]["estimates"]["RB1"]["node_h"]:.6f} |')
        lines += ['',f'RB1 core total: {rb1:.6f} node_h. Frozen interventions,analysis and planning overhead are additional.',
            '', '| RBc family | Estimated node_h/candidate |','| --- | ---: |']
        for family,cost in family_costs.items():lines.append(f'| {family} | {cost:.6f} |')
        lines += ['',f'RBc12-16candidate estimate: {sum(family_costs.values())*3:.6f}-{sum(family_costs.values())*4:.6f} node_h.',
            '', '| Latency case | Prefill tok/s | Decode p50 ms | Decode p95 ms | Peak CUDA GiB | Persistent adapter M+c bytes |',
            '| --- | ---: | ---: | ---: | ---: | ---: |']
        for label,r in latency.items():
            lines.append(f'| {label} | {r["prefill_tokens_per_second"]:.1f} | {r["decode_p50_ms"]:.3f} | {r["decode_p95_ms"]:.3f} | {r["peak_cuda_bytes"]/2**30:.3f} | {r["adapter_persistent_state"]["M_c_bytes"]} |')
        lines += ['',f'Complete R_resS1 dev pass: {anchor["full_eval_seconds"]:.3f}s. All21training cases have100stable steps;all7latency cases have512token prefill and128decode samples.',
            f'Measured T0022 usage through timing:1755MIGseconds={1755/14400:.6f}node_h. This tabulation job is additional until its scheduler end.','',*summary['limits'],'']
        (out/'cost-tables.md').write_text('\n'.join(lines))
        metrics={'training_cases':21,'stable_steps_per_case':100,'latency_cases':7,'decode_samples':896,
                 'RB1_core_node_h':rb1,'max_side_step_seconds':prefix_max,'profile_required':prefix_max>.4}
        artifacts=[{'path':str(out/name),'role':name} for name in ('cost-summary.json','cost-tables.md')]
        passed=True
    finally:
        write(out/'manifest.json',{'status':'complete' if passed else 'partial','contract':'E-0019@1',
            'spec_digest':os.environ['SR_SPEC_DIGEST'],'metrics':metrics,'artifacts':artifacts,
            'validator':{'name':'rb0_cost_integrity','passed':passed}})


if __name__=='__main__':main()
