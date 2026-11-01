"""Deterministic E-0002 selection and E-0003 planning; run on compute."""
import json
import math
import os
from pathlib import Path
import sys
from transformers import AutoTokenizer
from calibrate import MODEL, REVISION
from v1_run import prepare, plan_batches

root = Path('/work/xg24i002/x10041/bypass-lora')
out = Path(sys.argv[1]); out.mkdir(parents=True,exist_ok=True)
records = [json.loads((root/f'runs/ATT-{i:04d}-2/manifest.json').read_text()) for i in range(11,18)]
assert all(r['status']=='complete' and r['validator']['passed'] for r in records)
metrics = [r['metrics'] for r in records]
assert all(m['reload.identical'] and not m.get('run.failed_lr') for m in metrics)
assert len({m['plan.order_sha256'] for m in metrics})==1
assert all(m['train.input_tokens']==200280 and m['train.steps']==1103 for m in metrics[1:])
for i in range(11,18):
    predictions=json.loads((root/f'runs/ATT-{i:04d}-2/predictions.json').read_text())
    assert len(predictions)==6096
    assert all(math.isfinite(s) for p in predictions for s in p['scores'])
    identity=[(p['task'],p['split'],p['id'],p['gold']) for p in predictions]
    if i==11: reference=identity
    else: assert identity==reference
selected={}; comparisons={}
for family, first in [('lora',1),('additive',3),('concat',5)]:
    values=[metrics[first+j]['calibration.nll'] for j in range(2)]
    eligible=[j for j,v in enumerate(values) if math.isfinite(v) and v<=metrics[0]['calibration.nll']]
    assert eligible, f'{family}: both learning rates failed'
    best=min(eligible,key=lambda j:values[j])
    relative=abs(values[0]-values[1])/min(values)
    if len(eligible)==2 and relative<.01: best=0
    selected[family]=[1e-4,3e-4][best]
    comparisons[family]={'nlls':values,'relative_gap':relative,'selected':selected[family]}
slice_pass=all(metrics[0][f'dev.{t}.accuracy']<.9 for t in ('piqa','boolq')) and any(
    m['calibration.nll']<metrics[0]['calibration.nll'] for m in metrics[1:3])
tok=AutoTokenizer.from_pretrained(MODEL,revision=REVISION,local_files_only=True)
tok.pad_token_id=tok.eos_token_id
splits=prepare(root/'data/v1')
batches=plan_batches(splits,tok,2000000,2,max_epochs=3)
steps=len(batches); tokens=sum(len(e['input_ids']) for b in batches for e in b)
schedule={'steps':steps,'tokens':tokens,'overshoot':max(0,tokens-2000000),
          'examples':sum(map(len,batches)),'last_epoch':batches[-1][-1]['epoch'],
          'dev_steps':[math.ceil(steps*f) for f in (.25,.5,.75)],'effective_batch':2,'microbatch':1}
# Costs are planning estimates from calibration, not measured E-0003 runs.
costs={}
for arm, family, factor in [('F_out',3,1),('F_pre',3,1),('C_gate',5,1),('C_up',5,1),
                            ('C_dual',5,1),('linear',5,1),('lora',1,1),('lora28',1,7)]:
    pair=metrics[family:family+2]
    step_s=max(m['train.steady_step_seconds'] for m in pair)*factor
    # Dev evaluations at25/50/75% plus final full eval/reload. Final eval includes calibration
    # conservatively, though E3 need only dev. Scale interim dev by2000/6096 examples.
    evaluation=max(m['eval.seconds'] for m in pair)*factor
    expected=steps*step_s + 3*evaluation*2000/6096 + 2*evaluation + 15
    request=math.ceil(expected*1.5/60)*60
    costs[arm]={'step_seconds_proxy':step_s,'expected_seconds':expected,'walltime_seconds':request,
                'expected_mig_hours':expected/3600,'expected_node_hours':expected/14400,
                'requested_mig_hours':request/3600,'requested_node_hours':request/14400,
                'basis':'family measured proxy' if factor==1 else 'unmeasured conservative7x four-layer LoRA proxy; smoke required'}
report={'selected_lr':selected,'comparisons':comparisons,'slice_pass':slice_pass,
        'schedule':schedule,'costs':costs,'base':metrics[0],
        'training_tokens':sum(m.get('train.input_tokens',0) for m in metrics),
        'task_sizes':{t:{s:len(v) for s,v in data.items()} for t,data in splits['splits'].items()}}
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
(out/'manifest.json').write_text(json.dumps({'status':'complete','spec_digest':os.environ['SR_SPEC_DIGEST'],
    'metrics':{'slice_pass':slice_pass,**{f'lr.{k}':v for k,v in selected.items()},
               'e3.steps':steps,'e3.tokens':tokens,'train.input_tokens':0},
    'artifacts':[{'path':str(out/'report.json'),'role':'selection_and_release'}],
    'validator':{'name':'deterministic_v1_audit','passed':True}},indent=2)+'\n')
print(json.dumps(report,indent=2))
