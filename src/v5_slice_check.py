"""D-0015 diagnostic: replace two 1g calibrations and recompute a separate lock."""
import argparse
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

from v5_train import CONTRACT, file_sha256, lock_lrs, write_json


PAIRS = (('V5-C',1e-3),('V5-L',1e-4))


def analyze(manifests,original_lock,output):
    original_lock = Path(original_lock).resolve()
    before = file_sha256(original_lock)
    original = json.loads(original_lock.read_text())
    if original['contract'] != CONTRACT:
        raise ValueError('original lock contract mismatch')
    output = Path(output).resolve()
    output.mkdir(parents=True,exist_ok=True)
    check_path = output/'lr-lock-2g-check.json'
    if check_path.resolve() == original_lock:
        raise ValueError('refusing to overwrite the formal LR lock')
    index,plans = {},set()
    for path in manifests:
        path = Path(path).resolve()
        value = json.loads(path.read_text())
        metrics = value['metrics']
        if not metrics['hardware'].endswith('MIG 2g.24gb'):
            raise ValueError('all replacement-set calibrations must report2g.24gb')
        key = (value['arm'],metrics['learning_rate'])
        if key in index:
            raise ValueError('duplicate calibration candidate')
        index[key] = (path,value)
        plans.add((metrics['plan.order_sha256'],metrics['plan.steps'],metrics['plan.input_tokens']))
    if len(plans) != 1:
        raise ValueError('replacement-set orders or budgets differ')
    comparisons = []
    for arm,lr in PAIRS:
        refs = [ref for ref in original['calibration_manifests'] if ref['arm']==arm and ref['lr']==lr]
        if len(refs) != 1 or file_sha256(refs[0]['path']) != refs[0]['sha256']:
            raise ValueError('original calibration provenance mismatch')
        old_path = Path(refs[0]['path']).resolve()
        old = json.loads(old_path.read_text())
        new_path,new = index[(arm,lr)]
        if (not old['metrics']['hardware'].endswith('MIG 1g.24gb')
                or new['spec_digest'] != old['spec_digest']):
            raise ValueError('rerun must preserve original1g scientific RunSpec')
        profile_path = new_path.parent/'mig-profile.json'
        profile = json.loads(profile_path.read_text())
        if (profile.get('passed') is not True or profile.get('profile') != '2g.24gb'
                or profile.get('uuid') != profile.get('cuda_visible_devices')
                or str(profile_path) not in {a.get('path') for a in new['artifacts']}):
            raise ValueError('rerun requires recorded allocated2g guard evidence')
        old_nll,new_nll = old['metrics']['calibration.nll'],new['metrics']['calibration.nll']
        if not all(isinstance(v,(int,float)) and math.isfinite(v) for v in (old_nll,new_nll)):
            raise ValueError('finite NLL pair required')
        comparisons.append({'arm':arm,'lr':lr,'original_manifest':str(old_path),
                            'rerun_manifest':str(new_path),'nll_1g':old_nll,'nll_2g':new_nll,
                            'delta_2g_minus_1g':new_nll-old_nll,'guard':str(profile_path)})
    # Use the reviewed selection implementation; never pass it the formal path.
    lock_lrs(SimpleNamespace(manifests=[Path(p) for p in manifests],output=check_path))
    check = json.loads(check_path.read_text())
    if file_sha256(original_lock) != before:
        raise ValueError('formal LR lock changed during diagnostic')
    same = check['locked_lrs'] == original['locked_lrs']
    return {'decision':'D-0015','contract':CONTRACT,'comparisons':comparisons,
            'original_locked_lrs':original['locked_lrs'],'check_locked_lrs':check['locked_lrs'],
            'choices_match':same,'formal_lock':{'path':str(original_lock),'sha256':before},
            'formal_lock_unchanged':True,'check_lock':str(check_path),
            'interpretation':('LR selection robust to slice in this two-point diagnostic' if same else
                              'LR selection is slice-sensitive;retain formal lock and report limitation perD0015')}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifests',nargs='+',type=Path,required=True)
    parser.add_argument('--original-lock',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True,exist_ok=True)
    passed = False
    try:
        report = analyze(args.manifests,args.original_lock,out)
        write_json(out/'slice-sensitivity.json',report)
        passed = True
    finally:
        write_json(out/'manifest.json',{'status':'complete' if passed else 'partial',
                   'spec_digest':os.environ['SR_SPEC_DIGEST'],'metrics':{'slice_check.complete':passed},
                   'artifacts':[{'path':str(out/name),'role':name[:-5]} for name in
                                ('slice-sensitivity.json','lr-lock-2g-check.json')],
                   'validator':{'name':'v5_slice_sensitivity','passed':passed}})


if __name__ == '__main__':
    main()
