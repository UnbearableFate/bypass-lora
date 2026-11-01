"""RB A2 split and paired training order. No confirmation contents are exported."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random

from v1_data import derive_seed, encode_training_example

STUDY = 'residual_la_v21'
BASE_SEED = 20261101
V1_MAPPING_SHA256 = '47cf5c777d07863b8def6903487697eafc970b0cd5b5d80babb7f20ded98571b'


def canonical(value):
    return json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False)


def confirmation_manifest(splits):
    seed = derive_seed(BASE_SEED,STUDY,'RB5','confirmation')
    identities = {}
    for task in ('piqa','boolq'):
        rows = splits['splits'][task]['train']
        ids = [r['id'] for r in rows]
        if len(ids) != len(set(ids)) or len(ids) < 1000:
            raise ValueError('confirmation source needs at least1000 unique train IDs per task')
        selected = set(random.Random(seed).sample(ids,1000))
        identities[task] = [i for i in ids if i in selected]
    return {'schema_version':1,'study_id':STUDY,'decision':'D-0025 A2',
        'source_split_mapping_sha256':V1_MAPPING_SHA256,'seed':seed,
        'sampling':'Uniform sample of1000 IDs per task from V1 train;fresh Random(seed) per task;source order in manifest',
        'ids':identities,'counts':{t:len(ids) for t,ids in identities.items()},
        'ids_sha256':hashlib.sha256(canonical(identities).encode()).hexdigest(),
        'sealed':True,'model_evaluated':False}


def apply_confirmation(splits, manifest):
    # Regenerate IDs before any training encoding, rejecting altered/foreign maps.
    if manifest != confirmation_manifest(splits):
        raise ValueError('RB confirmation manifest differs from fixed A2 selection')
    result = {'splits':{},'metadata':deepcopy(splits['metadata'])}
    for task in ('piqa','boolq'):
        held = set(manifest['ids'][task])
        result['splits'][task] = dict(splits['splits'][task])
        result['splits'][task]['train'] = [r for r in splits['splits'][task]['train'] if r['id'] not in held]
        assert not held.intersection(r['id'] for r in result['splits'][task]['train'])
    result['metadata']['rb_confirmation_ids_sha256'] = manifest['ids_sha256']
    result['metadata']['rb_train_counts'] = {t:len(result['splits'][t]['train']) for t in ('piqa','boolq')}
    return result


def training_stream(splits, phase, seed, epoch, excluded=()):
    rng = random.Random(derive_seed(BASE_SEED,STUDY,phase,'train',seed)+epoch)
    excluded = set(excluded)
    tasks = {}
    for task in ('piqa','boolq'):
        rows = list(splits['splits'][task]['train'])
        rng.shuffle(rows)
        tasks[task] = [r for r in rows if (task,r['id']) not in excluded]
    result = []
    for index in range(max(map(len,tasks.values()),default=0)):
        for task in tasks:
            if index < len(tasks[task]):
                result.append(tasks[task][index])
    return result


def plan_batches(splits, tok, phase, seed, excluded=(), token_budget=2000000, max_epochs=3):
    if token_budget is not None and token_budget <= 0 or max_epochs != 3:
        raise ValueError('RB requires positive token budget (or epochs) and3epoch cap')
    batches, pending, cache, tokens = [],[],{},0
    for epoch in range(max_epochs):
        stream = training_stream(splits,phase,seed,epoch,excluded)
        for index,row in enumerate(stream):
            key = row['task'],row['id']
            if key not in cache:
                cache[key] = encode_training_example(tok,row,max_length=512)
            pending.append(dict(cache[key],epoch=epoch,epoch_end=index==len(stream)-1))
            if len(pending) == 2:
                batches.append(pending)
                tokens += sum(len(e['input_ids']) for e in pending)
                pending=[]
                if token_budget is not None and tokens >= token_budget:
                    return batches
    if pending:
        batches.append(pending)
    return batches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    from v1_run import prepare
    import os
    manifest = confirmation_manifest(prepare(args.data_dir))
    path = args.output/'rb-confirmation.json'
    path.write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    (args.output/'manifest.json').write_text(json.dumps({'status':'complete',
        'spec_digest':os.environ['SR_SPEC_DIGEST'],'metrics':{'confirmation.rows':2000,
        'model_evaluated':False,'ids_sha256':manifest['ids_sha256']},
        'artifacts':[{'path':str(path.resolve()),'role':'sealed_confirmation_ids'}],
        'validator':{'name':'rb_A2_ids_only','passed':True}},indent=2)+'\n')


if __name__ == '__main__':
    main()
