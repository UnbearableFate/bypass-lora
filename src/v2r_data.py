"""E-0014 T1 fact-count variants E in {1, 4, 8}: new data versions, V2R seed namespace.

Rendering, templates, the V2 seen name pool (checksum-pinned bytes), single-token
answer checks and the semantic content hash are those of V2 T1; only the number of
facts per episode changes. The V2 generator and its released data are untouched.
Confirmation is recorded as identities only: no content is generated for it.

Deduplication (cross-split, as in V2): dev, then calibration, then train are
generated. A dev or calibration candidate whose semantic content already exists in
dev or calibration is redrawn with the next attempt number; a train candidate whose
content exists in dev or calibration is redrawn. Repeated contents inside train are
kept and counted: at E=1 there are only 92*4*92 possible contents, so forcing
unique train contents would distort the frequency-weighted tail distribution.
Run generation inside PBS.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random

from v1_data import derive_seed
from v2_data import (MODEL, REVISION, RELATIONS, canonical, digest, encode,
                     file_digest, visible_delimiters)

VARIANTS = (1, 4, 8)
SPLITS = ('train', 'calibration', 'dev', 'confirmation')
RELEASED = ('dev', 'calibration', 'train')  # generation order; see module docstring
SIZES = {'train': 16384, 'calibration': 2048, 'dev': 1000, 'confirmation': 2000}
NAME_POOLS_SHA256 = 'a6bde954098e6bee7452abba1eabc56e7d0bf811b4ab220a5106c95c216ef57b'
MAX_ATTEMPTS = 1000
# metadata.json SHA-256 of each released full dataset; pinned after the
# generation run so that training refuses any other bytes.
METADATA_SHA256 = {1: None, 4: None, 8: None}


def version(n_facts):
    if n_facts not in VARIANTS:
        raise ValueError('E-0014 variants are E in {1, 4, 8}; E=16 is the V2 data')
    return f'concat-loop-v2-t1-e{n_facts}-1'


def identity(split, index, n_facts, *, support_order=None):
    if split not in SPLITS or type(index) is not int or index < 0:
        raise ValueError('invalid split/index')
    support_order = ('natural' if index % 2 == 0 else 'shuffled') if support_order is None else support_order
    if support_order not in ('natural', 'shuffled'):
        raise ValueError('invalid support order')
    return {'study_id': 'concat_loop_v2', 'phase': 'V2R', 'generator_version': version(n_facts),
            'task': 'T1', 'layout': {'hop': 1, 'n_facts': n_facts, 'relations': list(RELATIONS),
                                     'name_pool': 'seen', 'support_order': support_order},
            'base_seed': derive_seed(20261029, 'concat_loop_v2', 'V2R', split, n_facts),
            'split': split, 'index': index}


def load_name_pools(path):
    if file_digest(path) != NAME_POOLS_SHA256:
        raise ValueError('V2 name-pool checksum mismatch')
    return json.loads(Path(path).read_text())


def relations_for(rng, n_facts):
    """E=1: one relation uniformly; E=4: each once; E=8: each twice (shuffled)."""
    if n_facts == 1:
        return [rng.choice(list(RELATIONS))]
    relations = list(RELATIONS) * (n_facts // 4)
    rng.shuffle(relations)
    return relations


def episode_content(ident, pools, attempt=0):
    if ident != identity(ident['split'], ident['index'], ident['layout']['n_facts'],
                         support_order=ident['layout']['support_order']):
        raise ValueError('identity does not match the generator contract')
    if type(attempt) is not int or not 0 <= attempt < MAX_ATTEMPTS:
        raise ValueError('invalid generation attempt')
    layout = ident['layout']
    n = layout['n_facts']
    core_layout = {k: v for k, v in layout.items() if k != 'support_order'}
    rng = random.Random(derive_seed(ident['base_seed'], ident['generator_version'], core_layout,
                                    ident['index'], attempt))
    names = pools['pools']['seen']
    heads = rng.sample([x['name'] for x in names], n)  # distinct within the episode
    # Tails are frequency-weighted and independent of heads (self edges allowed).
    tails = rng.choices([x['name'] for x in names], weights=[x['sampling_weight'] for x in names], k=n)
    relations = relations_for(rng, n)
    facts = [{'head': h, 'relation': r, 'tail': t} for h, r, t in zip(heads, relations, tails)]
    target = rng.randrange(n)
    query = dict(facts[target])
    order = list(range(n))
    if layout['support_order'] == 'shuffled':
        random.Random(derive_seed(ident['base_seed'], 'support-order', ident['index'])).shuffle(order)
    ordered = [facts[i] for i in order]
    content_hash = digest({'facts': sorted((x['head'], x['relation'], x['tail']) for x in facts),
                           'question': {'head': query['head'], 'relation': query['relation'], 'hop': 1}})
    return ordered, order.index(target), query, content_hash


def make_episode(tok, pools, ident, attempt=0):
    if ident['split'] == 'confirmation':
        raise ValueError('confirmation content is sealed; record identities only')
    n = ident['layout']['n_facts']
    facts, target, query, content_hash = episode_content(ident, pools, attempt)
    support = ''.join(f"{f['head']} {f['relation']} {f['tail']} ;\n" for f in facts)
    question = f"Who does {query['head']} {RELATIONS[query['relation']]}?\nAnswer:"
    support_ids, positions = visible_delimiters(tok, support)
    if len(positions) != n:
        raise ValueError('each fact must have one visible delimiter')
    answer_text = ' ' + query['tail']
    answer_ids = encode(tok, answer_text)
    if len(answer_ids) != 1:
        raise ValueError('answer is not a single token')
    question_ids = encode(tok, question)
    full_ids = encode(tok, support + question)
    for prompt, ids in ((question, question_ids), (support + question, full_ids)):
        if encode(tok, prompt + answer_text) != ids + answer_ids:
            raise ValueError('answer boundary changes prompt tokens')
    for fact, position in zip(facts, positions):
        fact.update(position)
    info = {x['name']: x for x in pools['pools']['seen']}
    return {'identity': ident, 'episode_id': digest(ident), 'content_sha256': content_hash,
            'generation_attempt': attempt,
            'support_text': support, 'support_input_ids': support_ids,
            'question_text': question, 'question_input_ids': question_ids, 'facts': facts,
            'query': {'head': query['head'], 'relation': query['relation'], 'target_fact_index': target,
                      'answer_text': answer_text, 'answer_token_id': answer_ids[0]},
            'axes': {'hop': 1, 'n_facts': n, 'n_distractors': n - 1, 'name_pool': 'seen',
                     'support_order': ident['layout']['support_order'], 'relation': query['relation'],
                     'answer_frequency_stratum': info[query['tail']]['frequency_stratum'],
                     'answer_bare_tokens': info[query['tail']]['bare_tokens'],
                     'answer_leading_space_tokens': 1, 'support_tokens': len(support_ids),
                     'question_tokens': len(question_ids), 'full_prompt_tokens': len(full_ids),
                     'support_characters': len(support), 'question_characters': len(question)}}


def sealed_record(ident):
    if ident['split'] != 'confirmation':
        raise ValueError('sealed_record requires confirmation')
    return {'identity': ident, 'episode_id': digest(ident)}


def released_rows(tok, pools, n_facts, sizes):
    """Yield (split, row, rejected_attempts) in generation order with deduplication."""
    evaluation = set()
    for split in RELEASED:
        for index in range(sizes[split]):
            ident = identity(split, index, n_facts)
            for attempt in range(MAX_ATTEMPTS):
                # Content hash first: rendering is only needed for accepted draws.
                content_hash = episode_content(ident, pools, attempt)[3]
                if content_hash not in evaluation:
                    break
            else:
                raise ValueError('deduplication exhausted the attempt budget')
            if split != 'train':
                evaluation.add(content_hash)
            yield split, make_episode(tok, pools, ident, attempt), attempt


def generate_dataset(tok, output, n_facts, *, v2_data_dir, sizes=None, pools=None):
    """Write released splits and the confirmation identity registry; generate twice."""
    output, v2_data_dir = Path(output), Path(v2_data_dir)
    output.mkdir(parents=True, exist_ok=True)
    sizes = dict(SIZES if sizes is None else sizes)
    if set(sizes) != set(SPLITS) or any(type(n) is not int or n <= 0 for n in sizes.values()):
        raise ValueError('all four splits must have positive integer sizes')
    if pools is None:
        pools = load_name_pools(v2_data_dir / 'name_pools.json')
        (output / 'name_pools.json').write_bytes((v2_data_dir / 'name_pools.json').read_bytes())
    else:  # tests only: an explicit pool is written as given
        (output / 'name_pools.json').write_text(canonical(pools) + '\n')
    metadata = {'version': version(n_facts), 'phase': 'V2R', 'n_facts': n_facts, 'model': MODEL,
                'tokenizer_revision': REVISION, 'sizes': sizes, 'splits': {},
                'seed_rule': 'base_seed = derive_seed(20261029, "concat_loop_v2", "V2R", split, E)',
                'write_scope': 'visible_fact_delimiter', 'read_scope': 'all_valid_question',
                'name_pools_sha256': file_digest(output / 'name_pools.json'),
                'frequency_definition': pools['frequency_definition'],
                'generation_order': list(RELEASED),
                'deduplication': ('dev/calibration contents unique within and across both; train '
                                  'contents disjoint from dev/calibration; repeated train contents kept'),
                'confirmation': 'identity registry only; no content generated or hashed'}
    handles = {s: (output / f'{s}.jsonl').open('wb') for s in RELEASED}
    hashes = {s: hashlib.sha256() for s in RELEASED}
    stats = {s: {'axes': defaultdict(Counter), 'entities': Counter(), 'tokens': 0, 'contents': Counter(),
                 'rejected_attempts': 0, 'rows_redrawn': 0, 'relations': Counter(), 'n_facts': Counter()}
             for s in RELEASED}
    ids, answer_counts, dev_answers = set(), Counter(), []
    try:
        for split, row, attempt in released_rows(tok, pools, n_facts, sizes):
            if row['episode_id'] in ids:
                raise ValueError('duplicate episode identity')
            ids.add(row['episode_id'])
            line = (canonical(row) + '\n').encode()
            handles[split].write(line)
            hashes[split].update(line)
            st = stats[split]
            st['rejected_attempts'] += attempt
            st['rows_redrawn'] += attempt > 0
            st['contents'][row['content_sha256']] += 1
            st['n_facts'][len(row['facts'])] += 1
            for axis, value in row['axes'].items():
                st['axes'][axis][str(value)] += 1
            for fact in row['facts']:
                st['entities'][fact['head']] += 1
                st['entities'][fact['tail']] += 1
                st['relations'][fact['relation']] += 1
            st['tokens'] += len(row['support_input_ids']) + len(row['question_input_ids']) + 1
            if split == 'train':
                answer_counts[str(row['query']['answer_token_id'])] += 1
            elif split == 'dev':
                dev_answers.append(str(row['query']['answer_token_id']))
    finally:
        for handle in handles.values():
            handle.close()
    second = {s: hashlib.sha256() for s in RELEASED}
    for split, row, _ in released_rows(tok, pools, n_facts, sizes):
        second[split].update((canonical(row) + '\n').encode())
    released_contents = {s: set(stats[s]['contents']) for s in RELEASED}
    if released_contents['train'] & (released_contents['dev'] | released_contents['calibration']):
        raise ValueError('train shares semantic content with an evaluation split')
    if released_contents['dev'] & released_contents['calibration']:
        raise ValueError('dev and calibration share semantic content')
    for split in RELEASED:
        st = stats[split]
        path = output / f'{split}.jsonl'
        if hashes[split].hexdigest() != second[split].hexdigest() or file_digest(path) != hashes[split].hexdigest():
            raise ValueError('repeated generation hash mismatch')
        if split != 'train' and max(st['contents'].values()) > 1:
            raise ValueError('evaluation split repeats a semantic content')
        metadata['splits'][split] = {
            'path': str(path.resolve()), 'count': sizes[split], 'sha256': hashes[split].hexdigest(),
            'second_generation_sha256': second[split].hexdigest(), 'input_tokens': st['tokens'],
            'unique_contents': len(st['contents']),
            'repeated_content_rows': sizes[split] - len(st['contents']),
            'rows_redrawn': st['rows_redrawn'], 'rejected_attempts': st['rejected_attempts'],
            'fact_relation_counts': dict(st['relations']), 'facts_per_episode': dict(st['n_facts']),
            'axes': {k: dict(v) for k, v in st['axes'].items()},
            'support_entity_counts': dict(st['entities']), 'delimiter_checks': n_facts * sizes[split]}
    path = output / 'confirmation.registry.jsonl'
    with path.open('wb') as handle:
        for index in range(sizes['confirmation']):
            ident = identity('confirmation', index, n_facts)
            record = sealed_record(ident)
            if record['episode_id'] in ids:
                raise ValueError('duplicate episode identity')
            ids.add(record['episode_id'])
            handle.write((canonical(record) + '\n').encode())
    metadata['splits']['confirmation'] = {'path': str(path.resolve()), 'count': sizes['confirmation'],
                                          'sha256': file_digest(path), 'sealed': True}
    constant = min(answer_counts, key=lambda t: (-answer_counts[t], int(t)))
    metadata['prior'] = {'answer_token_id': int(constant), 'train_answer_counts': dict(answer_counts),
                         'dev_em': sum(x == constant for x in dev_answers) / len(dev_answers)}
    metadata['unique_identities'] = len(ids)
    (output / 'metadata.json').write_text(canonical(metadata) + '\n')
    return metadata


def load_variant(directory, n_facts, split):
    """Verify pinned metadata, split bytes, identities and counts before use."""
    if split not in ('train', 'calibration', 'dev'):
        raise ValueError('only train/calibration/dev are released')
    pinned = METADATA_SHA256.get(n_facts)
    directory = Path(directory)
    if pinned is None or file_digest(directory / 'metadata.json') != pinned:
        raise ValueError(f'E={n_facts} metadata is not the pinned release')
    metadata = json.loads((directory / 'metadata.json').read_text())
    entry = metadata['splits'][split]
    if metadata['version'] != version(n_facts) or metadata['n_facts'] != n_facts:
        raise ValueError('variant version mismatch')
    payload = (directory / f'{split}.jsonl').read_bytes()
    if hashlib.sha256(payload).hexdigest() != entry['sha256']:
        raise ValueError(f'{split} data hash mismatch')
    rows = [json.loads(line) for line in payload.splitlines() if line.strip()]
    if len(rows) != entry['count'] or len({r['episode_id'] for r in rows}) != len(rows):
        raise ValueError('data split/count mismatch')
    for row in rows:
        ident = row['identity']
        if (ident['split'] != split or ident['generator_version'] != version(n_facts)
                or len(row['facts']) != n_facts or ident['phase'] != 'V2R'):
            raise ValueError('row is not a released E-0014 variant row')
    return metadata, rows
