"""Deterministic concat_loop_v2 T1 data. No model inference or legacy code.

Only V2 T1 seen-name data is released here. Other layouts have identities but
cannot be materialized. Confirmation is emitted only as identities and hashes.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random

from v1_data import derive_seed

MODEL = 'Qwen/Qwen3-0.6B-Base'
REVISION = 'da87bfb608c14b7cf20ba1ce41287e8de496c0cd'
VERSION = 'concat-loop-v2-t1-1'
SPLITS = ('train', 'calibration', 'dev', 'confirmation')
SIZES = {'train': 16384, 'calibration': 2048, 'dev': 1000, 'confirmation': 2000}
RELATIONS = {'emails': 'email', 'photographs': 'photograph', 'tutors': 'tutor', 'rescues': 'rescue'}
# New hand-curated candidates, not copied from the old repository. Tokenizer
# eligibility, not model performance, determines the retained pool.
CANDIDATES = '''Aaron Adam Adrian Alan Albert Alex Alexander Alice Amanda Amy
Andrew Angela Anna Anne Anthony Arthur Ashley Austin Barbara Benjamin Betty
Blake Bob Bobby Brian Bruce Caleb Carl Carlos Carol Caroline Catherine Charles
Charlotte Chris Christian Christina Christopher Claire Clara Daniel David Dean
Dennis Diana Donald Donna Dorothy Douglas Edward Elizabeth Emily Emma Eric Ethan
Eva Evelyn Frank Gabriel Gary George Grace Gregory Hannah Harry Helen Henry Ian
Isaac Jack Jacob James Jane Janet Jason Jean Jeffrey Jennifer Jeremy Jerry Jesse
Jessica Joan Joe John Jonathan Jordan Joseph Joshua Joyce Julia Julie Justin
Karen Katherine Kathleen Kathryn Keith Kelly Kenneth Kevin Laura Lauren Lawrence
Linda Lisa Logan Louis Lucas Lucy Luke Madison Margaret Maria Marie Marilyn Mark
Martha Martin Mary Mason Matthew Megan Melissa Michael Michelle Nancy Natalie
Nathan Nicholas Nicole Noah Oliver Olivia Oscar Pamela Patricia Patrick Paul
Peter Philip Rachel Ralph Raymond Rebecca Richard Robert Roger Ronald Rose Roy
Ruby Russell Ruth Ryan Samuel Sandra Sarah Scott Sean Sharon Shirley Simon
Sophia Stephanie Stephen Steven Susan Teresa Terry Thomas Timothy Tina Tyler
Victoria Vincent Virginia Walter Wayne William Zachary Zoe'''.split()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def encode(tok, text):
    return tok.encode(text, add_special_tokens=False)


def make_name_pools(tok):
    """Stratify by bare-token length and *designed synthetic* frequency.

    Frequency is a controlled 3:1 answer-sampling weight, not an estimate of
    pretraining/name popularity. Leading-space token length is always one.
    """
    buckets = defaultdict(list)
    rejected = []
    for name in sorted(set(CANDIDATES)):
        ids = encode(tok, ' ' + name)
        if len(ids) != 1 or tok.decode(ids) != ' ' + name:
            rejected.append(name)
            continue
        buckets[len(encode(tok, name))].append({'name': name, 'answer_token_id': ids[0],
                                               'bare_tokens': len(encode(tok, name)),
                                               'leading_space_tokens': 1})
    pools = {'seen': [], 'unseen': []}
    for length, names in sorted(buckets.items()):
        names.sort(key=lambda x: (derive_seed(20261029, 'concat_loop_v2', 'V2', 'names', x['name']), x['name']))
        for i, entry in enumerate(names):
            # Each block of four has both frequency strata in each name pool.
            entry = dict(entry, frequency_stratum='high' if i % 2 == 0 else 'low',
                         sampling_weight=3 if i % 2 == 0 else 1)
            pools['seen' if i % 4 < 2 else 'unseen'].append(entry)
    for pool in pools.values():
        pool.sort(key=lambda x: x['name'])
        if len(pool) < 32:
            raise ValueError('not enough tokenizer-validated names in a pool')
    return {'pools': pools, 'rejected_candidates': rejected,
            'frequency_definition': 'controlled synthetic answer weight high:low=3:1; not corpus frequency',
            'tokenizer_revision': REVISION}


def identity(split, index, *, hop=1, n_facts=16, name_pool='seen', support_order='natural'):
    if split not in SPLITS or not isinstance(index, int) or isinstance(index, bool) or index < 0:
        raise ValueError('invalid split/index')
    allowed = (1, 2, 3, 4, 6, 8) if split == 'confirmation' else (1, 2, 3)
    if hop not in allowed or n_facts < hop or n_facts % 4:
        raise ValueError('invalid hop or fact count')
    if name_pool not in ('seen', 'unseen') or (name_pool == 'unseen' and split != 'confirmation'):
        raise ValueError('unseen names are sealed')
    if support_order not in ('natural', 'shuffled'):
        raise ValueError('invalid support order')
    return {'study_id': 'concat_loop_v2', 'phase': 'V2', 'generator_version': VERSION,
            'task': 'T1' if hop == 1 else 'typed_walk',
            'layout': {'hop': hop, 'n_facts': n_facts, 'relations': list(RELATIONS),
                       'name_pool': name_pool, 'support_order': support_order},
            'base_seed': derive_seed(20261029, 'concat_loop_v2', 'V2', split),
            'split': split, 'index': index}


def episode_content(ident, pools):
    """Pure structural content; order RNG never changes the graph or question."""
    layout = ident['layout']
    if ident['phase'] != 'V2' or layout['hop'] != 1 or layout['n_facts'] != 16 or layout['name_pool'] != 'seen':
        raise ValueError('only V2 T1 E16 seen layout is materialized; other layouts are reserved')
    core_layout = {k: v for k, v in layout.items() if k != 'support_order'}
    rng = random.Random(derive_seed(ident['base_seed'], VERSION, core_layout, ident['index']))
    names = pools['pools']['seen']
    heads = rng.sample([x['name'] for x in names], 16)
    # Draw tails independently of heads and query selection (self edges allowed).
    tails = rng.choices([x['name'] for x in names], weights=[x['sampling_weight'] for x in names], k=16)
    relations = list(RELATIONS) * 4
    rng.shuffle(relations)
    facts = [{'head': h, 'relation': r, 'tail': t} for h, r, t in zip(heads, relations, tails)]
    target = rng.randrange(16)
    query = dict(facts[target])
    order = list(range(16))
    if layout['support_order'] == 'shuffled':
        random.Random(derive_seed(ident['base_seed'], 'support-order', ident['index'])).shuffle(order)
    ordered = [facts[i] for i in order]
    # Canonical semantic hash excludes rendering/order/identity. It detects a
    # duplicate even when support is permuted, split/seed/index are changed.
    content_hash = digest({'facts': sorted((x['head'], x['relation'], x['tail']) for x in facts),
                           'question': {'head': query['head'], 'relation': query['relation'], 'hop': 1}})
    return ordered, order.index(target), query, content_hash


def visible_delimiters(tok, support_text):
    """Return positions from text only; no target/answer is accepted."""
    encoded = tok(support_text, add_special_tokens=False, return_offsets_mapping=True)
    positions = []
    for char, value in enumerate(support_text):
        if value != ';':
            continue
        matches = [i for i, (a, b) in enumerate(encoded['offset_mapping']) if a <= char < b]
        if len(matches) != 1:
            raise ValueError('delimiter must be covered by exactly one token')
        positions.append({'delimiter_char_span': [char, char + 1], 'delimiter_token_index': matches[0]})
    if len({p['delimiter_token_index'] for p in positions}) != len(positions):
        raise ValueError('multiple facts share a delimiter token')
    return list(encoded['input_ids']), positions


def make_episode(tok, pools, ident):
    if ident['split'] == 'confirmation':
        raise ValueError('confirmation content is sealed; use sealed_record')
    facts, target, query, content_hash = episode_content(ident, pools)
    support = ''.join(f"{f['head']} {f['relation']} {f['tail']} ;\n" for f in facts)
    question = f"Who does {query['head']} {RELATIONS[query['relation']]}?\nAnswer:"
    support_ids, positions = visible_delimiters(tok, support)
    if len(positions) != 16:
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
            'support_text': support, 'support_input_ids': support_ids,
            'question_text': question, 'question_input_ids': question_ids, 'facts': facts,
            'query': {'head': query['head'], 'relation': query['relation'], 'target_fact_index': target,
                      'answer_text': answer_text, 'answer_token_id': answer_ids[0]},
            'axes': {'hop': 1, 'n_facts': 16, 'n_distractors': 15, 'name_pool': 'seen',
                     'support_order': ident['layout']['support_order'], 'relation': query['relation'],
                     'answer_frequency_stratum': info[query['tail']]['frequency_stratum'],
                     'answer_bare_tokens': info[query['tail']]['bare_tokens'],
                     'answer_leading_space_tokens': 1, 'support_tokens': len(support_ids),
                     'question_tokens': len(question_ids), 'full_prompt_tokens': len(full_ids),
                     'support_characters': len(support), 'question_characters': len(question)}}


def sealed_record(ident, pools):
    if ident['split'] != 'confirmation':
        raise ValueError('sealed_record requires confirmation')
    _, _, _, content_hash = episode_content(ident, pools)
    return {'identity': ident, 'episode_id': digest(ident), 'content_sha256': content_hash}


def generate_dataset(tok, output, *, sizes=None):
    """Stream released data twice, checking hashes without saving a second copy."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    sizes = dict(SIZES if sizes is None else sizes)
    if set(sizes) != set(SPLITS) or any(type(n) is not int or n <= 0 for n in sizes.values()):
        raise ValueError('all four splits must have positive integer sizes')
    pools = make_name_pools(tok)
    (output / 'name_pools.json').write_text(canonical(pools) + '\n')
    ids, contents = set(), set()
    metadata = {'version': VERSION, 'model': MODEL, 'tokenizer_revision': REVISION,
                'sizes': sizes, 'splits': {}, 'write_scope': 'visible_fact_delimiter',
                'read_scope': 'all_valid_question', 'frequency_definition': pools['frequency_definition'],
                'name_pools_sha256': file_digest(output / 'name_pools.json'),
                'legacy_overlap': {'count': 0, 'method': 'construction: disjoint relation labels',
                                   'scope': 'labeled fact-set plus question; not graph isomorphism',
                                   'legacy_commit': 'a255e814740ff85ab3b4ec4836c88f0a8f802b2d',
                                   'proof': 'research/notes/T-0005-data-contract.md'}}
    answer_counts = Counter()
    dev_answers = []
    for split in SPLITS:
        path = output / (split + ('.registry.jsonl' if split == 'confirmation' else '.jsonl'))
        axes = defaultdict(Counter)
        entity_counts = Counter()
        tokens = 0
        first_hash = hashlib.sha256()
        with path.open('wb') as handle:
            for i in range(sizes[split]):
                ident = identity(split, i, support_order='natural' if i % 2 == 0 else 'shuffled')
                row = sealed_record(ident, pools) if split == 'confirmation' else make_episode(tok, pools, ident)
                if row['episode_id'] in ids or row['content_sha256'] in contents:
                    raise ValueError('duplicate episode identity or semantic content across/within splits')
                ids.add(row['episode_id'])
                contents.add(row['content_sha256'])
                line = (canonical(row) + '\n').encode()
                handle.write(line)
                first_hash.update(line)
                if split != 'confirmation':
                    for axis, value in row['axes'].items():
                        axes[axis][str(value)] += 1
                    for fact in row['facts']:
                        entity_counts[fact['head']] += 1
                        entity_counts[fact['tail']] += 1
                    tokens += len(row['support_input_ids']) + len(row['question_input_ids']) + 1
                    if split == 'train':
                        answer_counts[str(row['query']['answer_token_id'])] += 1
                    elif split == 'dev':
                        dev_answers.append(str(row['query']['answer_token_id']))
        second_hash = hashlib.sha256()
        for i in range(sizes[split]):
            ident = identity(split, i, support_order='natural' if i % 2 == 0 else 'shuffled')
            row = sealed_record(ident, pools) if split == 'confirmation' else make_episode(tok, pools, ident)
            second_hash.update((canonical(row) + '\n').encode())
        if first_hash.hexdigest() != second_hash.hexdigest() or file_digest(path) != first_hash.hexdigest():
            raise ValueError('repeated generation hash mismatch')
        entry = {'path': str(path.resolve()), 'count': sizes[split], 'sha256': first_hash.hexdigest(),
                 'second_generation_sha256': second_hash.hexdigest()}
        if split != 'confirmation':
            entry.update(input_tokens=tokens, axes={k: dict(v) for k, v in axes.items()},
                         support_entity_counts=dict(entity_counts), delimiter_checks=16 * sizes[split])
        metadata['splits'][split] = entry
    # Train only chooses the constant; dev merely scores that frozen choice.
    constant = min(answer_counts, key=lambda t: (-answer_counts[t], int(t)))
    metadata['prior'] = {'answer_token_id': int(constant), 'train_answer_counts': dict(answer_counts),
                         'dev_em': sum(x == constant for x in dev_answers) / len(dev_answers)}
    metadata['train_budget'] = {'arms_share_identical_data': 8, 'tokens_per_arm': 4000000,
                                'max_epochs': 3, 'available_tokens_3_epochs': 3 * metadata['splits']['train']['input_tokens']}
    metadata['unique_identities'] = len(ids)
    metadata['unique_semantic_contents'] = len(contents)
    # Future hops/unseen layouts are schema-only. No OOD episodes are generated.
    metadata['reserved_layouts'] = [{'split': split, 'hop': hop, 'name_pool': pool}
        for split, hops, pool in [('train', (2, 3), 'seen'), ('dev', (2, 3), 'seen'),
                                  ('confirmation', (4, 6, 8), 'seen'), ('confirmation', (1, 2, 3), 'unseen')]
        for hop in hops]
    (output / 'metadata.json').write_text(canonical(metadata) + '\n')
    return metadata
