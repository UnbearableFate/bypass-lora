"""Phase-parameterized T2 endpoint data; no inference or sealed text export.

Run generation and validation inside PBS. The V2 utilities and name-pool bytes
are shared, but the V2 generator and its released artifacts are never modified.
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

VERSION = 'concat-loop-v3-t2-1'
NAME_POOLS_SHA256 = 'a6bde954098e6bee7452abba1eabc56e7d0bf811b4ab220a5106c95c216ef57b'
RELEASED = ('train', 'calibration', 'dev')
SEALED = ('confirmation', 'ood_h4', 'ood_h6', 'ood_h8', 'ood_unseen')
SPLITS = RELEASED + SEALED
SIZES = {'train': None, 'calibration': 2048, 'dev': 1000, 'confirmation': 2000,
         'ood_h4': 1000, 'ood_h6': 1000, 'ood_h8': 1000, 'ood_unseen': 1000}
QUESTION_TEMPLATE = ('Starting from {head}, keep following the {relation} facts until '
                     'there is no further {relation} fact. Who is reached?\nAnswer:')


def compositions(total=12, maximum=4):
    """Ordered compositions, uniformly sampled (not uniform integer partitions)."""
    if total == 0:
        return [()]
    return [(n,) + rest for n in range(1, min(total, maximum) + 1)
            for rest in compositions(total - n, maximum)]


TRAIN_COMPOSITIONS = tuple(c for c in compositions() if sum(n >= 3 for n in c) >= 2)
OOD_COMPOSITIONS = {h: tuple(c for c in compositions(maximum=8) if max(c) >= h)
                    for h in (4, 6, 8)}


def identity(split, index, *, phase='V3', hop=None, support_order=None):
    if split not in SPLITS or type(index) is not int or index < 0:
        raise ValueError('invalid split/index')
    if phase not in ('V3', 'V4'):
        raise ValueError('phase must be V3 or V4')
    expected = int(split[-1]) if split.startswith('ood_h') else 1 + index % 3
    hop = expected if hop is None else hop
    if type(hop) is not int or (hop not in (1, 2, 3) if not split.startswith('ood_h') else hop != expected):
        raise ValueError('hop is outside this slice')
    support_order = ('natural' if index % 2 == 0 else 'shuffled') if support_order is None else support_order
    if support_order not in ('natural', 'shuffled'):
        raise ValueError('invalid support order')
    return {'study_id': 'concat_loop_v2', 'phase': phase, 'generator_version': VERSION,
            'task': 'T2', 'layout': {'hop': hop, 'n_facts': 16, 'relations': list(RELATIONS),
                                  'name_pool': 'unseen' if split == 'ood_unseen' else 'seen',
                                  'support_order': support_order},
            'base_seed': derive_seed(20261029, 'concat_loop_v2', phase, split),
            'split': split, 'index': index}


def _validate_identity(ident):
    expected = identity(ident['split'], ident['index'], phase=ident['phase'],
                        hop=ident['layout']['hop'], support_order=ident['layout']['support_order'])
    if ident != expected:
        raise ValueError('identity does not match the generator contract')


def load_name_pools(path):
    if file_digest(path) != NAME_POOLS_SHA256:
        raise ValueError('V2 name-pool checksum mismatch')
    return json.loads(Path(path).read_text())


def _weighted_sample(rng, entries, count):
    remaining = list(entries)
    result = []
    for _ in range(count):
        chosen = rng.choices(remaining, weights=[x['sampling_weight'] for x in remaining], k=1)[0]
        result.append(chosen['name'])
        remaining.remove(chosen)
    return result


def semantic_hash(facts, head, relation, *, endpoint=True):
    question = {'head': head, 'relation': relation}
    if endpoint:
        question['termination'] = 'no_further_queried_relation'
    return digest({'facts': sorted((f['head'], f['relation'], f['tail']) for f in facts),
                   'question': question})


def _structure(ident, pools):
    """Internal transient graph, also used to hash sealed records without rendering."""
    _validate_identity(ident)
    hop, index, base = ident['layout']['hop'], ident['index'], ident['base_seed']
    rng = random.Random(derive_seed(base, VERSION, 'graph', index))
    # Neither hop nor presentation order enters the in-distribution graph RNG.
    choices = TRAIN_COMPOSITIONS if hop <= 3 else OOD_COMPOSITIONS[hop]
    lengths = rng.choice(choices)
    relation = rng.choice(list(RELATIONS))
    entries = pools['pools'][ident['layout']['name_pool']]
    names = _weighted_sample(rng, entries, 12 + len(lengths))
    chains, offset = [], 0
    for length in lengths:
        chains.append(names[offset:offset + length + 1])
        offset += length + 1
    query_rng = random.Random(derive_seed(base, VERSION, 'query-start', index, hop))
    eligible = [i for i, chain in enumerate(chains) if len(chain) > hop]
    chosen = query_rng.choice(eligible)  # exactly one eligible start per chain
    path = chains[chosen][-hop - 1:]
    r_facts = [{'head': a, 'relation': relation, 'tail': b}
               for chain in chains for a, b in zip(chain, chain[1:])]
    noise_rng = random.Random(derive_seed(base, VERSION, 'distractors', index, hop))
    terminal_noise = index % 2 == 0
    other_heads = [name for i, chain in enumerate(chains) if i != chosen for name in chain]
    heads = ([path[-1]] if terminal_noise else []) + noise_rng.sample(other_heads, 3 if terminal_noise else 4)
    distractors = []
    for i, head in enumerate(heads):
        # The designated terminal edge must really make an untyped traversal
        # wrong: its tail is outside all chains, and hence has no outgoing fact.
        tail_pool = [x for x in entries if x['name'] not in names] if terminal_noise and i == 0 else entries
        tail = noise_rng.choices(tail_pool, weights=[x['sampling_weight'] for x in tail_pool], k=1)[0]['name']
        distractors.append({'head': head, 'relation': noise_rng.choice([r for r in RELATIONS if r != relation]),
                            'tail': tail})
    order_rng = random.Random(derive_seed(base, VERSION, 'support-order', index))
    noise_positions = set(order_rng.sample(range(16), 4))
    r_iter, d_iter = iter(r_facts), iter(distractors)
    facts = [next(d_iter) if i in noise_positions else next(r_iter) for i in range(16)]
    if ident['layout']['support_order'] == 'shuffled':
        order_rng.shuffle(facts)
    indices = [next(i for i, f in enumerate(facts)
                    if (f['head'], f['relation'], f['tail']) == (a, relation, b))
               for a, b in zip(path, path[1:])]
    return {'facts': facts, 'path': path, 'path_fact_indices': indices, 'relation': relation,
            'chains': chains, 'chain_lengths': lengths, 'terminal_distractor': terminal_noise,
            'content_sha256': semantic_hash(facts, path[0], relation)}


def make_episode(tok, pools, ident):
    if ident['split'] not in RELEASED:
        raise ValueError('sealed slices cannot be materialized')
    graph = _structure(ident, pools)
    facts, path = graph['facts'], graph['path']
    support = ''.join(f"{f['head']} {f['relation']} {f['tail']} ;\n" for f in facts)
    question = QUESTION_TEMPLATE.format(head=path[0], relation=graph['relation'])
    support_ids, positions = visible_delimiters(tok, support)
    question_ids, full_ids = encode(tok, question), encode(tok, support + question)
    answer_text = ' ' + path[-1]
    answer_ids = encode(tok, answer_text)
    if len(positions) != 16 or len(answer_ids) != 1:
        raise ValueError('invalid delimiters or answer token count')
    for prompt, ids in ((question, question_ids), (support + question, full_ids)):
        if encode(tok, prompt + answer_text) != ids + answer_ids:
            raise ValueError('single-token answer boundary mismatch')
    for fact, position in zip(facts, positions):
        fact.update(position)
    entry = next(x for x in pools['pools']['seen'] if x['name'] == path[-1])
    return {'identity': ident, 'episode_id': digest(ident), 'content_sha256': graph['content_sha256'],
            'support_text': support, 'support_input_ids': support_ids,
            'question_text': question, 'question_input_ids': question_ids, 'facts': facts,
            'query': {'head': path[0], 'relation': graph['relation'],
                      'target_fact_index': graph['path_fact_indices'][0],
                      'answer_text': answer_text, 'answer_token_id': answer_ids[0]},
            'path': {'entities': path, 'fact_indices': graph['path_fact_indices']},
            'axes': {'hop': len(path) - 1, 'n_facts': 16, 'n_distractors': 16 - (len(path) - 1),
                     'name_pool': 'seen', 'support_order': ident['layout']['support_order'],
                     'relation': graph['relation'], 'n_r_chains': len(graph['chains']),
                     'chain_length_composition': ','.join(map(str, sorted(graph['chain_lengths']))),
                     'terminal_distractor': graph['terminal_distractor'],
                     'distractor_positions': ','.join(str(i) for i, f in enumerate(facts) if f['relation'] != graph['relation']),
                     'answer_frequency_stratum': entry['frequency_stratum'],
                     'answer_bare_tokens': entry['bare_tokens'], 'answer_leading_space_tokens': 1,
                     'support_tokens': len(support_ids), 'question_tokens': len(question_ids),
                     'full_prompt_tokens': len(full_ids), 'support_characters': len(support),
                     'question_characters': len(question)}}


def sealed_record(ident, pools):
    if ident['split'] not in SEALED:
        raise ValueError('sealed_record requires a sealed slice')
    graph = _structure(ident, pools)
    return {'identity': ident, 'episode_id': digest(ident), 'content_sha256': graph['content_sha256']}


def _r_chains(row):
    relation = row['query']['relation']
    outgoing = {f['head']: f['tail'] for f in row['facts'] if f['relation'] == relation}
    tails = set(outgoing.values())
    chains = []
    for root in sorted(set(outgoing) - tails):
        chain = [root]
        while chain[-1] in outgoing:
            chain.append(outgoing[chain[-1]])
        chains.append(chain)
    return chains, outgoing


def shortcut_scores(row, constant):
    chains, outgoing = _r_chains(row)
    answer = row['query']['answer_text'].lstrip()
    endpoints = {chain[-1] for chain in chains}
    longest = [c[-1] for c in chains if len(c) == max(map(len, chains))]
    # Ignore relation labels completely. Last presented edge breaks a tie;
    # cycles count as incorrect. Generation makes the queried trajectory
    # unambiguous, with exactly half diverted at its endpoint.
    untyped = {f['head']: f['tail'] for f in row['facts']}
    current, visited = row['query']['head'], set()
    while current in untyped and current not in visited:
        visited.add(current)
        current = untyped[current]
    untyped_correct = current not in untyped and current == answer
    last = [f['tail'] for f in row['facts'] if f['relation'] == row['query']['relation'] and f['tail'] in endpoints][-1]
    return {'prior': float(row['query']['answer_token_id'] == constant),
            'random_r_endpoint': 1 / len(chains), 'ignore_relation': float(untyped_correct),
            'longest_r_chain': longest.count(answer) / len(longest),
            'last_r_endpoint': float(last == answer),
            'first_bridge': float(outgoing[row['query']['head']] == answer)}


def shortcut_report(rows, constant):
    sums, counts = defaultdict(Counter), Counter()
    for row in rows:
        hop, order = row['axes']['hop'], row['axes']['support_order']
        groups = ('all', f'hop={hop}', f'order={order}', f'hop={hop},order={order}')
        values = shortcut_scores(row, constant)
        for group in groups:
            counts[group] += 1
            sums[group].update(values)
    groups = {k: {'examples': counts[k], **{m: v / counts[k] for m, v in sums[k].items()}}
              for k in counts}
    checks = {f'hop={h}.{metric}': groups[f'hop={h}'][metric] <= limit
              for h in (2, 3) for metric, limit in
              [('prior', .05), ('random_r_endpoint', .30), ('ignore_relation', .55)]}
    return {'groups': groups, 'gates': checks, 'gates_passed': all(checks.values()),
            'ignore_relation_rule': 'last-presented outgoing edge; stop at no edge; cycles fail',
            'order_terminal_coupling': 'contract alternates both by index: natural has terminal distractor; shuffled does not'}


def _v2_inventory(directory):
    """Read original hashes and also normalize away differing task/hop schemas."""
    directory = Path(directory)
    metadata = json.loads((directory / 'metadata.json').read_text())
    actual, normalized, counts = set(), set(), {}
    for split, entry in metadata['splits'].items():
        path = directory / (split + ('.registry.jsonl' if split == 'confirmation' else '.jsonl'))
        if file_digest(path) != entry['sha256']:
            raise ValueError('V2 split checksum mismatch: ' + split)
        count = 0
        with path.open() as handle:
            for line in handle:
                row = json.loads(line)
                actual.add(row['content_sha256'])
                if 'facts' in row:
                    normalized.add(semantic_hash(row['facts'], row['query']['head'], row['query']['relation'], endpoint=False))
                count += 1
        if count != entry['count']:
            raise ValueError('V2 split count mismatch')
        counts[split] = count
    return actual, normalized, counts


def generate_dataset(tok, output, *, v2_data_dir, phase='V3', sizes=None, minimum_train_prompt_tokens=5500000):
    """Stream all slices twice. Sealed rows never contain text, paths or answers."""
    output, v2_data_dir = Path(output), Path(v2_data_dir)
    output.mkdir(parents=True, exist_ok=True)
    sizes = dict(SIZES if sizes is None else sizes)
    if set(sizes) != set(SPLITS) or any(type(n) is not int or n < 6 for k, n in sizes.items() if not (k == 'train' and n is None)):
        raise ValueError('all slices require at least six rows; train may be automatic')
    pools = load_name_pools(v2_data_dir / 'name_pools.json')
    (output / 'name_pools.json').write_bytes((v2_data_dir / 'name_pools.json').read_bytes())
    v2_actual, v2_normalized, v2_counts = _v2_inventory(v2_data_dir)
    ids, contents, normalized = set(), set(), set()
    metadata = {'version': VERSION, 'phase': phase, 'model': MODEL, 'tokenizer_revision': REVISION,
                'question_template': QUESTION_TEMPLATE, 'write_scope': 'visible_fact_delimiter',
                'read_scope': 'all_valid_question', 'name_pools_sha256': file_digest(output / 'name_pools.json'),
                'frequency_definition': pools['frequency_definition'], 'sizes': sizes, 'splits': {},
                'composition_sampling': 'uniform ordered compositions of 12; maximum 4 and at least two lengths >=3',
                'ood_composition_sampling': 'uniform ordered compositions of 12; maximum 8 and maximum >= requested hop',
                'v2_overlap': {'actual_hashes': 0, 'normalized_facts_query': 0, 'source_counts': v2_counts,
                               'sealed_normalized_proof': 'V2 has four edges of each relation; T2 has twelve of the queried relation'},
                'legacy_overlap': {'count': 0, 'method': 'construction: disjoint relation labels',
                                   'proof': 'research/notes/T-0005-data-contract.md; equally applies to T2 fact sets'}}
    answer_counts = Counter()
    for split in SPLITS:
        released = split in RELEASED
        path = output / (split + ('.jsonl' if released else '.registry.jsonl'))
        axes, graph_axes, entity_counts = defaultdict(Counter), defaultdict(Counter), Counter()
        first = hashlib.sha256()
        prompt_tokens = tokens = i = 0
        automatic = split == 'train' and sizes[split] is None
        with path.open('wb') as handle:
            while (i == 0 or i % 1024 or 3 * prompt_tokens < minimum_train_prompt_tokens) if automatic else i < sizes[split]:
                ident = identity(split, i, phase=phase)
                row = make_episode(tok, pools, ident) if released else sealed_record(ident, pools)
                graph = row if released else _structure(ident, pools)
                graph_facts = graph['facts']
                head = row['query']['head'] if released else graph['path'][0]
                relation = row['query']['relation'] if released else graph['relation']
                norm = semantic_hash(graph_facts, head, relation, endpoint=False)
                if row['episode_id'] in ids or row['content_sha256'] in contents or norm in normalized:
                    raise ValueError('duplicate identity or semantic content across/within slices')
                if row['content_sha256'] in v2_actual or norm in v2_normalized:
                    raise ValueError('V2 semantic overlap')
                ids.add(row['episode_id'])
                contents.add(row['content_sha256'])
                normalized.add(norm)
                line = (canonical(row) + '\n').encode()
                handle.write(line)
                first.update(line)
                if released:
                    for axis, value in row['axes'].items():
                        axes[axis][str(value)] += 1
                    for fact in row['facts']:
                        entity_counts.update((fact['head'], fact['tail']))
                    prompt_tokens += len(row['support_input_ids']) + len(row['question_input_ids'])
                    tokens += len(row['support_input_ids']) + len(row['question_input_ids']) + 1
                    if split == 'train':
                        answer_counts[row['query']['answer_token_id']] += 1
                else:
                    graph_axes['n_r_chains'][str(len(graph['chains']))] += 1
                    graph_axes['chain_length_composition'][','.join(map(str, sorted(graph['chain_lengths'])))] += 1
                i += 1
        sizes[split] = i
        second = hashlib.sha256()
        for index in range(i):
            ident = identity(split, index, phase=phase)
            row = make_episode(tok, pools, ident) if released else sealed_record(ident, pools)
            second.update((canonical(row) + '\n').encode())
        if first.hexdigest() != second.hexdigest() or file_digest(path) != first.hexdigest():
            raise ValueError('independent regeneration hash mismatch')
        entry = {'path': str(path.resolve()), 'count': i, 'sha256': first.hexdigest(),
                 'second_generation_sha256': second.hexdigest(), 'sealed': not released}
        if released:
            entry.update(input_tokens=tokens, prompt_input_tokens=prompt_tokens,
                         axes={k: dict(v) for k, v in axes.items()}, support_entity_counts=dict(entity_counts),
                         delimiter_checks=16 * i)
        else:
            entry['aggregate_graph_axes'] = {k: dict(v) for k, v in graph_axes.items()}
        metadata['splits'][split] = entry
    constant = min(answer_counts, key=lambda t: (-answer_counts[t], t))
    metadata['prior'] = {'answer_token_id': constant, 'train_answer_counts': dict(answer_counts)}
    shortcuts = {}
    for split in ('calibration', 'dev'):
        with (output / (split + '.jsonl')).open() as handle:
            shortcuts[split] = shortcut_report((json.loads(line) for line in handle), constant)
    metadata['shortcuts'] = shortcuts
    metadata['prior']['dev_em'] = shortcuts['dev']['groups']['all']['prior']
    metadata['train_budget'] = {'max_epochs': 3, 'minimum_prompt_tokens_3_epochs': minimum_train_prompt_tokens,
                                'available_prompt_tokens_3_epochs': 3 * metadata['splits']['train']['prompt_input_tokens'],
                                'available_tokens_3_epochs': 3 * metadata['splits']['train']['input_tokens']}
    metadata.update(unique_identities=len(ids), unique_semantic_contents=len(contents))
    (output / 'metadata.json').write_text(canonical(metadata) + '\n')
    return metadata
