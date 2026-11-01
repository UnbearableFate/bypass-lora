"""T2 contract and boundary regressions. Execute only inside a PBS allocation."""
import copy
import json
from collections import Counter

import pytest
from transformers import AutoTokenizer

from v2_data import generate_dataset as v2_generate
from v2_eval import prompt_tokens
from v3_data import (MODEL, REVISION, NAME_POOLS_SHA256, QUESTION_TEMPLATE, SEALED,
                     SPLITS, TRAIN_COMPOSITIONS, _structure, digest, file_digest,
                     generate_dataset, identity, load_name_pools, make_episode,
                     sealed_record, shortcut_report, shortcut_scores)


@pytest.fixture(scope='module')
def tok():
    return AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)


@pytest.fixture(scope='module')
def v2_dir(tok, tmp_path_factory):
    directory = tmp_path_factory.mktemp('v2-reference')
    v2_generate(tok, directory, sizes={'train': 12, 'calibration': 12, 'dev': 12, 'confirmation': 12})
    return directory


@pytest.fixture(scope='module')
def pools(v2_dir):
    return load_name_pools(v2_dir / 'name_pools.json')


def check_graph(ident, graph):
    facts, relation = graph['facts'], graph['relation']
    assert len(facts) == 16
    pairs = [(f['head'], f['relation']) for f in facts]
    assert len(set(pairs)) == 16
    r_facts = [f for f in facts if f['relation'] == relation]
    assert len(r_facts) == 12
    assert len({f['head'] for f in r_facts}) == len({f['tail'] for f in r_facts}) == 12
    outgoing = {f['head']: f['tail'] for f in r_facts}
    for head in outgoing:
        seen, node = set(), head
        while node in outgoing:
            assert node not in seen
            seen.add(node)
            node = outgoing[node]
    path = graph['path']
    assert len(path) == ident['layout']['hop'] + 1
    assert path[0] != path[-1] and path[-1] not in outgoing
    for a, b, index in zip(path, path[1:], graph['path_fact_indices']):
        assert facts[index] == {'head': a, 'relation': relation, 'tail': b}
    query_chain = next(c for c in graph['chains'] if path[0] in c)
    noise = [f for f in facts if f['relation'] != relation]
    endpoint_noise = [f for f in noise if f['head'] == path[-1]]
    assert len(endpoint_noise) == int(ident['index'] % 2 == 0)
    assert all(f['head'] not in query_chain for f in noise if f not in endpoint_noise)
    if endpoint_noise:
        assert endpoint_noise[0]['tail'] not in {x for c in graph['chains'] for x in c}


def test_graph_constraints_walk_distance_and_sealed_layouts(pools):
    for split in ('dev',) + SEALED:
        for i in range(48):
            ident = identity(split, i)
            graph = _structure(ident, pools)
            check_graph(ident, graph)
            if ident['layout']['hop'] <= 3:
                assert max(graph['chain_lengths']) <= 4
                assert sum(n >= 3 for n in graph['chain_lengths']) >= 2
            else:
                assert max(graph['chain_lengths']) <= 8
                assert max(graph['chain_lengths']) >= ident['layout']['hop']


def test_hop_does_not_choose_graph_or_composition(pools):
    assert len(TRAIN_COMPOSITIONS) == len(set(TRAIN_COMPOSITIONS))
    assert all(sum(c) == 12 and max(c) <= 4 and sum(n >= 3 for n in c) >= 2 for c in TRAIN_COMPOSITIONS)
    assert len({len(c) for c in TRAIN_COMPOSITIONS}) > 1
    for index in range(48):
        graphs = [_structure(identity('dev', index, hop=h), pools) for h in (1, 2, 3)]
        triples = [sorted((f['head'], f['relation'], f['tail']) for f in g['facts']
                          if f['relation'] == g['relation']) for g in graphs]
        assert triples[0] == triples[1] == triples[2]
        assert graphs[0]['chain_lengths'] == graphs[1]['chain_lengths'] == graphs[2]['chain_lengths']
        assert len({g['relation'] for g in graphs}) == 1


def test_rotation_joint_balance_and_terminal_alternation():
    identities = [identity('train', i) for i in range(1000)]
    counts = Counter((x['layout']['hop'], x['layout']['support_order']) for x in identities)
    assert len(counts) == 6 and max(counts.values()) - min(counts.values()) <= 1
    for index, ident in enumerate(identities):
        assert ident['layout']['hop'] == 1 + index % 3
        assert ident['layout']['support_order'] == ('natural' if index % 2 == 0 else 'shuffled')


def test_support_order_does_not_change_graph_query_or_semantics(tok, pools):
    for index in range(24):
        rows = [make_episode(tok, pools, identity('dev', index, support_order=order))
                for order in ('natural', 'shuffled')]
        assert rows[0]['content_sha256'] == rows[1]['content_sha256']
        assert rows[0]['episode_id'] != rows[1]['episode_id']
        assert rows[0]['question_text'] == rows[1]['question_text']
        assert rows[0]['query']['answer_token_id'] == rows[1]['query']['answer_token_id']
        for row in rows:
            for a, b, i in zip(row['path']['entities'], row['path']['entities'][1:], row['path']['fact_indices']):
                fact = row['facts'][i]
                assert (fact['head'], fact['relation'], fact['tail']) == (a, row['query']['relation'], b)


def test_visible_positions_single_token_answer_and_identical_question_template(tok, pools):
    for i in range(36):
        row = make_episode(tok, pools, identity('dev', i))
        assert row['question_text'] == QUESTION_TEMPLATE.format(**row['query'])
        assert not any(char.isdigit() for char in row['question_text'])
        assert len(set(f['delimiter_token_index'] for f in row['facts'])) == 16
        for fact in row['facts']:
            start, end = fact['delimiter_char_span']
            assert row['support_text'][start:end] == ';'
            assert ';' in tok.decode([row['support_input_ids'][fact['delimiter_token_index']]])
        target = [row['query']['answer_token_id']]
        assert tok.encode(row['query']['answer_text'], add_special_tokens=False) == target
        for line in ('M', 'F'):
            ids = prompt_tokens(tok, row, line)
            text = (row['support_text'] if line == 'F' else '') + row['question_text']
            assert tok.encode(text + row['query']['answer_text'], add_special_tokens=False) == ids + target
        changed = copy.deepcopy(row)
        changed['path'] = {'entities': ['unobservable'], 'fact_indices': [999]}
        changed['facts'] = []
        changed['support_text'] = 'different support'
        assert prompt_tokens(tok, row, 'M') == prompt_tokens(tok, changed, 'M')


def test_sealing_phase_independence_identity_validation(tok, pools):
    for split in SEALED:
        ident = identity(split, 2)
        assert set(sealed_record(ident, pools)) == {'identity', 'episode_id', 'content_sha256'}
        with pytest.raises(ValueError, match='sealed'):
            make_episode(tok, pools, ident)
    with pytest.raises(ValueError):
        identity('dev', 0, hop=4)
    with pytest.raises(ValueError):
        identity('ood_h6', 0, hop=4)
    v3, v4 = (identity('dev', 7, phase=p) for p in ('V3', 'V4'))
    assert digest(v3) != digest(v4)
    assert make_episode(tok, pools, v3)['content_sha256'] != make_episode(tok, pools, v4)['content_sha256']
    changed = copy.deepcopy(v3)
    changed['layout']['name_pool'] = 'unseen'
    with pytest.raises(ValueError, match='identity'):
        make_episode(tok, pools, changed)


def test_shortcuts_independently_known_cases(tok, pools):
    rows = [make_episode(tok, pools, identity('calibration', i)) for i in range(60)]
    for i, row in enumerate(rows):
        constant = row['query']['answer_token_id']
        scores = shortcut_scores(row, constant)
        assert scores['prior'] == 1
        assert scores['first_bridge'] == int(row['axes']['hop'] == 1)
        assert scores['ignore_relation'] == int(i % 2 == 1)
        assert scores['random_r_endpoint'] == 1 / row['axes']['n_r_chains']
    report = shortcut_report(rows, -1)
    for hop in (1, 2, 3):
        assert report['groups'][f'hop={hop}']['ignore_relation'] == .5
        assert report['groups'][f'hop={hop}']['prior'] == 0
    assert report['groups']['order=natural']['ignore_relation'] == 0
    assert report['groups']['order=shuffled']['ignore_relation'] == 1
    assert report['gates_passed']
    # Three chains, one strictly longest. Presentation-last endpoint differs
    # from the longest, and the direct bridge differs from the true endpoint.
    fixture = {'facts': [{'head': a, 'relation': 'r', 'tail': b}
                         for a, b in [('a', 'b'), ('b', 'c'), ('c', 'd'), ('e', 'f'), ('f', 'g'), ('h', 'i')]],
               'query': {'head': 'b', 'relation': 'r', 'answer_text': ' d', 'answer_token_id': 5}}
    scores = shortcut_scores(fixture, 9)
    assert scores == {'prior': 0., 'random_r_endpoint': 1/3, 'ignore_relation': 1.,
                      'longest_r_chain': 1., 'last_r_endpoint': 0., 'first_bridge': 0.}


def test_streaming_hashes_uniqueness_prior_overlap_and_no_sealed_text(tok, v2_dir, tmp_path):
    sizes = dict.fromkeys(SPLITS, 12)
    sizes['train'] = 24
    metadata = generate_dataset(tok, tmp_path, v2_data_dir=v2_dir, sizes=sizes)
    assert metadata['unique_identities'] == metadata['unique_semantic_contents'] == sum(sizes.values())
    assert file_digest(tmp_path / 'name_pools.json') == NAME_POOLS_SHA256
    assert (tmp_path / 'name_pools.json').read_bytes() == (v2_dir / 'name_pools.json').read_bytes()
    for split, entry in metadata['splits'].items():
        assert entry['sha256'] == entry['second_generation_sha256']
        suffix = '.registry.jsonl' if split in SEALED else '.jsonl'
        rows = [json.loads(line) for line in (tmp_path / (split + suffix)).read_text().splitlines()]
        if split in SEALED:
            assert all(set(row) == {'identity', 'episode_id', 'content_sha256'} for row in rows)
        else:
            assert entry['prompt_input_tokens'] == sum(len(r['support_input_ids']) + len(r['question_input_ids']) for r in rows)
    train = [json.loads(line) for line in (tmp_path / 'train.jsonl').read_text().splitlines()]
    counts = Counter(r['query']['answer_token_id'] for r in train)
    assert metadata['prior']['answer_token_id'] == min(counts, key=lambda x: (-counts[x], x))
    assert metadata['v2_overlap']['actual_hashes'] == metadata['v2_overlap']['normalized_facts_query'] == 0


def test_checksum_failure_and_automatic_train_size(tok, v2_dir, tmp_path):
    bad = tmp_path / 'bad-pools.json'
    bad.write_text('{}')
    with pytest.raises(ValueError, match='checksum'):
        load_name_pools(bad)
    sizes = dict.fromkeys(SPLITS, 6)
    sizes['train'] = None
    meta = generate_dataset(tok, tmp_path / 'automatic', v2_data_dir=v2_dir, sizes=sizes,
                            minimum_train_prompt_tokens=1)
    assert meta['sizes']['train'] == 1024
    assert meta['train_budget']['available_prompt_tokens_3_epochs'] >= 1
