"""E-0014 variant data invariants (E in 1, 4, 8); execute in PBS only."""
from collections import Counter
import json
from pathlib import Path

import pytest
from transformers import AutoTokenizer

from v1_data import derive_seed
from v2_data import RELATIONS, identity as v2_identity
from v2r_data import (MODEL, REVISION, RELEASED, VARIANTS, episode_content, generate_dataset,
                      identity, load_name_pools, make_episode, sealed_record, version)

V2_DATA = Path('/work/xg24i002/x10041/bypass-lora/runs/ATT-0039-1/data')
SMALL = {'train': 48, 'calibration': 12, 'dev': 12, 'confirmation': 8}


@pytest.fixture(scope='module')
def tok():
    return AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)


@pytest.fixture(scope='module')
def pools():
    return load_name_pools(V2_DATA / 'name_pools.json')


def test_identity_namespace_and_version():
    for n in VARIANTS:
        ident = identity('dev', 3, n)
        assert ident['generator_version'] == f'concat-loop-v2-t1-e{n}-1'
        assert ident['phase'] == 'V2R' and ident['layout']['n_facts'] == n
        assert ident['base_seed'] == derive_seed(20261029, 'concat_loop_v2', 'V2R', 'dev', n)
        assert ident['base_seed'] != v2_identity('dev', 3)['base_seed']
        assert ident['layout']['support_order'] == 'shuffled'
    assert len({identity('train', 0, n)['base_seed'] for n in VARIANTS}) == 3
    for bad in (0, 2, 16):
        with pytest.raises(ValueError):
            version(bad)


@pytest.mark.parametrize('n', VARIANTS)
def test_generation_rules(tok, pools, n):
    weights = {x['name']: x['sampling_weight'] for x in pools['pools']['seen']}
    targets, relation_draws = Counter(), Counter()
    for index in range(200):
        row = make_episode(tok, pools, identity('train', index, n))
        facts = row['facts']
        assert len(facts) == n and row['axes']['n_facts'] == n
        assert len({f['head'] for f in facts}) == n
        assert all(f['tail'] in weights for f in facts)
        relations = Counter(f['relation'] for f in facts)
        if n == 1:
            relation_draws.update(relations)
        else:
            assert relations == Counter({r: n // 4 for r in RELATIONS})
        target = row['query']['target_fact_index']
        assert facts[target]['head'] == row['query']['head']
        assert facts[target]['relation'] == row['query']['relation']
        assert row['query']['answer_text'] == ' ' + facts[target]['tail']
        assert tok.encode(row['query']['answer_text'], add_special_tokens=False) == [row['query']['answer_token_id']]
        assert row['question_text'] == f"Who does {row['query']['head']} {RELATIONS[row['query']['relation']]}?\nAnswer:"
        delimiters = [f['delimiter_token_index'] for f in facts]
        assert delimiters == sorted(set(delimiters)) and len(delimiters) == n
        assert all(row['support_input_ids'][d] == row['support_input_ids'][delimiters[0]] for d in delimiters)
        targets[target] += 1
    assert set(targets) == set(range(n))
    if n == 1:
        assert set(relation_draws) == set(RELATIONS)


def test_order_axis_preserves_content_and_tracks_target(tok, pools):
    for index in range(20):
        a = make_episode(tok, pools, identity('dev', index, 8, support_order='natural'))
        b = make_episode(tok, pools, identity('dev', index, 8, support_order='shuffled'))
        assert a['content_sha256'] == b['content_sha256']
        assert a['facts'][a['query']['target_fact_index']]['head'] == b['facts'][b['query']['target_fact_index']]['head']


def test_attempts_change_content_and_identity_is_checked(pools):
    ident = identity('train', 5, 4)
    assert episode_content(ident, pools, 0)[3] != episode_content(ident, pools, 1)[3]
    assert episode_content(ident, pools, 1) == episode_content(ident, pools, 1)
    forged = json.loads(json.dumps(ident))
    forged['base_seed'] += 1
    with pytest.raises(ValueError, match='contract'):
        episode_content(forged, pools)
    with pytest.raises(ValueError, match='sealed'):
        make_episode(None, pools, identity('confirmation', 0, 4))
    assert set(sealed_record(identity('confirmation', 0, 4))) == {'identity', 'episode_id'}


@pytest.mark.parametrize('n', VARIANTS)
def test_small_dataset_deterministic_and_deduplicated(tok, tmp_path, n):
    first = generate_dataset(tok, tmp_path / 'a', n, v2_data_dir=V2_DATA, sizes=SMALL)
    second = generate_dataset(tok, tmp_path / 'b', n, v2_data_dir=V2_DATA, sizes=SMALL)
    for split in ('train', 'calibration', 'dev', 'confirmation'):
        assert first['splits'][split]['sha256'] == second['splits'][split]['sha256']
        assert (tmp_path / 'a' / Path(first['splits'][split]['path']).name).read_bytes() == \
            (tmp_path / 'b' / Path(first['splits'][split]['path']).name).read_bytes()
    registry = [json.loads(x) for x in (tmp_path / 'a' / 'confirmation.registry.jsonl').read_text().splitlines()]
    assert len(registry) == SMALL['confirmation'] and all(set(r) == {'identity', 'episode_id'} for r in registry)
    contents = {s: [json.loads(x)['content_sha256'] for x in (tmp_path / 'a' / f'{s}.jsonl').read_text().splitlines()]
                for s in RELEASED}
    assert not set(contents['train']) & (set(contents['dev']) | set(contents['calibration']))
    assert len(set(contents['dev'] + contents['calibration'])) == SMALL['dev'] + SMALL['calibration']


def test_collision_redraw_with_tiny_pool(tok, pools, tmp_path):
    # Two names give 2*4*2 = 16 E=1 contents: evaluation rows must redraw, train may repeat.
    tiny = dict(pools, pools={'seen': pools['pools']['seen'][:2], 'unseen': []})
    sizes = {'train': 40, 'calibration': 6, 'dev': 6, 'confirmation': 2}
    meta = generate_dataset(tok, tmp_path, 1, v2_data_dir=V2_DATA, sizes=sizes, pools=tiny)
    assert meta['splits']['dev']['rows_redrawn'] + meta['splits']['calibration']['rows_redrawn'] > 0
    assert meta['splits']['train']['repeated_content_rows'] > 0
    assert meta['splits']['dev']['unique_contents'] == 6 and meta['splits']['calibration']['unique_contents'] == 6
    rows = {s: [json.loads(x) for x in (tmp_path / f'{s}.jsonl').read_text().splitlines()] for s in RELEASED}
    evaluation = {r['content_sha256'] for s in ('dev', 'calibration') for r in rows[s]}
    assert not evaluation & {r['content_sha256'] for r in rows['train']}
    for split, split_rows in rows.items():
        for row in split_rows:
            again = make_episode(tok, tiny, row['identity'], row['generation_attempt'])
            assert again == row
