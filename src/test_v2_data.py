"""Regression invariants for the V2 data contract; execute in PBS only."""
import copy
import json

import pytest
import torch
from transformers import AutoTokenizer

from v2_data import (MODEL, REVISION, RELATIONS, digest, episode_content, generate_dataset,
                     identity, make_episode, make_name_pools, sealed_record, visible_delimiters)
from v2_eval import left_pad, prompt_tokens, score_logits


@pytest.fixture(scope='module')
def tok():
    return AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)


@pytest.fixture(scope='module')
def pools(tok):
    return make_name_pools(tok)


def test_pool_disjointness_tokenization_and_frequency(tok, pools):
    seen, unseen = pools['pools']['seen'], pools['pools']['unseen']
    assert set(x['name'] for x in seen).isdisjoint(x['name'] for x in unseen)
    for pool in (seen, unseen):
        assert {x['frequency_stratum'] for x in pool} == {'high', 'low'}
        for name in pool:
            assert tok.encode(' ' + name['name'], add_special_tokens=False) == [name['answer_token_id']]
            assert name['bare_tokens'] == len(tok.encode(name['name'], add_special_tokens=False))


def test_order_is_independent_axis_and_target_tracks_permutation(tok, pools):
    for index in range(20):
        natural = make_episode(tok, pools, identity('dev', index))
        shuffled = make_episode(tok, pools, identity('dev', index, support_order='shuffled'))
        assert natural['content_sha256'] == shuffled['content_sha256']
        assert natural['question_text'] == shuffled['question_text']
        assert natural['query']['answer_token_id'] == shuffled['query']['answer_token_id']
        assert natural['episode_id'] != shuffled['episode_id']
        for row in (natural, shuffled):
            q = row['query']
            target = row['facts'][q['target_fact_index']]
            assert (target['head'], target['relation'], ' ' + target['tail']) == (q['head'], q['relation'], q['answer_text'])
            assert len({f['head'] for f in row['facts']}) == 16
            assert set(f['relation'] for f in row['facts']) == set(RELATIONS)


def test_delimiters_are_text_only_and_one_per_fact(tok, pools):
    row = make_episode(tok, pools, identity('dev', 3))
    ids, positions = visible_delimiters(tok, row['support_text'])
    assert ids == row['support_input_ids']
    assert len(positions) == len({p['delimiter_token_index'] for p in positions}) == 16
    for fact, position in zip(row['facts'], positions):
        start, end = position['delimiter_char_span']
        assert row['support_text'][start:end] == ';'
        assert fact['delimiter_token_index'] == position['delimiter_token_index']
        assert ';' in tok.decode([ids[position['delimiter_token_index']]])


def test_m_prompt_cannot_observe_support_or_offline_targets(tok, pools):
    row = make_episode(tok, pools, identity('dev', 4))
    changed = copy.deepcopy(row)
    changed['support_text'] = 'arbitrary replacement support'
    changed['facts'] = []
    changed['query']['target_fact_index'] = 999
    assert prompt_tokens(tok, row, 'M') == prompt_tokens(tok, changed, 'M') == row['question_input_ids']
    assert prompt_tokens(tok, row, 'F') != prompt_tokens(tok, changed, 'F')


def test_sealing_and_reserved_layouts(tok, pools):
    ident = identity('confirmation', 0)
    record = sealed_record(ident, pools)
    assert set(record) == {'identity', 'episode_id', 'content_sha256'}
    with pytest.raises(ValueError, match='sealed'):
        make_episode(tok, pools, ident)
    for split in ('train', 'dev'):
        for hop in (2, 3):
            reserved = identity(split, 0, hop=hop)
            with pytest.raises(ValueError, match='reserved'):
                episode_content(reserved, pools)
        for hop in (4, 6, 8):
            with pytest.raises(ValueError):
                identity(split, 0, hop=hop)
    assert len({digest(identity('confirmation', 0, hop=h)) for h in (1, 2, 3, 4, 6, 8)}) == 6
    with pytest.raises(ValueError, match='sealed'):
        identity('dev', 0, name_pool='unseen')


def test_streaming_reproducibility_registry_and_prior(tok, tmp_path):
    metadata = generate_dataset(tok, tmp_path, sizes={'train': 32, 'calibration': 16, 'dev': 16, 'confirmation': 16})
    assert metadata['unique_identities'] == metadata['unique_semantic_contents'] == 80
    for split, entry in metadata['splits'].items():
        assert entry['sha256'] == entry['second_generation_sha256']
    sealed = [json.loads(s) for s in (tmp_path / 'confirmation.registry.jsonl').read_text().splitlines()]
    assert all(set(s) == {'identity', 'episode_id', 'content_sha256'} for s in sealed)
    dev = [json.loads(s) for s in (tmp_path / 'dev.jsonl').read_text().splitlines()]
    constant = metadata['prior']['answer_token_id']
    assert metadata['prior']['dev_em'] == sum(x['query']['answer_token_id'] == constant for x in dev) / len(dev)
    row = dev[0]
    row['identity']['split'] = 'confirmation'
    with pytest.raises(ValueError, match='restricted'):
        prompt_tokens(tok, row, 'M')


def test_full_vocabulary_metric_and_nonfinite_failure():
    logits = torch.tensor([[0., 1., 2., 9.], [5., 1., 0., -1.]])
    targets = torch.tensor([1, 0])
    predictions, nll = score_logits(logits, targets)
    assert predictions.tolist() == [3, 0]  # token 3 need not belong to the name pool
    assert torch.allclose(nll, logits.logsumexp(-1) - logits[torch.arange(2), targets])
    logits[0, 3] = float('nan')
    with pytest.raises(ValueError, match='nonfinite'):
        score_logits(logits, targets)


def test_left_padding_uses_positions_not_token_value():
    batch = left_pad([[7, 0], [8]], 0, 'cpu')
    assert batch['input_ids'].tolist() == [[7, 0], [0, 8]]
    assert batch['attention_mask'].tolist() == [[1, 1], [0, 1]]
    assert batch['position_ids'].tolist() == [[0, 1], [0, 0]]
