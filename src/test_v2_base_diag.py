"""E-0007 contract regression tests; run through PBS."""
import copy

import pytest
import torch
from transformers import AutoTokenizer

from v2_data import MODEL, REVISION, encode, identity, make_episode, make_name_pools
from v2_eval import prompt_tokens
from v2_base_diag import (FACT_COUNTS, candidate_score, candidate_sets, category,
                          describe, fact_view, summarize, token_names)


@pytest.fixture(scope='module')
def fixture():
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    pools = make_name_pools(tok)
    rows = [make_episode(tok, pools, identity('calibration', i, support_order='shuffled' if i % 2 else 'natural')) for i in range(20)]
    return tok, rows


def test_explicit_split_and_confirmation_rejection(fixture):
    tok, rows = fixture
    row = rows[0]
    with pytest.raises(ValueError, match='restricted'):
        prompt_tokens(tok, row, 'F')
    assert prompt_tokens(tok, row, 'F', split='calibration')
    for actual in ('dev', 'calibration', 'confirmation'):
        r = copy.deepcopy(row)
        r['identity']['split'] = actual
        for requested in ('dev', 'calibration', 'confirmation', 'train'):
            if requested in ('dev', 'calibration') and actual == requested:
                assert prompt_tokens(tok, r, 'M', split=requested) == r['question_input_ids']
            else:
                with pytest.raises(ValueError, match='restricted'):
                    prompt_tokens(tok, r, 'M', split=requested)
    r['identity']['split'] = 'dev'
    assert prompt_tokens(tok, r, 'M') == r['question_input_ids']


def test_subsets_render_order_target_boundary_and_no_mutation(fixture):
    tok, rows = fixture
    for row in rows:
        before = copy.deepcopy(row)
        previous = set()
        for count in FACT_COUNTS:
            view = fact_view(tok, row, count)
            assert view == fact_view(tok, row, count)
            chosen = view['diagnostic']['source_fact_indices']
            assert chosen == sorted(chosen) and len(chosen) == count
            assert previous.issubset(chosen)
            previous = set(chosen)
            assert row['query']['target_fact_index'] in chosen
            fact = view['facts'][view['query']['target_fact_index']]
            assert fact['head'] == row['query']['head']
            assert ' ' + fact['tail'] == row['query']['answer_text']
            ids = prompt_tokens(tok, view, 'F', split='calibration')
            assert encode(tok, view['support_text'] + view['question_text'] + row['query']['answer_text']) == ids + [row['query']['answer_token_id']]
            if count == 16:
                assert view['support_text'] == row['support_text']
                assert view['support_input_ids'] == row['support_input_ids']
        assert before == row
    broken = copy.deepcopy(rows[0])
    broken['support_text'] += ' '
    with pytest.raises(ValueError, match='rerender'):
        fact_view(tok, broken, 16)


def test_candidate_absent_truth_empty_and_ties():
    logits = torch.tensor([1., 5., 5., 2.])
    assert candidate_score(logits, [2, 1], 1)['prediction'] == 1
    empty = candidate_score(logits, [], 1)
    assert empty['prediction'] is None and not empty['correct']
    assert empty['random_nonempty'] is None and empty['random_attainable'] == 0
    absent = candidate_score(logits, [2, 3], 1)
    assert not absent['covered'] and not absent['correct']
    assert absent['random_nonempty'] == .5 and absent['random_attainable'] == 0


def test_category_priority_overlap_and_name_universe():
    names = {n: i for i, n in enumerate(['Answer', 'Subject', 'Same', 'Other', 'Head', 'Absent'])}
    row = {'query': {'head': 'Subject', 'relation': 'emails', 'answer_token_id': 0},
           'facts': [{'head': 'Subject', 'relation': 'emails', 'tail': 'Answer'},
                     {'head': 'Same', 'relation': 'emails', 'tail': 'Same'},
                     {'head': 'Head', 'relation': 'tutors', 'tail': 'Other'},
                     {'head': 'Other', 'relation': 'tutors', 'tail': 'Same'}]}
    assert [category(i, row, names) for i in range(7)] == [
        'correct', 'question_subject', 'other_tail_same_relation', 'other_tail_other_relation',
        'other_head', 'name_outside_support', 'non_name_token']
    row['query']['answer_token_id'] = 1
    assert category(1, row, names) == 'correct'


def test_scoring_summary_self_edges_and_full_vocab(fixture):
    tok, rows = fixture
    names = token_names(tok)
    row = fact_view(tok, rows[0], 1)
    head = row['query']['head']
    row['facts'][0]['tail'] = head
    row['query'].update(answer_token_id=names[head], answer_text=' ' + head)
    sets, excluded = candidate_sets(tok, row)
    assert sets['a'] == [names[head]] and sets['b'] == []
    assert all(not v for v in excluded.values())
    logits = torch.zeros(tok.vocab_size)
    logits[names[head]] = 3
    rec = describe(logits, tok, row, names, 'F')
    assert rec['answer_rank'] == 1 and rec['top5'][0]['token_id'] == names[head]
    assert rec['correct'] and rec['subject_copy'] and rec['category'] == 'correct'
    summary = summarize([rec])
    assert summary['answer_is_subject_count'] == 1 and summary['nonself_em'] is None
    assert summary['candidates']['b'] == {'accuracy': 0., 'coverage': 0., 'empty_count': 1,
        'random_nonempty': None, 'random_attainable': 0., 'conditional_accuracy': None}
    logits[0] = 5
    rec = describe(logits, tok, row, names, 'F')
    assert rec['prediction'] == 0 and not rec['correct'] and rec['answer_rank'] == 2
    assert rec['candidates']['a']['correct']  # Restricted scores cannot replace full vocab EM.
    logits[0] = float('nan')
    with pytest.raises(ValueError, match='nonfinite'):
        describe(logits, tok, row, names, 'F')
