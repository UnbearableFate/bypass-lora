"""Protocol and grouping regressions for E-0012; execute inside PBS only."""
import pytest
import torch

from v2_layer_diagnostics import (BilinearAddress, constant_baseline, name_end,
                                  selection_key, slot_summary, standardizer, token_positions)


class CharacterTokenizer:
    def __call__(self, text, **kwargs):
        return {'input_ids': [ord(c) for c in text], 'offset_mapping': [(i, i + 1) for i in range(len(text))]}


def test_offsets_with_repeated_and_multitoken_names():
    support = 'Alexander emails Alice ;\nAlice tutors Alexander ;\n'
    question = 'Who does Alexander email?\nAnswer:'
    facts = [{'head': 'Alexander', 'relation': 'emails', 'tail': 'Alice', 'delimiter_token_index': support.index(';')},
             {'head': 'Alice', 'relation': 'tutors', 'tail': 'Alexander', 'delimiter_token_index': support.rindex(';')}]
    row = {'support_text': support, 'support_input_ids': [ord(c) for c in support], 'facts': facts,
           'question_text': question, 'question_input_ids': [ord(c) for c in question], 'query': {'head': 'Alexander'}}
    positions = token_positions(CharacterTokenizer(), row)
    assert positions['head'] == [len('Alexander') - 1, support.index('\n') + len('Alice')]
    assert positions['tail'] == [support.index('Alice') + len('Alice') - 1, support.rindex('Alexander') + len('Alexander') - 1]
    assert positions['question'] == [len('Who does Alexander') - 1, len(question) - 1]
    bad = {**row, 'question_input_ids': row['question_input_ids'][:-1]}
    with pytest.raises(ValueError, match='question tokenization'):
        token_positions(CharacterTokenizer(), bad)
    with pytest.raises(ValueError, match='exactly one'):
        name_end([(0, 1), (2, 3)], 0, 2)


def test_standardization_uses_fitting_input_only_and_handles_constant_channel():
    x = torch.tensor([[1., 3.], [3., 3.]])
    mean, scale = standardizer(x)
    torch.testing.assert_close(mean, torch.tensor([2., 3.]))
    torch.testing.assert_close(scale, torch.tensor([1., 1e-5]))
    heldout = torch.tensor([[1000., 99.]])
    assert ((heldout - mean) / scale)[0, 0] == 998.


def test_selection_ties_and_frequency_baseline():
    a = {'correct': 9, 'n': 10, 'nll': .2, 'l2': 0., 'epoch': 8}
    b = {**a, 'l2': 1e-4}
    assert selection_key(b) > selection_key(a)
    assert selection_key({**b, 'epoch': 4}) > selection_key(b)
    assert selection_key({**b, 'nll': .1}) > selection_key(b)
    assert selection_key({**b, 'correct': 10, 'nll': 1.}) > selection_key(b)
    baseline = constant_baseline(torch.tensor([[1, 1], [1, 0]]), torch.tensor([0, 1, 2]), 3)
    assert baseline == {'class': 1, 'train_count': 3, 'train_n': 4, 'correct': 1, 'n': 3, 'accuracy': 1 / 3}


def test_address_logits_preserve_episode_ownership():
    model = BilinearAddress(2, 2, 2, 'cpu')
    with torch.no_grad():
        model.U.copy_(torch.eye(2))
        model.V.copy_(torch.eye(2))
    query = torch.tensor([[1., 0.], [1., 0.]])
    keys = torch.tensor([[[5., 0.], [0., 3.]], [[0., 3.], [5., 0.]]])
    logits = model((query, keys))
    torch.testing.assert_close(logits, torch.tensor([[5., 0.], [0., 5.]]))
    assert logits.argmax(-1).tolist() == [0, 1]


def test_slot_histograms_exclude_padding_and_use_recipient_labels():
    argmax = torch.tensor([[0, 1, 1], [1, 0, 0]])
    mask = torch.tensor([[True, True, False], [True, False, False]])
    relations = torch.tensor([[0, 1], [0, 1]])
    result = slot_summary(argmax, mask, relations, torch.tensor([1, 1]), torch.tensor([0, 1]))
    assert result['n'] == 3
    assert result['slot_counts'] == [1, 2]
    assert result['same_relation_count'] == 2
    assert result['target_count'] == 2
    assert sum(result['slot_frequencies']) == 1
