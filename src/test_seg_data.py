import pytest

from seg_data import encode_segments, seam_edits
from seg_train import tokenizer
from v1_data import encode_candidates, _encode


@pytest.fixture(scope='module')
def tok():
    return tokenizer()


@pytest.mark.parametrize('task,long', [('piqa', False), ('piqa', True), ('boolq', False), ('boolq', True)])
def test_separate_prompt_tokens_same_candidates_total_budget_and_boolq_prefix(tok, task, long):
    row = {'task': task, 'id': 'synthetic', 'label': 0 if task == 'piqa' else True,
           'goal': 'Open the sealed glass jar. '*(400 if long else 1), 'solutions': ['Use a lid grip.', 'Eat the lid.'],
           'passage': 'The river crosses the valley. '*(400 if long else 1), 'question': 'Does the river cross the valley?'}
    original, segmented = encode_candidates(tok, row), encode_segments(tok, row)
    first = segmented[0]['segment1_ids']
    for old, new in zip(original, segmented):
        assert len(first)+len(new['input_ids']) <= 512
        assert [v for v in old['labels'] if v != -100] == [v for v in new['labels'] if v != -100]
        assert new['segment1_ids'] == first
        assert all(v == -100 for v in new['labels'][:new['prompt_length']])
    prompt2 = segmented[0]['input_ids'][:segmented[0]['prompt_length']]
    if task == 'boolq':
        assert first+prompt2 == original[0]['input_ids'][:original[0]['prompt_length']]
    else:
        assert prompt2 == _encode(tok, 'Solution:')
        reserve = max(e['completion_length'] for e in original)
        assert first == _encode(tok, f"Goal: {row['goal']}\n")[-(512-reserve-len(prompt2)):]
    assert segmented[0]['seam'] == segmented[1]['seam']


def test_seam_counts_do_not_count_unchanged_prefix_and_suffix():
    assert seam_edits([1, 2, 3], [1, 2, 3])['v1_changed_tokens'] == 0
    audit = seam_edits([1, 2, 8, 9, 3], [1, 2, 7, 3])
    assert audit['common_prefix'] == 2 and audit['common_suffix'] == 1
    assert audit['v1_changed_tokens'] == 2 and audit['seg_changed_tokens'] == 1
