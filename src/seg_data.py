"""E-0025 independent segment encodings, retaining the RB1 sample plan.

Total length (both segments) remains <=512. Candidate IDs and answer masks
come directly from V1. Only independently tokenizing the PIQA seam can change
prompt IDs. BoolQ's retained Passage prefix is exactly V1's.
"""
from v1_data import _encode, encode_candidates

ORDER_SHA256 = '33759bc20322d0c9981128dd8574438546fc3439194ab32b7c79e35e0f0fc2bf'
STEPS, EXAMPLES = 13115, 26230


def seam_edits(reference, segmented):
    """Count changed middle spans after removing equal prefix and suffix."""
    common = 0
    while common < min(len(reference), len(segmented)) and reference[common] == segmented[common]:
        common += 1
    tail = 0
    while (tail < min(len(reference), len(segmented))-common and
           reference[len(reference)-1-tail] == segmented[len(segmented)-1-tail]):
        tail += 1
    return {'identical': reference == segmented, 'common_prefix': common, 'common_suffix': tail,
            'v1_changed_tokens': len(reference)-common-tail,
            'seg_changed_tokens': len(segmented)-common-tail,
            'v1_prompt_tokens': len(reference), 'seg_prompt_tokens': len(segmented)}


def encode_segments(tok, row, max_length=512):
    old = encode_candidates(tok, row, max_length)
    suffixes = [e['input_ids'][e['prompt_length']:] for e in old]
    budget = max_length-max(map(len, suffixes))
    old_prompt = old[0]['input_ids'][:old[0]['prompt_length']]
    if row['task'] == 'piqa':
        first_full = _encode(tok, f"Goal: {row['goal']}\n")
        second = _encode(tok, 'Solution:')
        room = budget-len(second)
        if room <= 0:
            raise ValueError('segmented PIQA needs a nonempty Goal segment')
        first = first_full[-room:]
    elif row['task'] == 'boolq':
        first_full = _encode(tok, f"Passage: {row['passage']}\n")
        second = _encode(tok, f"Question: {row['question']}\nAnswer:")
        first = first_full[:budget-len(second)]
        if old_prompt != first+second:
            raise ValueError('segmented BoolQ must retain exactly the V1 Passage tokens')
    else:
        raise ValueError('unknown segmented task')
    if not first or not second:
        raise ValueError('segmented encoding requires two nonempty segments')
    audit = {**seam_edits(old_prompt, first+second), 'task': row['task'], 'id': row['id'],
             'segment1_tokens': len(first), 'segment2_prompt_tokens': len(second),
             'full_segment1_tokens': len(first_full), 'candidate_tokens': list(map(len, suffixes))}
    result = []
    for previous, suffix in zip(old, suffixes):
        current = dict(previous, input_ids=second+suffix, labels=[-100]*len(second)+suffix,
                       prompt_length=len(second), segment1_ids=first.copy(), seam=audit)
        assert [x for x in current['labels'] if x != -100] == suffix
        assert len(first)+len(current['input_ids']) <= max_length
        result.append(current)
    return result


def segment_plan(rb_batches, splits, tok):
    lookup = {(r['task'], r['id']): r for task in ('piqa', 'boolq')
              for r in splits['splits'][task]['train']}
    cache, batches = {}, []
    for batch in rb_batches:
        current = []
        for old in batch:
            key = old['task'], old['id']
            if key not in cache:
                row = lookup[key]
                gold = int(row['label']) if row['task'] == 'piqa' else (0 if row['label'] else 1)
                cache[key] = encode_segments(tok, row)[gold]
            current.append(dict(cache[key], epoch=old['epoch'], epoch_end=old['epoch_end']))
        batches.append(current)
    return batches
