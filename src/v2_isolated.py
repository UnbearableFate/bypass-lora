"""Isolated encoding of support facts, shared by E-0015 (diagnostics) and E-0016 D4/D5.

Each fact is encoded alone in the E=1 data format ("Head relation Tail ;\\n", the
same text and visible_delimiters tokenization as the E=1 generator); its delimiter is
its last token. v2_memory.isolated_batch turns the annotation into single-fact support
sequences; isolated_writes (offline) and isolated_support (differentiable) consume it.
"""
from v2_data import visible_delimiters


def fact_text(fact):
    return f"{fact['head']} {fact['relation']} {fact['tail']} ;\n"


def annotate_isolated(tok, rows):
    """Attach row['_isolated'] (per fact, in support order: {'ids', 'delimiter'}).

    Checks the support is the concatenated fact format and that a one-fact support
    equals its isolated encoding. Returns how many facts' tokens inside the contextual
    support differ from their isolated tokens (descriptive; 0 on the released data).
    """
    differing = 0
    for row in rows:
        if row['support_text'] != ''.join(fact_text(f) for f in row['facts']):
            raise ValueError('support text is not the concatenated fact format')
        isolated, previous = [], -1
        for fact in row['facts']:
            ids, positions = visible_delimiters(tok, fact_text(fact))
            if len(positions) != 1 or positions[0]['delimiter_token_index'] != len(ids) - 1:
                raise ValueError('isolated fact must end at its single delimiter')
            isolated.append({'ids': ids, 'delimiter': positions[0]['delimiter_token_index']})
            differing += row['support_input_ids'][previous + 1:fact['delimiter_token_index'] + 1] != ids
            previous = fact['delimiter_token_index']
        if len(row['facts']) == 1 and (isolated[0]['ids'] != row['support_input_ids']
                                       or isolated[0]['delimiter'] != row['facts'][0]['delimiter_token_index']):
            raise ValueError('isolated encoding differs from the E=1 data format')
        row['_isolated'] = isolated
    return differing
