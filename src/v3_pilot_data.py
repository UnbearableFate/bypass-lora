"""Pinned V3 dataset selection and offline strata for the shared F-line runner."""
from pathlib import Path

from v3_eval import grouped_scores, load_released_data

# Reviewed full generation ATT-0077-1, E-0011@2. Includes all split checksums,
# exact V2 pool bytes, empirical shortcut gates and 5.5M prompt-token coverage.
METADATA_SHA256 = '2ad8cc358ec5f6005e832125e3fa9995a0aa0a7c73f77c8fec6da6b7416a178f'


def load_data(path, split):
    path = Path(path)
    if split not in ('train', 'calibration', 'dev') or path.name != split + '.jsonl':
        raise ValueError('only the explicitly selected released V3 split is authorized')
    _, rows = load_released_data(path.parent, METADATA_SHA256, split)
    return rows


def stratify(rows, records, constant):
    """Join diagnostics after evaluation; axes never enter a model forward."""
    by_id = {r['episode_id']: r for r in rows}
    if len(by_id) != len(rows) or len(records) != len(rows):
        raise ValueError('prediction/data count or identity mismatch')
    seen, scored = set(), []
    for record in records:
        ident = record['episode_id']
        if ident not in by_id or ident in seen:
            raise ValueError('prediction identity missing or duplicated')
        seen.add(ident)
        row = by_id[ident]
        if record['target'] != row['query']['answer_token_id']:
            raise ValueError('prediction target does not match data')
        scored.append({'hop': row['axes']['hop'], 'support_order': row['axes']['support_order'],
                       'em': record['prediction'] == record['target'], 'nll': record['nll'],
                       'subject_copy': record['prediction'] == record['subject'],
                       'prior': record['target'] == constant})
    groups = grouped_scores(scored)
    longer = [r for r in scored if r['hop'] in (2, 3)]
    if longer:
        groups['hops2-3'] = grouped_scores(longer)['all']
    metrics = {f'{group}.{metric}': value for group, values in groups.items() if group != 'all'
               for metric, value in values.items()}
    for group in ('hop=2', 'hop=3', 'hops2-3'):
        if group in groups:
            metrics[group + '.gate'] = groups[group]['em'] >= groups[group]['prior'] + .20
    return metrics
