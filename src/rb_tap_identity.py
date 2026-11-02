"""E-0028 rule 1 identity checks under coordinator decision D-0060."""

PINNED_PROFILE='NVIDIA GH200 120GB MIG 2g.24gb'
LEGACY_COMMIT='7bf091d4af24c2a9fdc666a5f885c31817b97174'


def prediction_keys(rows):
    return [(r['task'],r['id'],r['gold'],r['prediction']) for r in rows]


def reference_check(actual,pinned,hardware,pinned_hardware=PINNED_PROFILE,same_allocation=None):
    """Strict predictions, using pinned rows on the same profile, else local rows.

    Both comparisons are reported. No numeric tolerance or slice restriction is
    introduced. The caller validates row membership and local-reference provenance.
    """
    if not hardware or hardware=='unknown' or not pinned_hardware or pinned_hardware=='unknown':
        raise ValueError('known hardware strings are required for identity reference selection')
    identities=lambda rows:[(r['task'],r['id'],r['gold']) for r in rows]
    if not actual or identities(actual)!=identities(pinned):
        raise ValueError('identity check requires paired prediction rows')
    if same_allocation is not None and identities(actual)!=identities(same_allocation):
        raise ValueError('same-allocation prediction rows are not paired')
    same_profile=hardware==pinned_hardware
    if not same_profile and same_allocation is None:
        raise ValueError('different hardware requires a same-allocation reference')
    pinned_equal=prediction_keys(actual)==prediction_keys(pinned)
    local_equal=None if same_allocation is None else prediction_keys(actual)==prediction_keys(same_allocation)
    return {'passed':pinned_equal if same_profile else local_equal,
            'gate_reference':'pinned' if same_profile else 'same_allocation',
            'hardware':hardware,'pinned_hardware':pinned_hardware,
            'predictions_equal_pinned':pinned_equal,'predictions_equal_same_allocation':local_equal,
            'prediction_mismatches_vs_pinned':sum(a['prediction']!=b['prediction'] for a,b in zip(actual,pinned)),
            'cross_job_scope':'gate' if same_profile else 'descriptive',
            'decision':'D-0060'}
