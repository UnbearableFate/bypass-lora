# T-0007 E-0007@2 diagnostic implementation

Post-merge baseline: ATT-0046-1 / RES-0038 / EV-0035, commit
3ccde7d9dfe595bf5bbcf88b55c597be10f5a210: 155 passed, no skips/failures,
46.759s pytest, 49s PBS, 0.013611 MIG-h = 0.00340275 node-h.

Explicit split parameter in v2_eval defaults to dev. Only dev and calibration
are allowed, and every row must match the requested split; confirmation always
fails. No identity relabeling. Diagnostic entry reads and checks only the locked
calibration file (2048 unique IDs). Original evaluate runs at batch 1 and batch 8
on calibration; diagnostic E16 batch 8 is checked against the original results.

Subsets use derive_seed(20261029, 'concat_loop_v2', 'V2', 'diag-fact-subset'), then
an episode-ID-derived random permutation of distractors. Prefixes of that
permutation yield nested E=1,2,4,8,16 views; all retain the target and preserve
source relative order. Source identities remain unchanged and source fact indices
are recorded. Rendering repeats the exact literal template in make_episode
(which has no standalone renderer); E16 equality with original text and tokens
is checked on every row. Token boundary checks use the shared prompt_tokens.

Rank is competition rank 1 + number of strictly greater logits, with tie count
reported. Top-5 and candidate argmax break ties by ascending token ID. Category
priority: correct, subject, same-relation tail, other tail, other head, other name,
non-name. The name universe is all eligible entries in v2_data.CANDIDATES, including
both seen and unseen pools; it is not an open-ended linguistic name classifier.
Subject copy is separately measured, including correct self edges. Conditional
and unconditional candidate scores, coverage, empties, nonempty random baseline,
and attainable random baseline follow E-0007@2. E1 candidate b is not interpreted;
nonself E1 candidate a is reported as the two-choice control. Synthetic frequency
strata denote generator sampling weights, not pretraining frequencies.

Five examples are fixed to the first five calibration identities with E=1,16,1,16,16,
independent of their scores. Output examples.md is copied into final notes.
No training, dev inference, confirmation reads, or H-0003 inference.

Smoke ATT-0047-1 passed 13 tests but stopped on the strict original-path check.
ATT-0048-1 independently confirmed that moving the identical logits to CPU before
CE changes its reduction result: first M batch max difference 0.00006485;
smoke M/F max differences 0.00007248/0.00006962, zero top-1 differences. Fixed
by computing score_logits on the original CUDA batch before transferring logits
for diagnostic summaries. No tolerance, metric, protocol, or data change.
