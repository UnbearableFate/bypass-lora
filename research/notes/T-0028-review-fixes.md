# RV-0078 low findings: scoped fixes before formal training

RV-0078 passed with three low findings and no medium/high findings. All three
are addressed before any formal SEG run or SEG dev quality inspection.

1. Preserve `pending_r128` while the primary task awaits its conditional
   width comparison, even if the secondary supports at r32. Secondary results
   remain visible; the replication flag waits until the primary is decided.
   After the width comparison, D-0044(c) is explicitly labelled
   `descriptive_inconclusive_D0044c`. Exact numeric gates are unchanged.
2. Remove `select` from the analysis API, CLI and launcher. Analyze the complete
   r32 grid after its interventions, which can return `pending_r128`; analyze
   the expanded grid only after a valid release. Missing interventions remain
   an explicit incomplete-input error, without a false implementation report.
3. Require deterministic-check `source_commit` and intervention
   `evaluation_source` to equal the immutable formal training source. A fresh
   deterministic regression at that same commit supplies the check receipt.
   Require an explicit complete terminal task-attempt list and measured usage
   for every listed attempt; infer mandatory Base/train/intervention/check
   attempt IDs from their roots and always require ATT-0271-1 smoke usage.
   The list includes failed attempts and any additional regression/reviewer
   attempts. The report states cost coverage, total, 3-node-hour cap and whether
   it is within cap. The analysis job's own usage is explicitly pending until
   terminal import and is added in delivery.

Targeted regressions include the pending-primary/secondary-success boundary,
D-0044(c), mismatched check/intervention source pins, missing usage for Base,
training, intervention, smoke, deterministic checks and failed attempts,
deduplication and reporting of budget excess. The full deterministic protocol
suite is replayed at the exact planned training source, without dev scoring.

No model, core state operator, tokenization, optimizer, training, candidate
scoring, checkpoint, donor-generation or bootstrap numeric path changed.
ATT-0271-1's seven-arm cost measurements remain applicable. The unchanged
default-path regression at that smoke is not repeated: this regression instead
replays all SEG identities/gradients and the unchanged core tests. Hardware
profiles are metadata only, with no slice gate or slice-related analysis.

PBS regression result and measured usage are pending. After it passes, request
the one focused `--recheck-of RV-0078`. All scoped files then remain frozen
until the answer. Full projected cost before this short regression is
2.538108543 node-hours including the optional r128 branch and review reserve;
its measured usage is charged once after termination.
