# T-0028 initial grid verified; frozen interventions submitted

All five initial runs are complete and independently audited in PBS by the
reviewed artifact validator. Scientific source remains **7d8b346**, covered
by RV-0079 and the D-0049 inherited-test-only exception. Audit wrapper
13b03b1 only calls those validators and descriptive summary functions; no
training, scoring, operator, donor, checkpoint or gate code changed.

ATT-0286-1 / PBS3771591 / RES-0264 passed in 17 scheduler seconds. It verifies
all inventories and checkpoint hashes, exact source/model/tokenizer/env/LR
pins, all 13,115 logged step LRs, token/budget/accounting totals, final paired
1000 dev rows/task, all three full-dev curves and candidate-token counts,
the common exclusion PIQA5528, full same-job reload, SEG-02 initialization
identity and the exact-source no-channel receipt ATT-0276-1. Rule 1(a),
iv_off/Base identity, remains pending until the intervention job completes.
The audit, all raw data checks and numerical summaries ran only in PBS.

Native terminal observations and usage were imported once through the
structured ledger export, without grep or manual duplicate usage. Results:

| Arm | Attempt | Result | Scheduler seconds | PIQA % | BoolQ % | Macro % |
|---|---|---|---:|---:|---:|---:|
| SEG-00 | ATT-0277-1 | RES-0259 | 179 | 66.1 | 60.9 | 63.50 |
| SEG-01 | ATT-0278-1 | RES-0260 | 3562 | 66.9 | 63.8 | 65.35 |
| SEG-02 | ATT-0279-1 | RES-0261 | 7393 | 69.4 | 65.8 | 67.60 |
| SEG-03 | ATT-0280-1 | RES-0262 | 3866 | 67.8 | 61.2 | 64.50 |
| SEG-04 | ATT-0281-1 | RES-0263 | 3857 | 67.4 | 64.2 | 65.80 |

These are seed0 dev point summaries, with no hypothesis direction assigned.
Intervals and the locked rule reducer await the complete intervention grid.
All four training arms have 26,230 presentations and 13,115 steps at the
locked R1 order. The keep arm executes 1,433,313 first-segment and 566,756
second-segment tokens (2,000,069 total); the other arms execute only the
566,756 second-segment tokens. T0/keep/reset have 663,552 stored/trainable/
with-gradient parameters, LoRA 630,784. Only keep carries state across the
boundary: 16,896 FP32 M+c bytes plus 32 int64 count bytes. Per-row token-seam
counts are retained in every formal run's seams-train.json and evaluation
artifact; include their descriptive summaries or links in the final report.

Initial formal executions total **18,857 MIG seconds**. Adding the 477-second
smoke, 55-second regression and 17-second audit gives **19,406 MIG seconds =
1.347638889 node-hours**. Native six-decimal MIG-hour accounting gives
1.347638750 node-hours. The remaining projection includes SEG-02 interventions,
the entire optional r128 branch and analysis/review headroom and stays below
the 3-node-hour cap. No failed compute attempts have occurred in this task.

Frozen SEG-02 interventions are submitted as **ATT-0287-1 / PBS3771652** at
the same reviewed 7d8b346 source and authenticated ATT-0279-1 final adapter.
One job runs Base_seg, normal, iv_off, iv_swap and iv_reset on full dev. The
part-model expectation is 1619 seconds; walltime 00:50:00 (1.5x rounded to ten
minutes), short-mig, one MIG instance, profile metadata only. Notification
**N-20261102-114201-1adf** is sealed at one job, with early failures enabled.

The r32 analysis input is prepared at
research/artifacts/T-0028/r32-analysis-input.json in the main project. It
contains every task attempt's cost and the exact-source deterministic FileRef.
It is not executable until ATT-0287-1 is complete and its terminal export
exists. After verifying/importing that job, run the locked 10,000-sample
analysis at 7d8b346. Release SEG-05..07 only if the exact primary-task rule 6
fires, under D-0044 and the updated budget. No confirmation data, hypothesis
status or coordinator-owned route decision was changed.
