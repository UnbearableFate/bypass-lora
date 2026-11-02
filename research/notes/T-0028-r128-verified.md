# T-0028 r128 grid verified; SEG-05 interventions pending

All three conditional width runs are complete at scientific source
7d8b3464b6145504d155268f4c667fa3ecc53b00, covered by RV-0079 and D-0049.
The recorded inherited-test-only stale label does not change that coverage.
Native terminal facts and measured usage were imported once. Results:

| Arm | Attempt | Result | PBS job | Scheduler seconds | PIQA % | BoolQ % | Macro % |
|---|---|---|---|---:|---:|---:|---:|
| SEG-05 keep r128 | ATT-0290-1 | RES-0275 | 3772011.opbs | 7676 | 69.3 | 65.9 | 67.60 |
| SEG-06 reset r128 | ATT-0291-1 | RES-0276 | 3772012.opbs | 3568 | 66.9 | 61.5 | 64.20 |
| SEG-07 T0 r128 | ATT-0292-1 | RES-0277 | 3772013.opbs | 3403 | 67.9 | 61.2 | 64.55 |

PBS artifact audit ATT-0301-1 / RES-0278 / PBS3773187 passed in 17 scheduler
seconds. Its read-only wrapper source 8b3ef43 calls the reviewed validators;
scientific training, interventions and the locked analysis remain 7d8b346.
The audit authenticates all eight final arms, 13,115 steps / 26,230
presentations for every trained arm, R1 order, all step LRs, exact lock/input/
source/environment/model pins, full 1000 dev rows per task, candidate counts,
curves, checkpoints and full same-job reloads. SEG-05 also passes initial
Base_seg identity. Each r128 arm has 2,752,512 stored/trainable/with-gradient
parameters and LR .003. SEG-05 carries 264,192 FP32 M+c bytes plus 32 int64
count bytes, and executes 1,433,313 first-segment + 566,756 second-segment
training tokens. SEG-06/07 execute only the 566,756 second-segment tokens.
The common exclusion remains PIQA5528 (537-token complete candidates).

The expanded audit also verifies identical encoding across all arms, the
512-token total budget with V1 longest-candidate reserve, and per-row
agreement with the retained V1 prompt. There are **zero seam-edited prompts
or changed tokens** in both training and dev for both tasks:

| Split/task | Presentations | Unique IDs | Encoded segment1 tokens | Segment2 prompt tokens | First-segment cropped rows/tokens |
|---|---:|---:|---:|---:|---:|
| Train PIQA | 16458 | 12064 | 174958 | 32916 | 1 / 7 |
| Train BoolQ | 9772 | 5379 | 1258355 | 146187 | 18 / 1641 |
| Dev PIQA | 1000 | 1000 | 10640 | 2000 | 0 / 0 |
| Dev BoolQ | 1000 | 1000 | 129803 | 14968 | 2 / 298 |

These are encoding counts. The no-channel arms skip segment1 execution.
Training counts include repeated presentations. The crops are recorded
alongside zero changes versus retained V1 prompt tokens; they are not
additional information removed at the seam. Full summaries and FileRefs are
in [audit.json](/work/xg24i002/x10041/bypass-lora/runs/ATT-0301-1/audit.json).
Per-row records include
[seams-train.json](/work/xg24i002/x10041/bypass-lora/runs/ATT-0290-1/seams-train.json)
and [evaluation-audit.json](/work/xg24i002/x10041/bypass-lora/runs/ATT-0290-1/evaluation-audit.json);
the audit inventories equivalent files for every other arm.

Usage through the expanded audit is **35,662 MIG seconds = 2.476527778
node-hours**, using the exact scheduler seconds and factor .25. The native
six-decimal usage totals will also be reported. No failed compute attempts
have occurred. The remaining measured-class projection is 1671 seconds for
SEG-05 controls plus the locked analysis; the complete projection remains
under the unchanged 3-node-hour cap.

SEG-05 interventions are submitted as **ATT-0302-1 / PBS3773252.opbs**, frozen
7d8b346 source and validated ATT-0290-1 adapter. It runs Base_seg, normal,
iv_off, iv_swap and iv_reset on all 1000 dev rows per task. Expected 1671s;
walltime 00:50:00, one short-mig instance. Profiles are metadata only.
Notification **N-20261102-144810-2db4** is sealed at one job, with early
failure enabled and notifier daemon running. The exact listener is in
T-0028-r128-intervention-job.json and sent to the coordinator.

The final analysis input is prepared in the main project at
research/artifacts/T-0028/final-analysis-input.json. Its 15-attempt terminal
cost inventory includes both preflights, every formal arm, both audits,
both intervention jobs and the preserved r32 analysis. It is not runnable
until ATT-0302-1 completes and its native terminal usage export exists.
Run the reviewed 10,000-sample task-stratified bootstrap and exact D-0044
rules there, then append the current analysis terminal cost in delivery.
Preserve ATT-0288-1 / RES-0266 and T-0028-r32-outcomes separately.

The width triplet was released by sealed rule6 (primary PIQA gain32 +1.6pp).
D-0052's exploratory fallback was unused. These are seed0 dev screening
observations; no final hypothesis direction, route, confirmation access or
status change is recorded here. The final rule assessment and per-rule
evidence await the complete intervention grid and locked PBS analysis.
