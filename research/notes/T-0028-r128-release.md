# T-0028 preregistered r128 release

The r32 result is frozen in ATT-0288-1 / RES-0266 and T-0028-r32-outcomes.
All implementation gates passed; the primary PIQA gain is exact2/125
(+1.6 pp), so the sealed0<gain<3 pp rule6 releases SEG-05..07 once. These
runs are preregistered, and do not use D-0052's exploratory fallback. The
r32 direction remains pending_r128; no hypothesis status or threshold changes.

| Arm | Attempt | PBS job | Expected s | Walltime | Parameters |
|---|---|---|---:|---|---:|
| SEG-05 | ATT-0290-1 | 3772011.opbs | 7393 | 03:10:00 | 2,752,512 |
| SEG-06 | ATT-0291-1 | 3772012.opbs | 3866 | 01:40:00 | 2,752,512 |
| SEG-07 | ATT-0292-1 | 3772013.opbs | 3562 | 01:30:00 | 2,752,512 |

All three use exactly reviewed7d8b346, R1 seed0 initialization/order,
13,115steps/26,230presentations, the committed familyLR0.003 and unchanged
total512 segmented encoding/V1 candidate objective. SEG-05 keeps state;
SEG-06 resets and skips segment1; SEG-07 is the paired T0. They start fresh,
not from r32 weights. Complete dev curves, final1000rows/task, accounting,
checkpoint inventories and same-job reload are required. SEG-05 additionally
checks same-job initial/Base identity. D-0049 covers the inherited test-only
stale label; RV-0079 covers every executed path. Profiles are metadata only.

Usage through r32 analysis is20,998MIGs=1.458194444nodeh. The earlier smoke
projection was2.598351472nodeh including the width branch and reserve. New
full-r32 measurements now allow a more conservative sizing: for each r128
arm take the larger of its own smoke part model and the observed full-r32
same-protocol duration. This raises the projected total to **2.653472222
node-hours**, including all three trainings, SEG-05 controls (1671expected
seconds) and0.05nodeh final analysis/review reserve. This is below the existing
3nodeh cap, so no budget revision is needed. Requested walltimes are1.5times
these measured class estimates, rounded up to10minutes. Queue wait is unknown.

The three-job group **N-20261102-122622-0842** is sealed with early-failure
notification; the notifier daemon is confirmed running. The coordinator will
listen and resume this researcher. On completion verify/import every output
and usage once, then SEG-05 off/swap/reset and final locked10000-bootstrap
analysis at the same7d8b346 source. Add every terminal task attempt, including
r32 analysis and any descriptive audit, to the final cost list. Preserve the
r32 analysis and evidence context separately. Final report must include all
rule outcomes, descriptive intervals, per-row seam records/summary, actual
state/tokens/parameter/timing counts and full cost. No confirmation data or
hypothesis-status changes are part of this task.
