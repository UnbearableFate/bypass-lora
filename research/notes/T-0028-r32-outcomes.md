# T-0028 locked r32 outcomes, recorded before r128

ATT-0288-1 / PBS3771942 / RES-0266 is complete and
validator-passed. All paired raw rows, the five SEG-02 conditions, exact
off/Base identity, training/checkpoint/source references, donor derangement,
all initial formal artifacts and the complete nine-attempt cost input pass.
The 10,000-sample shared task-stratified bootstrap uses seed20261101. The
16-second terminal observation, measured usage and result were imported once.

| Rule | Recorded r32 outcome |
|---|---|
| 1 implementation | All four checks pass: off/Base, initialization, no-channel, reload. Diagnostic. |
| 2 dynamic range | Both tasks informative: full-context LoRA minus SEG-04 is +7.0 pp PIQA and +19.3 pp BoolQ. |
| 3 primary | PIQA. |
| 4 learnability | T0 is below Base+1 pp; keep/reset/LoRA are above. The two no-history controls are not both unlearned, so screening interpretation remains available. |
| 5 r32 comparison | PIQA gain = +1.6 pp (exact 2/125), swap drop = 4.2 pp; gain is below the +3 pp support gate. Descriptive at r32. |
| 6 width | **Release SEG-05..07 once**, because 0 < 1.6 < 3 pp on primary PIQA. D-0044 remains unchanged. |
| 7 r32 challenge | Does not fire: gain is positive. |
| 8 secondary | BoolQ gain +2.0 pp, swap drop5.3 pp; descriptive, no secondary-only support and no replication flag while the primary width comparison is pending. |
| 9 summaries | Macro and NLL descriptive only; no hypothesis status change. |

The max-control gain's descriptive 95% interval is [-0.3,+3.5] pp on PIQA
and [-0.8,+4.9] pp on BoolQ. These are data intervals conditional on the
trained seed0 models. They exclude seed and task/width-selection uncertainty,
are not simultaneous intervals, and are not the numerical release gates.
Threshold decisions use exact point-estimate fractions.

The r32 direction is **pending_r128**, with no supports/challenges edge.
These r32 outcomes are preserved separately from the later expanded analysis.
D-0052's requested exploratory fallback is not used because the original
preregistered release rule itself fires. No sealed contract or threshold is
changed in response to the results. Evidence records and interpretation of the
completed width comparison remain screening; hypothesis status is owned by
the coordinator after conclusion review.

Total measured usage through this analysis is **20,998 MIG seconds =
1.458194444 node-hours**. The complete r128 branch (three trainings plus
SEG-05 interventions) is projected at1.090157028 node-hours. Including0.05
node-hour remaining analysis/review headroom gives **2.598351472 node-hours**,
under3. No budget revision is needed. SEG-05..07 use the reviewed7d8b346 source,
2752512 parameters per arm, unchanged R1 initialization/order/13115steps/
26230examples and the RBc R/T LR0.003. Profiles remain metadata only; no
confirmation data or hypothesis-status change is introduced.
