# T-0034 / E-0031 delivery

H-0030 and H-0031 are both **inconclusive** under the sealed exact thresholds.
H-QVU improves the combined macro score over X-01 by only0.016404pp and is
0.574147pp below LoRA RB1-07; S1 exceeds its paired T0 by0.098425pp. The
1.90pp swap drop cannot substitute for the required1pp S1-minus-T0 margin.
Neither new arm passes the practical gate, so rule6 does not release formal
U-25/T0-U-25 controls and rule7 does not trigger more seeds. The computation
and implementation work is complete. Rule10 requires the coordinator's
conclusion review before a route decision; hypothesis states are unchanged.

Scientific execution source: `49f70094b78474ef0158dd631e3b25ea6ebdd174`.
Reviewed by RV-0107, with inherited scope coverage RV-0094/RV-0095.
Locked analysis: ATT-0424-1 / PBS3788665 / RES-0405, complete/passed in10s.
Its initial analysis is also the final task analysis because rule6 is false.
No additional analysis or formal conditional training is required.

Main artifact: [analysis.json](/work/xg24i002/x10041/bypass-lora/runs/ATT-0424-1/analysis.json).
The source inputs are frozen in
[analysis-input-initial.json](/work/xg24i002/x10041/bypass-lora/research/artifacts/T-0034/analysis-input-initial.json).

## Primary scores and comparisons

Combined dev+calibration uses3048rows per task, first computes each task's
accuracy, then their macro average. Accuracy values below are percentages;
NLL is the macro average of task answer-token NLL. Decisions use exact
fractions in the artifact, not these rounded table values.

| Arm | PIQA | BoolQ | Macro | Answer NLL |
| --- | ---: | ---: | ---: | ---: |
| H-QVU | 73.3924 | 82.9396 | 78.1660 | 1.545885 |
| T0-QVU | 73.4580 | 82.6772 | 78.0676 | 1.538858 |
| X-01 | 73.7205 | 82.5787 | 78.1496 | 1.541527 |
| RB1-07 all-7 LoRA r7 | 73.6877 | 83.7927 | 78.7402 | 1.549140 |
| RB1-09 | 73.5236 | 82.8412 | 78.1824 | 1.542337 |
| RB1-01 | 73.0971 | 82.7100 | 77.9035 | 1.497269 |
| RB1-02 | 72.5394 | 82.4475 | 77.4934 | 1.504046 |

| Comparison | Difference, pp | Descriptive95%CI, pp |
| --- | ---: | --- |
| H-QVU minus X-01 | +0.016404 | [-0.557743,+0.590551] |
| H-QVU minus RB1-07 | -0.574147 | [-1.213911,+0.049213] |
| H-QVU minus T0-QVU | +0.098425 | [-0.524934,+0.688976] |
| H-QVU normal minus swap, dev only | +1.900000 | [+0.850000,+2.950000] |
| T0-QVU minus X-01 | -0.082021 | [-0.721785,+0.557743] |
| T0-QVU minus RB1-07 | -0.672572 | [-1.279528,-0.082021] |

H-0030 is inconclusive because d_X is strictly positive but below+1pp,
and d_L is below0. H-0031 is inconclusive because d_h is strictly positive
but below+1pp, regardless of the swap check. T0 fails the practical gate.
These are the locked classifications, not post-hoc significance decisions.

Dev-only macro accuracy is H-QVU78.10%,T0-QVU78.05%,X-01 78.15%,
X-02 78.30%,RB1-07 78.95%,Base70.15%. H-minus-X is-0.05pp,
H-minus-LoRA-0.85pp, and H-minus-T0+0.05pp (95%CI[-1.05,+1.15]).
Both new arms satisfy learnability: Base+7.95/+7.90pp.

## Fixed dev curves

| Training fraction / step | H macro,% | H NLL | T0 macro,% | T0 NLL |
| --- | ---: | ---: | ---: | ---: |
| 25% /3279 | 76.20 | 1.619104 | 75.65 | 1.591184 |
| 50% /6558 | 77.25 | 1.546737 | 76.90 | 1.545648 |
| 75% /9837 | 77.85 | 1.562433 | 78.25 | 1.546444 |
| 100% /13115 | 78.10 | 1.562880 | 78.05 | 1.556738 |

## Implementation and producer validation

The shared X-tap read precedes attention. The same z writes q before
q_norm/RoPE, v before GQA expansion and up before the FFN product; original
attention and FFN execute once. All three exits start at zero. CPU initialization
is elementwise paired for H/T0; shared SHA is
`d177ba7ed7a7c19a2787858edc6e53a7a50018cddd7c92625b9fa37a1d4024ca`.
The90PBS tests cover ordering, zero exits, gradients/frozen backbone,
suffix causality, left/right padding, reload and missing diagnostic checkpoint
handling. Real first-step probe gradients are nonzero at every exit; the
backbone is frozen. Stored/trainable/with-grad counts are629000 for both
new arms. U25 smoke counts619400; no formal U25 control was released.

Both formal arms use actual2g.24gb, identical R1 order, seed0, LR1e-3,
eta0.25,13115steps,26230examples,2000069inputtokens and387653answer-tokens.
Full step0/iv_off dev predictions equal their independent same-allocation
Base references; dev/cal checkpoint reloads match exactly including scores.
Runtime steady step seconds are0.231128618(H)/0.206598341(T0), with peak
CUDA3367694848/3364760576bytes. These are run costs, not a controlled latency
benchmark. Formal results are RES-0390/RES-0391; implementation evidence EV-0178.

Every historical reevaluation is separate from the smoke and uses its
training profile, full1000dev/2048calibration rows per task and0dev flips:

| Arm | Producer reevaluation | Profile | Result |
| --- | --- | --- | --- |
| RB1-01 | ATT-0411-1 /3786999 | 2g | RES-0388 |
| RB1-02 | ATT-0412-3 /3788306 | 1g | RES-0400 |
| RB1-07 | ATT-0413-1 /3787002 | 2g | RES-0389 |
| RB1-09 | ATT-0414-2 /3787113 | 2g | RES-0392 |

The four compatibility entries in the final analysis all show
same_profile=true/flips0. Actual guards, their SHA references, provenance
and the mandatory pre-analysis checks are in
[T-0034-profile-binding-audit.md](T-0034-profile-binding-audit.md), EV-0180,
and the immutable profile binding artifact referenced by the input.
No unavailable exception was used. RB1-02 had3wrong-profile rejections,
then two successes. The minimum successful job ID3788306 was selected as
pre-specified in MSG-1061. Duplicate ATT-0412-5/3788308 (RES-0401) has
byte-identical full dev/calibration JSON, including every likelihood score
and NLL; its measured usage is counted.

H intervention ATT-0421-1/3788305 (RES-0399) uses successful producer
ATT-0409-3 and actual2g. The normal condition reproduces the training dev
predictions; same-task deranged donor history swap retains current-token
contribution. The old afterok child ATT-0420-1 never started and is cancelled.

## Rule8 resolution diagnostics

| Combined comparison | Difference, pp | Descriptive95%CI, pp |
| --- | ---: | --- |
| RB1-02 minus RB1-01 | -0.410105 | [-1.033465,+0.213255] |
| RB1-07 minus RB1-09 | +0.557743 | [-0.098835,+1.213911] |
| RB1-07 minus X-01 | +0.590551 | [-0.049213,+1.246719] |

## Per-rule evidence and continuation

| Rule | Result | Evidence |
| --- | --- | --- |
| 1 | All implementation, reload, matching-profile compatibility checks pass | EV-0185 |
| 2 | Both new arms >=Base+1pp on dev | EV-0186 |
| 3 | H-0030 inconclusive | EV-0187 |
| 4 | H-0031 inconclusive; swap alone is insufficient | EV-0188 |
| 5 | T0 practical gate false | EV-0189 |
| 6 | Conditional controls not triggered; no formal U25/T0-U25 | EV-0190 |
| 7 | Extra-seed/practical-T0 replication not triggered | EV-0191 |
| 8 | All retained checkpoints available; intervals above | EV-0192 |
| 9 | Fixed strata/seed; screening scope/selection limitations retained | EV-0193 |
| 10 | Coordinator stop-review condition true; no route decision made here | EV-0194 |

EV-0187/0188 are neutral screening records about E-0031. No directional
supports/challenges edge or hypothesis-state change was added for these
inconclusive outcomes. Include these IDs and RES-0405/analysis artifact in
the coordinator's conclusion-review proposal; the experiment's complete
arm table is also available in the review package. The coordinator owns
the conclusion review and subsequent route decision.

All intervals use10000paired bootstrap draws, seed20261101, dev/calibration
separately stratified by task and common indices across arms. They describe
row sampling only. Calibration rows are disjoint from training but were
previously used for LR/eta selection; q/v LR was not separately calibrated.
Scope is only this ordinary RB setting, model, seed0 and LR1e-3. A2 and
official validation stayed sealed. No wider/tap/eta/initialization search,
new seeds, mixed LoRA route or ranking-loss experiment was started.

## Measured usage and closure

Final native ledger totals: **4.180837MIG-instance hours x0.25 =
1.04520925node-hours**, below initial1.3 and total2.0. The18charged jobs
used15051scheduler seconds, including all7wrong-profile rejections, both
passed smokes, two trainings, five legacy reevaluations including the duplicate,
the successful intervention and this locked analysis. Requested walltimes
are not used as measured costs. Native six-decimal ledger rounding is kept.

All19submitted job references are ended:11success,7guard failures,1cancelled
FINISH no-start. The cancelled dependency had no workload/allocation or
exported usage; two earlier prepared-only registrations were retired before
any submission. Every available native accounting_ref has been imported
once. See T-0034-final-cost.json and terminal observation files for details.

No workload, scientific analysis or tests ran on login. No merge/rebase/push
or project cleanup was performed. No further run is required by this task;
coordinator acceptance/merge and independent conclusion review are next.
