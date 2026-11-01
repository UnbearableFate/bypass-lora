# T-0003 delivery: E-0003@2 eight-arm screening

All eight requested seed-0 arms completed the locked schedule and passed application validation and exact checkpoint reload. The audited formal analysis selects **F*=F_out (V1-01)** and **C*=C_up (V1-04)**. **Concat did not win** under the preregistered rule: the best concat arm, C_dual, exceeds F_out by only 0.35 pp, below the required 1 pp. Both LoRA baselines outperform C* in this sample. This is single-seed screening evidence against the broad H-0002 advantage in this tokenwise setting, not a confirmation or a verdict on memory-based V2. Hypothesis state and any release of V2 remain the coordinator/user's decisions after conclusion review.

## Results and fixed selections

| Arm | PIQA accuracy | BoolQ accuracy | Macro accuracy | Dev macro NLL | ATT | RES | EV |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| F_out | 73.80% | 81.80% | 77.80% | 1.576770 | ATT-0025-1 | RES-0044 | EV-0040 |
| F_pre | 73.00% | 81.10% | 77.05% | 1.567340 | ATT-0026-1 | RES-0045 | EV-0041 |
| C_gate | 73.40% | 80.80% | 77.10% | 1.577798 | ATT-0027-1 | RES-0046 | EV-0042 |
| C_up | 73.60% | 81.80% | 77.70% | 1.565245 | ATT-0028-1 | RES-0047 | EV-0043 |
| C_dual | 73.90% | 82.40% | 78.15% | 1.561985 | ATT-0029-1 | RES-0048 | EV-0044 |
| Linear C_dual | 73.50% | 82.50% | 78.00% | 1.558246 | ATT-0030-1 | RES-0049 | EV-0045 |
| LoRA4 rank10 | 74.80% | 83.60% | 79.20% | 1.549602 | ATT-0031-1 | RES-0050 | EV-0046 |
| LoRA28 rank2 | 74.90% | 83.20% | 79.05% | 1.553573 | ATT-0032-1 | RES-0051 | EV-0047 |
| Base | 72.30% | 68.00% | 70.15% | 2.337478 | ATT-0024-1 | RES-0024 | EV-0022 |

F_out exceeds F_pre by 0.75 pp, so they are approximately equal under the strict <1 pp rule and F_out is selected. All three C arms are effective (at least 1 pp above Base). C_dual is best, but C_up is only 0.45 pp behind and uses fewer parameters, so C_up is selected. C_gate is 1.05 pp behind C_dual and falls outside the near-best set. The fixed C_dual-versus-C_gate flag is therefore true; this does not override C*=C_up or the overall concat-not-win flag.

The conditional statement that PIQA supplied no distinguishing signal does **not** trigger: not every arm is within Base±1 pp. PIQA changes are modest, while BoolQ accounts for most improvement over Base; the primary metric remains the specified macro average. None of the final arms is near random or at a 90% accuracy ceiling.

## Paired comparisons

Intervals are descriptive 95% percentile intervals from 10,000 task-stratified paired row resamples, seed20261029, with common indices across all arms. They do not include training-seed variation, adjust for selection/multiple comparisons, or replace the fixed point-estimate rules. Positive accuracy differences favor the left arm; negative NLL differences favor the left arm.

| Comparison | Macro accuracy difference pp [95% CI] | Macro NLL difference [95% CI] |
| --- | ---: | ---: |
| C_gate − F_out | -0.700 [-1.700, 0.350] | 0.001029 [-0.019101, 0.021217] |
| C_up − F_out | -0.100 [-0.950, 0.750] | -0.011525 [-0.028371, 0.004816] |
| C_dual − F_out | 0.350 [-0.600, 1.350] | -0.014785 [-0.035271, 0.006015] |
| C_gate − F_pre | 0.050 [-0.850, 0.950] | 0.010459 [-0.006840, 0.027728] |
| C_up − F_pre | 0.650 [-0.300, 1.550] | -0.002095 [-0.018066, 0.014049] |
| C_dual − F_pre | 1.100 [0.100, 2.100] | -0.005355 [-0.025283, 0.015040] |
| C_dual − C_gate | 1.050 [0.100, 2.001] | -0.015814 [-0.036085, 0.004956] |
| C_dual − linear | 0.150 [-0.700, 1.000] | 0.003739 [-0.011733, 0.019508] |
| C* (C_up) − LoRA4 | -1.500 [-2.400, -0.600] | 0.015643 [-0.005661, 0.036544] |
| C* (C_up) − LoRA28 | -1.350 [-2.500, -0.200] | 0.011672 [-0.016174, 0.039528] |

The nonlinear proxy has no clearly separated advantage over its linear degradation control here. Interface parameter counts differ, so these results do not establish parameter efficiency. Per the contract, concat-not-win leaves C_dual as a possible V2 interaction probe; no V2 work was started or authorized by this delivery.

The complete deterministic tables, including per-task NLL, all 36 pairwise comparisons, task-specific learning curves, parameter counts and throughput, are in `research/notes/T-0003-tables.md`. Machine-readable analysis, curves and costs are under `/work/xg24i002/x10041/bypass-lora/runs/ATT-0052-1/`.

## Learning curves and checkpoint rule

| Arm | Macro accuracy at 3270 | At 6539 | At 9808 | Final 13077 |
| --- | ---: | ---: | ---: | ---: |
| F_out | 76.90% | 77.65% | 77.30% | 77.80% |
| F_pre | 75.95% | 76.80% | 76.70% | 77.05% |
| C_gate | 76.20% | 77.15% | 77.30% | 77.10% |
| C_up | 76.60% | 77.60% | 77.80% | 77.70% |
| C_dual | 77.10% | 77.55% | 77.80% | 78.15% |
| Linear | 77.50% | 77.80% | 77.45% | 78.00% |
| LoRA4 | 76.75% | 78.35% | 77.95% | 79.20% |
| LoRA28 | 76.30% | 78.20% | 78.70% | 79.05% |

Intermediate dev records have the same ordered task/id/gold identities as each arm's final record. Their reported accuracy/NLL was recomputed and checked in PBS. They are diagnostic curves only; final checkpoint selection was never changed. Several early NLL values are lower than final NLL, but no early checkpoint was substituted.

## Schedule, data and implementation evidence

Every trained arm: 13,077 steps, 26,154 examples, **2,000,040 non-padding input tokens**, 40-token threshold overshoot, 388,094 supervised tokens, no padding tokens in microbatch1, last_epoch=1 (second epoch). Eight-arm training totals: 16,000,320 input tokens, 3,104,752 supervised tokens, 209,232 examples. Effective batch2, LR3e-4 all families, 654 warmup steps then cosine to zero, identical data-order SHA256 `fcfed383efd9a1672f6e210b8eb52fc2a6039db5c6d868926fb2ddec0e49d376`.

Exactly one train row excluded in each arm: PIQA official train id5528, candidate lengths [537,537]; 1/13065=0.007654% of PIQA train, BoolQ0/6379. Removal occurs after each task's shuffle and before alternation; relative order retained. Both rates are below1%. Dev/calibration unchanged. Locked split-map SHA256 `47cf5c777d07863b8def6903487697eafc970b0cd5b5d80babb7f20ded98571b`. Official validation was not read or evaluated.

Environment unchanged: Qwen/Qwen3-0.6B-Base model/tokenizer revision `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`; lock blob `3c916b51337de3cba594d645e606a31d4f39c443`; Python3.13.13, torch2.10.0+cu128, transformers4.57.1, CUDA module12.8, GH200 MIG2g.24gb, eager attention. FP32 adapters, BF16 frozen Base. Formal runtime snapshot `1fe7d3da9c7dc936ecad99d08f832015e1988b91`.

Reviews: RV-0007 accepted runtime/selection but found missing formal-input checks; fixed and rechecked in **RV-0008 pass, zero findings**. ATT-0022-1/RES-0022/EV-0020:126 checks plus all8 real smokes, exact initial Base equality, parameter counts and reload. ATT-0023-1/RES-0023/EV-0021:145 checks including full-input validation and the actual smoke-path rejection. ATT-0021-1/RES-0021/EV-0019 was a failed test fixture, corrected without weakening the guard; its cost remains counted.

Base rerun ATT-0024-1/RES-0024 passed, and ATT-0033-1/RES-0025/EV-0022 verified exact equality of all2000 dev records, metrics and split map against ATT-0011-2. Final analysis ATT-0052-1/RES-0052 ran the unchanged RV-0008-reviewed `v1_analysis.analyze` in PBS; source commit cd05e46 adds a reporting wrapper only. The wrapper tabulates the existing metrics/ledger and audits curve/exclusion artifacts; it introduces no selection/bootstrap rule. The analysis manifest is complete/validator-passed.

## Resources

| Arm | PBS job | Measured walltime | MIG instance-h | Node-h |
| --- | --- | --- | ---: | ---: |
| F_out | 3759532.opbs | 00:55:07 | .918611 | .22965275 |
| F_pre | 3759533.opbs | 00:53:08 | .885556 | .22138900 |
| C_gate | 3759534.opbs | 00:56:16 | .937778 | .23444450 |
| C_up | 3759535.opbs | 00:53:49 | .896944 | .22423600 |
| C_dual | 3759537.opbs | 00:56:15 | .937500 | .23437500 |
| Linear | 3759538.opbs | 00:55:19 | .921944 | .23048600 |
| LoRA4 | 3759541.opbs | 01:01:06 | 1.018333 | .25458325 |
| LoRA28 | 3759546.opbs | 01:52:04 | 1.867778 | .46694450 |

Eight formal arms: **8.384444 MIG instance-h = 2.096111 node-h**. Additional T-0003 costs: failed fixture .005833333333333333 MIG-h; eight-arm smoke .122778; regression .007778; Base .085278; Base comparison .000278; final analysis .000556. Total **8.606945333333333 MIG instance-h = 2.151736333333333 node-h**, below E-0003's10node-h ceiling. Manual failed-fixture usage differs from the later ledger's six-decimal rounding by0.000000333MIG-h; it is counted once. All other values use ledger rounding. Total T-0003 training input tokens including smoke: **16,031,216**. Tokens are reported facts, not resource ceilings.

No remaining engineering failure or open review finding blocks this delivery. Scientific limitations remain: one seed, one fixed dev split, unequal adapter counts, and no official validation/confirmation. Coordinator conclusion review and the user's V2 budget decision are outstanding downstream actions, not claims completed here.

Combined comparison evidence: **EV-0048 screening/valid, challenges H-0002 in the stated scope**; supported by ATT-0052-1/RES-0052 and all eight arm results. This evidence link does not change H-0002's state. The coordinator should obtain conclusion review before recording the hypothesis or route decision.
