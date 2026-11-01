# T-0003 implementation and preregistered analysis

E-0003 uses the sealed task contract and the E-0002 locked environment (uv.lock blob 3c916b51337de3cba594d645e606a31d4f39c443; cuda/12.8). No official validation is accessed.

Implementation: optional E-0003 train exclusion audit, followed by removal after each task's deterministic shuffle and before alternation. Original split map remains unchanged. Any per-task exclusion fraction strictly above 1% stops the run. Each run archives excluded row task/id/candidate token lengths/reason. The planned full schedule must reproduce 13,077 steps, 2,000,040 input tokens, and dev points 3270/6539/9808. Final checkpoint is the only primary result. Intermediate dev evaluation restores all module modes and Python/NumPy/Torch CPU/CUDA RNG states; it does not touch the optimizer or data iterator.

All eight arms use LR 3e-4, batch 2, microbatch 1. Parameter reporting distinguishes total stored model parameters (including frozen Base), adapter stored parameters, trainable parameters, and parameters with an allocated gradient observed during training (zero-valued gradients count as gradients). Adapter stored/trainable/with-gradient counts must equal the contract. Smoke checks exact zero-exit initialization and exact checkpoint reload for every arm. Evaluation now accepts explicit split names; therefore Base will be rerun, and its dev predictions checked against ATT-0011-2 before using the reference.

Before any formal run: bootstrap uses 10,000 replicates, NumPy default_rng seed 20261029, sampling with replacement within each task (1,000 rows each). Every replicate shares task row indices across all arms and Base. Report two-sided percentile 95% intervals for paired macro-accuracy differences, with raw per-task accuracy and token-weighted task NLL and their macro means. No seed-level uncertainty or confirmatory significance claim is inferred. F* uses the contract's <1 pp equivalence rule with F_out preferred. The exact operational definition of an "effective" C candidate and the equal-parameter C_gate/C_up tie remain for coordinator clarification before formal runs.

Smoke timing is measured after the first 10 of 20 steps for all eight arms. Requested walltime will use measured training/evaluation/save/load durations times 1.5; MIG usage is reported both as instance-hours and node-hours (multiply by .25). E-0003 total ceiling is 10 node-hours including tests and diagnostics.

Status: implementation in progress; no T-0003 PBS checks or formal runs have yet executed.

## E-0003@2 clarification before formal runs (MSG-0063)

Effective means dev macro accuracy at least 1 pp above the rerun Base. Among effective C arms, retain those strictly less than 1 pp below the best effective C arm, then prefer fewer parameters. Equal parameter counts use higher macro accuracy, then lower dev macro NLL, then C_gate. No effective C arm means no C*. Concat wins only when C* exists and the highest scoring concat arm is at least 1 pp above F*. C_dual improves on C_gate only at a difference of at least 1 pp. All decisions use point estimates; bootstrap intervals remain descriptive. This supersedes the unresolved definition above. Formal registrations use E-0003@2.

ATT-0021-1 / RES-0021: 116 tests passed, one new test fixture failed before any smoke training. The fixture supplied one invalid row out of one, correctly triggering the >1% guard while expecting an audit record. Corrected to one invalid row out of 100; the production guard is unchanged. Also corrected the held-out comparison dictionary to key by task and split so both dev and calibration are checked. PBS 3759153.opbs, mg1098, Exit_status 1, 21 s = 0.005833333333333333 MIG instance-h = 0.0014583333333333333 node-h; zero training tokens. Retest pending.

## Successful pre-review PBS validation

ATT-0022-1 / RES-0022 executed immutable b933e23cc64514fe4145e4bad6f49e19c52c15cc under E-0003@2 / DOC-0040@2. PBS 3759211.opbs on GH200 MIG 2g.24gb: 126/126 tests passed; all eight real 20-step smokes passed, including exact initial logits against Base, intermediate dev at steps 5/10/15, finite final dev, exact reload, and adapter stored/trainable/with-gradient counts matching the contract. Each consumed 3,862 input tokens and 590 supervised tokens; aggregate 30,896 input tokens. These tiny dev measurements are engineering checks, not interface evidence.

Every arm reproduced the full schedule: 13,077 steps, 2,000,040 input tokens (40 overshoot), dev points 3270/6539/9808, and order SHA256 fcfed383efd9a1672f6e210b8eb52fc2a6039db5c6d868926fb2ddec0e49d376. Every exclusion artifact contains only PIQA id 5528 with candidate lengths [537,537].

| Arm | Adapter stored = trainable = with gradient | Total model stored | Steady step seconds (last 10) | Final 32/task dev seconds | Reload seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| F_out | 262144 | 596312064 | .203224 | 4.811680 | 5.088158 |
| F_pre | 262144 | 596312064 | .223815 | 5.485355 | 5.804770 |
| C_gate | 524288 | 596574208 | .207837 | 5.125572 | 5.425015 |
| C_up | 524288 | 596574208 | .205219 | 4.914017 | 5.076872 |
| C_dual | 917504 | 596967424 | .210398 | 5.166461 | 5.683372 |
| linear | 917504 | 596967424 | .217786 | 5.012502 | 5.252097 |
| LoRA4 | 901120 | 596951040 | .223529 | 4.879263 | 5.293577 |
| LoRA28 | 1261568 | 597311488 | .435291 | 8.368681 | 8.515534 |

Ledger measured cost ATT-0022-1: 442 seconds, 0.122778 MIG instance-h = 0.0306945 node-h. Prior failed fixture job: 21 seconds, manual usage U-0034 recorded 0.005833333333333333 MIG instance-h; ledger later confirmed rounded 0.005833. The prior cost is counted once; its matching ledger observation is archived without re-importing a second usage accounting ref. Combined observed cost is about 0.032153 node-h. No formal E-0003 training run or full Base rerun has yet executed.

Authorship: parent Codex model identity not externally observed (unknown); data/analysis/smoke helper used gpt-6-luna at high reasoning. Parent inspected helper edits; PBS performed all runtime validation. Code-review scope includes analysis decision rules and unchanged adapter/LoRA dependencies. Formal runs remain gated on review.
