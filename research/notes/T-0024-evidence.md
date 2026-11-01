# E-0020 RBc learning-rate calibration evidence

Diagnostic evidence about E-0020@1 (DOC-0231@1) and preparation for H-0013@1. This evidence records the registered LR choices; it supplies neither support nor challenge for the efficacy hypothesis. No hypothesis state is changed.

## Result and exact selection rule

The reviewed rb_train.py lock CLI completed in PBS ATT-0236-1 / RES-0214 / 3769071.opbs at source 77f1e9b1f200940d733ba76afd807af4a7516498. All sixteen candidates are complete, finite and valid. Each initial raw finite minimum was at the upper initial endpoint, so exactly one upper point was added per family. The final choice is the smallest LR whose relative excess over that family's finite minimum is at most 0.01.

| Family | Initial grid | Required extension | Final raw minimum LR / NLL | Locked LR | Selected NLL | Boundary minimum |
| --- | --- | --- | --- | --- | --- | --- |
| RBc-L, LoRA4 r7 | 1e-4,3e-4,1e-3 | 3e-3 | 1e-3 / 1.583488263911061 | 3e-4 | 1.5967402533514987 | false |
| RBc-T, R_res+T0 S1 | 1e-4,3e-4,1e-3 | 3e-3 | 3e-3 / 1.5528074713722433 | 3e-3 | 1.5528074713722433 | true: grid boundary |
| RBc-R, R_res+KernelMatrix S1 | 3e-4,1e-3,3e-3 | 1e-2 | 3e-3 / 1.559168157001273 | 3e-3 | 1.559168157001273 | false |
| RBc-C, C_dual+KernelMatrix S1 | 3e-4,1e-3,3e-3 | 1e-2 | 3e-3 / 1.585611915833004 | 1e-3 | 1.5936240153439594 | false |

Decisions, from the PBS report's unrounded relative excess values:
- L: 3e-4 is 0.008368858641052754 above the minimum, within 1%; 1e-4 is 0.035725292306728426 above it and excluded.
- T: the new 3e-3 point is the minimum; the next smaller 1e-3 is 0.02827225990134682 above it. Lock 3e-3, label grid boundary, and do not expand again.
- R: 3e-3 is the minimum; 1e-3 is 0.016880123914515544 above it and excluded. The extension 1e-2 is 0.01027321697864741 above it, also outside 1%.
- C: 1e-3 is 0.005053001576836797 above the minimum, within 1%; 3e-4 is 0.018555928997850385 above it and excluded.

Planned RB1 mapping is L -> RB1-07, T -> RB1-01, R -> RB1-02..06, C -> RB1-08 and RB1-09. S2/S4, fixed-query and eta/4 inherit the R-family LR; C_up inherits C_dual's LR, as registered. This calibration has not independently optimized those variants.

## Locked artifacts

Final committed lock: research/artifacts/T-0024/lr-lock.json, 3646 bytes, SHA256
38b8ee24af3ded8aee398a4cc3dfaa5a1dfece4737a6cde34642fbff7bf1012d.
It includes all sixteen input manifest paths/bytes/SHA256 values, experiment E-0020@1, family LRs, boundary flags and the A2 split SHA.

Raw reviewed CLI output: research/artifacts/T-0024/lr-lock-reviewed.json, 3555 bytes, SHA256
3bbcd9fbf2feb8f3e2cf19f04da5fa338085e2190043b77ddd190763e5c86503.
The final lock differs only by the authenticated plan_order_sha256 field, preserving every reviewed LR decision.

A2 split manifest SHA256: 34277aa4627271dbf4e0bcb9592e3014c6130d87b1766caad17c2d6b97a8a7cc.
Common V0 plan order SHA256: 053284831520676982857bf9677a9fa3e338b0bbad4d2ce2af1c3f88d20457d3.

Complete candidate table (family, LR, NLL/finite, steps, tokens, stable-step seconds and exact per-run node·h): [T-0024-candidate-costs.md](T-0024-candidate-costs.md). It is byte-identical to ATT-0236-1/candidate-table.md, SHA256 85c283f61ed9f5d8fbcb25a0895d6e0a49a694748fc88ed44af28f1247a0cb24.

Full verification and rule output: research/artifacts/T-0024/final-report.json, 43304 bytes, SHA256
9fd7c94bcc473ea79760e2f40057efa2fc1ea4e01c4db93bacf16e2bd35e6b93.
It is byte-identical to ATT-0236-1/report.json.

## Validation and scope

All workloads ran through Miyabi PBS with startup allocation checks. Initial twelve calibrations executed source 6df9e479b229feb3d5edfac588b06264612c7839; four endpoints executed ef1f2d549913ef67fcd4d450d2913b04a97d2829. Their scientific trainer, core/model, data, scoring and uv.lock files were checked against RV-0062's reviewed snapshot 183ad685785b8249d8a1c679a09e8dd5280d8cd5. No dependencies, tolerances or training/scoring definitions changed. The only new src file is rbc_report.py, an artifact checker that calls the unchanged reviewed extension_lr and lock CLI.

Initial artifact validation ATT-0222-1/RES-0205 verified all12 candidates and generated the required extensions. Final ATT-0236-1/RES-0214 verified all16 independently before calling the lock CLI: correct registered spec/source/arm/family/LR/seed0/effective phaseV0/stagecalibration; complete/passed manifests; pinned Qwen3-0.6B-Base model/tokenizer revision; expected stored/trainable/with-gradient parameters; matching split and plan order; complete-step budget and side write/read accounting; no dev monitoring, export, checkpoint or reload.

Every candidate has 1096 optimizer steps, 2192 example presentations and 200229 nonpadding input tokens. The penultimate step is below 200000, so the final complete step is the first crossing. Across16 candidates this is 17536 steps,35072 presentations and3203664 input tokens; the contract's3.2M is the rounded complete-step estimate.

Each evaluation scores all2048 PIQA and2048 BoolQ calibration examples. The report reconstructed each task NLL from saved gold CE sums divided by gold-token count, then the two-task arithmetic mean, and required exact equality with the reported calibration.nll. Training retains V1's separate per-example supervised-token mean objective, divided by step example count before backward. Initialization and training-order derivations use residual_la_v21/V0/seed0. No dev, RB confirmation or official validation was evaluated; no RB1 was trained.

The prior RB0 unit/model/CLI regression suite was not rerun because its reviewed scientific code is unchanged. This task validated the actual full calibration entry and real lock CLI. There is no calibration checkpoint/reload claim: RBc deliberately performs one calibration pass and no adapter export/reload; checkpoint behavior was covered by RB0. Slice profiles are metadata only under D-0018/D-0019; no slice analysis or reruns were performed. Stable step times are descriptive training metrics, not controlled timing comparisons.

This seed0 short-budget LR selection does not estimate a treatment effect or test H-0013's efficacy thresholds. T's boundary result does not establish a global optimum. No implementation issue remains; RB1 release and task acceptance belong to the coordinator.

## Run/result/accounting inventory

| Attempt | Result | PBS job | Exact MIG seconds | node·h |
| --- | --- | --- | ---: | ---: |
| ATT-0210-1 | RES-0193 | 3768869.opbs | 584 | 0.040555555556 |
| ATT-0211-1 | RES-0194 | 3768871.opbs | 606 | 0.042083333333 |
| ATT-0212-1 | RES-0195 | 3768872.opbs | 615 | 0.042708333333 |
| ATT-0213-1 | RES-0196 | 3768874.opbs | 588 | 0.040833333333 |
| ATT-0214-1 | RES-0197 | 3768875.opbs | 569 | 0.039513888889 |
| ATT-0215-1 | RES-0198 | 3768876.opbs | 580 | 0.040277777778 |
| ATT-0216-1 | RES-0199 | 3768878.opbs | 587 | 0.040763888889 |
| ATT-0217-1 | RES-0200 | 3768879.opbs | 607 | 0.042152777778 |
| ATT-0218-1 | RES-0201 | 3768880.opbs | 654 | 0.045416666667 |
| ATT-0219-1 | RES-0202 | 3768881.opbs | 603 | 0.041875000000 |
| ATT-0220-1 | RES-0203 | 3768882.opbs | 647 | 0.044930555556 |
| ATT-0221-1 | RES-0204 | 3768883.opbs | 590 | 0.040972222222 |
| ATT-0223-1 | RES-0209 | 3768984.opbs | 606 | 0.042083333333 |
| ATT-0224-1 | RES-0210 | 3768985.opbs | 588 | 0.040833333333 |
| ATT-0225-1 | RES-0211 | 3768987.opbs | 593 | 0.041180555556 |
| ATT-0226-1 | RES-0212 | 3768988.opbs | 630 | 0.043750000000 |
| ATT-0222-1 | RES-0205 | 3768977.opbs | 13 | 0.000902777778 |
| ATT-0236-1 | RES-0214 | 3769071.opbs | 17 | 0.001180555556 |

Sixteen training candidates:9647 MIG seconds =0.6699305555555556 node·h. Two artifact reports:13+17=30seconds =0.0020833333333333333 node·h. Full T-0024 total:9677 MIG seconds /14400 =0.6720138888888889 node·h, below1.5.

All eighteen execution ends, result imports and exact usage entries were recorded once; accounting keys are miyabi-pbs:<full job id>. Current complete metadata: research/notes/T-0024-final-jobs.json. Twelve withdrawn prepared registrations ATT-0198-1..ATT-0209-1 never submitted a job and consumed no compute; the immutable descriptive metric error was corrected before execution. They are absent from all candidate/lock/evidence inputs.
