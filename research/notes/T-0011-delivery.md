# T-0011 delivery: E-0010 V2 M-line factorial

M-line seed 0 screening; final dev decisions; no confirmation read

| Arm | EM normal / zero / shuffled / permuted (%) | NLL normal / zero / shuffled / permuted | Target weight | n_eff |
| --- | --- | --- | --- | --- |
| V2-01 | 0.90 / 1.10 / 1.90 / 1.40 | 4.427822 / 9.248186 / 4.427827 / 4.426864 | 0.060240 | 9.9692 |
| V2-02 | 1.70 / 1.10 / 1.50 / 2.00 | 4.448453 / 9.248186 / 4.449490 / 4.452334 | 0.059004 | 6.2745 |
| V2-03 | 1.60 / 1.10 / 1.20 / 1.00 | 4.422075 / 9.248186 / 4.423024 / 4.604354 | 0.053784 | 1.1108 |
| V2-04 | 1.90 / 1.10 / 1.30 / 1.30 | 4.442024 / 9.249403 / 4.441777 / 4.452406 | 0.054220 | 1.0335 |
| V2-05 | 0.80 / 1.10 / 1.50 / 1.00 | 4.427029 / 9.248186 / 4.426557 / 4.432025 | 0.058403 | 5.6185 |
| V2-06 | 1.80 / 1.10 / 1.20 / 1.50 | 4.446569 / 9.248186 / 4.447419 / 4.447202 | 0.057738 | 4.0445 |
| V2-07 | 1.70 / 1.10 / 1.10 / 1.50 | 4.425355 / 9.248186 / 4.423455 / 4.431249 | 0.055640 | 2.5380 |
| V2-08 | 2.00 / 1.10 / 1.50 / 2.60 | 4.445451 / 9.248186 / 4.447843 / 4.445822 | 0.058278 | 5.7746 |
| V2-09 | 2.10 / 1.10 / 1.50 / 1.10 | 4.610641 / 9.249403 / 4.609989 / 4.611436 | 0.059095 | 7.0042 |
| V2-10 | 1.80 / 1.10 / 1.90 / 0.90 | 4.497515 / 9.248186 / 4.499404 / 4.607800 | 0.052307 | 1.0511 |

H-0003: challenged (screening). Qualified: []; nondegenerate eligible: [].
H-0004: inconclusive (screening); SoftmaxSlot wins 0/4, KernelMatrix wins 0/4.
V3 selection: None; finalists: [].

Full rule inputs, paired intervals, oracle results, strata, curves, parameter counts and source hashes are in factorial-report.json.
No numerical high/low threshold is sealed. Report oracle EM/NLL, errors and paired differences; coordinator interprets them without adding a decision gate.
Application node-hours exclude PBS overhead. Actual budget accounting comes from the Miyabi ledger.

## Delivery and validation

All ten released training runs and the full reviewed analysis completed. EV-0070 challenges H-0003 at screening level; EV-0071 records H-0004's inconclusive inputs without a supports/challenges link. These are the locked analysis outputs and evidence; the coordinator owns conclusion review, hypothesis-status decisions and the next route.

Source snapshot: `9145309f16231d52bb121118779c986e875951e5`. This task adds only `src/v2_factorial_analysis.py`, `src/test_v2_factorial_analysis.py` and `src/v2_factorial_check.py`. Training/evaluation, data generation and dependencies are unchanged. Training registrations reported coverage by RV-0019. Analysis RV-0023 passed with zero findings. Exact author model identity was unobserved and recorded as unknown.

ATT-0078-1 / RES-0080 / EV-0066: 21 synthetic regressions passed, zero failures/errors/skips, 2.060 seconds application time. PBS3762004.opbs on confirmed compute host mg1098 finished exit0 with 3 seconds measured walltime. Tests cover literal threshold boundaries, C_dual inclusion/exclusion, strict V3 ties, shared bootstrap draws and chunk invariance, factorial normalization, formal inputs, pairing/reload and summary corruption.

ATT-0093-1 / RES-0094: complete ten-arm CLI/report execution from the reviewed snapshot. PBS3762550.opbs on mg1098 finished exit0 with 5 seconds measured walltime. The analysis validated all ten formal manifests/configurations, released LRs, schedules and finite histories; full dev1000/calibration2048 outputs and summaries; ordered episode pairing across arms/controls, disjoint split identities, actual reload prediction equality and all curve milestones. It ran10,000 paired episode bootstrap replicates, seed20261029, separately for each released split, with identical resampled indices across arms/controls within a split. No confirmation content was opened.

Each training run used seed0,6000steps,4000470input tokens,warmup300,batch8/microbatch8,eval16, final dev/calibration six-control evaluation and dev normal reload. Intermediate25/50/75% dev normal/zero/shuffled results are descriptive only. Base revision, data hashes, lock and CUDA environment match E-0010@2 Release. All12 jobs checked compute allocation and live RUNNING state before workloads. All ten training imports and the analysis import have no RunSpec mismatches; omitted manifest identity fields remain listed by rl rather than claimed independently verified.

## Literal rule inputs

N/Z/S/P in the generated table means normal/zero/shuffled/permuted. Address/read diagnostics use valid-question-token means; EM/NLL and bootstrap use episodes. Final dev prior EM is1.50%. H-0003 requires EM_N-max(EM_Z,EM_S) at least10pp, NLL_N strictly below both controls, and target weight at least0.125 for eligibility. The released E-0008 positive control passed (ATT-0069-1). C_dual probes count for H3/V3.

| Arm | H3 EM margin pp | NLL_N below both? | Target weight >=0.125? | Normalized read error |
| --- | --- | --- | --- | --- |
| V2-01 | -1.0 | yes | no | 0.165140 |
| V2-02 | +0.2 | yes | no | 0.354737 |
| V2-03 | +0.4 | yes | no | 2.492829 |
| V2-04 | +0.6 | no | no | 0.488782 |
| V2-05 | -0.7 | no | no | 0.392928 |
| V2-06 | +0.6 | yes | no | 0.433630 |
| V2-07 | +0.6 | no | no | 0.289613 |
| V2-08 | +0.5 | yes | no | 0.245538 |
| V2-09 | +0.6 | no | no | 0.061087 |
| V2-10 | -0.1 | yes | no | 0.894542 |

No arm reaches the10pp EM threshold; qualified0/10, nondegenerate0/10, eligible0/10. No V3 configuration exists under the sealed selection rule. This task does not start V3 on these checkpoints.

H-0004 uses four kernel/soft pairs only: (01,03),(02,04),(05,07),(06,08). Soft normal EM is higher in all four, but its target weight is lower in the first three. In the fourth its n_eff is higher (5.774581 versus4.044499). Therefore no cell meets all three soft conditions; kernel EM is lower in all four, so none meets the kernel condition. Soft0/4 and kernel0/4, both below3/4: inconclusive. C_dual probes are excluded.

## Paired uncertainty and descriptive effects

Percentile95% intervals use10,000 episode draws with seed20261029 and common indices across every arm/control. They describe this seed/checkpoint/split, not seed replication or confirmation, and do not change the decisions.

| Arm | N-Z EM pp [95% CI] | N-S EM pp [95% CI] | N-P EM pp [95% CI] |
| --- | --- | --- | --- |
| V2-01 | -0.2 [-1.1,0.7] | -1.0 [-1.8,-0.2] | -0.5 [-1.2,0.2] |
| V2-02 | +0.6 [-0.4,1.7] | +0.2 [-0.5,0.9] | -0.3 [-0.8,0.2] |
| V2-03 | +0.5 [-0.5,1.5] | +0.4 [-0.4,1.2] | +0.6 [-0.4,1.6] |
| V2-04 | +0.8 [-0.3,1.9] | +0.6 [-0.2,1.4] | +0.6 [-0.4,1.6] |
| V2-05 | -0.3 [-1.1,0.5] | -0.7 [-1.6,0.1] | -0.2 [-0.9,0.5] |
| V2-06 | +0.7 [-0.4,1.8] | +0.6 [0.0,1.3] | +0.3 [-0.4,1.0] |
| V2-07 | +0.6 [-0.4,1.7] | +0.6 [-0.4,1.6] | +0.2 [-0.8,1.2] |
| V2-08 | +0.9 [-0.2,2.0] | +0.5 [-0.2,1.3] | -0.6 [-1.6,0.4] |
| V2-09 | +1.0 [-0.1,2.1] | +0.6 [-0.4,1.6] | +1.0 [0.2,1.9] |
| V2-10 | +0.7 [-0.3,1.8] | -0.1 [-1.2,1.0] | +0.9 [-0.1,1.9] |

All normal-minus-shuffled NLL intervals include zero. Normal NLL is much lower than zero-memory NLL, while normal and shuffled NLL remain close. This is consistent with prior fitting without useful episode-specific retrieval; the zero comparison alone does not demonstrate memory use. Full NLL intervals, oracle contrasts and calibration comparisons are in JSON.

Factorial effects use V2-01..08 only and the documented conventional normalization `2/8 * sum(product of contrast signs * response)`, including interactions. Positive levels: qknorm_pre_rope, soft_slot and C_up.

| Effect | Normal EM effect pp [95% CI] |
| --- | --- |
| Feature | +0.05 [-0.375,0.475] |
| Reader | +0.50 [-0.05,1.05] |
| Fusion | +0.60 [-0.05,1.30] |
| Feature x reader | +0.05 [-0.275,0.375] |
| Feature x fusion | +0.05 [-0.375,0.50] |
| Reader x fusion | -0.30 [-0.775,0.175] |
| Feature x reader x fusion | -0.05 [-0.40,0.30] |

C_up normal-NLL main effect is+0.020054 [0.010469,0.030111], worse NLL on this comparison. Reader normal-minus-shuffled EM effect is+0.75pp [0.225,1.30], but its NLL effect is+0.000056 [-0.003077,0.003169]; this small descriptive EM difference does not meet H3 or H4. Remaining factorial outcomes/interactions are in JSON.

| Probe minus matching comparator | Normal EM difference pp [95% CI] | Normal NLL difference [95% CI] |
| --- | --- | --- |
| V2-09 minus V2-05 (kernel F_out) | +1.30 [0.30,2.40] | +0.183611 [0.165581,0.201856] |
| V2-09 minus V2-06 (kernel C_up) | +0.30 [-0.90,1.50] | +0.164071 [0.143955,0.184652] |
| V2-10 minus V2-07 (soft F_out) | +0.10 [-1.00,1.20] | +0.072161 [0.061264,0.083004] |
| V2-10 minus V2-08 (soft C_up) | -0.20 [-1.30,0.90] | +0.052064 [0.039103,0.065136] |

JSON contains relation, target-fact-position and support-order strata for every control and both splits. Strata/subsets have point estimates, not additional bootstrap intervals. In the answer-not-subject dev subset (987 episodes), normal correct counts are9/17/16/19/8/18/17/20/21/18 in arm order (EM approximately0.81–2.13%); zero-memory correct counts are0 throughout. The primary gate still uses all1000 episodes.

## Oracle diagnosis and learning curves

| Arm | Key-oracle EM% / NLL | Value-oracle EM% / NLL |
| --- | --- | --- |
| V2-01 | 1.7 / 4.427536 | 1.4 / 4.426904 |
| V2-02 | 1.9 / 4.449441 | 1.9 / 4.450851 |
| V2-03 | 1.2 / 4.571719 | 1.3 / 4.576288 |
| V2-04 | 1.7 / 4.446609 | 1.5 / 4.448434 |
| V2-05 | 1.1 / 4.428592 | 1.3 / 4.427768 |
| V2-06 | 2.0 / 4.444413 | 2.3 / 4.445240 |
| V2-07 | 1.9 / 4.426663 | 1.1 / 4.427398 |
| V2-08 | 2.0 / 4.443414 | 1.8 / 4.454114 |
| V2-09 | 2.0 / 4.611339 | 2.1 / 4.609486 |
| V2-10 | 1.2 / 4.500572 | 1.6 / 4.537459 |

Value oracle does not rescue these checkpoints (1.1–2.3% EM versus normal0.8–2.1%). Under the contract's qualitative rule, both low points toward value-writing or fusion diagnosis. This is an interpretation, not a new numeric gate, isolation of either component, or proof that addressing is adequate. Addressing remains unresolved: every target weight is below0.125 and even below uniform1/16, including concentrated soft reads. Evaluation-only oracle substitution does not establish what an independently oracle-trained path could learn. No claim attributes reader differences to matrix compression.

| Arm | Dev normal EM% at25/50/75/100% steps |
| --- | --- |
| V2-01 | 1.2/1.0/1.6/0.9 |
| V2-02 | 1.3/1.4/2.2/1.7 |
| V2-03 | 1.5/1.0/1.5/1.6 |
| V2-04 | 1.1/1.4/1.7/1.9 |
| V2-05 | 1.4/1.0/1.5/0.8 |
| V2-06 | 1.3/1.4/2.0/1.8 |
| V2-07 | 1.5/0.9/1.5/1.7 |
| V2-08 | 1.0/1.5/2.0/2.0 |
| V2-09 | 1.6/1.0/1.5/2.1 |
| V2-10 | 1.4/1.3/1.7/1.8 |

Normal/zero/shuffled EM and NLL curves are in JSON. Calibration normal EM is approximately1.03–1.76%; calibration prior1.9043%. Calibration and intermediate checkpoints were not used for selection.

## Records, parameters and actual cost

P below is each of stored/trainable/parameters-with-nonzero-gradient; the three counts are equal for every arm. The last count means a nonzero gradient was observed at least once. EV-0070 is the shared aggregate H3 screening evidence for all ten arms; EV-0071 records H4 inputs.

| Arm (tap/reader/fusion) | Attempt | Result | PBS job | P | Measured MIG-h |
| --- | --- | --- | --- | --- | --- |
| 01 direct/kernel/F_out | ATT-0080-1 | RES-0084 | 3762042.opbs | 557056 | 0.403333 |
| 02 direct/kernel/C_up | ATT-0081-1 | RES-0085 | 3762043.opbs | 819200 | 0.403333 |
| 03 direct/soft/F_out | ATT-0082-1 | RES-0086 | 3762044.opbs | 557056 | 0.409444 |
| 04 direct/soft/C_up | ATT-0083-1 | RES-0087 | 3762045.opbs | 819200 | 0.415278 |
| 05 qknorm/kernel/F_out | ATT-0084-1 | RES-0088 | 3762046.opbs | 688128 | 0.402500 |
| 06 qknorm/kernel/C_up | ATT-0085-1 | RES-0089 | 3762047.opbs | 950272 | 0.415556 |
| 07 qknorm/soft/F_out | ATT-0086-1 | RES-0090 | 3762048.opbs | 688128 | 0.423333 |
| 08 qknorm/soft/C_up | ATT-0087-1 | RES-0091 | 3762049.opbs | 950272 | 0.407778 |
| 09 qknorm/kernel/C_dual | ATT-0088-1 | RES-0092 | 3762050.opbs | 1343488 | 0.416389 |
| 10 qknorm/soft/C_dual | ATT-0089-1 | RES-0093 | 3762051.opbs | 1343488 | 0.430556 |

Kernel LR0.01 and soft LR0.003 are the released family-wide values, including probes. Throughput is approximately3248–3490 input tokens/s; peak allocated memory approximately1.488–1.500GB. Training job walltime24:09–25:50; requested36:41–38:36 from measured forecasts times1.5. Detailed per-arm timings/gradients/throughput are in original artifacts and JSON.

Twelve jobs: ten trainings plus regression3s and analysis5s. Ledger total **4.129722 MIG-h**, conversion times0.25 = **1.0324305 node-h**, below the E-0010 budget7node-h. Observations/usage were imported with stable accounting references; application timing fields were not added again. Coordinator acknowledged the single ten-job notification N-20261101-175131-1cd7.

Durable artifacts:

- Full report: `/work/xg24i002/x10041/bypass-lora/runs/ATT-0093-1/factorial-report.json`; generated concise table `factorial-report.md` alongside it.
- Original outputs/checkpoints: `/work/xg24i002/x10041/bypass-lora/runs/<ATT>/`. Report source entries retain input SHA256 hashes.
- Maps: `/work/xg24i002/x10041/bypass-lora/research/artifacts/T-0011/training-run-map.json` and `imported-results.json`.
- Ledger exports in that same artifacts directory: `analysis-check-observations.json`, `training-submission-observations.json`, `training-ended-observations.json`, `analysis-ended-observations.json`, `final-usage-summary.json`.

No unresolved code-review findings or required execution checks remain. This is one-seed screening with qualitative oracle localization, not a general architectural impossibility result. Coordinator conclusion review, hypothesis decisions and the next diagnostic route remain outside this task.
