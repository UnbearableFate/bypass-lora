# T-0008 delivery: formal M training and learning-rate calibration

E-0009 implementation and calibration complete. **KernelMatrix LR = 0.01; SoftmaxSlot LR = 0.003.** Both are finite, selected from the initial grid plus the prescribed single endpoint extension using calibration normal answer NLL only. RV-0019 passes with zero findings. Coordinator owns acceptance/merge and E-0010 release; no E-0010 training was performed here.

Final source commit `631f87e65157af1012326ffce0b1c78df10ecfe4`. Training/evaluation source was reviewed at `d8cc3d4011529fa5c2fcbaa4277d91981a077fed`; the later source commit adds only the deterministic calibration reporting CLI. Initial six calibration runs used `f7c4d2bdde6b2167355e131c757e230d23c80a00`; its sole later correction is the nonblocking reconstruction diagnostic aggregation described below. Later commits change this note only.

## Implementation and validation

- `src/v2_train.py`: formal M-line training, batch8 with correctly weighted microbatch accumulation, answer-only CE, verified data hashes, seed0, FP32 new parameters/BF16 frozen Base, AdamW default betas/eps/zero weight decay, clip1, warmup/cosine. Twelve supported tap/reader/fusion configurations include C_up. Support writes each delimiter once; query reads without support backbone KV.
- `src/v2_schedule.py`: shared verified data loading, fixed three-epoch order, complete-step token stop, LR schedule and finite-candidate/tie selection. Researcher-3 copied the unchanged module and independently verified order/counts in PBS ATT-0054-2 (MSG-0164).
- `src/v2_memory.py`: adds C_up to the allowed arms and explicit shuffled donor batches. Global donor i-1 modulo the full split is fixed before batch16 evaluation, so changing batch boundaries does not change donor identity. Recipient-referenced read errors remain available; target-dependent shuffled addressing metrics remain suppressed.
- Final evaluation includes all six controls, full-vocabulary EM/NLL, address/read diagnostics, relation/target-position/support-order strata, nonself-answer subsets, prior baseline, subject-copy and answer rank. Curves at25/50/75% use only dev normal/zero/shuffled. Export/reload saves all adapter tensors/config and requires identical per-episode dev normal predictions.
- `src/v2_calibration_report.py`: reproducible artifact validation and final LR selection. It checks complete/passing manifests, representative arm, schedule/hash,300 finite loss records, reload and2048 calibration episodes, and retains each metrics artifact SHA256.
- No changes to bypass core/model, v1 code, v2 data generator, dependencies or frozen tolerances. No confirmation content read.

ATT-0055-1 / RES-0055 / EV-0050: **146 tests passed**, including all12 protocol/zero-exit arms, microbatch8/3/1 accumulation, global donors across batch16 boundary, checkpoint exact tensor/config/logit reload, and batch-composition tests. Composition BF16 logit differences0.2265625–0.375 are within the pre-existing0.50002 tolerance; this is tolerance-bounded agreement, not exact invariance.

ATT-0056-1 / RES-0057 / EV-0052: **all12 arms completed32 steps**,21357 input tokens each. Every arm has exact zero-exit Base logits and identical dev predictions after reload. All controls/curves ran on16dev+16calibration examples. This is an engineering smoke, not scientific evidence.

RV-0017 passed under the low-only rule, identifying one reporting issue: split `read.reconstruction_max_abs` was a token-weighted mean of batch maxima. The d8cc3d4 fix uses max and documents the exception. ATT-0068-1 / RES-0062 / EV-0054: focused two-batch regression passed (one test;124 deselected), preserving ordinary means/EM/NLL/predictions. **RV-0019 pass,0 findings** resolves it. Broad146 and12-arm smoke were not rerun for this isolated reporting correction. No training rerun required per reviewer. Historical f7c4d2b multi-batch reconstruction fields retain their old weighted-batch-max meaning; they must not be claimed as corrected split maxima and are unused in LR selection. Current source and extension runs use the correction.

## Shared schedule

Order SHA256 `0d4b55ed8655d91cf0d0532bdac1820a518b5c5730ef70e48f264da6b4bc9e79`: UTF-8 episode_id followed by LF, including final LF, over all three epochs. Independent M/F match was recorded before calibration. Each episode's count is support plus question tokens; answer is a label only.

| Budget | Steps | Actual input tokens | Warmup steps |
| --- | --- | --- | --- |
| 200000 | 300 | 200170 | 15 |
| 4000000 | 6000 | 4000470 | 300 |

First optimizer update uses target_lr/warmup, warmup endpoint target_lr, last update0. Verified in ATT-0053-1 / RES-0053 / EV-0049 and independently by researcher-3 ATT-0054-2. Training data: sealed ATT-0039-1 train16384/calibration2048/dev1000; hashes checked in every training job.

## Calibration and selection

Representative qknorm_pre_rope + F_out, layer14/r128/S1, seed0; all8 candidates300steps/200170tokens, batch8, eval16. EM values below are fractions. Each run completed all controls, full dev/calibration evaluation, curves and exact dev normal reload prediction comparison. The reporting artifacts include strata and per-episode predictions; choice uses only calibration normal NLL.

| Reader | LR | Normal NLL | Zero NLL | Shuffled NLL | Normal EM | Zero EM | Shuffled EM | Gradient norm mean / max | Clip fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Kernel | 0.0003 | 5.138646 | 9.233152 | 5.139059 | 0.015137 | 0.008301 | 0.014160 | 6.9006 /13.3615 | 1.0000 |
| Kernel | 0.001 | 4.704182 | 9.230779 | 4.703015 | 0.017090 | 0.008789 | 0.018555 | 3.5623 /8.0075 | 1.0000 |
| Kernel | 0.003 | 4.627133 | 9.230779 | 4.628094 | 0.019531 | 0.008789 | 0.018066 | 1.6733 /16.1801 | 0.9800 |
| Kernel | **0.01** | **4.584629** | 9.230779 | 4.584687 | 0.019043 | 0.008789 | 0.014160 | 2.2575 /100.8155 | 0.2167 |
| Soft | 0.0003 | 5.130321 | 9.230779 | 5.130886 | 0.011719 | 0.008789 | 0.013672 | 7.0968 /13.0115 | 1.0000 |
| Soft | 0.001 | 4.682995 | 9.233152 | 4.684171 | 0.019531 | 0.008301 | 0.015137 | 3.6312 /9.4449 | 1.0000 |
| Soft | **0.003** | **4.627491** | 9.230779 | 4.628111 | 0.014160 | 0.008789 | 0.017090 | 1.7339 /17.2646 | 0.9933 |
| Soft | 0.01 | 4.606243 | 9.230779 | 4.606002 | 0.016602 | 0.008789 | 0.014160 | 1.9022 /43.9344 | 0.4667 |

Both initial-grid choices were upper endpoint0.003, so each received the required one extension0.01. Kernel selects0.01. Soft0.01 has lower raw NLL, but0.003 is within0.5% of that minimum; the sealed tie rule chooses the smaller LR0.003. No further extensions are prescribed. Diagnostic calibration only: normal and shuffled are close, and these results neither support nor challenge H-0003.

Final reproducible table: `/work/xg24i002/x10041/bypass-lora/runs/ATT-0074-1/calibration-report.json`. Initial table: ATT-0070-1. Both generated in PBS from immutable snapshots.

| Attempt | Purpose | Result | Evidence | PBS |
| --- | --- | --- | --- | --- |
| ATT-0053-1 | Schedule | RES-0053 | EV-0049 | 3761319 |
| ATT-0055-1 | Broad146 checks | RES-0055 | EV-0050 | 3761353 |
| ATT-0056-1 | Twelve-arm smoke | RES-0057 | EV-0052 | 3761393 |
| ATT-0061-1 | Kernel0.0003 | RES-0064 | EV-0056 | 3761516 |
| ATT-0062-1 | Kernel0.001 | RES-0065 | EV-0056 | 3761517 |
| ATT-0063-1 | Kernel0.003 | RES-0066 | EV-0056 | 3761519 |
| ATT-0064-1 | Soft0.0003 | RES-0067 | EV-0056 | 3761520 |
| ATT-0065-1 | Soft0.001 | RES-0068 | EV-0056 | 3761523 |
| ATT-0066-1 | Soft0.003 | RES-0069 | EV-0056 | 3761526 |
| ATT-0068-1 | Reporting regression + forecasts | RES-0062 | EV-0054 | 3761611 |
| ATT-0070-1 | Initial selection | RES-0070 | EV-0057 | 3761647 |
| ATT-0071-1 | Kernel0.01 | RES-0072 | EV-0060 | 3761654 |
| ATT-0072-1 | Soft0.01 | RES-0073 | EV-0060 | 3761655 |
| ATT-0074-1 | Final selection | RES-0074 | EV-0061 | 3761797 |

## Timing and costs

Smoke measurements and PBS-computed forecast: `/work/xg24i002/x10041/bypass-lora/runs/ATT-0068-1/smoke-cost-report.json`. Peak allocated bytes rounded to decimal GB. Forecast includes6000steps, full dev+calibration six-control final evaluation, dev curves/reload,10s fixed allowance; requests use1.5x. These are extrapolations from32steps and16examples/split, not measured full4M runs.

| Tap | Reader | Fusion | Step seconds | Tokens/sec | Peak GB | Estimated4M seconds | PBS request seconds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| direct | Kernel | F_out | 0.198479 | 3343.6 | 1.477 | 1490.3 | 2236 |
| direct | Kernel | C_up | 0.195923 | 3416.9 | 1.482 | 1472.7 | 2210 |
| direct | Kernel | C_dual | 0.196148 | 3414.0 | 1.485 | 1473.0 | 2210 |
| direct | Soft | F_out | 0.203995 | 3282.6 | 1.477 | 1535.3 | 2304 |
| direct | Soft | C_up | 0.203485 | 3290.0 | 1.482 | 1534.9 | 2303 |
| direct | Soft | C_dual | 0.203992 | 3282.1 | 1.486 | 1539.9 | 2310 |
| qknorm | Kernel | F_out | 0.196052 | 3411.5 | 1.478 | 1472.0 | 2209 |
| qknorm | Kernel | C_up | 0.195694 | 3419.3 | 1.483 | 1470.7 | 2207 |
| qknorm | Kernel | C_dual | 0.195623 | 3420.8 | 1.487 | 1467.2 | 2201 |
| qknorm | Soft | F_out | 0.204668 | 3271.5 | 1.479 | 1543.6 | 2316 |
| qknorm | Soft | C_up | 0.204320 | 3275.5 | 1.484 | 1537.9 | 2307 |
| qknorm | Soft | C_dual | 0.204504 | 3272.8 | 1.488 | 1543.5 | 2316 |

E-0010 requests need short-mig (~36.7–38.6minutes), exceeding debug-mig30min. Initial calibration app walltimes316.5–347.4s; extension299.2/337.3s, consistent with the estimate and9min requests. Individual training throughput/gradient stats are retained in the final calibration report.

Actual task usage: **2920 MIG seconds =0.811111 MIG-hours =0.202778 node-hours**, below1node-hour. Fourteen PBS jobs, all exit0, guarded compute allocations. Ledger observations/usage exports are retained under `research/artifacts/T-0008/`. Initial schedule job mistakenly used namespace pbs:miyabi rather than miyabi-pbs in its manual observation; its2s usage was recorded manually U-0066 and not duplicated. Last analysis job1s was recorded from fresh job_get with the ledger accounting_ref while the terminal ledger event lagged. All other usage came from ledger exports; do not double-count these manual entries.

Main training exposure: eight calibrations x200170 plus twelve smoke arms x21357 =1857644 nonpadding input tokens. Protocol tests also execute small backward checks and are included in measured compute, not this main-run token total. No formal4M run was launched.

## Remaining limits and next action

No unresolved code findings. Root exact author model identifier was unavailable and recorded as unknown in reviews; helper implementation/tests used gpt-6.1-sol at medium reasoning. Scientific effects, the F-line gate and E-0010 release remain coordinator decisions. The selected LR for each reader applies to all its E-0010 arms, including C_dual, per contract. Accept and merge sr/researcher-2/T-0008; do not interpret calibration as H-0003 evidence. Historical reconstruction-maximum limitation and forecast uncertainty remain explicitly disclosed.
