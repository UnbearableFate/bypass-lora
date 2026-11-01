# T-0019 results: E-0017@1, seed replication of the D5 recipe under the isolated write

This task measures and reports only. The H-0012 judgement and whether to reopen the H-0003 route are the coordinator's. Confirmation data was never read.

The isolated write is the protocol registered in E-0016. The gate uses the E-0010@2 H3 thresholds through the same `h3_gate` as E-0016 D5, and is reported separately from E-0010.

## Replication verdict (P1 and P2 only)

- **P1 and P2 both meet H3,** in both target-position strata.
- With E-0016 D5 at seed 0 (EV-0095), that is 3 of 3 seeds for the full recipe: stage A R1 retrained at seed s, then stage B D5 continuation at seed s.
- Under E-0017's outcome_actions this is the replication branch. The coordinator makes the decision and the conclusion review.
- **C1 and C2 also meet H3.** They are descriptive only and do not enter the verdict.

## Stage B: E=16 dev (1,000 episodes), isolated write

| run | attempt | result | H3 | EM normal / zero / shuffled | margin | NLL normal / zero / shuffled | target weight all / last | last-pos n_eff | most common last-pos slot rate | t0 (n=52) EM / margin / weight | t≥1 (n=948) EM / margin / weight | slice |
|---|---|---|---|---|---:|---|---|---:|---:|---|---|---|
| P1 (stage A seed 1 → cont. seed 1) | ATT-0172-1 | RES-0170 | pass | 49.7 / 1.1 / 1.9 | 239/500 = 47.8 pp | 2.279 / 9.249 / 8.583 | 0.162 / 0.610 | 1.97 | 0.076 | 50.0 / 48.1 / 0.152 | 49.7 / 47.8 / 0.162 | 1g.24gb |
| P2 (stage A seed 2 → cont. seed 2) | ATT-0173-1 | RES-0171 | pass | 48.0 / 1.1 / 1.8 | 231/500 = 46.2 pp | 2.378 / 9.248 / 8.241 | 0.158 / 0.600 | 2.01 | 0.077 | 50.0 / 48.1 / 0.143 | 47.9 / 46.1 / 0.159 | 2g.24gb |
| C1 (R1 seed 0 → cont. seed 1) | ATT-0170-1 | RES-0168 | pass | 45.3 / 1.1 / 2.3 | 43/100 = 43.0 pp | 2.406 / 9.249 / 8.205 | 0.150 / 0.589 | 2.07 | 0.078 | 42.3 / 34.6 / 0.131 | 45.5 / 43.5 / 0.151 | 1g.24gb |
| C2 (R1 seed 0 → cont. seed 2) | ATT-0171-1 | RES-0169 | pass | 46.5 / 1.1 / 2.3 | 221/500 = 44.2 pp | 2.394 / 9.248 / 8.245 | 0.157 / 0.598 | 2.02 | 0.074 | 44.2 / 38.5 / 0.152 | 46.6 / 44.5 / 0.157 | 2g.24gb |
| E-0016 D5 (R1 seed 0 → cont. seed 0) | ATT-0159-1 | RES-0153 | pass | 42.8 / 1.1 / 1.7 | 411/1000 = 41.1 pp | 2.514 / 9.248 / 7.977 | 0.162 / 0.571 | 2.12 | 0.080 | 44.2 / 38.5 / 0.140 | 42.7 / 41.2 / 0.163 | 2g.24gb |

| stage A | attempt | result | E=1 dev EM normal / zero / shuffled | E=1 NLL normal / zero / shuffled | E=16 dev contextual normal EM | adapter.pt sha256 | slice |
|---|---|---|---|---|---|---|---|
| R1 seed 0 (E-0014, reference) | ATT-0122-1 | RES-0123 | 83.6 / 0.9 / 1.0 | 0.538 / 9.265 / 18.229 | 0.4 | 9ea30adb36d914fb09e689fddbb0f27d91e32323231ac359245fcbd3024eb616 (via ATT-0159-1) | not recorded |
| P1 stage A, seed 1 | ATT-0166-1 | RES-0162 | 82.7 / 0.9 / 1.1 | 0.523 / 9.265 / 18.670 | 1.5 | b5b2aec9dffd352910990043cb84370fb500081b3bf869918f8215d885740797 | 2g.24gb |
| P2 stage A, seed 2 | ATT-0167-1 | RES-0163 | 81.1 / 0.9 / 0.9 | 0.580 / 9.265 / 18.874 | 1.8 | aa3919e2540ea6fedc3c0dfa2accbdcdb07fe8fdf626f134f88dad4a247a38b4 | 2g.24gb |

**Observations (measured facts, not judgements):**
- **Gate.** All five runs of the recipe pass with margins between 41 and 48 pp, including E-0016 D5.
  - The all-position target weight is between 0.150 and 0.162.
  - The last-position weight is between 0.57 and 0.61.
  - Last-position n_eff is about 2.
  - The most common last-position argmax slot covers 7–8% of episodes, so the read does not collapse to a fixed slot.
- **Starting point vs continuation seed.**
  - Retrained stage A (P1/P2): 48.0–49.7 EM.
  - Same R1 start, other continuation seeds (C1/C2): 45.3–46.5 EM.
  - E-0016 D5 (seed 0): 42.8 EM.
  - The spread is a few EM points with one run each; no interval was computed.
- **Contextual encoding.** After stage B it stays near zero past the first fact, as in E-0016 D5. For P1: normal 4.9, with 67.3 at target 0 and 1.5 at target ≥ 1. The adapters work only with isolated writes.

## Stage A: R1 recipe on E=1 (P1, P2)

| stage A | attempt | result | E=1 dev EM normal / zero / shuffled | E=1 NLL normal / zero / shuffled | E=16 dev contextual normal EM | adapter.pt sha256 | slice |
|---|---|---|---|---|---|---|---|
| R1 seed 0 (E-0014, reference) | ATT-0122-1 | RES-0123 | 83.6 / 0.9 / 1.0 | 0.538 / 9.265 / 18.229 | 0.4 | 9ea30adb36d914fb09e689fddbb0f27d91e32323231ac359245fcbd3024eb616 (via ATT-0159-1) | not recorded |
| P1 stage A, seed 1 | ATT-0166-1 | RES-0162 | 82.7 / 0.9 / 1.1 | 0.523 / 9.265 / 18.670 | 1.5 | b5b2aec9dffd352910990043cb84370fb500081b3bf869918f8215d885740797 | 2g.24gb |
| P2 stage A, seed 2 | ATT-0167-1 | RES-0163 | 81.1 / 0.9 / 0.9 | 0.580 / 9.265 / 18.874 | 1.8 | aa3919e2540ea6fedc3c0dfa2accbdcdb07fe8fdf626f134f88dad4a247a38b4 | 2g.24gb |

Both new seeds learn the E=1 value path as R1 does: the margin over zero and shuffled is above 80 pp.

## Per-run tables (stage B)

Every run is listed. The controls are under the isolated write; the contextual normal/oracle_value rows and the E=16 calibration rows (`calibration_isolated`, 2,048 episodes) are descriptive only. Each run also shows its curves at 25/50/75/100% and its training CE in 500-step blocks.

### P1-B (ATT-0172-1)

Strata: target position 0 n=52, ≥1 n=948.

**H3 gate (isolated write, E-0010@2 thresholds)**

| stratum | EM normal | EM zero | EM shuffled | margin (≥10 pp) | NLL normal | NLL zero | NLL shuffled | target weight all (≥0.125) | target weight last | pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| all | 49.7 | 1.1 | 1.9 | 239/500 = 47.8 pp | 2.279 | 9.249 | 8.583 | 0.162 | 0.610 | True |
| position0 | 50.0 | 0.0 | 1.9 | 25/52 = 48.1 pp | 2.594 | 9.399 | 7.839 | 0.152 | 0.572 | True |
| position1plus | 49.7 | 1.2 | 1.9 | 151/316 = 47.8 pp | 2.262 | 9.241 | 8.624 | 0.162 | 0.612 | True |

**Controls and diagnostics (E=16 dev, 1000 episodes); EM shown as all (t0 / t≥1)**

| protocol | control | EM | NLL (t0 / t≥1) | target weight all (t0 / t≥1) | target weight last | n_eff all | n_eff last | last-pos argmax: most common slot rate (t0 / t≥1) | fusion/residual mean last / all |
|---|---|---|---|---|---|---:|---:|---|---|
| isolated | normal | 49.7 (50.0 / 49.7) | 2.279 (2.594 / 2.262) | 0.162 (0.152 / 0.162) | 0.610 (0.572 / 0.612) | 3.319 | 1.967 | 0.076 (0.712 / 0.080) | 16.332 / 14.982 |
| isolated | zero | 1.1 (0.0 / 1.2) | 9.249 (9.399 / 9.241) | 0.000 (0.000 / 0.000) | 0.000 (0.000 / 0.000) | 0.000 | 0.000 | — (— / —) | 0.000 / 0.000 |
| isolated | shuffled | 1.9 (1.9 / 1.9) | 8.583 (7.839 / 8.624) | — (— / —) | — (— / —) | 3.326 | 2.082 | 0.080 (0.154 / 0.080) | 16.113 / 14.971 |
| isolated | permuted | 2.3 (1.9 / 2.3) | 8.424 (8.478 / 8.421) | 0.162 (0.152 / 0.162) | 0.610 (0.572 / 0.612) | 3.319 | 1.967 | 0.076 (0.712 / 0.080) | 16.560 / 14.962 |
| isolated | oracle_key | 23.2 (26.9 / 23.0) | 3.289 (3.291 / 3.288) | 0.347 (0.344 / 0.347) | 0.347 (0.344 / 0.347) | 3.560 | 3.560 | 0.075 (0.500 / 0.077) | 15.537 / 13.841 |
| isolated | oracle_value | 77.1 (84.6 / 76.7) | 0.818 (0.685 / 0.826) | 0.162 (0.152 / 0.162) | 0.610 (0.572 / 0.612) | 3.319 | 1.967 | 0.076 (0.712 / 0.080) | 17.852 / 15.903 |
| contextual | normal | 4.9 (67.3 / 1.5) | 14.979 (1.394 / 15.724) | 0.090 (0.394 / 0.074) | 0.237 (0.916 / 0.200) | 2.950 | 1.684 | 0.228 (0.942 / 0.189) | 26.820 / 24.727 |
| contextual | oracle_value | 7.2 (82.7 / 3.1) | 14.393 (0.685 / 15.144) | 0.090 (0.394 / 0.074) | 0.237 (0.916 / 0.200) | 2.950 | 1.684 | 0.228 (0.942 / 0.189) | 31.141 / 27.712 |
| calibration_isolated | normal | 49.5 (41.7 / 50.1) | 2.260 (2.505 / 2.240) | 0.160 (0.144 / 0.161) | 0.604 (0.570 / 0.607) | 3.316 | 1.997 | 0.070 (0.709 / 0.072) | 16.175 / 14.948 |
| calibration_isolated | zero | 0.8 (0.0 / 0.9) | 9.233 (9.283 / 9.229) | 0.000 (0.000 / 0.000) | 0.000 (0.000 / 0.000) | 0.000 | 0.000 | — (— / —) | 0.000 / 0.000 |
| calibration_isolated | shuffled | 1.6 (3.3 / 1.5) | 8.420 (7.900 / 8.461) | — (— / —) | — (— / —) | 3.333 | 2.086 | 0.075 (0.086 / 0.075) | 16.193 / 14.971 |

Isolated normal last-position argmax slot counts (slots 0–15): [57, 57, 68, 62, 67, 76, 69, 68, 63, 46, 51, 57, 61, 75, 58, 65]; t0: [37, 2, 1, 0, 3, 0, 2, 2, 0, 0, 1, 3, 0, 1, 0, 0]

**Training (answer CE, isolated write) and learning curves (isolated controls)**

Train CE mean, steps 1–100: 6.347; around 25% (1450–1549): 4.330; 50%: 3.468; 75%: 2.471; last 100: 1.946. Steps 6000, stable step 0.153 s, clip fraction 0.997, grad norm mean/max 4.07/38.97.

Train CE, mean per 500-step block (ending at step): 500: 5.093, 1000: 4.525, 1500: 4.337, 2000: 4.179, 2500: 3.848, 3000: 3.569, 3500: 3.266, 4000: 3.004, 4500: 2.625, 5000: 2.326, 5500: 2.161, 6000: 2.126.

| point | EM normal (t0 / t≥1) | EM zero | EM shuffled | NLL normal | NLL zero | NLL shuffled | target weight all | target weight last |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 25% | 6.0 (5.8 / 6.0) | 1.1 | 2.3 | 4.396 | 9.249 | 5.027 | 0.110 | 0.256 |
| 50% | 22.7 (19.2 / 22.9) | 1.1 | 1.2 | 3.416 | 9.249 | 5.707 | 0.142 | 0.455 |
| 75% | 41.1 (46.2 / 40.8) | 1.1 | 1.8 | 2.542 | 9.249 | 7.453 | 0.151 | 0.572 |
| 100% | 49.7 (50.0 / 49.7) | 1.1 | 1.9 | 2.279 | 9.249 | 8.583 | 0.162 | 0.610 |

Reload: True / True; start replay (record-only): {'episodes': 1000, 'prediction_mismatches': 4, 'nll_max_abs_difference': 0.1959991455078125}; start adapter b5b2aec9dffd = pin b5b2aec9dffd; initial zero-exit vs Base max abs: None; context≠isolated tokens: {'train': 0, 'dev': 0, 'calibration': 0}; wall 1338 s, 0.3715 MIG-h = 0.0929 node·h.

### P2-B (ATT-0173-1)

Strata: target position 0 n=52, ≥1 n=948.

**H3 gate (isolated write, E-0010@2 thresholds)**

| stratum | EM normal | EM zero | EM shuffled | margin (≥10 pp) | NLL normal | NLL zero | NLL shuffled | target weight all (≥0.125) | target weight last | pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| all | 48.0 | 1.1 | 1.8 | 231/500 = 46.2 pp | 2.378 | 9.248 | 8.241 | 0.158 | 0.600 | True |
| position0 | 50.0 | 0.0 | 1.9 | 25/52 = 48.1 pp | 2.554 | 9.402 | 7.699 | 0.143 | 0.555 | True |
| position1plus | 47.9 | 1.2 | 1.8 | 437/948 = 46.1 pp | 2.368 | 9.240 | 8.271 | 0.159 | 0.602 | True |

**Controls and diagnostics (E=16 dev, 1000 episodes); EM shown as all (t0 / t≥1)**

| protocol | control | EM | NLL (t0 / t≥1) | target weight all (t0 / t≥1) | target weight last | n_eff all | n_eff last | last-pos argmax: most common slot rate (t0 / t≥1) | fusion/residual mean last / all |
|---|---|---|---|---|---|---:|---:|---|---|
| isolated | normal | 48.0 (50.0 / 47.9) | 2.378 (2.554 / 2.368) | 0.158 (0.143 / 0.159) | 0.600 (0.555 / 0.602) | 3.450 | 2.006 | 0.077 (0.654 / 0.079) | 17.578 / 16.277 |
| isolated | zero | 1.1 (0.0 / 1.2) | 9.248 (9.402 / 9.240) | 0.000 (0.000 / 0.000) | 0.000 (0.000 / 0.000) | 0.000 | 0.000 | — (— / —) | 0.000 / 0.000 |
| isolated | shuffled | 1.8 (1.9 / 1.8) | 8.241 (7.699 / 8.271) | — (— / —) | — (— / —) | 3.474 | 2.153 | 0.073 (0.173 / 0.073) | 17.407 / 16.277 |
| isolated | permuted | 2.0 (1.9 / 2.0) | 8.121 (7.780 / 8.140) | 0.158 (0.143 / 0.159) | 0.600 (0.555 / 0.602) | 3.450 | 2.006 | 0.077 (0.654 / 0.079) | 17.889 / 16.157 |
| isolated | oracle_key | 23.2 (25.0 / 23.1) | 3.340 (3.244 / 3.345) | 0.353 (0.351 / 0.353) | 0.353 (0.351 / 0.353) | 3.442 | 3.442 | 0.074 (0.558 / 0.076) | 16.767 / 14.934 |
| isolated | oracle_value | 75.6 (78.8 / 75.4) | 0.880 (0.811 / 0.884) | 0.158 (0.143 / 0.159) | 0.600 (0.555 / 0.602) | 3.450 | 2.006 | 0.077 (0.654 / 0.079) | 19.215 / 17.115 |
| contextual | normal | 4.3 (57.7 / 1.4) | 14.641 (1.867 / 15.342) | 0.089 (0.443 / 0.070) | 0.222 (0.932 / 0.183) | 3.354 | 1.755 | 0.250 (0.942 / 0.212) | 27.428 / 24.197 |
| contextual | oracle_value | 6.5 (78.8 / 2.5) | 14.482 (0.813 / 15.232) | 0.089 (0.443 / 0.070) | 0.222 (0.932 / 0.183) | 3.354 | 1.755 | 0.250 (0.942 / 0.212) | 31.786 / 28.285 |
| calibration_isolated | normal | 46.7 (43.7 / 47.0) | 2.367 (2.531 / 2.354) | 0.157 (0.144 / 0.158) | 0.592 (0.562 / 0.594) | 3.469 | 2.048 | 0.068 (0.702 / 0.072) | 17.488 / 16.212 |
| calibration_isolated | zero | 0.9 (0.0 / 0.9) | 9.231 (9.276 / 9.227) | 0.000 (0.000 / 0.000) | 0.000 (0.000 / 0.000) | 0.000 | 0.000 | — (— / —) | 0.000 / 0.000 |
| calibration_isolated | shuffled | 1.4 (1.3 / 1.4) | 8.169 (7.670 / 8.209) | — (— / —) | — (— / —) | 3.481 | 2.145 | 0.077 (0.099 / 0.078) | 17.450 / 16.218 |

Isolated normal last-position argmax slot counts (slots 0–15): [56, 59, 64, 54, 67, 74, 77, 74, 62, 57, 55, 50, 63, 70, 52, 66]; t0: [34, 3, 1, 1, 3, 0, 2, 3, 0, 0, 1, 3, 0, 1, 0, 0]

**Training (answer CE, isolated write) and learning curves (isolated controls)**

Train CE mean, steps 1–100: 6.129; around 25% (1450–1549): 4.283; 50%: 3.506; 75%: 2.453; last 100: 2.046. Steps 6000, stable step 0.150 s, clip fraction 0.999, grad norm mean/max 3.94/37.67.

Train CE, mean per 500-step block (ending at step): 500: 5.036, 1000: 4.477, 1500: 4.397, 2000: 4.183, 2500: 3.943, 3000: 3.641, 3500: 3.376, 4000: 3.126, 4500: 2.648, 5000: 2.486, 5500: 2.252, 6000: 2.248.

| point | EM normal (t0 / t≥1) | EM zero | EM shuffled | NLL normal | NLL zero | NLL shuffled | target weight all | target weight last |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 25% | 4.1 (1.9 / 4.2) | 1.1 | 2.2 | 4.839 | 9.248 | 5.413 | 0.103 | 0.262 |
| 50% | 15.4 (11.5 / 15.6) | 1.1 | 1.5 | 3.902 | 9.248 | 6.060 | 0.132 | 0.413 |
| 75% | 39.0 (40.4 / 38.9) | 1.1 | 2.1 | 2.767 | 9.248 | 7.590 | 0.150 | 0.546 |
| 100% | 48.0 (50.0 / 47.9) | 1.1 | 1.8 | 2.378 | 9.248 | 8.241 | 0.158 | 0.600 |

Reload: True / True; start replay (record-only): {'episodes': 1000, 'prediction_mismatches': 0, 'nll_max_abs_difference': 0.0}; start adapter aa3919e2540e = pin aa3919e2540e; initial zero-exit vs Base max abs: None; context≠isolated tokens: {'train': 0, 'dev': 0, 'calibration': 0}; wall 1272 s, 0.3534 MIG-h = 0.0883 node·h.

### C1 (ATT-0170-1)

Strata: target position 0 n=52, ≥1 n=948.

**H3 gate (isolated write, E-0010@2 thresholds)**

| stratum | EM normal | EM zero | EM shuffled | margin (≥10 pp) | NLL normal | NLL zero | NLL shuffled | target weight all (≥0.125) | target weight last | pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| all | 45.3 | 1.1 | 2.3 | 43/100 = 43.0 pp | 2.406 | 9.249 | 8.205 | 0.150 | 0.589 | True |
| position0 | 42.3 | 0.0 | 7.7 | 9/26 = 34.6 pp | 2.752 | 9.399 | 7.530 | 0.131 | 0.527 | True |
| position1plus | 45.5 | 1.2 | 2.0 | 103/237 = 43.5 pp | 2.387 | 9.241 | 8.242 | 0.151 | 0.593 | True |

**Controls and diagnostics (E=16 dev, 1000 episodes); EM shown as all (t0 / t≥1)**

| protocol | control | EM | NLL (t0 / t≥1) | target weight all (t0 / t≥1) | target weight last | n_eff all | n_eff last | last-pos argmax: most common slot rate (t0 / t≥1) | fusion/residual mean last / all |
|---|---|---|---|---|---|---:|---:|---|---|
| isolated | normal | 45.3 (42.3 / 45.5) | 2.406 (2.752 / 2.387) | 0.150 (0.131 / 0.151) | 0.589 (0.527 / 0.593) | 3.258 | 2.072 | 0.078 (0.673 / 0.082) | 17.011 / 15.573 |
| isolated | zero | 1.1 (0.0 / 1.2) | 9.249 (9.399 / 9.241) | 0.000 (0.000 / 0.000) | 0.000 (0.000 / 0.000) | 0.000 | 0.000 | — (— / —) | 0.000 / 0.000 |
| isolated | shuffled | 2.3 (7.7 / 2.0) | 8.205 (7.530 / 8.242) | — (— / —) | — (— / —) | 3.268 | 2.191 | 0.079 (0.135 / 0.081) | 16.892 / 15.556 |
| isolated | permuted | 2.5 (3.8 / 2.4) | 7.912 (7.497 / 7.935) | 0.150 (0.131 / 0.151) | 0.589 (0.527 / 0.593) | 3.258 | 2.072 | 0.078 (0.673 / 0.082) | 17.373 / 15.704 |
| isolated | oracle_key | 20.7 (13.5 / 21.1) | 3.547 (3.638 / 3.542) | 0.309 (0.301 / 0.310) | 0.309 (0.301 / 0.310) | 4.067 | 4.067 | 0.074 (0.423 / 0.076) | 16.024 / 14.277 |
| isolated | oracle_value | 72.7 (75.0 / 72.6) | 0.949 (0.881 / 0.952) | 0.150 (0.131 / 0.151) | 0.589 (0.527 / 0.593) | 3.258 | 2.072 | 0.078 (0.673 / 0.082) | 18.681 / 16.643 |
| contextual | normal | 4.2 (57.7 / 1.3) | 15.415 (2.332 / 16.133) | 0.085 (0.569 / 0.059) | 0.213 (0.882 / 0.176) | 2.694 | 1.735 | 0.239 (0.885 / 0.204) | 26.273 / 21.637 |
| contextual | oracle_value | 6.5 (76.9 / 2.6) | 15.116 (0.878 / 15.897) | 0.085 (0.569 / 0.059) | 0.213 (0.882 / 0.176) | 2.694 | 1.735 | 0.239 (0.885 / 0.204) | 30.673 / 27.294 |
| calibration_isolated | normal | 44.8 (41.7 / 45.0) | 2.405 (2.497 / 2.398) | 0.147 (0.139 / 0.147) | 0.582 (0.563 / 0.583) | 3.255 | 2.114 | 0.072 (0.735 / 0.073) | 16.966 / 15.517 |
| calibration_isolated | zero | 0.8 (0.0 / 0.9) | 9.233 (9.283 / 9.229) | 0.000 (0.000 / 0.000) | 0.000 (0.000 / 0.000) | 0.000 | 0.000 | — (— / —) | 0.000 / 0.000 |
| calibration_isolated | shuffled | 1.5 (2.0 / 1.4) | 8.049 (7.552 / 8.088) | — (— / —) | — (— / —) | 3.265 | 2.204 | 0.074 (0.113 / 0.075) | 16.980 / 15.537 |

Isolated normal last-position argmax slot counts (slots 0–15): [55, 57, 64, 59, 73, 78, 75, 70, 65, 50, 49, 54, 61, 72, 50, 68]; t0: [35, 4, 1, 1, 3, 0, 3, 1, 0, 0, 1, 2, 0, 1, 0, 0]

**Training (answer CE, isolated write) and learning curves (isolated controls)**

Train CE mean, steps 1–100: 6.266; around 25% (1450–1549): 4.410; 50%: 3.678; 75%: 2.699; last 100: 2.118. Steps 6000, stable step 0.163 s, clip fraction 0.983, grad norm mean/max 3.64/39.31.

Train CE, mean per 500-step block (ending at step): 500: 5.086, 1000: 4.601, 1500: 4.418, 2000: 4.286, 2500: 4.019, 3000: 3.796, 3500: 3.494, 4000: 3.210, 4500: 2.818, 5000: 2.488, 5500: 2.350, 6000: 2.279.

| point | EM normal (t0 / t≥1) | EM zero | EM shuffled | NLL normal | NLL zero | NLL shuffled | target weight all | target weight last |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 25% | 5.3 (7.7 / 5.2) | 1.1 | 1.9 | 4.419 | 9.249 | 4.836 | 0.133 | 0.248 |
| 50% | 17.2 (13.5 / 17.4) | 1.1 | 1.5 | 3.599 | 9.249 | 5.297 | 0.132 | 0.404 |
| 75% | 37.7 (40.4 / 37.6) | 1.1 | 1.5 | 2.722 | 9.249 | 6.893 | 0.143 | 0.543 |
| 100% | 45.3 (42.3 / 45.5) | 1.1 | 2.3 | 2.406 | 9.249 | 8.205 | 0.150 | 0.589 |

Reload: True / True; start replay (record-only): {'episodes': 1000, 'prediction_mismatches': 4, 'nll_max_abs_difference': 0.3117866516113281}; start adapter 9ea30adb36d9 = pin 9ea30adb36d9; initial zero-exit vs Base max abs: None; context≠isolated tokens: {'train': 0, 'dev': 0, 'calibration': 0}; wall 1388 s, 0.3855 MIG-h = 0.0964 node·h.

### C2 (ATT-0171-1)

Strata: target position 0 n=52, ≥1 n=948.

**H3 gate (isolated write, E-0010@2 thresholds)**

| stratum | EM normal | EM zero | EM shuffled | margin (≥10 pp) | NLL normal | NLL zero | NLL shuffled | target weight all (≥0.125) | target weight last | pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| all | 46.5 | 1.1 | 2.3 | 221/500 = 44.2 pp | 2.394 | 9.248 | 8.245 | 0.157 | 0.598 | True |
| position0 | 44.2 | 0.0 | 5.8 | 5/13 = 38.5 pp | 2.625 | 9.402 | 7.770 | 0.152 | 0.552 | True |
| position1plus | 46.6 | 1.2 | 2.1 | 211/474 = 44.5 pp | 2.382 | 9.240 | 8.271 | 0.157 | 0.600 | True |

**Controls and diagnostics (E=16 dev, 1000 episodes); EM shown as all (t0 / t≥1)**

| protocol | control | EM | NLL (t0 / t≥1) | target weight all (t0 / t≥1) | target weight last | n_eff all | n_eff last | last-pos argmax: most common slot rate (t0 / t≥1) | fusion/residual mean last / all |
|---|---|---|---|---|---|---:|---:|---|---|
| isolated | normal | 46.5 (44.2 / 46.6) | 2.394 (2.625 / 2.382) | 0.157 (0.152 / 0.157) | 0.598 (0.552 / 0.600) | 3.212 | 2.017 | 0.074 (0.654 / 0.077) | 17.287 / 16.337 |
| isolated | zero | 1.1 (0.0 / 1.2) | 9.248 (9.402 / 9.240) | 0.000 (0.000 / 0.000) | 0.000 (0.000 / 0.000) | 0.000 | 0.000 | — (— / —) | 0.000 / 0.000 |
| isolated | shuffled | 2.3 (5.8 / 2.1) | 8.245 (7.770 / 8.271) | — (— / —) | — (— / —) | 3.237 | 2.149 | 0.077 (0.154 / 0.076) | 17.174 / 16.313 |
| isolated | permuted | 2.5 (1.9 / 2.5) | 8.072 (8.002 / 8.076) | 0.157 (0.152 / 0.157) | 0.598 (0.552 / 0.600) | 3.212 | 2.017 | 0.074 (0.654 / 0.077) | 17.573 / 15.849 |
| isolated | oracle_key | 25.8 (25.0 / 25.8) | 3.201 (3.133 / 3.205) | 0.374 (0.370 / 0.374) | 0.374 (0.370 / 0.374) | 3.292 | 3.292 | 0.071 (0.596 / 0.074) | 16.620 / 14.798 |
| isolated | oracle_value | 75.1 (82.7 / 74.7) | 0.898 (0.768 / 0.905) | 0.157 (0.152 / 0.157) | 0.598 (0.552 / 0.600) | 3.212 | 2.017 | 0.074 (0.654 / 0.077) | 19.044 / 16.957 |
| contextual | normal | 5.3 (75.0 / 1.5) | 14.902 (1.675 / 15.627) | 0.086 (0.551 / 0.061) | 0.209 (0.940 / 0.169) | 2.379 | 1.755 | 0.262 (0.962 / 0.224) | 27.730 / 22.559 |
| contextual | oracle_value | 6.3 (82.7 / 2.1) | 15.140 (0.765 / 15.928) | 0.086 (0.551 / 0.061) | 0.209 (0.940 / 0.169) | 2.379 | 1.755 | 0.262 (0.962 / 0.224) | 32.698 / 29.095 |
| calibration_isolated | normal | 47.2 (42.4 / 47.6) | 2.374 (2.506 / 2.363) | 0.153 (0.133 / 0.154) | 0.590 (0.562 / 0.592) | 3.216 | 2.053 | 0.069 (0.735 / 0.072) | 17.204 / 16.288 |
| calibration_isolated | zero | 0.9 (0.0 / 0.9) | 9.231 (9.276 / 9.227) | 0.000 (0.000 / 0.000) | 0.000 (0.000 / 0.000) | 0.000 | 0.000 | — (— / —) | 0.000 / 0.000 |
| calibration_isolated | shuffled | 1.4 (1.3 / 1.4) | 8.223 (7.670 / 8.267) | — (— / —) | — (— / —) | 3.228 | 2.125 | 0.074 (0.106 / 0.076) | 17.240 / 16.301 |

Isolated normal last-position argmax slot counts (slots 0–15): [54, 58, 64, 59, 68, 74, 71, 70, 67, 52, 58, 54, 63, 72, 52, 64]; t0: [34, 2, 1, 1, 3, 1, 2, 3, 1, 0, 1, 3, 0, 0, 0, 0]

**Training (answer CE, isolated write) and learning curves (isolated controls)**

Train CE mean, steps 1–100: 6.023; around 25% (1450–1549): 4.326; 50%: 3.465; 75%: 2.489; last 100: 2.080. Steps 6000, stable step 0.155 s, clip fraction 0.998, grad norm mean/max 3.90/37.21.

Train CE, mean per 500-step block (ending at step): 500: 5.037, 1000: 4.499, 1500: 4.414, 2000: 4.191, 2500: 3.968, 3000: 3.614, 3500: 3.371, 4000: 3.101, 4500: 2.662, 5000: 2.501, 5500: 2.265, 6000: 2.243.

| point | EM normal (t0 / t≥1) | EM zero | EM shuffled | NLL normal | NLL zero | NLL shuffled | target weight all | target weight last |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 25% | 4.9 (5.8 / 4.9) | 1.1 | 1.3 | 4.565 | 9.248 | 5.027 | 0.111 | 0.238 |
| 50% | 15.3 (11.5 / 15.5) | 1.1 | 1.2 | 3.755 | 9.248 | 5.946 | 0.136 | 0.414 |
| 75% | 38.4 (38.5 / 38.4) | 1.1 | 1.9 | 2.731 | 9.248 | 7.673 | 0.150 | 0.550 |
| 100% | 46.5 (44.2 / 46.6) | 1.1 | 2.3 | 2.394 | 9.248 | 8.245 | 0.157 | 0.598 |

Reload: True / True; start replay (record-only): {'episodes': 1000, 'prediction_mismatches': 0, 'nll_max_abs_difference': 0.0}; start adapter 9ea30adb36d9 = pin 9ea30adb36d9; initial zero-exit vs Base max abs: None; context≠isolated tokens: {'train': 0, 'dev': 0, 'calibration': 0}; wall 1324 s, 0.3676 MIG-h = 0.0919 node·h.

## Checks (all stage-B runs)
- **Start adapter:** the loaded adapter.pt equals its pinned producer digest.
  - C1/C2: 9ea30adb…6eb616.
  - P1: ATT-0166-1's b5b2aec9….
  - P2: ATT-0167-1's aa3919e2….
  - It was checked before and after loading.
- **Replay** of the start run's recorded contextual dev predictions:
  - 0/1000 mismatches for C2 and P2, both on 2g slices.
  - 4/1000 for C1 and P1, both on 1g slices; NLL max difference 0.312 and 0.196. These are recorded only, per D-0018/D-0019.
  - ATT-0146-1 had already recorded the same 4 mismatches for R1 on a 1g slice.
- **Reload:** saving and reloading gives identical full records and summaries for all three protocols.
- **Token format:** context tokens equal isolated tokens for every train, dev and calibration fact.
- **Slices** (metadata only): C1 and P1-B ran on 1g.24gb; C2, P2-B and both stage-A runs ran on 2g.24gb.

## Attempts and compute

| attempt | role | result | MIG-h |
|---|---|---|---:|
| ATT-0164-1 | checks at 64331a4 (init test failed, RV-0049 low) | — (failed manifest) | 0.0367 |
| ATT-0165-1 | checks at f01b036 (RV-0049 fixes) | RES-0158 | 0.0500 |
| ATT-0168-1 | RV-0050 regression, cases did not run (job-script cd) | — (failed manifest) | 0.0203 |
| ATT-0168-2 | RV-0050 regression at 7d53d5d | RES-0164 | 0.0642 |
| ATT-0166-1 | P1 stage A (R1 seed 1) | RES-0162 | 0.2850 |
| ATT-0167-1 | P2 stage A (R1 seed 2) | RES-0163 | 0.2864 |
| ATT-0172-1 | P1 stage B | RES-0170 | 0.3756 |
| ATT-0173-1 | P2 stage B | RES-0171 | 0.3575 |
| ATT-0170-1 | C1 | RES-0168 | 0.3897 |
| ATT-0171-1 | C2 | RES-0169 | 0.3717 |
| **total** | | | **2.2369 MIG-h = 0.559 node·h** (estimate 0.55, cap 0.8) |

**Evidence** (screening, E-0017, each supports H-0012 because its H3 passes):
- EV-0102: P1, stage A included.
- EV-0103: P2, stage A included.
- EV-0104: C1, descriptive.
- EV-0105: C2, descriptive.

## Open issues
- **Spec mismatch on the stage-B results.** RES-0168 to RES-0171 import with `spec_match=False`, and `rl evidence add` warned that they "cannot serve the pre-specified comparison". The cause is the arm label:
  - the runner writes the program arm `arm: "D5"` into its manifest;
  - I registered the runs under the E-0017 arm names `P1-B`, `P2-B`, `C1`, `C2`.

  Seed, code, contract and data agree. E-0016's D5 (spec arm D5) and the stage-A runs (no manifest arm) match. I did not edit any output manifest; the coordinator decides how to treat it.
- **Single run per arm.** Each of the three seeds of the recipe has one run, so this is screening only.
- **Stage-A seed rule.** For seed s ≥ 1, s is appended to the recorded seed-0 order parts. This is a documented interpretation, accepted in RV-0049; the contract did not fix where an absent seed term goes.
