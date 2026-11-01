# T-0018 results: E-0016@1 D4/D5, isolated-encoding write (seed 0 screening)

This task measures and reports only. The H-0012 judgement and the route are the coordinator's. Confirmation data was never read.

The isolated write is a newly registered protocol. The gate below uses the E-0010@2 H3 thresholds, but it is reported separately from E-0010 and is not the same H3.

**Code.** Branch sr/researcher-3/T-0018 at 415cac3. Details: `research/notes/T-0018-implementation.md`.

**Reviews.**
- RV-0046 (reviewer-3) raised one low finding, fixed in 415cac3. The recheck, RV-0048, passed.
- The shared T-0017 parts are covered by RV-0045.
- D-0017 accepted the merged `v2_train.py`, checked by ATT-0160-1.
- D-0019 made the MIG guard record-only for ATT-0158-2.

## Summary

| arm | attempt | result | evidence | H3 (isolated) | EM normal / zero / shuffled | margin | NLL normal / zero / shuffled | target weight all / last |
|---|---|---|---|---|---|---:|---|---|
| D4 (from scratch) | ATT-0158-2 | RES-0156 | EV-0096 | fail | 1.4 / 1.1 / 1.1 | 0.3 pp | 4.423 / 9.248 / 4.424 | 0.058 / 0.063 |
| D5 (from R1) | ATT-0159-1 | RES-0153 | EV-0095 (supports H-0012) | pass | 42.8 / 1.1 / 1.7 | 41.1 pp | 2.514 / 9.248 / 7.977 | 0.162 / 0.571 |

**Observations. These are measured facts, not judgements.**
- **D5 passes all three thresholds,** in both target-position strata.
  - R1's own isolated-write E=16 EM in E-0015 was 2.5% (iso_kv, ATT-0141-1, on a different E-0015 evaluation set). Continued training under the isolated write raised it to 42.8%.
  - The isolated oracle_value reaches 73.6%. Oracle_key (24.5) is below normal (42.8).
  - After this training, the contextual-encoding normal is 5.7%: 65.4 at target 0 and 2.4 at target ≥1, with NLL 15.0. The adapter is not usable with contextual writes past the first fact.
- **D4 learned no address from answer CE alone in 6000 steps.**
  - The last-position read is uniform: weight 0.0625 = 1/16 and n_eff 16.0 in every stratum.
  - Normal and shuffled have equal NLL (4.423 vs 4.424), and oracle_value does not help (1.4). The memory branch acts as a content-independent bias that lowers NLL against zero (9.248) without retrieving.
  - Training CE plateaued at about 4.41 from about step 2500, and the clip fraction was 0.136. In D5 the clip fraction was 0.991 at the same LR and clip.
  - The fusion/residual ratio is high: 44.7 at the last position, against 18.2 in D5.

## Checks (both arms)
- **Token format:** context tokens equal isolated tokens for every train and dev fact (0 differ).
- **Gradients:** all six adapter parameters receive nonzero gradient (557,056 of 557,056).
- **Reload:** the saved and reloaded adapter gives identical full records and summaries for both protocols.
- **D4 initialization:** the zero-exit logits equal Base (max abs 0.0).
- **D5 resume:**
  - adapter only, with no optimizer state;
  - fresh 300-step warmup then cosine;
  - order seed 5813166871386195579;
  - replaying R1's recorded contextual dev predictions gave 0/1000 mismatches, NLL max abs 0.0;
  - R1 adapter sha256 9ea30adb…6eb616.
- **Slices (metadata only, D-0018/D-0019):** both formal runs landed on 2g.24gb slices, D4 on mg1110/2 and D5 on mg1110/1.

## Arm tables

### D4 (ATT-0158-2)

Strata: target position 0 n=52, ≥1 n=948.

**H3 gate (isolated write, E-0010@2 thresholds)**

| stratum | EM normal | EM zero | EM shuffled | margin (≥10 pp) | NLL normal | NLL zero | NLL shuffled | target weight all (≥0.125) | target weight last | pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| all | 1.4 | 1.1 | 1.1 | 3/1000 = 0.3 pp | 4.423 | 9.248 | 4.424 | 0.058 | 0.063 | False |
| position0 | 1.9 | 0.0 | 0.0 | 1/52 = 1.9 pp | 4.318 | 9.402 | 4.320 | 0.049 | 0.063 | False |
| position1plus | 1.4 | 1.2 | 1.2 | 1/474 = 0.2 pp | 4.428 | 9.240 | 4.430 | 0.058 | 0.063 | False |

**Controls and diagnostics (E=16 dev, 1000 episodes); EM shown as all (t0 / t≥1)**

| protocol | control | EM | NLL (t0 / t≥1) | target weight all (t0 / t≥1) | target weight last | n_eff all | n_eff last | last-pos argmax: most common slot rate (t0 / t≥1) | fusion/residual mean last / all |
|---|---|---|---|---|---|---:|---:|---|---|
| isolated | normal | 1.4 (1.9 / 1.4) | 4.423 (4.318 / 4.428) | 0.058 (0.049 / 0.058) | 0.063 (0.063 / 0.063) | 3.734 | 16.000 | 0.076 (0.115 / 0.078) | 44.652 / 48.924 |
| isolated | zero | 1.1 (0.0 / 1.2) | 9.248 (9.402 / 9.240) | 0.000 (0.000 / 0.000) | 0.000 (0.000 / 0.000) | 0.000 | 0.000 | — (— / —) | 0.000 / 0.000 |
| isolated | shuffled | 1.1 (0.0 / 1.2) | 4.424 (4.320 / 4.430) | — (— / —) | — (— / —) | 3.734 | 16.000 | 0.080 (0.096 / 0.081) | 44.651 / 48.924 |
| isolated | permuted | 1.7 (0.0 / 1.8) | 4.436 (4.337 / 4.442) | 0.058 (0.049 / 0.058) | 0.063 (0.063 / 0.063) | 3.734 | 16.000 | 0.076 (0.115 / 0.078) | 44.658 / 39.195 |
| isolated | oracle_key | 1.4 (1.9 / 1.4) | 4.447 (4.336 / 4.453) | 0.088 (0.114 / 0.086) | 0.088 (0.114 / 0.086) | 5.879 | 5.879 | 0.078 (0.173 / 0.080) | 34.319 / 30.568 |
| isolated | oracle_value | 1.4 (3.8 / 1.3) | 4.416 (4.316 / 4.421) | 0.058 (0.049 / 0.058) | 0.063 (0.063 / 0.063) | 3.734 | 16.000 | 0.076 (0.115 / 0.078) | 44.496 / 39.630 |
| contextual | normal | 2.2 (3.8 / 2.1) | 4.474 (4.344 / 4.481) | 0.058 (0.141 / 0.053) | 0.063 (0.062 / 0.063) | 4.309 | 16.000 | 0.116 (0.192 / 0.113) | 24.491 / 26.138 |
| contextual | oracle_value | 1.4 (3.8 / 1.3) | 4.493 (4.316 / 4.503) | 0.058 (0.141 / 0.053) | 0.063 (0.062 / 0.063) | 4.309 | 16.000 | 0.116 (0.192 / 0.113) | 24.147 / 21.508 |

Isolated normal last-position argmax slot counts (slots 0–15): [61, 64, 57, 56, 53, 63, 61, 67, 67, 65, 76, 45, 60, 69, 62, 74]; t0: [6, 0, 2, 6, 2, 2, 6, 3, 3, 5, 2, 2, 2, 3, 4, 4]

**Training (answer CE, isolated write) and learning curves (isolated controls)**

Train CE mean, steps 1–100: 6.430; around 25% (1450–1549): 4.588; 50%: 4.491; 75%: 4.438; last 100: 4.389. Steps 6000, stable step 0.152 s, clip fraction 0.136, grad norm mean/max 0.88/10.31.

Train CE, mean per 500-step block (ending at step): 500: 5.312, 1000: 4.703, 1500: 4.630, 2000: 4.564, 2500: 4.539, 3000: 4.525, 3500: 4.486, 4000: 4.474, 4500: 4.449, 5000: 4.443, 5500: 4.412, 6000: 4.413.

| point | EM normal (t0 / t≥1) | EM zero | EM shuffled | NLL normal | NLL zero | NLL shuffled | target weight all | target weight last |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 25% | 1.0 (1.9 / 0.9) | 1.1 | 1.4 | 4.666 | 9.248 | 4.666 | 0.058 | 0.063 |
| 50% | 1.0 (0.0 / 1.1) | 1.1 | 1.0 | 4.509 | 9.248 | 4.511 | 0.057 | 0.063 |
| 75% | 1.8 (1.9 / 1.8) | 1.1 | 1.2 | 4.461 | 9.248 | 4.462 | 0.058 | 0.063 |
| 100% | 1.4 (1.9 / 1.4) | 1.1 | 1.1 | 4.423 | 9.248 | 4.424 | 0.058 | 0.063 |

Reload: True / True; R1 replay: None; initial zero-exit vs Base max abs: 0.0; context≠isolated tokens: {'train': 0, 'dev': 0}; wall 1161 s, 0.3225 MIG-h = 0.0806 node·h.

### D5 (ATT-0159-1)

Strata: target position 0 n=52, ≥1 n=948.

**H3 gate (isolated write, E-0010@2 thresholds)**

| stratum | EM normal | EM zero | EM shuffled | margin (≥10 pp) | NLL normal | NLL zero | NLL shuffled | target weight all (≥0.125) | target weight last | pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| all | 42.8 | 1.1 | 1.7 | 411/1000 = 41.1 pp | 2.514 | 9.248 | 7.977 | 0.162 | 0.571 | True |
| position0 | 44.2 | 0.0 | 5.8 | 5/13 = 38.5 pp | 2.976 | 9.402 | 7.349 | 0.140 | 0.525 | True |
| position1plus | 42.7 | 1.2 | 1.5 | 391/948 = 41.2 pp | 2.489 | 9.240 | 8.011 | 0.163 | 0.574 | True |

**Controls and diagnostics (E=16 dev, 1000 episodes); EM shown as all (t0 / t≥1)**

| protocol | control | EM | NLL (t0 / t≥1) | target weight all (t0 / t≥1) | target weight last | n_eff all | n_eff last | last-pos argmax: most common slot rate (t0 / t≥1) | fusion/residual mean last / all |
|---|---|---|---|---|---|---:|---:|---|---|
| isolated | normal | 42.8 (44.2 / 42.7) | 2.514 (2.976 / 2.489) | 0.162 (0.140 / 0.163) | 0.571 (0.525 / 0.574) | 3.343 | 2.119 | 0.080 (0.538 / 0.082) | 18.234 / 16.598 |
| isolated | zero | 1.1 (0.0 / 1.2) | 9.248 (9.402 / 9.240) | 0.000 (0.000 / 0.000) | 0.000 (0.000 / 0.000) | 0.000 | 0.000 | — (— / —) | 0.000 / 0.000 |
| isolated | shuffled | 1.7 (5.8 / 1.5) | 7.977 (7.349 / 8.011) | — (— / —) | — (— / —) | 3.351 | 2.238 | 0.077 (0.135 / 0.077) | 18.138 / 16.626 |
| isolated | permuted | 2.4 (1.9 / 2.4) | 7.703 (7.657 / 7.705) | 0.162 (0.140 / 0.163) | 0.571 (0.525 / 0.574) | 3.343 | 2.119 | 0.080 (0.538 / 0.082) | 18.545 / 16.848 |
| isolated | oracle_key | 24.5 (15.4 / 25.0) | 3.091 (3.110 / 3.090) | 0.358 (0.349 / 0.358) | 0.358 (0.349 / 0.358) | 3.575 | 3.575 | 0.076 (0.596 / 0.079) | 17.437 / 15.530 |
| isolated | oracle_value | 73.6 (80.8 / 73.2) | 0.952 (0.782 / 0.962) | 0.162 (0.140 / 0.163) | 0.571 (0.525 / 0.574) | 3.343 | 2.119 | 0.080 (0.538 / 0.082) | 19.973 / 17.788 |
| contextual | normal | 5.7 (65.4 / 2.4) | 15.046 (1.556 / 15.786) | 0.090 (0.419 / 0.072) | 0.223 (0.937 / 0.184) | 3.000 | 1.624 | 0.237 (0.942 / 0.198) | 27.988 / 25.344 |
| contextual | oracle_value | 6.6 (80.8 / 2.5) | 14.725 (0.781 / 15.490) | 0.090 (0.419 / 0.072) | 0.223 (0.937 / 0.184) | 3.000 | 1.624 | 0.237 (0.942 / 0.198) | 32.271 / 28.714 |

Isolated normal last-position argmax slot counts (slots 0–15): [49, 58, 65, 57, 68, 80, 73, 77, 73, 57, 49, 51, 58, 68, 56, 61]; t0: [28, 2, 1, 1, 4, 2, 3, 2, 0, 3, 1, 3, 1, 1, 0, 0]

**Training (answer CE, isolated write) and learning curves (isolated controls)**

Train CE mean, steps 1–100: 6.263; around 25% (1450–1549): 4.292; 50%: 3.679; 75%: 2.742; last 100: 2.311. Steps 6000, stable step 0.159 s, clip fraction 0.991, grad norm mean/max 3.75/40.79.

Train CE, mean per 500-step block (ending at step): 500: 5.079, 1000: 4.573, 1500: 4.331, 2000: 4.190, 2500: 4.005, 3000: 3.741, 3500: 3.478, 4000: 3.160, 4500: 2.795, 5000: 2.508, 5500: 2.425, 6000: 2.439.

| point | EM normal (t0 / t≥1) | EM zero | EM shuffled | NLL normal | NLL zero | NLL shuffled | target weight all | target weight last |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 25% | 4.3 (7.7 / 4.1) | 1.1 | 1.8 | 4.246 | 9.248 | 4.719 | 0.124 | 0.234 |
| 50% | 20.6 (21.2 / 20.6) | 1.1 | 2.4 | 3.579 | 9.248 | 5.679 | 0.149 | 0.426 |
| 75% | 35.3 (36.5 / 35.2) | 1.1 | 2.4 | 2.850 | 9.248 | 7.266 | 0.155 | 0.524 |
| 100% | 42.8 (44.2 / 42.7) | 1.1 | 1.7 | 2.514 | 9.248 | 7.977 | 0.162 | 0.571 |

Reload: True / True; R1 replay: {'episodes': 1000, 'prediction_mismatches': 0, 'nll_max_abs_difference': 0.0}; initial zero-exit vs Base max abs: None; context≠isolated tokens: {'train': 0, 'dev': 0}; wall 1221 s, 0.3392 MIG-h = 0.0848 node·h.

## Attempts and compute

| attempt | role | result | MIG-h |
|---|---|---|---:|
| ATT-0151-1 | tests + default-off regression vs master (regression identical; test bug, not imported) | — | 0.0192 |
| ATT-0153-1 | smoke, landed on 1g, guard exit 3, no model work | — | 0.0003 |
| ATT-0153-2 | tests (29 passed) + D4/D5 40-step smoke | RES-0141 | 0.0267 |
| ATT-0156-1 | tests for the RV-0046 fix | RES-0142 | 0.0061 |
| ATT-0160-1 | combined suite (67 passed) + default identity, D-0017 | RES-0143 | 0.0261 |
| ATT-0158-1 | D4, landed on 1g, guard exit 3, no model work (failed) | — | 0.0003 |
| ATT-0158-2 | D4 formal | RES-0156 | 0.3250 |
| ATT-0159-1 | D5 formal | RES-0153 | 0.3433 |
| **total** | | | **0.7469 MIG-h = 0.187 node·h** (budget 0.25) |

## Open issues
- Both arms are single-seed screening only.
- The shuffled control has no target-weight diagnostic, because the donor episode has no target in the read memory. It is shown as "—".
- The contextual-encoding comparison is descriptive. The E-0015 R1 numbers come from a different evaluation set (E-0015 calibration), so they are not directly comparable.
