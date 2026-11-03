# E-0032@1 seed-0 analysis (T-0035)

Analysis job ATT-0423-1 (PBS 3788639, commit 9aa4d79, RES-0404), output
/work/xg24i002/x10041/bypass-lora/research/artifacts/ATT-0423-1/analysis.json.
Input research/artifacts/T-0035/analysis-input.json (sha256 7d02f55d...).
All numbers are BoolQ dev (1,000 rows), exact fractions in pp, unless noted.
No hypothesis status was changed. The contract changes statuses only after the
conclusion review.

## Runs (training source 9aa4d79; review round RV-0108/RV-0109 pass)

| arm | attempt | result | slice | BoolQ | PIQA | Keep iv_swap BoolQ |
|---|---|---|---|---|---|---|
| SEG-W1 | ATT-0415-1 | RES-0394 | 1g | 62.9 | 66.7 | 61.7 |
| SEG-W2 | ATT-0416-1 | RES-0395 | 2g | 63.7 | 67.3 | 63.8 |
| SEG-X5 | ATT-0417-1 | RES-0396 | 1g | 65.1 | 68.2 | 58.7 |
| SEG-X6 | ATT-0418-1 | RES-0397 | 1g | 61.6 | 66.2 | |
| SEG-X7 | ATT-0419-1 | RES-0398 | 2g | 61.2 | 52.9 | |

The references are E-0025 seed 0, pinned by SHA256: SEG-05 65.9, SEG-06 61.5,
SEG-07 61.2, SEG-04 64.2, SEG-00 60.9, SEG-05 iv_swap 59.4; PIQA SEG-05 69.3.
The slice is metadata only (D-0018/D-0019). Every in-job identity check
compares two passes on the same slice.

## Rule 1 (diagnostic): passed, 38/38 checks

- **Smoke ATT-0407-1** (rule1_smoke at b0e7a8f, RES-0384): 359 tests pass,
  among them the mask-identity outputs and gradients and the Reset boundary
  test.
  - Write masks: W1 writes once per example (train 26,230; dev 2,000). W2
    follows the frozen sentence rule (train 58,315 writes, 1-39 per example;
    dev 5,393).
  - W1/W2 initial digests equal the RES-0281 SEG-05 digests for seeds 0-2.
  - X5/X6/X7 initial values are identical per seed.
  - The replay of ATT-0379-2 on 2g has identical rows and 0 flips.
- **Checks ATT-0408-1** (9aa4d79, RES-0383): 362 tests pass.
- **Source check:** b0e7a8f..9aa4d79 changes only seg_w_analysis.py and two
  test files, so the rule-1 source shows no training-path difference from
  the training source.
- **Each training run:**
  - step 0 and iv_off equal the same-allocation Base_seg;
  - reload is identical;
  - stored, trainable and with-grad parameter counts match (2,752,512 for W;
    2,228,224 for X);
  - segment-2 accounting holds: writes = reads = tokens, and 0 for T0;
  - W segment-1 writes equal the mask totals;
  - the in-job iv_swap is present for the Keep arms;
  - the X triplet initial digest 7a21d194... is equal across the three arms.

## Rule 2 (quantities, seed 0)

| V | Keep | Δ_V(0) | gain_V(0) | Δg_V(0) | lora_V | swap_V | controls (learned?) |
|---|---|---|---|---|---|---|---|
| SEG-W1 | 62.9 | -3.0 | +1.4 | -3.0 | -1.3 | +1.2 | SEG-06 61.5 / SEG-07 61.2 (no) |
| SEG-W2 | 63.7 | -2.2 | +2.2 | -2.2 | -0.5 | -0.1 | SEG-06 61.5 / SEG-07 61.2 (no) |
| SEG-X5 | 65.1 | -0.8 | +3.5 | -0.9 (descriptive) | +0.9 | +6.4 | X6 61.6 / X7 61.2 (no) |

The values are gain_SEG05(0) = +4.4 and G_SEG05 = 88/3000. The learnability
line is 61.9. At seed 0, neither the SEG-05 controls nor the X controls
learned.

## Rule 3 (screening)

| V | condition | result |
|---|---|---|
| W1 | Δ ≥ +1.5 | fails |
| W2 | Δ ≥ +1.5 | fails |
| X5 | Δ ≥ +1.5 | fails |
| X5 | gain_X(0) ≥ 133/3000 (4.433) | fails (3.5) |

No variant passed.

## Rule 4

Nothing was selected, so seeds 1-2 are not run (the contract says "没有变体通过 -> 不复现").

## Rule 5

Rule 5 was not evaluated because there was no replication.

## Rule 6 (screening)

| V | Δ_V(0) | Δ_V(0) ≤ 0? |
|---|---|---|
| W1 | -3.0 | yes |
| W2 | -2.2 | yes |
| X5 | -0.8 | yes |

- **H-0032:** W1 and W2 both have Δ ≤ 0, so the rule-6 outcome is
  **challenges (screening)**.
- **H-0033:** X has Δ ≤ 0, so the outcome is **challenges (screening)**. As the
  contract requires, Δg_X(0) is not used for this.

## Rule 8 (descriptive intervals)

Paired, task-stratified bootstrap, 95% interval, 10,000 draws, seed 20261101,
max control recomputed per draw. The intervals cover row sampling only, not
seed or selection uncertainty.

| V | Δ(0) | gain(0) | swap(0) |
|---|---|---|---|
| W1 | [-5.7, -0.4] | [-0.4, +2.7] | [+0.3, +2.1] |
| W2 | [-4.8, +0.4] | [-0.1, +4.0] | [-1.3, +1.2] |
| X5 | [-2.8, +1.2] | [+0.6, +5.9] | [+3.1, +9.7] |

For X5, Δg(0) is [-3.3, +1.5].

PIQA Δ(0), descriptive:

| V | PIQA Δ(0) | interval |
|---|---|---|
| W1 | -2.6 | [-4.2, -1.0] |
| W2 | -2.0 | [-3.6, -0.4] |
| X5 | -1.1 | [-2.5, +0.3] |

## Rule 7 (mechanism description, not a gate)

Job ATT-0422-1 (RES-0403) measures the segment-1 share of the ρᵀc
normalizer. The share is computed at the segment-2 answer positions, per
layer and gold candidate, as a row mean. The reference model is the retained
E-0030 SEG-05 checkpoint at seed 3, not seed 0.

| model | BoolQ L6 / L13 / L20 / L27 | PIQA L6 / L13 / L20 / L27 |
|---|---|---|
| SEG-W1 (seed 0) | 0.004 / 0.007 / 0.010 / 0.001 | 0.040 / 0.007 / 0.006 / 0.000 |
| SEG-W2 (seed 0) | 0.192 / 0.005 / 0.015 / 0.000 | 0.048 / 0.005 / 0.005 / 0.000 |
| SEG-05 (seed 3, ATT-0379-2) | 0.548 / 0.628 / 0.558 / 0.996 | 0.528 / 0.547 / 0.515 / 0.498 |

The contract's three compatibility statements each have a precondition, and
none of the preconditions holds:
- "W1 comparable to SEG-05" requires |Δ_W1(0)| < 1.5; the value is 3.0.
- "W2 clearly higher than SEG-05" does not hold; Δ is -2.2.
- "Δg_X > 0" does not hold; Δg_X(0) is -0.9, and it is descriptive only.

Description: under W1/W2, segment-1 writes carry at most 1.5% of the ρᵀc
normalizer at layers 13-27. At layer 6 on BoolQ, W1 carries 0.4% and W2 19%;
on PIQA, W1 carries 4.0% and W2 4.8%.
For SEG-05 the share is 50-100%. The lower Keep scores and the near-zero W
swap effects (W1 +1.2, W2 -0.1, against SEG-05's +6.5) are 与“第 1 段的少量写入在读出归一化中几乎不占质量、第 2 段主要读自己的写入”相容.
This is not the only explanation: W1/W2 also change what the segment-1 state
summarizes, and the reference checkpoint comes from another seed.

## Cost (measured, node-h = MIG-h × 0.25)

E-0032 total including failures is 2.274236 node-h. This is the analysis's
2.273403 over 11 attempts plus 0.000833 for the analysis job itself.

| item | node-h |
|---|---|
| seed-0 training (five arms) | 2.041806 |
| smoke, checks and failed attempts | 0.196806 |
| rule-7 mechanism job | 0.034792 |
| analysis job | 0.000833 |

The total is within the seed-0 cap of 2.35 (D-0077) and the 5.0 total cap.
There is no replication spend.

## Limitations

- Dev both screens and judges. The gates are exact point estimates.
- At seed 0, the SEG-05 controls and the X controls are both below the
  learnability line, so seed-0 gains are biased upward (contract rule 2/3).
- Three arms ran on 1g and two on 2g. Per D-0018/D-0019 this is metadata only.
- SEG-X7 PIQA (52.9) vs SEG-07 (67.9): closed by the steps diagnostic below.

## SEG-X7 training failure (diagnostic, MSG-1065; ATT-0425-1, RES-0406)

A read-only PBS diagnostic at 5b0fad0 read steps.jsonl from the five seed-0
runs and from the 12 retained E-0030 runs (seeds 3-5, SEG-04/05/06/07).
Windows are 100 steps.

- **SEG-X7's spike.** After warmup reaches the peak LR, the window mean
  example_mean_nll jumps from 2.495 (steps 601-700) to 4.561 (701-800).
  preclip_norm peaks at 178.8 at step 792.
- **The loss never recovers.** From step 801 on, SEG-X7 minus SEG-X6 is at
  least 0.77 nat in every window, with a mean of 1.33. Over the last 2,000
  steps it is 2.953 against X6 2.083, X5 1.879, W1 1.969 and W2 1.965.
- **Dev NLL rises with it.** SEG-X7's PIQA dev NLL is 5.18; the other 16
  runs fall between 2.84 and 3.07.
- **A large gradient norm alone does not mark a failure.** Norms above 50
  also appear in SEG-X6 (159.9), SEG-X5 (86.7) and SEG-W2 (86.4). The
  E-0030 SEG-07 runs reach 579-992, yet their loss tracks SEG-06: the mean
  window difference is about 0, and over the last 2,000 steps it is -0.014
  to -0.018. Only SEG-X7 stays above its partner after the spike.
- **The other jump is not arm-specific.** The jump at the window starting at
  step 5401 (+1.0 to +1.3 nat) appears in every examined run, across arms
  and seeds.

Conclusion: SEG-X7 is a failed control. Its loss spiked and did not recover,
so its BoolQ 61.2 and PIQA 52.9 are not valid estimates of T0_X(0). No rule
outcome changes:
- gain_X(0) already uses max(Reset_X, T0_X) = Reset_X (61.6).
- Rule 3 for X fails on Δ_X(0) = -0.8 alone. A better-trained T0_X could
  only lower gain_X.
- Rule 6 uses Δ_X(0) only.
- The X controls' learnability at seed 0 is descriptive, and the analysis
  does not use it.
