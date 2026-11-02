# T-0027 results: E-0024@1 (RB1 position-0 share and constant-read index)

This is a zero-training diagnostic. It does not change any hypothesis status. Scope: RB1-02, RB1-03, RB1-08 and RB1-09, seed 0, 1,000 dev rows per task. The forwards cover gold candidates only, one row at a time, under normal, iv_strict and iv_window8. No confirmation set was opened, no training ran and no checkpoint was deleted.

## Outcome (preregistered reading)

**Reading: mixed.** Under the contract's rule for mixed or inconsistent layers, this means a per-layer description and no unified conclusion.

- In every concentrated residual cell (DOC-0263 normal answer-position n_eff < 2), the read is dominated by position 0: share0 is 0.81–0.98, against a uniform 1/n of 0.057 (PIQA) or 0.0087 (BoolQ).
- The read vector is near-constant across episodes (C > 0.9) only at PIQA layer 27: C is 0.987 for RB1-02 and 0.995 for RB1-03.
- In the other concentrated cells, C is 0.50–0.68.
- The "sink read" criterion (share0 > 0.5 and C > 0.9 in all concentrated layers) therefore fails on C. The "content read" criterion (share0 < 0.2) fails everywhere.

Final files:
- summary: `runs/ATT-0285-1/summary.json`, manifest complete, validator passed
- per-arm outputs: RB1-02 from `runs/ATT-0283-1/RB1-02/sink.json`; RB1-03, RB1-08 and RB1-09 from `runs/ATT-0275-1/RB1-0x/sink.json`

## Attempts, commits and checks

| Arm | Attempt | Commit | Gate reference | Gate result |
| --- | --- | --- | --- | --- |
| RB1-03, RB1-08, RB1-09 | ATT-0275-1 | ee5d8f6 | recorded E-0021 run (2g) | gold NLL and every read summary: 0 difference, all 3 conditions |
| RB1-02 | ATT-0283-1 | ec88f08 | E-0021 `evaluate_history` rerun on the same 2g allocation (D-0050) | 0 difference, all 3 conditions |
| summary and identity | ATT-0285-1 | 582b444 | — | identity passed; checks_passed true |

- **Checkpoints.** Every arm's training manifest and checkpoint-inventory SHA256 matched the E-0021 provenance; a mismatch would have stopped the run.
- **Position 0.** PIQA position 0 is token 39992 ("Goal") in all 1,000 rows. BoolQ position 0 is token 12187 ("Pass") in all 1,000 rows.
- **RB1-02 against its recorded 1g E-0021 run.** This is descriptive only: slice rounding under D-0018/D-0019.

  | Condition | Gold NLL max diff | Gold NLL mean diff | Read-summary n_eff rel. diff |
  | --- | ---: | ---: | ---: |
  | normal | 1.86 | 0.079 | 1.9e-3 |
  | iv_strict | 1.77 | 0.078 | 4.1e-4 |
  | iv_window8 | 2.83 | 0.164 | 2.2e-5 |

  On the same allocation the difference is exactly 0.
- **iv_window8 check.** Passes for all arms: share0 at answer positions ≥ 8 has a maximum of 0.0. Some PIQA answer positions lie below 8: 291 per layer-step. There position 0 is still readable, so the iv_window8 question-level share0 is 0.02–0.05 rather than 0.
- **iv_strict check.** Amended by D-0051, *after* the preregistered bound failed in ATT-0283-1.
  - The original bound required |strict − normal| question-level share0 < 0.01. It failed at 4 cells:

    | Arm | Cell | Difference |
    | --- | --- | ---: |
    | RB1-02 | BoolQ L13 | −0.0102 |
    | RB1-03 | PIQA L20, both steps | −0.0108 |
    | RB1-09 | PIQA L20 | +0.0152 |

    RB1-08's largest difference was 0.0095. These values stay in the summary as `strict_bound_0.01_ok`.
  - Cause (coordinator: a design error):
    - Removing the current slot renormalizes share0 by 1/(1−cw), where cw is up to 0.047.
    - From layer 13 onward, iv_strict at earlier adapted layers also changes the hidden states.
  - Replacement: the exact layer-6 step-1 identity share0_strict = share0_normal/(1−cw_normal), checked per position. On RB1-02, first 50 dev rows per task, over 9,330 positions, the maximum absolute error is 1.8e-7 (tolerance 1e-5).
- **Reviews.** RV-0076 (2 medium, 1 low), then the recheck RV-0077 (one low remaining, no blocking findings).
  - RV-0077's remaining low: a NaN can vanish inside `max()`. It is not fixed. A newly computed NaN would still fail when the sink.json is written, because write_json rejects NaN (allow_nan=False).
  - Later changes accepted by the coordinator without a new review round:
    - ee5d8f6 (D-0048): the gold NLL is summed exactly as E-0021 sums it.
    - ec88f08 (D-0050): the same-allocation gate. Its test proves the gate's reference is `evaluate_history` and that perturbed outputs fail.
    - 582b444 (D-0051): the identity check and the amended summary.
  - `rl` reports coverage as stale for `src/test_rb1_artifacts.py`. That file was changed only by 8bf0146, a 16-line test addition on main from T-0025. No E-0024 run executes it (MSG-0739).
- **Batching.** Batching was rejected. ATT-0270-1 showed that BF16 results depend on tensor shape even on plain Base: padding a single row changes BoolQ gold NLL by up to 1.37. Every E-0024 forward is therefore batch-one.
- **Tests.** ATT-0284-1: 58 passed. This includes the GPU test that batch-one gold forwards match the E-0021 path exactly.

## Residual group, normal condition (question-level means; * = concentrated cell)

Column key:
- share0, share0-2: read share on position 0, and on positions 0–2.
- argmax0: fraction of questions whose largest read weight is on position 0.
- C: constant-read index, the mean pairwise cosine of read vectors across 200 sampled questions.
- C_value: the same cosine for the current-token value vectors.

RB1-03 step 2 is within 0.001 of step 1 in every cell, so the table shows step 1.

| Arm | Task | Layer | share0 | 1/n | share0-2 | argmax0 | C | C_value | n_eff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RB1-02 | BoolQ | 6 | 0.038 | 0.0087 | 0.049 | 1.000 | 0.780 | 0.993 | 111.0 |
| RB1-02 | BoolQ | 13 | 0.534 | 0.0087 | 0.548 | 1.000 | 0.500 | 0.753 | 4.82 |
| RB1-02 | BoolQ | 20 | 0.598 | 0.0087 | 0.605 | 1.000 | 0.697 | 0.876 | 3.12 |
| RB1-02 | BoolQ | 27* | 0.919 | 0.0087 | 0.920 | 1.000 | 0.677 | 0.927 | 1.19 |
| RB1-02 | PIQA | 6* | 0.831 | 0.0565 | 0.856 | 0.996 | 0.541 | 0.988 | 1.73 |
| RB1-02 | PIQA | 13* | 0.812 | 0.0565 | 0.833 | 1.000 | 0.572 | 0.607 | 1.79 |
| RB1-02 | PIQA | 20* | 0.869 | 0.0565 | 0.883 | 1.000 | 0.554 | 0.671 | 1.52 |
| RB1-02 | PIQA | 27* | 0.967 | 0.0565 | 0.970 | 1.000 | **0.987** | 0.816 | 1.11 |
| RB1-03 | BoolQ | 6 | 0.050 | 0.0087 | 0.057 | 1.000 | 0.831 | 0.991 | 91.8 |
| RB1-03 | BoolQ | 13 | 0.660 | 0.0087 | 0.675 | 1.000 | 0.471 | 0.873 | 2.37 |
| RB1-03 | BoolQ | 20 | 0.666 | 0.0087 | 0.675 | 1.000 | 0.688 | 0.922 | 2.41 |
| RB1-03 | BoolQ | 27* | 0.922 | 0.0087 | 0.923 | 1.000 | 0.512 | 0.858 | 1.18 |
| RB1-03 | PIQA | 6* | 0.865 | 0.0565 | 0.886 | 0.999 | 0.504 | 0.985 | 1.50 |
| RB1-03 | PIQA | 13 | 0.681 | 0.0565 | 0.715 | 1.000 | 0.425 | 0.684 | 2.53 |
| RB1-03 | PIQA | 20* | 0.884 | 0.0565 | 0.896 | 1.000 | 0.499 | 0.560 | 1.44 |
| RB1-03 | PIQA | 27* | 0.982 | 0.0565 | 0.984 | 1.000 | **0.995** | 0.833 | 1.06 |

How to read these columns:
- 1/n is the uniform-read share0, averaged the same way.
- Position-weighted share0 (the DOC-0263 convention) differs from the question-level value only for PIQA, where questions have several answer positions. RB1-02 PIQA L13/L20/L27 give 0.776/0.829/0.950, and RB1-03 0.645/0.847/0.972.
- The recomputed n_eff equals DOC-0263 exactly: the read summaries match E-0021 with zero difference.

Under iv_strict the concentrated-cell values move by at most 0.011 in share0. iv_window8 removes position 0 for late answer positions, and its n_eff is 5–8.

## Concat group (descriptive only; no concentrated cells; normal condition)

| Arm | Task | Layer | share0 | 1/n | argmax0 | C | n_eff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RB1-08 | BoolQ | 6 / 13 / 20 / 27 | 0.007 / 0.011 / 0.011 / 0.373 | 0.0087 | 0 / 0 / 0 / 1.000 | 0.905 / 0.615 / 0.813 / 0.664 | 134 / 133 / 139 / 9.1 |
| RB1-08 | PIQA | 6 / 13 / 20 / 27 | 0.111 / 0.073 / 0.245 / 0.754 | 0.0565 | 0.856 / 0.620 / 0.962 / 1.000 | 0.814 / 0.564 / 0.345 / 0.707 | 34.3 / 35.4 / 23.2 / 2.69 |
| RB1-09 | BoolQ | 6 / 13 / 20 / 27 | 0.004 / 0.006 / 0.011 / 0.551 | 0.0087 | 0 / 0 / 0 / 1.000 | 0.915 / 0.554 / 0.937 / 0.471 | 112 / 135 / 106 / 3.76 |
| RB1-09 | PIQA | 6 / 13 / 20 / 27 | 0.085 / 0.049 / 0.283 / 0.702 | 0.0565 | 0.765 / 0 / 0.902 / 1.000 | 0.807 / 0.630 / 0.492 / 0.679 | 34.7 / 36.2 / 17.1 / 3.24 |

The concat reads are close to the uniform 1/n at BoolQ layers 6–20 and PIQA layers 6–13. Layer 27 puts 0.37–0.75 of the read on position 0.

## Per-layer description for the stage report (no unified conclusion)

These statements are observations from the table above:
- **Position-0 concentration.** The concentrated residual reads (n_eff < 2) put most of their weight on position 0. That token is the same in every episode of a task ("Goal" or "Pass"), and as the first causal token its key and value do not depend on the episode.
- **Constancy across episodes.** The read vector at the first answer position is near-constant only at PIQA layer 27 (C 0.987 for RB1-02, 0.995 for RB1-03). At the other concentrated cells C is 0.50–0.68, so the read output still varies across episodes despite 81–92% position-0 weight.
- **Why C stays moderate (untested).** One possible mechanism: if the position-0 value vector is small next to the remaining 8–19% of weight on content positions, those content positions set the read direction. Value norms were not measured here, so this is not tested.
- **Swap, permute and window effects.** The preregistered sink explanation (small swap drop, large permute and window drops) is supported only at PIQA layer 27. It is not supported as a description of all concentrated layers.

## Cost

All costs are measured on MIG 2g.24gb and counted as node·h = MIG instance·h × 0.25. D-0047 raised the cap to 0.15 node·h and D-0048 raised it to 0.25 node·h.

| Attempt(s) | Purpose | MIG instance·h |
| --- | --- | ---: |
| ATT-0268-1, ATT-0272-1, ATT-0274-1, ATT-0282-1, ATT-0284-1 | tests | 0.070 |
| ATT-0269-1, ATT-0270-1 | batch checks | 0.016 |
| ATT-0273-1 | stopped at the E-0021 gate (NLL summation) | 0.147 |
| ATT-0275-1 | stopped at RB1-02 (1g reference) | 0.393 |
| ATT-0283-1 | RB1-02, then the summary failed the strict bound | 0.299 |
| ATT-0285-1 | identity and amended summary | 0.010 |
| **Total** | | **0.935 = 0.234 node·h** |

Usage records: U-0277 to U-0282 and U-0284 to U-0288.

Measured per-condition times, batch-one gold forwards: 121–165 s. RB1-02 needed 345–349 s per condition including the same-allocation evaluator.

## Open

- The preregistered C criterion separates layer 27 from the other concentrated layers. The value-norm explanation above is not tested.
- RB1-02's recorded E-0021 run on the 1g slice differs from 2g by slice rounding. The E-0021 conclusions use the recorded numbers.
