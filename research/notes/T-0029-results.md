# T-0029 / E-0026 results: r128 width probe in the ordinary RB1 setting

Locked analysis: ATT-0300-1 / RES-0274 (PBS 3772900, source 3678f60 = reviewed
81ad3de + input files). Full tables are in
`/work/xg24i002/x10041/bypass-lora/runs/ATT-0300-1/analysis.md`, sealed as DOC-0292@1,
with `analysis.json` alongside. Intervals are descriptive paired task-stratified bootstrap
95% intervals (10,000 draws, default_rng 20261101, indices shared across all labels).
They are conditional on the trained models and exclude seed and same-dev selection uncertainty.

## Arms (seed 0; RB1 init/order; 13,115 steps, 26,230 examples, 2,000,069 tokens)

| Arm | Attempt / result | Params | LR | PIQA % | BoolQ % | Macro % |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Base (ATT-0024-1) | – | 0 | – | 72.30 | 68.00 | 70.15 |
| W-01 R_res+T0 S1 r128 | ATT-0295-1 / RES-0270 | 2,752,512 | 3e-3 | 72.30 | 81.30 | 76.80 |
| W-02 R_res+KernelMatrix S1 r128 | ATT-0296-1 / RES-0271 | 2,752,512 | 3e-3 | 72.70 | 81.20 | 76.95 |
| W-03 four-layer all-7 LoRA r31 | ATT-0297-1 / RES-0272 | 2,793,472 | 3e-4 | 74.50 | 83.00 | 78.75 |
| RB1-01 / RB1-02 / RB1-07 (r32/r7 reference) | E-0021 | – | – | – | – | 77.30 / 77.10 / 78.95 |

## Pre-registered rules

| Rule | Outcome | Evidence |
| --- | --- | --- |
| 1 implementation | Pass: iv_off = same-allocation and pinned Base; step-0 = Base (all arms); reload identical; params = registration (ATT-0294-1, c9f78b9); W-01/W-02 init identical | EV-0124 (diagnostic) |
| 2 learnability | All >= Base+1: W-01 +6.65 [4.85,8.45], W-02 +6.80 [5.00,8.55], W-03 +8.60 [6.85,10.35]. LoRA learned, so the level is screening | EV-0125 |
| 3 H-0018 | **challenges** (screening, r128 only): W-02 >= Base+1 holds, but d_L = W-02 − W-03 = −1.80 pp (−9/500) [−3.10,−0.50] < −1 | EV-0123 (challenges H-0018) |
| 4 practical advantage | Not recorded (d_L −1.80 < +1) | EV-0126 |
| 5 history | Descriptive only; H-0014 not reopened. d_T = W-02 − W-01 = +0.15 [−1.00,1.35]; iv_swap drop 2.25 [1.15,3.40]; iv_permute drop 32.30 [29.40,35.20] | EV-0127 |
| 6 width (descriptive) | W-02 − RB1-02 −0.15 [−1.15,0.85]; W-01 − RB1-01 −0.50 [−1.50,0.50]; W-03 − RB1-07 −0.20 [−1.20,0.80] | EV-0128 |
| 7 per-task/NLL/curves | See EV-0129 and analysis.md. W-02 dips below Base at 25% (68.25) | EV-0129 |
| 8 | Rule 3 challenges, so no further widening or tap change in the ordinary setting | in EV-0123 |

The W-02 intervention normal condition reproduces the training predictions
(0 mismatches, max score difference 0). Its iv_off condition is
row-identical to Base.

## Cost (measured, MIG-h × 0.25)

| Attempt | Purpose | MIG-h | node·h |
| --- | --- | ---: | ---: |
| ATT-0289-1 | smoke at 44f880c (failed on one test message) | 0.062222 | 0.015556 |
| ATT-0293-1 | recheck tests at e07495e (failed on a test typo) | 0.011389 | 0.002847 |
| ATT-0294-1 | preflight tests/registration/smoke at c9f78b9 | 0.063056 | 0.015764 |
| ATT-0295-1 | W-01 training | 0.939444 | 0.234861 |
| ATT-0296-1 | W-02 training | 1.117500 | 0.279375 |
| ATT-0297-1 | W-03 training | 1.101667 | 0.275417 |
| ATT-0298-1 | RV-0082 tests at 81ad3de | 0.011389 | 0.002847 |
| ATT-0299-1 | W-02 interventions | 0.440278 | 0.110070 |
| ATT-0300-1 | locked analysis | 0.004722 | 0.001181 |
| Total | | 3.751667 | 0.937917 |

This is within the 1.5 node·h budget.

## Records and caveats

- Reviews:
  - RV-0080 then RV-0081 (one round) cover the training source c9f78b9.
  - RV-0082 (new round, analysis fix 81ad3de) passed with 0 findings.
  - Coverage exceptions D-0053 (training) and D-0054 (intervention) apply because only test files were stale. ATT-0300-1 had the same stale test files; it ran at reviewed 81ad3de plus the committed input JSON.
- RES-0273 (ATT-0299-1) is flagged as differing from its RunSpec arm. The manifest says `W-02` and the RunSpec says `W-02-interventions`. The outputs are valid and consumed by the analysis through provenance hashes.
- The failed attempts ATT-0289-1 and ATT-0293-1 are kept as engineering history; their usage is included above.
- Limits:
  - Single seed, and dev also serves as the decision set.
  - 1,000 dev rows per task resolve only differences of about 2 pp or more.
  - r128 reuses the r32 learning-rate lock.
  - T0 is a current-token gate, not history-free attention.
- A2 confirmation and official validation were not opened. No hypothesis status was changed.
