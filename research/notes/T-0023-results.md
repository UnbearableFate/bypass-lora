# T-0023 results: E-0022@2 common start stage (V3-S0) fails the hop-1 H3 gate

## Run
- **ATT-0196-1 / RES-0192.** PBS 3768686, commit e53478f (reviewed: RV-0058 → RV-0059 pass).
- **Duration:** 21 min 39 s wall, 0.3608 MIG-h (0.090 node·h).
- **Training:** D5 seed-0 adapter ATT-0159-1 (`adapter.pt` `c4f3f9f8…`, checked before and after loading) continued on T2 train hop-1 questions.
  - 6,000 steps, 4,797,567 tokens, stable step 0.185 s.
  - Order seed `derive_seed(20261029, "concat_loop_v2", "V3", "start", 0)`, 8 epochs of the 6,144 hop-1 rows.
  - Every side parameter received a gradient. 76.6% of steps were clipped.
  - Loss averaged 2.82 over the last 100 steps; all losses finite.
- **Evaluation:** final checkpoint only, with the reviewed E-0016 evaluator (isolated write, six controls) on all T2 dev rows (334/333/333).
- **Reload:** the reloaded checkpoint gives byte-identical prediction files for all six controls and an identical `dev.metrics.json` (verified with `cmp`).

## Gate (E-0010@2 H3, hop 1, S=1): **not passed**
| Criterion | Value | Threshold | Pass |
| --- | --- | --- | --- |
| EM_normal − max(EM_zero, EM_shuffled) | 89/334 − 4/334 = 85/334 = 25.4 pp | ≥ 10 pp | yes |
| NLL_normal < min(NLL_zero, NLL_shuffled) | 3.626 < 8.175 | | yes |
| Target weight, all question positions | 0.0655 | ≥ 0.125 | **no** |
| Target weight, last position (reported) | 0.384 | | |

I recomputed the gate independently from the saved hop-1 records: margin 85/334, NLL 3.626 vs 8.175, token-weighted target weight 0.06554. This matches `metrics.hop1_gate`.

Following E-0022 rule 1, the ten arms were **not** trained. The start stage stops here and goes to consultation.

## Consultation data (all from saved outputs; no new runs)

### Per hop, final checkpoint, S=1, reviewed evaluator
Target weight here is the weight on `query.target_fact_index`, which is the hop-1 fact for every hop.

| Hop | Control | EM | NLL | Target weight (all positions) | Target weight (last) | n_eff (all) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | normal | 89/334 = 26.6% | 3.626 | 0.0655 | 0.384 | 1.83 |
| 1 | zero | 0.0% | 9.582 | 0 | 0 | 0 |
| 1 | shuffled | 1.2% | 8.175 | — | — | 1.78 |
| 1 | permuted | 2.7% | 6.688 | 0.0655 | 0.384 | 1.83 |
| 1 | oracle_key | 12.0% | 3.928 | 0.175 | 0.175 | 8.73 |
| 1 | oracle_value | 66.8% | 1.209 | (normal) | (normal) | |
| 2 | normal | 4.2% | 6.894 | 0.0533 | 0.424 | 1.83 |
| 2 | shuffled / permuted / oracle_key / oracle_value | 0.6 / 0.9 / 3.9 / 0.3% | | | | |
| 3 | normal | 0.9% | 6.486 | 0.0559 | 0.362 | 1.86 |
| 3 | shuffled / permuted / oracle_key / oracle_value | 1.8 / 3.3 / 2.7 / 0.0% | | | | |

On hops 2 and 3 the oracle_value control gives the hop-1 tail, which is not the answer, so it is near 0 by construction.

### Target weight by question position (hop-1 normal records)
- The reviewed evaluator saves two per-example quantities: the token-weighted mean over all valid question positions, and the last position. The mean over non-last positions is derived from them as (all·T − last)/(T − 1).
- Hop 1 (T = 23–25 question tokens, mean 23.5):
  - all positions 0.066 (per-example mean);
  - **last position 0.384**;
  - non-last positions 0.052.
- Last-position argmax equals the target fact for 50.6% of hop-1 rows.
- 33.8% of rows put weight ≥ 0.5 on the target at the last position; 68 of those 113 are correct.
- Address sharpness is high (n_eff 1.83), but at most positions the weight sits on non-target slots.

### T1 vs T2 question length (same protocol, D5 seed 0, ATT-0159-1 on T1 E=16 dev)
| | Question tokens | Target weight (all) | Target weight (last) | Non-last mean | EM |
| --- | --- | --- | --- | --- | --- |
| T1 (ATT-0159-1, passed H3) | 7 (all rows) | 0.162 | 0.571 | 0.094 | 42.8% |
| T2 hop 1, start stage (ATT-0196-1) | 23–25 | 0.066 | 0.384 | 0.052 | 26.6% |
| T2 hop 1, D5 seed 0 before the start stage (E-0018, ATT-0182-1) | 23–25 | 0.036 | 0.054 | | 2.7% |

The all-position criterion averages over every question token. T2's question template has more than three times as many tokens as T1's, and the target read is concentrated at the end. On T1 the last position is 0.571 and the all-position mean 0.162; on T2 they are 0.384 and 0.066. This is an observation about the metric's scope, not a recommendation: the gate's scope is part of the contract, and the coordinator owns the decision.

### Contrast with the D5 seed-0 T2 baseline (E-0018, ATT-0182-1, hop 1)
| Measure | E-0018 baseline | Start stage | Change |
| --- | --- | --- | --- |
| normal EM | 9/334 = 2.7% | 26.6% | |
| EM margin | 3/334 | 85/334 | |
| normal NLL | 7.745 | 3.626 | |
| target weight, all positions | 0.036 | 0.066 | |
| target weight, last position | 0.054 | 0.384 | |
| oracle_key EM | 7.2% | 12.0% | |
| oracle_key target weight | 0.135 | 0.175 | |
| oracle_value EM | 75.8% | 66.8% | down |

Training changed the keys and the question-side addressing at the answer position. The decoder's use of the correct value dropped somewhat (oracle_value 75.8% → 66.8%).

### Curves (dev hop 1, descriptive, not used for selection)
| Point | normal EM | normal NLL | Target weight (all) | shuffled EM |
| --- | --- | --- | --- | --- |
| 25% | 3.0% | 4.453 | 0.050 | 1.5% |
| 50% | 9.6% | 4.166 | 0.058 | 1.2% |
| 75% | 22.5% | 3.989 | 0.064 | 1.2% |
| 100% | 26.6% | 3.626 | 0.066 | 1.2% |

EM and the all-position target weight were still rising at the end. The all-position weight moved slowly, from 0.050 to 0.066.

## Cost
| Item | node·h |
| --- | --- |
| Tests and smokes (ATT-0190-1, ATT-0192-1, ATT-0195-1) | 0.050 |
| Start stage (ATT-0196-1) | 0.090 |
| **T-0023 total** | **0.140** (budget 3) |

## Status
Stopped at the start stage per E-0022 rule 1; WAIT for the coordinator's consultation. The ten-arm code (reviewed) is ready, and the arm job template takes the start run and its pinned digest.
