# T-0026 results: E-0023@1 (readiness R, stage A, rule 4)

The full run lies on one path: readiness R held, stage A gave case A3 (stage B not released), and the rule-4 frozen-transfer diagnostic was followed by the D-0035 repair pilot. Evidence is screening only.

Readiness-rule origin, as recorded with every evidence entry: R was fixed after the start-stage dev results were seen, and it was checked once on the previously unused T2 calibration split.

The code is in `research/notes/T-0026-implementation.md`. Reviews: RV-0065→RV-0067 (start, diagnostics, stage A), RV-0070 (rule-4 diagnostic), RV-0071→RV-0072 (repair pilot).

## Stage 0: readiness R (ATT-0227-1 / RES-0206 / EV-0113)
Common start ATT-0196-1 (`adapter.pt` `f70ea08e…`), final checkpoint, T2 calibration 2,048 rows (683/683/682), reviewed E-0016 evaluator, S = 1.

| Criterion | Value | Threshold | Pass |
| --- | --- | --- | --- |
| hop1 EM margin | 181/683 = 26.5 pp | ≥ 10 pp | yes |
| hop1 NLL, normal < min(zero, shuffled) | 3.466 < min(9.541, 7.848) | | yes |
| Answer-position weight on the first fact, hops 1/2/3 | 0.411 / 0.394 / 0.408 | ≥ 1/8 each | yes |
| hop1 most common answer-position slot | 53/683 = 7.8% | ≤ 1/4 | yes |

The all-position target weight was 0.057 (descriptive). R holds, so stage A was released.

## Stage A (ATT-0233/0234/0235-1, RES-0219/0220/0221; rule ATT-0251-1 / RES-0223; EV-0115)
- Setup: 6,000 steps on T2 train hops 1-3 from the common start, the same order for every arm. Evaluation on T2 dev (334/333/333) at S_eval = S_train.
- All three runs were validated against the readiness-checked start and their reloads gave identical records. V3-01's first-edge oracle equals the reviewed `oracle_value` exactly.
- V3-01 ran on a 1g slice (metadata only, D-0018).

| Arm | S | hop1 EM | hop2 EM | hop3 EM | Primary (hop2+hop3)/2 | shuffled hops 2-3 | permuted hops 2-3 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| V3-01 | 1 | 48/334 = 14.37% | 12/333 | 10/333 | 1100/333 = 3.30 pp | 6/666 | 12/666 |
| V3-02 | 2 | 39/334 = 11.68% | 13/333 | 13/333 | 1300/333 = 3.90 pp | 8/666 | 10/666 |
| V3-03 | 4 | 35/334 = 10.48% | 13/333 | 9/333 | 1100/333 = 3.30 pp | 4/666 | 12/666 |

The zero-memory read gives 0 correct on every hop for every arm.

- S_M* = V3-02, since V3-03 is not at least V3-02 + 0.5 pp.
- Gain over V3-01: 200/333 = +0.60 pp. Hop1 difference: −450/167 = −2.69 pp.
- Rule 3 gives **case A3**: descriptive only, stage B not released.
- Paired hop-stratified bootstrap, 95% (descriptive):
  - V3-02 − V3-01: primary [−0.60, +1.80] pp, hop1 [−5.09, −0.60] pp;
  - V3-03 − V3-01: primary [−1.20, +1.20] pp.

### Diagnostics on the stage-A checkpoints (dev, descriptive)
Value oracles (correct out of 334/333/333):

| Arm | First-edge oracle, hops 1/2/3 | End-edge oracle, hops 1/2/3 |
| --- | --- | --- |
| V3-01 | 141 / 4 / 1 | 141 / 149 / 155 |
| V3-02 | 141 / 4 / 0 | 141 / 154 / 152 |
| V3-03 | 141 / 5 / 2 | 141 / 151 / 155 |

All three hop-1 totals are 141. The per-example records differ: V3-02 and V3-03 each share 271 of 334 predictions with V3-01, and 128 and 126 correct rows respectively.

Answer-position weight of step s on path fact j, written step s: (fact 1, fact 2[, fact 3]):

| Arm | hop2 | hop3 |
| --- | --- | --- |
| V3-01 | s1: (0.222, 0.062) | s1: (0.186, 0.072, 0.068) |
| V3-02 | s1: (0.225, 0.061); s2: (0.203, 0.069) | s1: (0.189, 0.071, 0.067); s2: (0.175, 0.076, 0.071) |
| V3-03 | s1 (0.214, 0.063); s2 (0.210, 0.066); s3 (0.191, 0.072); s4 (0.169, 0.078) | s1 (0.184, 0.072, 0.068); s2 (0.181, 0.075, 0.070); s3 (0.167, 0.079, 0.072); s4 (0.151, 0.082, 0.075) |

- Rule 5 (secondary, reported):
  - hop2 step 2 on fact 2: 0.0685 (V3-02) and 0.0660 (V3-03). Both are above 1/16 and below 1/8.
  - V3-03 hop3 step 3 on fact 3: 0.0721.
- Later steps stay on fact 1 rather than moving to the next fact.
- Step-1 weight on the first fact fell from about 0.40 at the start (R) to about 0.19–0.23.
- EM by the answer name's train frequency (median count 280.5; high vs low, out of 500 each):
  - V3-01: 54 vs 16;
  - V3-02: 51 vs 14;
  - V3-03: 45 vs 12.

## Rule 4: frozen-transfer diagnostic (ATT-0253-1 / RES-0225 / EV-0116; D-0034; RV-0070)
Setup: V3-02 frozen, T2 calibration hops 2-3 (683 + 682 = 1,365 rows), answer position, S = 2, chain end as label.

| D-0034 criterion | Raw value | Threshold | Holds |
| --- | --- | --- | --- |
| (a) end-edge oracle EM − max(EM_zero, EM_shuffled) | 581/1365 − max(0/1365, 22/1365) = 860/21 = 40.95 pp | ≥ 10 pp | yes |
| (b) bridge-as-subject, step 1 on path fact 2 | 0.2025 | ≥ 1/8 | yes |
| (b) wrong bridge (same relation) / donor bridge | 0.0724 / 0.0846 | both < 1/8 | yes |
| (c) native step 2 on path fact 2 | 0.0761 | < 1/8 | yes |

The wrong-bridge mean covers 1,365 rows; no row lacked a same-relation off-path fact.

Mapping: (a), (b) and (c) all hold, which selects the successor-slot CE S4 pilot with the permuted-label control.

Descriptive results:
- Native EM is 55/1365.
- Bridge-subject EM is 104/1365 (hop2 81/683, hop3 23/682). Its step-1 argmax falls on fact 2 in 35.4% of rows, against 6.1% native, 8.4% donor and 6.4% wrong bridge.
- The wrong bridge's own off-path fact gets 0.208 at step 1.
- forced_first (z^(1) set to the correct first-edge read, later steps native): EM 22/1365, step 2 on fact 2 0.0991.
  - Its step-1 entries are the native step-1 weights that the forced state replaced, not a forced read (RV-0070 low finding).

In short, the frozen reader addresses the next edge when it is given the bridge entity as the subject, and the end value decodes. Its own step-2 transition, however, does not move to fact 2.

## Rule 4 repair: successor-slot CE S4 pilot (ATT-0256-1 / ATT-0257-1, comparison ATT-0259-1; RES-0238/0239/0240; EV-0117; D-0035)
This is extra supervision, reported separately. It is not unsupervised H-0005 evidence.

Setup:
- Both arms start from ATT-0196-1 (readiness ATT-0227-1) in the V3-03 configuration (S = 4).
- 2,000 steps on the first 16,000 stage-A episodes, with identical init and order. Code at 6cc7124.
- Each arm sees 16,013 successor labels: hop2 5,329 and hop3 10,684.
- Both runs: reload records identical, answer-position replay error at most 3e-6 relative during training, 2g slice.
- `compare` admitted both runs against the trusted reference, which binds the configuration, the 2,000-step prefix, the label counts, the released data and the dev identities.

| D-0035 criterion (T2 dev, S = 4) | Correct labels | Permuted labels | Threshold | Holds |
| --- | --- | --- | --- | --- |
| (i) primary (EM_hop2 + EM_hop3)/2 | (16+14)/666 = 1500/333 = 4.50 pp | (11+8)/666 = 950/333 = 2.85 pp | difference 550/333 = +1.65 pp ≥ +3 pp | no |
| (ii) hop-2 step 2 on path fact 2 | 0.0834 | 0.0624 | correct ≥ 1/8 and permuted < 1/8 | no |

**Correct labels do not beat permuted labels, so the repair route stops** (AD-0003 drop condition).

Descriptive:
- hop1 EM is 11/334 (correct) and 10/334 (permuted), a difference of 50/167 pp. Both are far below V3-01 (48/334) and the start (26.5% hop1 EM on calibration).
- NLL for hops 1/2/3 is 4.35/4.33/4.31 (correct) and 4.41/4.39/4.38 (permuted).
- Controls at S = 4 (hop1/hop2/hop3 correct):
  - correct arm: zero 0/0/0, shuffled 3/3/3, permuted memory 12/17/11;
  - permuted arm: zero 0/0/0, shuffled 4/8/3, permuted memory 8/10/7.
- The answer-position read became nearly flat in both arms. Uniform over 16 slots is 0.0625.
  - Correct arm: every step puts 0.082–0.087 on each path fact, and step 1 on fact 1 is 0.086.
  - Permuted arm: 0.062–0.068, with step 1 on fact 1 at 0.067.
  - The start had about 0.40 on fact 1 at step 1, and the stage-A arms about 0.19–0.23.
- Successor CE over the last 100 steps was 2.51 (correct) and 2.77 (permuted), against ln 16 = 2.77. The first 100 steps were 3.57 and 4.79.
  - The correct labels were learned only slightly beyond uniform.
  - In both arms the read lost its first-edge addressing, rather than gaining a successor transition.
- V3-03 for reference (6,000 steps, no successor CE): primary 1100/333 = 3.30 pp, hop2 step 2 on fact 2 0.066.

## Summary for the conclusion review
- R held, and stage A gave A3: a gain of +200/333 pp, below +3 pp, with hop1 at −450/167 pp. Stage B was not released.
- The frozen diagnostic localizes the stage-A failure to the loop transition, as D-0034 (a), (b) and (c) all hold:
  - the end value decodes;
  - a bridge-entity query addresses the next edge;
  - the native step 2 stays on fact 1.
- The single allowed repair (D-0035, successor-slot CE at coefficient 1, 2,000 steps) did not separate from its permuted control on either criterion. It flattened the read and collapsed hop1 in both arms.
- Under E-0023, the A3 path is now complete: descriptive results only, no further arms.

## Cost (ledger, MIG-h × 0.25 = node·h)
| Item | Attempts | MIG-h | node·h |
| --- | --- | --- | --- |
| Readiness R | ATT-0227-1 | 0.0453 | 0.0113 |
| Regressions and pre-review tests | ATT-0229/0231/0232/0237/0238/0252/0254/0255-1 | 0.2097 | 0.0524 |
| Stage A: V3-01 / V3-02 / V3-03 | ATT-0233/0234/0235-1 | 0.3739 / 0.4308 / 0.5786 | 0.0935 / 0.1077 / 0.1447 |
| Stage-A rule and diagnostics | ATT-0251-1 | 0.0389 | 0.0097 |
| Frozen-transfer diagnostic | ATT-0253-1 | 0.0400 | 0.0100 |
| Repair pilots: correct / permuted | ATT-0256/0257-1 | 0.2164 / 0.2225 | 0.0541 / 0.0556 |
| Pilot comparison | ATT-0259-1 | 0.0031 | 0.0008 |
| **Total** | | **2.159** | **0.540** of the 2.6 cap |

Correction: EV-0117's summary gives the pilot cost as 0.23 MIG-h = 0.058 node·h. The ledger shows 0.442 MIG-h = 0.110 node·h for the two pilots and the comparison.
