# T-0015 results: E-0014 memory-route repair screening (seed 0)

Code: c6037d3 (review round RV-0040 → RV-0041 pass). Arms: ATT-0122-1 … ATT-0127-1 (RES-0123 … RES-0128).
Analysis: ATT-0138-1 (RES-0129), `runs/ATT-0138-1/report.{json,md}`. Evidence: EV-0080 … EV-0085 (screening). None supports or challenges H-0009 (no H3 gate passed). Confirmation was never read.

## E=16 dev, H3 gate (E-0010@2): EM margin ≥ 10 pp, NLL_normal < min(zero, shuffled), target weight ≥ 0.125

| arm | change | correct normal / zero / shuffled | EM margin | NLL normal / zero / shuffled | target weight | n_eff | H3 |
|---|---|---|---:|---|---:|---:|---|
| R1 | E=1 train | 4 / 11 / 6 | −0.7 pp | 18.737 / 9.248 / 18.904 | 0.061 | 13.03 | fail |
| R2 | E=4 train | 23 / 11 / 8 | +1.2 pp | 6.222 / 9.248 / 6.724 | 0.091 | 9.33 | fail |
| R3 | curriculum 1→4→8→16 | 10 / 11 / 12 | −0.2 pp | 4.4315 / 9.248 / 4.4296 | 0.052 | 1.00 | fail |
| R4 | prior path, frozen | 6 / 11 / 12 | −0.6 pp | 4.4303 / 4.7403 / 4.4303 | 0.061 | 1.80 | fail |
| R5 | address CE λ=1 (**extra supervision**) | 62 / 11 / 16 | +4.6 pp | 4.102 / 9.248 / 4.970 | 0.143 | 2.09 | fail (EM only) |
| R6 | centered cosine, learnable τ | 15 / 11 / 13 | +0.2 pp | 4.4205 / 9.248 / 4.4244 | 0.060 | 5.73 | fail |
| V2-03 | reference (ATT-0082-1) | 16 / 11 / 12 | +0.4 pp | 4.4221 / 9.248 / 4.4230 | 0.054 | 1.11 | fail |

Descriptive 95% CIs (paired bootstrap, 10,000 draws), normal − shuffled EM:
- R5: 3.1 to 6.2 pp
- R2: 0.5 to 2.6 pp
- all other arms include 0.

## Variant gates

| arm | split | correct normal / zero / shuffled | EM margin | NLL normal / zero / shuffled | target weight | gate |
|---|---|---|---:|---|---:|---|
| R1 | E=1 dev | 836 / 9 / 10 | +82.6 pp (CI 80.2–84.9) | 0.538 / 9.265 / 18.229 | 1.000 | R1 (≥ 50 pp): **pass** |
| R2 | E=4 dev | 82 / 0 / 16 | +6.6 pp | 3.687 / 9.282 / 5.955 | 0.371 | R2 (≥ 10 pp, NLL, weight ≥ 0.5): fail |
| R3 | E=1 dev | 64 / 9 / 20 | +4.4 pp | 4.317 / 9.265 / 4.438 | 1.000 | descriptive |
| R3 | E=4 dev | 30 / 0 / 10 | +2.0 pp | 4.395 / 9.282 / 4.417 | 0.251 | descriptive |

R3 calibration monitors at each segment end (normal EM):
- E=1: 15.3% at step 1000
- E=4: 1.6% at step 2000
- E=8: 1.6% at step 3000
- E=16: 2.1% at step 6000

## Last question position, E=16 dev, normal memory

| arm | argmax = target | modal slot (rate) | same relation | target weight | fusion / residual, mean | prior / residual |
|---|---:|---|---:|---:|---:|---:|
| R1 | 4.9% | 2 (15.3%) | 22.6% | 0.061 | 18.5 | |
| R2 | 24.2% | 1 (17.0%) | 99.8% | 0.231 | 20.2 | |
| R3 | 5.2% | 0 (100%) | 24.9% | 0.052 | 50.7 | |
| R4 | 5.2% | 0 (100%) | 24.9% | 0.055 | 21.2 | 1.10 |
| R5 | 60.2% | 6 (7.9%) | 100% | 0.518 | 24.4 | |
| R6 | 5.2% | 0 (86.3%) | 25.2% | 0.059 | 41.7 | |

Oracle controls on E=16 dev (EM, NLL):

| arm | oracle_value | oracle_key |
|---|---|---|
| R5 | 0.107, 3.64 | 0.067, 4.04 |
| R1 | 0.058, 15.97 | |
| R2 | 0.037, 5.73 | |
| all others | ≤ 0.021 | |

## Observations (measurement, not route decisions)

1. **The value path works when addressing is trivial.** R1 at E=1 answers 83.6% from memory, against about 1% under the zero and shuffled controls. Writing, reading and decoding are usable in isolation, so E-0014's "R1 未达标" branch does not apply.
2. **No unsupervised E=16 arm reaches H3**, and none approaches it:
   - R3 (curriculum), R4 (prior path) and R6 (centered cosine) reproduce the V2-03 collapse: the last-position argmax is slot 0 in 86–100% of cases, and normal NLL equals shuffled.
   - Removing the answer prior (R4: the prior path alone reaches NLL 4.74 at zero memory) does not stop the collapse.
   - The R3 curriculum reached only 15% E=1 calibration EM after 1000 steps. After the E=16 phase it is back at the V2 state, with E=1 dev at 6.4%.
3. **R2 (E=4) learns relation addressing, not head addressing.** At E=16, 99.8% of the last-position argmaxes pick a fact with the queried relation, but only 24% pick the target. On E=4 (one fact per relation) that would make addressing trivial; R2's E=4 target weight is 0.37 and its EM 8.2%.
4. **R5 (with address supervision) partly learns addressing** (last-position target weight 0.52, argmax = target 60%), yet E=16 EM is only 6.2%. Even oracle_value reaches only 10.7%. At E=16 the value/decoder side is also weak: it was trained mostly under diffuse reads, unlike R1's clean single-slot reads.
   - R5's answer CE was still falling at the end: the 75% learning-curve EM was 4.3%, and the final EM is 6.2%.
   - In E-0014's own outcome terms, R5 is not a pass, so the "只有 R5 达标" branch is not triggered either.
5. **The fusion increment remains 18–51× the layer-14 residual** at the last question position in every arm, the same over-large exit V2 diagnosed (EV-0072).
6. **R1 on E=16 dev is overconfident and wrong** (NLL 18.7). Its address parameters received no gradient (one slot at E=1), so the E=16 read is a near-uniform mixture of 16 values.

## Cost (miyabi ledger, MIG at ¼ node)

| item | MIG-h |
|---|---:|
| R1 | 0.281 |
| R2 | 0.306 |
| R3 | 0.361 |
| R4 | 0.414 |
| R5 | 0.392 |
| R6 | 0.411 |
| data (ATT-0112-1) | 0.015 |
| checks (ATT-0115-1) | 0.061 |
| RV-0040 regression (ATT-0121-1) | 0.009 |
| analysis (ATT-0138-1) | 0.001 |
| **total** | **2.251 MIG-h = 0.563 node·h** (budget 1.5) |
