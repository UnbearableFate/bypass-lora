# T-0021 / E-0018@1 results: D5 read transfer to T2, the no-training S_eval scan, the loop implementation check and timing

**Status.** Every job requested by the contract is complete. Evidence level: diagnostic.

**Data.**
- Splits: T2 dev only (1000 episodes; 334/333/333 at hops 1/2/3), plus T2 train for the timing smoke and T1 E=16 dev for the S=1 check.
- Confirmation and ood_h4/6/8/unseen were never opened.
- Code: reviewed commit 7f3b5a0 (RV-0057, pass, 0 findings).
- Implementation: `research/notes/T-0021-implementation.md`.

## Outcome under the E-0018 rules
- **Implementation check: pass.**
  - At S=1, all three adapters' loop records equal the reviewed E-0016 evaluator's records example by example for normal, zero, shuffled and permuted on T2 dev. Compared fields: prediction, correct, NLL, all/last target weight, all/last n_eff, last argmax.
  - On T1 E=16 dev, the P1 adapter (ATT-0172-1) matches in two ways:
    - against the reviewed evaluator in the same job: exact for all four controls;
    - against P1's recorded isolated predictions: 0 prediction mismatches and identical NLL for all four controls (1000 episodes).
  - Memory stays fixed: the state is unchanged and accounting shows S reads per question token and 0 writes, asserted in every batch.
  - The replay of the native S-step increment is within 1.7e-4 of ‖Z‖ (S=8), against a bound of 1e-3.
  - No non-finite values appeared.
- **Hop 1 at S_eval=1 fails H3 for all three D5 adapters.**
  - Normal EM on hop 1 is 2.7 / 3.0 / 4.5 % (seeds 0/1/2).
  - The EM margin over max(zero, shuffled) is 0.9 / 1.8 / 3.0 pp, against ≥ 10 required.
  - The all-position target weight is 0.036 / 0.037 / 0.038, against ≥ 0.125 required. That is below the uniform 1/16.
  - The NLL condition alone passes (normal 7.0–7.7 < shuffled 8.6–8.9 < zero 9.58).
  - **Contract action: V3 pre-registers a common start stage**, the same for every group (e.g. continued training on T2 hop 1 first). V3 groups do not start directly from the D5 adapter.
- **S_eval > 1: no descriptive signal.**
  - hops 2–3 EM does not rise above S_eval=1 (all stay at 2–5 %, within ±1–2 pp of S=1).
  - NLL rises with S in every hop.
  - The step-2 weight does not move toward the hop-2 target.
  - Any loop benefit can only be judged by V3's trained controls.
- **Timing (smoke, not evidence)** gives the V3 cost estimates below.

## Why hop 1 fails on T2: addressing, not decoding
- **oracle_value on hop 1** (the target fact's stored value replaces the read at every question position) gives EM 75.7 / 80.2 / 76.3 % with NLL 0.73–0.88. The value → answer path that D5 learned on T1 carries over to the new question template.
- **The read does not find the hop-1 fact.**
  - Target weight is about 0.036 at all positions and 0.054–0.069 at the last position.
  - The last-position argmax rate on the hop-1 fact is about 0.06 (≈ 1/16).
  - The S=1 last-position argmax spreads evenly over the 16 slots (seed 0 counts 54–77 per slot).
  - n_eff at the last position is about 2.1: each query's weight sits on a couple of slots, but not on the target.
  - On T1, the same adapters reach about 0.16 all-position / 0.6 last-position target weight (E-0017 RES-0168..0171).
  - So the T2 question template ("Starting from X, keep following the r facts …") does not produce a key-matching query from the head entity the way the T1 template does.
- **oracle_key on hop 1** (the target's stored key as query) raises the target weight only to 0.14–0.15 (n_eff ≈ 9–10). EM is 6.9–7.2 %, NLL about 4.2. The stored keys do not separate the facts sharply under this read either; that is descriptive, and E-0018 has no gate on it.
- **permuted** hop-1 EM (5.1–6.6 %) exceeds normal (2.7–4.5 %). With the address near uniform, the target slot is not the one being read, so rolling the values does not remove a working association.
- **Comparison with E-0011 baselines on dev hop 1** (structural shortcuts, model-free):

  | Baseline | Hop-1 EM |
  |---|---|
  | prior | 1.5 % |
  | random r endpoint | 16.7 % |
  | untyped traversal | 50.0 % |
  | first bridge | 100 % |

  M-line Base (ATT-0090-1) has EM 0 and NLL 9.58 / 9.61 / 9.55 per hop; the D5 zero control reproduces about 9.58 / 9.62 / 9.56.

## Cross-seed summary (normal; EM % / NLL; S=1 shuffled and permuted EM)

| seed | hop | S=1 | S=2 | S=4 | S=8 | shuffled S=1 | permuted S=1 |
|---|---|---|---|---|---|---:|---:|
| 0 | 1 | 2.7 / 7.74 | 3.6 / 9.23 | 3.3 / 9.69 | 3.6 / 9.58 | 1.8 | 6.3 |
| 0 | 2 | 3.9 / 8.23 | 3.9 / 9.79 | 4.2 / 10.14 | 4.5 / 9.89 | 1.2 | 3.3 |
| 0 | 3 | 3.6 / 7.97 | 3.9 / 9.54 | 3.0 / 10.10 | 2.7 / 10.08 | 0.9 | 2.1 |
| 1 | 1 | 3.0 / 7.37 | 3.9 / 8.75 | 3.9 / 9.26 | 4.2 / 9.25 | 1.2 | 6.6 |
| 1 | 2 | 4.8 / 7.75 | 4.2 / 9.33 | 3.6 / 9.92 | 3.0 / 9.92 | 1.8 | 2.4 |
| 1 | 3 | 2.4 / 7.77 | 2.4 / 9.29 | 2.4 / 9.84 | 2.4 / 9.86 | 1.8 | 2.1 |
| 2 | 1 | 4.5 / 7.01 | 4.2 / 8.35 | 3.9 / 8.87 | 3.0 / 8.81 | 1.5 | 5.1 |
| 2 | 2 | 3.0 / 7.22 | 3.3 / 8.73 | 2.7 / 9.32 | 3.3 / 9.25 | 1.2 | 2.4 |
| 2 | 3 | 3.3 / 7.14 | 3.6 / 8.61 | 3.9 / 9.19 | 3.9 / 9.15 | 1.2 | 2.1 |

- The zero control reads 0 at every step: EM 0 and NLL about 9.58 for all S. That holds by construction and is checked.
- **Depth trajectory** (seed 0, normal, last position):
  - The query cosine to step 1 falls slowly: 0.998 at step 2, 0.987 at step 4, 0.949 at step 8.
  - The read norm stays about 49–54.
  - The increment ‖Z‖ grows almost linearly in S: 52 / 103 / 201 / 384 at S = 1/2/4/8.
  - The state norm goes from 293 to 449 at S=8.
  - The weight on path facts 1–3 rises only from about 0.036 to 0.045 by step 8; it does not shift toward fact 2 or fact 3.
  - Seeds 1 and 2 are the same (full tables below).
- **Support order** (hop × natural/shuffled): normal EM stays at 1–5 % in every cell at S=1 and S=4. Order shows no structure while addressing is near chance.

## Timing (ATT-0185-1, smoke, MIG 2g.24gb confirmed by the mandatory guard)
Setup:
- Start: the seed-0 D5 adapter (ATT-0159-1).
- Data: T2 train, V3 train order (`make_step_schedule(phase='V3')`, order sha 2ea9d474…).
- 100 steps × batch 8 at LR 0.003 (5-step warmup, cosine), answer CE with the isolated write.
- 79,639 input tokens per 100 steps, i.e. 796.4 support + question tokens per step.

| S | stable step (s) | median / p90 (s) | tokens/s | peak allocated | grad norm mean / max | clip fraction | train CE first → last 20-step mean |
|---|---:|---|---:|---:|---|---:|---|
| 1 | 0.2054 | 0.2045 / 0.2067 | 3765 | 1.564 GB | 1.87 / 14.3 | 0.86 | 5.81 → 4.49 |
| 2 | 0.2440 | 0.2448 / 0.2469 | 3263 | 1.564 GB | 2.03 / 17.4 | 0.87 | 6.14 → 4.45 |
| 4 | 0.3275 | 0.3287 / 0.3330 | 2433 | 1.564 GB | 2.14 / 18.7 | 0.90 | 6.19 → 4.50 |

- All losses and gradients are finite, and all six parameter groups receive nonzero gradients.
- The step-time increment is about 0.04 s per extra read step.
- The peak memory is the same at all three S. The frozen backbone and isolated support pass set it, not the r=128 loop.
- The high clip fraction comes from continuing from the D5 adapter on a new template. It is not a V3 recommendation.

### V3 node·h estimate per group
Training only: steps = ceil(budget tokens / 796.39), times the stable step time, at 2g MIG, with node·h = MIG-h / 4.

| group cost class | 1M tokens (1256 steps) | 2M (2512) | 5M (6279) |
|---|---:|---:|---:|
| S1 (V3-01) | 0.018 | 0.036 | 0.090 |
| S2 (V3-02) | 0.021 | 0.043 | 0.106 |
| S4 (V3-03/04/05/09/10; V3-08 untied ≈ same compute) | 0.029 | 0.057 | 0.143 |

- **Evaluation overhead per trained group:**
  - one S_eval 1/2/4/8 scan of six controls costs about 275 s ≈ 0.019 node·h (ATT-0182-1/0184-1);
  - a reload re-evaluation would cost about the same;
  - so add about 0.04 node·h per group.
  - Learning curves on dev, if registered, would add on top.
- **Not measured here:** V3-06 (tokenwise kernel) and V3-07 (wider S1).
- **A common start stage** (the outcome above) adds the same cost once, or per seed. For example, 1M tokens of T2 hop-1 continuation at S=1 costs about 0.018 node·h plus evaluation.

## Records and cost

| Attempt | Purpose | PBS | Slice (metadata only, D-0018) | MIG-h (ledger) | Result | Evidence |
|---|---|---|---|---:|---|---|
| ATT-0177-1 | pre-review tests and smokes | 3767807 | 2g | 0.022778 | failed (replay bound; recorded, not imported) | — |
| ATT-0178-1 | pre-review rerun | 3767825 | 2g | 0.017222 | failed (replay bound; recorded, not imported) | — |
| ATT-0181-1 | pre-review tests and smokes, 38 passed | 3767840 | 2g | 0.0225 | RES-0175 | — |
| ATT-0182-1 | E1 seed 0 (ATT-0159-1) | 3767895 | 2g | 0.08 | RES-0178 | EV-0107 |
| ATT-0183-1 | E1 seed 1 / P1 (ATT-0172-1) + T1 check | 3767907 | 1g (ledger 2g) | 0.110556 | RES-0179 | EV-0108 |
| ATT-0184-1 | E1 seed 2 (ATT-0173-1) | 3767908 | 2g | 0.082222 | RES-0180 | EV-0109 |
| ATT-0185-1 | timing smoke S=1/2/4 | 3767909 | 2g (required) | 0.028611 | RES-0181 (smoke) | — |

- **Total:** 0.363889 MIG-h = **0.0910 node·h**, against an estimate of 0.2 and a cap of 0.4.
- **Pins:** each adapter was checked before and after loading against its producer manifest digest:
  - ATT-0159-1: c4f3f9f8…
  - ATT-0172-1: a223b7c3…
  - ATT-0173-1: c064416a…
- **Slice note:** ATT-0183-1 ran on a 1g slice (mg1110) while the ledger records 2g. Per D-0018 this is metadata only. Its T1 replay against P1's records was still exact; P1-B itself ran on 1g.

## Open points for the coordinator
- **V3 common start stage.** Its form and budget (data, hop mix, steps, whether it is per seed) are for the V3 contract.
  - The diagnosis points at the query side under the T2 template: oracle_value works, while the address is at or below uniform.
  - Training on T2 hop 1 updates A_q/Rq. Since the decoder already transfers, keeping the value path as is could be a cheaper variant. Both are options, not results.
- **S_eval > 1 diagnostics** are not interpretable while the hop-1 address is near chance. The trajectory tables show only that the loop adds increment norm without re-addressing.
- **oracle_key** reaches only about 0.14 target weight on T2. Whether stored keys of T2 facts are separable enough (16 facts sharing one relation and chained entities) is a question V3 may want to check after the start stage.

## Full per-adapter tables

### seed 0 (ATT-0182-1; adapter ATT-0159-1)

**S_eval=1, reviewed evaluator, all six controls (EM % / NLL)**

| hop | prior EM | normal | zero | shuffled | permuted | oracle_key | oracle_value | target weight all / last (normal) |
|---|---:|---|---|---|---|---|---|---|
| 1 | 1.5 | 2.7 / 7.74 | 0.0 / 9.58 | 1.8 / 8.81 | 6.3 / 7.49 | 7.2 / 4.15 | 75.7 / 0.88 | 0.036 / 0.054 |
| 2 | 0.3 | 3.9 / 8.23 | 0.0 / 9.62 | 1.2 / 8.39 | 3.3 / 7.95 | 3.0 / 4.74 | 0.0 / 11.52 | 0.033 / 0.067 |
| 3 | 1.5 | 3.6 / 7.97 | 0.0 / 9.56 | 0.9 / 8.33 | 2.1 / 7.90 | 2.4 / 4.69 | 0.0 / 11.65 | 0.040 / 0.071 |

Hop-1 H3 (E-0010@2 thresholds, isolated write): passed=False; EM margin 3/334 = 0.9 pp (≥10); NLL normal 7.745 vs zero 9.582 / shuffled 8.805 (nll_pass=True); target weight all 0.0363 (≥0.125), last 0.0539.

**Loop evaluator: EM % / NLL by hop and S_eval**

| control | hop | S=1 | S=2 | S=4 | S=8 |
|---|---|---|---|---|---|
| normal | hop1 | 2.7 / 7.74 | 3.6 / 9.23 | 3.3 / 9.69 | 3.6 / 9.58 |
| normal | hop2 | 3.9 / 8.23 | 3.9 / 9.79 | 4.2 / 10.14 | 4.5 / 9.89 |
| normal | hop3 | 3.6 / 7.97 | 3.9 / 9.54 | 3.0 / 10.10 | 2.7 / 10.08 |
| normal | all | 3.4 / 7.98 | 3.8 / 9.52 | 3.5 / 9.98 | 3.6 / 9.85 |
| zero | hop1 | 0.0 / 9.58 | 0.0 / 9.58 | 0.0 / 9.58 | 0.0 / 9.58 |
| zero | hop2 | 0.0 / 9.62 | 0.0 / 9.62 | 0.0 / 9.62 | 0.0 / 9.62 |
| zero | hop3 | 0.0 / 9.56 | 0.0 / 9.56 | 0.0 / 9.56 | 0.0 / 9.56 |
| zero | all | 0.0 / 9.58 | 0.0 / 9.58 | 0.0 / 9.58 | 0.0 / 9.58 |
| shuffled | hop1 | 1.8 / 8.81 | 0.9 / 10.65 | 0.6 / 11.29 | 0.6 / 11.35 |
| shuffled | hop2 | 1.2 / 8.39 | 0.9 / 10.05 | 0.9 / 10.57 | 0.6 / 10.49 |
| shuffled | hop3 | 0.9 / 8.33 | 1.2 / 9.97 | 1.5 / 10.50 | 1.5 / 10.49 |
| shuffled | all | 1.3 / 8.51 | 1.0 / 10.23 | 1.0 / 10.79 | 0.9 / 10.78 |
| permuted | hop1 | 6.3 / 7.49 | 6.6 / 8.96 | 5.7 / 9.47 | 4.8 / 9.47 |
| permuted | hop2 | 3.3 / 7.95 | 2.7 / 9.53 | 3.6 / 10.05 | 4.2 / 9.98 |
| permuted | hop3 | 2.1 / 7.90 | 2.4 / 9.55 | 2.1 / 10.19 | 1.8 / 10.25 |
| permuted | all | 3.9 / 7.78 | 3.9 / 9.35 | 3.8 / 9.90 | 3.6 / 9.90 |

**Per-step address (normal control): weight on path fact j, all positions / last position; last-position argmax = path fact j rate; n_eff last; query cosine to step 1 (last); state / read norm (last)**

| S_eval | hop | step | w(fact1) | w(fact2) | w(fact3) | argmax=fact1/2/3 | n_eff last | qcos last | |z| last | |a| last |
|---|---|---|---|---|---|---|---:|---:|---:|---:|
| 2 | 1 | 1 | 0.036 / 0.054 | — | — | 0.057 | 2.21 | 1.000 | 293.00 | 52.94 |
| 2 | 1 | 2 | 0.037 / 0.054 | — | — | 0.060 | 2.24 | 0.998 | 298.21 | 52.66 |
| 2 | 2 | 1 | 0.033 / 0.067 | 0.033 / 0.055 | — | 0.063/0.057 | 2.04 | 1.000 | 296.66 | 54.33 |
| 2 | 2 | 2 | 0.034 / 0.067 | 0.035 / 0.056 | — | 0.066/0.057 | 2.10 | 0.998 | 302.14 | 53.71 |
| 2 | 3 | 1 | 0.040 / 0.071 | 0.035 / 0.037 | 0.041 / 0.050 | 0.060/0.033/0.054 | 2.07 | 1.000 | 299.62 | 53.79 |
| 2 | 3 | 2 | 0.040 / 0.071 | 0.035 / 0.037 | 0.042 / 0.050 | 0.063/0.033/0.054 | 2.10 | 0.998 | 304.73 | 53.55 |
| 8 | 1 | 1 | 0.036 / 0.054 | — | — | 0.057 | 2.21 | 1.000 | 293.00 | 52.94 |
| 8 | 1 | 2 | 0.037 / 0.054 | — | — | 0.060 | 2.24 | 0.998 | 298.21 | 52.66 |
| 8 | 1 | 4 | 0.041 / 0.055 | — | — | 0.066 | 2.47 | 0.987 | 332.55 | 51.38 |
| 8 | 1 | 8 | 0.045 / 0.057 | — | — | 0.063 | 3.11 | 0.949 | 448.51 | 49.01 |
| 8 | 2 | 1 | 0.033 / 0.067 | 0.033 / 0.055 | — | 0.063/0.057 | 2.04 | 1.000 | 296.66 | 54.33 |
| 8 | 2 | 2 | 0.034 / 0.067 | 0.035 / 0.056 | — | 0.066/0.057 | 2.10 | 0.998 | 302.14 | 53.71 |
| 8 | 2 | 4 | 0.036 / 0.068 | 0.037 / 0.059 | — | 0.063/0.063 | 2.38 | 0.986 | 338.17 | 52.02 |
| 8 | 2 | 8 | 0.039 / 0.069 | 0.040 / 0.062 | — | 0.063/0.054 | 3.03 | 0.950 | 458.32 | 49.63 |
| 8 | 3 | 1 | 0.040 / 0.071 | 0.035 / 0.037 | 0.041 / 0.050 | 0.060/0.033/0.054 | 2.07 | 1.000 | 299.62 | 53.79 |
| 8 | 3 | 2 | 0.040 / 0.071 | 0.035 / 0.037 | 0.042 / 0.050 | 0.063/0.033/0.054 | 2.10 | 0.998 | 304.73 | 53.55 |
| 8 | 3 | 4 | 0.041 / 0.070 | 0.037 / 0.039 | 0.044 / 0.051 | 0.066/0.039/0.054 | 2.26 | 0.986 | 340.20 | 52.55 |
| 8 | 3 | 8 | 0.044 / 0.068 | 0.041 / 0.043 | 0.046 / 0.055 | 0.066/0.036/0.051 | 2.93 | 0.949 | 462.05 | 49.83 |

Increment norm |Z| / final state |z_S| at the last position (normal, all hops): S=1: 51.61 / 301.69; S=2: 102.52 / 315.93; S=4: 200.96 / 362.78; S=8: 383.78 / 490.33.

S=1 normal last-position argmax slot counts (slots 0–15): [65, 64, 56, 54, 69, 59, 61, 64, 54, 68, 77, 55, 60, 66, 67, 61].

Replay max ||ΔZ||/||Z||: S=1: 5.6e-05, S=2: 4.5e-05, S=4: 1.0e-04, S=8: 1.7e-04; wall 274 s (reviewed S=1 60 s; loop S=1 42 s, S=2 45 s, S=4 52 s, S=8 65 s).

### seed 1 (P1) (ATT-0183-1; adapter ATT-0172-1)

**S_eval=1, reviewed evaluator, all six controls (EM % / NLL)**

| hop | prior EM | normal | zero | shuffled | permuted | oracle_key | oracle_value | target weight all / last (normal) |
|---|---:|---|---|---|---|---|---|---|
| 1 | 1.5 | 3.0 / 7.37 | 0.0 / 9.58 | 1.2 / 8.89 | 6.6 / 7.33 | 6.9 / 4.25 | 80.2 / 0.73 | 0.037 / 0.056 |
| 2 | 0.3 | 4.8 / 7.75 | 0.0 / 9.62 | 1.8 / 8.21 | 2.4 / 7.57 | 4.2 / 4.81 | 0.3 / 12.21 | 0.034 / 0.072 |
| 3 | 1.5 | 2.4 / 7.77 | 0.0 / 9.56 | 1.8 / 8.49 | 2.1 / 7.67 | 1.5 / 4.81 | 0.0 / 12.25 | 0.042 / 0.070 |

Hop-1 H3 (E-0010@2 thresholds, isolated write): passed=False; EM margin 3/167 = 1.8 pp (≥10); NLL normal 7.368 vs zero 9.582 / shuffled 8.890 (nll_pass=True); target weight all 0.0368 (≥0.125), last 0.0555.

**Loop evaluator: EM % / NLL by hop and S_eval**

| control | hop | S=1 | S=2 | S=4 | S=8 |
|---|---|---|---|---|---|
| normal | hop1 | 3.0 / 7.37 | 3.9 / 8.75 | 3.9 / 9.26 | 4.2 / 9.25 |
| normal | hop2 | 4.8 / 7.75 | 4.2 / 9.33 | 3.6 / 9.92 | 3.0 / 9.92 |
| normal | hop3 | 2.4 / 7.77 | 2.4 / 9.29 | 2.4 / 9.84 | 2.4 / 9.86 |
| normal | all | 3.4 / 7.63 | 3.5 / 9.12 | 3.3 / 9.67 | 3.2 / 9.68 |
| zero | hop1 | 0.0 / 9.58 | 0.0 / 9.58 | 0.0 / 9.58 | 0.0 / 9.58 |
| zero | hop2 | 0.0 / 9.62 | 0.0 / 9.62 | 0.0 / 9.62 | 0.0 / 9.62 |
| zero | hop3 | 0.0 / 9.56 | 0.0 / 9.56 | 0.0 / 9.56 | 0.0 / 9.56 |
| zero | all | 0.0 / 9.58 | 0.0 / 9.58 | 0.0 / 9.58 | 0.0 / 9.58 |
| shuffled | hop1 | 1.2 / 8.89 | 1.2 / 10.73 | 1.2 / 11.32 | 0.9 / 11.37 |
| shuffled | hop2 | 1.8 / 8.21 | 1.8 / 9.83 | 1.5 / 10.44 | 1.5 / 10.55 |
| shuffled | hop3 | 1.8 / 8.49 | 1.5 / 10.24 | 1.2 / 10.88 | 1.5 / 11.01 |
| shuffled | all | 1.6 / 8.53 | 1.5 / 10.27 | 1.3 / 10.88 | 1.3 / 10.98 |
| permuted | hop1 | 6.6 / 7.33 | 6.0 / 8.75 | 6.3 / 9.28 | 6.0 / 9.25 |
| permuted | hop2 | 2.4 / 7.57 | 2.7 / 9.11 | 2.7 / 9.62 | 3.6 / 9.50 |
| permuted | hop3 | 2.1 / 7.67 | 2.7 / 9.21 | 2.4 / 9.72 | 2.4 / 9.66 |
| permuted | all | 3.7 / 7.53 | 3.8 / 9.02 | 3.8 / 9.54 | 4.0 / 9.47 |

**Per-step address (normal control): weight on path fact j, all positions / last position; last-position argmax = path fact j rate; n_eff last; query cosine to step 1 (last); state / read norm (last)**

| S_eval | hop | step | w(fact1) | w(fact2) | w(fact3) | argmax=fact1/2/3 | n_eff last | qcos last | |z| last | |a| last |
|---|---|---|---|---|---|---|---:|---:|---:|---:|
| 2 | 1 | 1 | 0.037 / 0.056 | — | — | 0.045 | 3.04 | 1.000 | 280.62 | 50.67 |
| 2 | 1 | 2 | 0.037 / 0.056 | — | — | 0.048 | 3.07 | 0.999 | 282.87 | 50.60 |
| 2 | 2 | 1 | 0.034 / 0.072 | 0.033 / 0.045 | — | 0.078/0.042 | 2.92 | 1.000 | 284.23 | 51.54 |
| 2 | 2 | 2 | 0.034 / 0.073 | 0.034 / 0.045 | — | 0.078/0.042 | 2.94 | 0.999 | 286.36 | 51.45 |
| 2 | 3 | 1 | 0.042 / 0.070 | 0.034 / 0.039 | 0.034 / 0.046 | 0.069/0.033/0.045 | 2.89 | 1.000 | 284.87 | 52.20 |
| 2 | 3 | 2 | 0.042 / 0.070 | 0.034 / 0.039 | 0.035 / 0.046 | 0.066/0.036/0.051 | 2.92 | 0.999 | 287.15 | 52.11 |
| 8 | 1 | 1 | 0.037 / 0.056 | — | — | 0.045 | 3.04 | 1.000 | 280.62 | 50.67 |
| 8 | 1 | 2 | 0.037 / 0.056 | — | — | 0.048 | 3.07 | 0.999 | 282.87 | 50.60 |
| 8 | 1 | 4 | 0.038 / 0.057 | — | — | 0.048 | 3.27 | 0.990 | 307.72 | 50.03 |
| 8 | 1 | 8 | 0.042 / 0.059 | — | — | 0.048 | 4.05 | 0.956 | 403.62 | 48.20 |
| 8 | 2 | 1 | 0.034 / 0.072 | 0.033 / 0.045 | — | 0.078/0.042 | 2.92 | 1.000 | 284.23 | 51.54 |
| 8 | 2 | 2 | 0.034 / 0.073 | 0.034 / 0.045 | — | 0.078/0.042 | 2.94 | 0.999 | 286.36 | 51.45 |
| 8 | 2 | 4 | 0.034 / 0.073 | 0.036 / 0.046 | — | 0.078/0.042 | 3.13 | 0.990 | 312.04 | 50.70 |
| 8 | 2 | 8 | 0.036 / 0.075 | 0.041 / 0.048 | — | 0.084/0.051 | 3.89 | 0.953 | 411.82 | 48.77 |
| 8 | 3 | 1 | 0.042 / 0.070 | 0.034 / 0.039 | 0.034 / 0.046 | 0.069/0.033/0.045 | 2.89 | 1.000 | 284.87 | 52.20 |
| 8 | 3 | 2 | 0.042 / 0.070 | 0.034 / 0.039 | 0.035 / 0.046 | 0.066/0.036/0.051 | 2.92 | 0.999 | 287.15 | 52.11 |
| 8 | 3 | 4 | 0.042 / 0.068 | 0.036 / 0.039 | 0.036 / 0.047 | 0.066/0.039/0.051 | 3.11 | 0.990 | 313.53 | 51.37 |
| 8 | 3 | 8 | 0.044 / 0.065 | 0.040 / 0.042 | 0.039 / 0.047 | 0.057/0.030/0.045 | 3.80 | 0.956 | 415.03 | 49.48 |

Increment norm |Z| / final state |z_S| at the last position (normal, all hops): S=1: 44.50 / 285.45; S=2: 88.84 / 295.19; S=4: 175.84 / 331.67; S=8: 340.01 / 439.44.

S=1 normal last-position argmax slot counts (slots 0–15): [59, 65, 52, 67, 67, 53, 59, 59, 58, 66, 82, 59, 81, 56, 59, 58].

Replay max ||ΔZ||/||Z||: S=1: 6.4e-05, S=2: 4.7e-05, S=4: 1.0e-04, S=8: 1.7e-04; wall 384 s (reviewed S=1 66 s; loop S=1 46 s, S=2 50 s, S=4 58 s, S=8 72 s).

### seed 2 (ATT-0184-1; adapter ATT-0173-1)

**S_eval=1, reviewed evaluator, all six controls (EM % / NLL)**

| hop | prior EM | normal | zero | shuffled | permuted | oracle_key | oracle_value | target weight all / last (normal) |
|---|---:|---|---|---|---|---|---|---|
| 1 | 1.5 | 4.5 / 7.01 | 0.0 / 9.58 | 1.5 / 8.60 | 5.1 / 7.07 | 7.2 / 4.23 | 76.3 / 0.80 | 0.038 / 0.069 |
| 2 | 0.3 | 3.0 / 7.22 | 0.0 / 9.62 | 1.2 / 8.06 | 2.4 / 7.23 | 3.3 / 4.83 | 0.6 / 11.90 | 0.037 / 0.090 |
| 3 | 1.5 | 3.3 / 7.14 | 0.0 / 9.56 | 1.2 / 8.41 | 2.1 / 7.34 | 3.3 / 4.82 | 0.0 / 12.03 | 0.042 / 0.086 |

Hop-1 H3 (E-0010@2 thresholds, isolated write): passed=False; EM margin 5/167 = 3.0 pp (≥10); NLL normal 7.011 vs zero 9.582 / shuffled 8.597 (nll_pass=True); target weight all 0.0379 (≥0.125), last 0.0689.

**Loop evaluator: EM % / NLL by hop and S_eval**

| control | hop | S=1 | S=2 | S=4 | S=8 |
|---|---|---|---|---|---|
| normal | hop1 | 4.5 / 7.01 | 4.2 / 8.35 | 3.9 / 8.87 | 3.0 / 8.81 |
| normal | hop2 | 3.0 / 7.22 | 3.3 / 8.73 | 2.7 / 9.32 | 3.3 / 9.25 |
| normal | hop3 | 3.3 / 7.14 | 3.6 / 8.61 | 3.9 / 9.19 | 3.9 / 9.15 |
| normal | all | 3.6 / 7.12 | 3.7 / 8.56 | 3.5 / 9.13 | 3.4 / 9.07 |
| zero | hop1 | 0.0 / 9.58 | 0.0 / 9.58 | 0.0 / 9.58 | 0.0 / 9.58 |
| zero | hop2 | 0.0 / 9.62 | 0.0 / 9.62 | 0.0 / 9.62 | 0.0 / 9.62 |
| zero | hop3 | 0.0 / 9.56 | 0.0 / 9.56 | 0.0 / 9.56 | 0.0 / 9.56 |
| zero | all | 0.0 / 9.58 | 0.0 / 9.58 | 0.0 / 9.58 | 0.0 / 9.58 |
| shuffled | hop1 | 1.5 / 8.60 | 1.2 / 10.37 | 1.5 / 10.95 | 0.9 / 10.87 |
| shuffled | hop2 | 1.2 / 8.06 | 1.5 / 9.73 | 1.8 / 10.34 | 1.2 / 10.35 |
| shuffled | hop3 | 1.2 / 8.41 | 1.5 / 10.17 | 1.8 / 10.78 | 2.1 / 10.81 |
| shuffled | all | 1.3 / 8.36 | 1.4 / 10.09 | 1.7 / 10.69 | 1.4 / 10.68 |
| permuted | hop1 | 5.1 / 7.07 | 5.1 / 8.49 | 4.5 / 9.01 | 4.5 / 8.94 |
| permuted | hop2 | 2.4 / 7.23 | 2.4 / 8.65 | 2.4 / 9.16 | 2.7 / 9.01 |
| permuted | hop3 | 2.1 / 7.34 | 2.4 / 8.81 | 2.1 / 9.37 | 3.0 / 9.35 |
| permuted | all | 3.2 / 7.21 | 3.3 / 8.65 | 3.0 / 9.18 | 3.4 / 9.10 |

**Per-step address (normal control): weight on path fact j, all positions / last position; last-position argmax = path fact j rate; n_eff last; query cosine to step 1 (last); state / read norm (last)**

| S_eval | hop | step | w(fact1) | w(fact2) | w(fact3) | argmax=fact1/2/3 | n_eff last | qcos last | |z| last | |a| last |
|---|---|---|---|---|---|---|---:|---:|---:|---:|
| 2 | 1 | 1 | 0.038 / 0.069 | — | — | 0.069 | 3.25 | 1.000 | 238.52 | 52.06 |
| 2 | 1 | 2 | 0.038 / 0.069 | — | — | 0.075 | 3.31 | 0.998 | 244.59 | 51.75 |
| 2 | 2 | 1 | 0.037 / 0.090 | 0.035 / 0.052 | — | 0.090/0.045 | 3.19 | 1.000 | 241.86 | 52.03 |
| 2 | 2 | 2 | 0.037 / 0.090 | 0.035 / 0.052 | — | 0.099/0.045 | 3.24 | 0.999 | 247.60 | 51.87 |
| 2 | 3 | 1 | 0.042 / 0.086 | 0.036 / 0.051 | 0.038 / 0.053 | 0.087/0.051/0.045 | 3.15 | 1.000 | 241.49 | 52.64 |
| 2 | 3 | 2 | 0.042 / 0.086 | 0.036 / 0.052 | 0.038 / 0.053 | 0.093/0.048/0.048 | 3.21 | 0.999 | 247.74 | 52.34 |
| 8 | 1 | 1 | 0.038 / 0.069 | — | — | 0.069 | 3.25 | 1.000 | 238.52 | 52.06 |
| 8 | 1 | 2 | 0.038 / 0.069 | — | — | 0.075 | 3.31 | 0.998 | 244.59 | 51.75 |
| 8 | 1 | 4 | 0.040 / 0.071 | — | — | 0.084 | 3.65 | 0.987 | 278.21 | 50.67 |
| 8 | 1 | 8 | 0.043 / 0.072 | — | — | 0.072 | 4.61 | 0.945 | 386.22 | 48.88 |
| 8 | 2 | 1 | 0.037 / 0.090 | 0.035 / 0.052 | — | 0.090/0.045 | 3.19 | 1.000 | 241.86 | 52.03 |
| 8 | 2 | 2 | 0.037 / 0.090 | 0.035 / 0.052 | — | 0.099/0.045 | 3.24 | 0.999 | 247.60 | 51.87 |
| 8 | 2 | 4 | 0.037 / 0.090 | 0.035 / 0.053 | — | 0.093/0.039 | 3.56 | 0.988 | 281.21 | 51.04 |
| 8 | 2 | 8 | 0.038 / 0.087 | 0.037 / 0.055 | — | 0.102/0.051 | 4.45 | 0.947 | 392.71 | 49.28 |
| 8 | 3 | 1 | 0.042 / 0.086 | 0.036 / 0.051 | 0.038 / 0.053 | 0.087/0.051/0.045 | 3.15 | 1.000 | 241.49 | 52.64 |
| 8 | 3 | 2 | 0.042 / 0.086 | 0.036 / 0.052 | 0.038 / 0.053 | 0.093/0.048/0.048 | 3.21 | 0.999 | 247.74 | 52.34 |
| 8 | 3 | 4 | 0.042 / 0.086 | 0.036 / 0.053 | 0.038 / 0.054 | 0.087/0.054/0.048 | 3.55 | 0.988 | 282.34 | 51.16 |
| 8 | 3 | 8 | 0.044 / 0.083 | 0.036 / 0.056 | 0.039 / 0.057 | 0.093/0.045/0.054 | 4.46 | 0.949 | 393.06 | 49.33 |

Increment norm |Z| / final state |z_S| at the last position (normal, all hops): S=1: 44.54 / 246.64; S=2: 88.65 / 260.60; S=4: 174.02 / 304.74; S=8: 332.92 / 421.63.

S=1 normal last-position argmax slot counts (slots 0–15): [52, 60, 50, 64, 76, 56, 65, 55, 59, 68, 94, 57, 70, 58, 61, 55].

Replay max ||ΔZ||/||Z||: S=1: 6.0e-05, S=2: 5.1e-05, S=4: 8.8e-05, S=8: 6.5e-05; wall 282 s (reviewed S=1 63 s; loop S=1 44 s, S=2 46 s, S=4 53 s, S=8 66 s).
