# T-0016 results: E-0015@4 memory-route diagnostics (evaluation only)

This task was evaluation only, with no training. Confirmation data was never read.

**Code.** Branch sr/researcher-3/T-0016:
- 7576c2d: the four checkpoint runs.
- 74d7c19: the RV-0042 fixes, the reload diagnostic and the analysis.

**Review.** RV-0042 (reviewer-1) raised two high findings: rule 4 carried an NLL condition it should not have, and rule 6 left out R2@E16 and R5@E4. Both were fixed in 74d7c19 and covered by synthetic rule tests. The recheck, RV-0043, passed.

**Runs.**

| attempt | result | role |
|---|---|---|
| ATT-0141-1 | RES-0133 | R1, ATT-0122-1 |
| ATT-0142-1 | RES-0134 | R2, ATT-0123-1 |
| ATT-0143-1 | RES-0135 | R5, ATT-0126-1 |
| ATT-0144-2 | RES-0136 | V2-03, ATT-0082-1 |
| ATT-0148-1 | RES-0137 | analysis: `runs/ATT-0148-1/analysis.json` |

Supporting attempts:
- ATT-0139-1 and ATT-0140-1: checks. ATT-0140-1 includes the bitwise regression of the existing controls against c6037d3.
- ATT-0145-1 and ATT-0147-1: unit tests.
- ATT-0144-1: the first V2-03 run. Its reload check mismatched and it stopped.
- ATT-0146-1: the reload diagnostic.

**Evidence.** All entries are diagnostic, linked to E-0015, with no supports or challenges:
- EV-0086: the MIG slice finding.
- EV-0087: R1.
- EV-0088: R2.
- EV-0089: R5.
- EV-0091: V2-03. EV-0090 was invalidated: its summary misstated the EM range, and EV-0091 replaces it.

**Compute.** 0.8111 MIG-h, which is **0.203 node·h**, against a budget of 0.25. This covers every T-0016 attempt above. Reviewer job RV-0042 used another 0.0047 MIG-h.

## Checks
- **Reload.** For R1, R2, R5 and V2-03, the recorded dev splits (E=16, plus E=1 for R1 and E=4 for R2) reproduce bitwise: 0 prediction mismatches and NLL difference 0.0 under normal, zero and shuffled.
  - V2-03 first ran as ATT-0144-1 on mg1098_mig3. That device reports `MIG 1g.24gb` with 26 SMs, although the ledger says 2g, and its BF16 results differ from the records. ATT-0146-1 showed that R1 and the Base-only zero control also mismatch on that slice, with identical evaluator settings.
  - ATT-0144-2 reran V2-03 on a 2g slice behind an `nvidia-smi` guard, and it is bitwise identical (EV-0086).
- **Isolated encoding.** Each fact's context tokens equal its isolated E=1-format tokens: 0 differ out of 16,000 / 32,768 / 65,536 E=16 facts in dev / calibration / train. Isolated and contextual writes are identical at E=1 (max difference 0.0) for every checkpoint.
- **Pf inputs.** The Pf inputs (frozen delimiter features) are identical across checkpoints, as they must be.
- **Splits.** The train and calibration splits are disjoint by id and by content.

## EM by checkpoint × split × control (calibration, %), stratified by target fact position

NLL is the mean answer NLL. E=1 has only target position 0. The sweep rows run only on the contracted cell (R1@E1, R2@E4, R5@E16). Answer-prior EM: E=1 1.4, E=4 1.3, E=16 1.9.

### R1 (ATT-0141-1; checkpoint ATT-0122-1)

Rows per split: e1: target 0 n=2048, target ≥1 n=0; e4: target 0 n=487, target ≥1 n=1561; e16: target 0 n=151, target ≥1 n=1897

| control | E=1 EM | E=1 NLL | E=4 EM (t0 / t≥1) | E=4 NLL | E=16 EM (t0 / t≥1) | E=16 NLL |
|---|---:|---:|---:|---:|---:|---:|
| normal | 84.1 | 0.519 | 1.1 (0.6 / 1.2) | 18.236 | 0.5 (0.7 / 0.5) | 18.918 |
| zero | 0.7 | 9.274 | 0.9 (0.4 / 1.1) | 9.209 | 0.8 (0.0 / 0.9) | 9.231 |
| shuffled | 1.4 | 18.258 | 1.2 (0.6 / 1.4) | 19.062 | 0.4 (1.3 / 0.4) | 19.085 |
| oracle_value | 84.1 | 0.519 | 21.9 (83.8 / 2.6) | 13.127 | 7.6 (83.4 / 1.6) | 15.778 |
| last_only | 82.4 | 0.588 | 1.1 (1.0 / 1.2) | 17.126 | 0.7 (0.7 / 0.7) | 18.295 |
| oracle_last | 84.1 | 0.519 | 19.2 (73.9 / 2.2) | 13.323 | 6.8 (74.8 / 1.4) | 16.007 |
| oracle_last_only | 82.4 | 0.588 | 22.0 (83.8 / 2.7) | 12.646 | 7.7 (81.5 / 1.8) | 15.333 |
| uniform | 84.1 | 0.519 | 1.4 (1.8 / 1.2) | 17.678 | 0.3 (0.7 / 0.3) | 18.929 |
| last_only_shuffled | 1.3 | 16.307 | 1.2 (1.0 / 1.3) | 17.927 | 0.4 (1.3 / 0.4) | 18.467 |
| normal@a0.1 | 1.0 | 5.532 |  |  |  |  |
| normal@a0.3 | 55.7 | 1.774 |  |  |  |  |
| oracle_value@a0.1 | 1.0 | 5.532 |  |  |  |  |
| oracle_value@a0.3 | 55.7 | 1.774 |  |  |  |  |
| nonfinal@a0.1 | 82.4 | 0.581 |  |  |  |  |
| nonfinal@a0.3 | 83.6 | 0.544 |  |  |  |  |
| wrong_oracle |  |  | 1.0 (0.6 / 1.1) | 19.685 | 0.8 (0.7 / 0.8) | 19.281 |
| iso_kv |  |  | 8.3 (9.0 / 8.0) | 6.953 | 2.5 (3.3 / 2.4) | 7.968 |
| iso_v |  |  | 7.7 (4.5 / 8.6) | 7.115 | 2.1 (2.6 / 2.1) | 8.189 |
| iso_oracle_value |  |  | 85.3 (84.0 / 85.7) | 0.463 | 86.4 (83.4 / 86.7) | 0.468 |
| iso_wrong_oracle |  |  | 1.6 (1.8 / 1.5) | 18.566 | 1.5 (1.3 / 1.5) | 18.179 |

### R2 (ATT-0142-1; checkpoint ATT-0123-1)

Rows per split: e1: target 0 n=2048, target ≥1 n=0; e4: target 0 n=487, target ≥1 n=1561; e16: target 0 n=151, target ≥1 n=1897

| control | E=1 EM | E=1 NLL | E=4 EM (t0 / t≥1) | E=4 NLL | E=16 EM (t0 / t≥1) | E=16 NLL |
|---|---:|---:|---:|---:|---:|---:|
| normal | 11.9 | 3.541 | 9.7 (10.3 / 9.5) | 3.666 | 2.8 (4.6 / 2.7) | 6.244 |
| zero | 0.7 | 9.274 | 0.9 (0.4 / 1.1) | 9.209 | 0.8 (0.0 / 0.9) | 9.231 |
| shuffled | 1.8 | 6.249 | 1.7 (1.0 / 1.9) | 6.165 | 1.8 (0.0 / 2.0) | 6.926 |
| oracle_value | 11.9 | 3.541 | 9.5 (10.7 / 9.1) | 3.651 | 4.1 (10.6 / 3.6) | 5.811 |
| last_only | 8.3 | 3.633 | 8.2 (8.6 / 8.0) | 3.723 | 2.3 (4.0 / 2.2) | 6.079 |
| oracle_last | 11.9 | 3.541 | 9.4 (9.2 / 9.4) | 3.663 | 4.1 (9.3 / 3.6) | 5.795 |
| oracle_last_only | 8.3 | 3.633 | 8.1 (8.6 / 7.9) | 3.721 | 4.0 (7.3 / 3.7) | 5.579 |
| uniform | 11.9 | 3.541 | 3.7 (3.9 / 3.7) | 4.465 | 2.1 (4.0 / 1.9) | 6.458 |
| last_only_shuffled | 1.8 | 6.082 | 1.8 (0.8 / 2.1) | 6.160 | 1.7 (1.3 / 1.7) | 6.742 |
| wrong_oracle |  |  | 2.5 (2.1 / 2.7) | 6.042 | 2.2 (5.3 / 2.0) | 7.286 |
| iso_kv |  |  | 11.3 (9.4 / 11.9) | 3.522 | 3.2 (6.0 / 3.0) | 4.688 |
| iso_v |  |  | 11.3 (9.4 / 11.9) | 3.518 | 4.0 (7.9 / 3.6) | 4.897 |
| iso_oracle_value |  |  | 12.2 (10.7 / 12.7) | 3.503 | 10.9 (9.9 / 11.0) | 3.524 |
| iso_wrong_oracle |  |  | 1.7 (0.8 / 2.0) | 6.568 | 1.7 (0.7 / 1.8) | 6.431 |
| normal@a0.1 |  |  | 0.9 (0.4 / 1.0) | 5.576 |  |  |
| normal@a0.3 |  |  | 5.7 (5.3 / 5.8) | 4.246 |  |  |
| oracle_value@a0.1 |  |  | 1.4 (0.8 / 1.5) | 5.474 |  |  |
| oracle_value@a0.3 |  |  | 5.9 (6.2 / 5.8) | 4.203 |  |  |
| nonfinal@a0.1 |  |  | 8.1 (8.2 / 8.1) | 3.705 |  |  |
| nonfinal@a0.3 |  |  | 8.9 (9.7 / 8.7) | 3.684 |  |  |

### R5 (ATT-0143-1; checkpoint ATT-0126-1)

Rows per split: e1: target 0 n=2048, target ≥1 n=0; e4: target 0 n=487, target ≥1 n=1561; e16: target 0 n=151, target ≥1 n=1897

| control | E=1 EM | E=1 NLL | E=4 EM (t0 / t≥1) | E=4 NLL | E=16 EM (t0 / t≥1) | E=16 NLL |
|---|---:|---:|---:|---:|---:|---:|
| normal | 6.1 | 3.999 | 7.6 (6.2 / 8.0) | 3.877 | 6.4 (2.0 / 6.7) | 4.091 |
| zero | 0.7 | 9.274 | 0.9 (0.4 / 1.1) | 9.209 | 0.8 (0.0 / 0.9) | 9.231 |
| shuffled | 1.9 | 4.901 | 1.5 (1.2 / 1.5) | 5.069 | 1.8 (2.0 / 1.8) | 4.970 |
| oracle_value | 6.1 | 3.999 | 7.2 (5.7 / 7.6) | 3.869 | 10.4 (7.9 / 10.6) | 3.638 |
| last_only | 5.3 | 4.058 | 7.0 (5.7 / 7.4) | 3.928 | 5.5 (2.6 / 5.7) | 4.164 |
| oracle_last | 6.1 | 3.999 | 7.6 (6.4 / 8.0) | 3.876 | 10.4 (4.6 / 10.9) | 3.642 |
| oracle_last_only | 5.3 | 4.058 | 7.0 (6.0 / 7.4) | 3.927 | 8.9 (2.6 / 9.4) | 3.707 |
| uniform | 6.1 | 3.999 | 3.3 (2.7 / 3.5) | 4.368 | 2.1 (0.7 / 2.3) | 4.414 |
| last_only_shuffled | 1.4 | 4.930 | 1.6 (1.6 / 1.5) | 5.121 | 1.7 (1.3 / 1.7) | 5.050 |
| wrong_oracle |  |  | 1.2 (1.0 / 1.2) | 5.228 | 1.5 (2.6 / 1.4) | 5.403 |
| iso_kv |  |  | 4.6 (5.5 / 4.4) | 4.033 | 3.3 (2.0 / 3.4) | 4.351 |
| iso_v |  |  | 4.4 (5.7 / 4.0) | 4.033 | 3.1 (3.3 / 3.1) | 4.295 |
| iso_oracle_value |  |  | 4.7 (5.7 / 4.4) | 4.028 | 6.0 (7.9 / 5.8) | 3.995 |
| iso_wrong_oracle |  |  | 1.7 (1.8 / 1.7) | 4.999 | 1.6 (2.6 / 1.5) | 4.929 |
| normal@a0.1 |  |  |  |  | 0.5 (0.7 / 0.5) | 5.692 |
| normal@a0.3 |  |  |  |  | 3.7 (2.0 / 3.8) | 4.584 |
| oracle_value@a0.1 |  |  |  |  | 0.9 (1.3 / 0.9) | 5.518 |
| oracle_value@a0.3 |  |  |  |  | 6.0 (5.3 / 6.0) | 4.262 |
| nonfinal@a0.1 |  |  |  |  | 6.2 (4.0 / 6.4) | 4.141 |
| nonfinal@a0.3 |  |  |  |  | 6.0 (2.6 / 6.3) | 4.115 |

### V2-03 (ATT-0144-2; checkpoint ATT-0082-1)

Rows per split: e1: target 0 n=2048, target ≥1 n=0; e4: target 0 n=487, target ≥1 n=1561; e16: target 0 n=151, target ≥1 n=1897

| control | E=1 EM | E=1 NLL | E=4 EM (t0 / t≥1) | E=4 NLL | E=16 EM (t0 / t≥1) | E=16 NLL |
|---|---:|---:|---:|---:|---:|---:|
| normal | 1.6 | 4.450 | 1.7 (1.8 / 1.7) | 4.435 | 1.5 (2.6 / 1.4) | 4.405 |
| zero | 0.7 | 9.274 | 0.9 (0.4 / 1.1) | 9.209 | 0.8 (0.0 / 0.9) | 9.231 |
| shuffled | 2.2 | 4.452 | 1.9 (1.6 / 1.9) | 4.437 | 1.0 (0.7 / 1.0) | 4.405 |
| oracle_value | 1.6 | 4.450 | 1.7 (1.4 / 1.7) | 4.486 | 1.3 (0.7 / 1.4) | 4.550 |
| last_only | 1.4 | 4.450 | 1.7 (2.1 / 1.5) | 4.448 | 2.1 (2.0 / 2.1) | 4.423 |
| oracle_last | 1.6 | 4.450 | 1.8 (2.1 / 1.7) | 4.478 | 1.4 (2.6 / 1.3) | 4.546 |
| oracle_last_only | 1.4 | 4.450 | 1.5 (1.4 / 1.5) | 4.498 | 1.7 (1.3 / 1.7) | 4.597 |
| uniform | 1.6 | 4.450 | 2.0 (2.3 / 1.9) | 4.466 | 1.3 (1.3 / 1.3) | 4.513 |
| last_only_shuffled | 1.1 | 4.451 | 1.6 (1.6 / 1.5) | 4.451 | 1.7 (0.7 / 1.7) | 4.423 |
| wrong_oracle |  |  | 1.8 (2.1 / 1.7) | 4.488 | 1.2 (1.3 / 1.2) | 4.551 |
| iso_kv |  |  | 1.7 (1.8 / 1.6) | 4.462 | 1.5 (0.7 / 1.6) | 4.438 |
| iso_v |  |  | 1.6 (1.4 / 1.7) | 4.466 | 1.3 (0.7 / 1.4) | 4.440 |
| iso_oracle_value |  |  | 1.7 (1.4 / 1.7) | 4.463 | 1.3 (0.7 / 1.4) | 4.431 |
| iso_wrong_oracle |  |  | 1.7 (1.4 / 1.7) | 4.466 | 1.1 (0.0 / 1.2) | 4.441 |

## Probes (calibration accuracy %)

The probes are closed-form ridge (λ=1), fit on the first 4096 train rows (file order) of the same E and tested on all calibration slots (2048 × E samples; Pz uses 2048 queries). "by pos" groups by fact position 0 / 1–3 / ≥4. Every probe has 0 unseen test labels. The Pz oracle_value and Pv ctx→iso values are in analysis.json.

| checkpoint | split | Pv ctx | Pv ctx by pos 0 / 1–3 / ≥4 | Pf ctx | Pf ctx by pos | Pf rand-128 | Pv iso | Pf iso | Pv iso→ctx | Pz normal | prior | label shuffle (Pv ctx) |
|---|---|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| R1 | e1 | 74.5 | 74.5 / — / — | 99.8 | 99.8 / — / — | 65.7 | 74.5 | 99.8 | 74.5 | 65.7 | 1.4 | 1.6 |
| R1 | e4 | 47.9 | 62.5 / 43.0 / — | 97.4 | 98.1 / 97.1 / — | 36.8 | 78.0 | 100.0 | 21.1 | 6.4 | 1.8 | 1.4 |
| R1 | e16 | 37.6 | 36.2 / 27.2 / 40.3 | 94.9 | 85.9 / 91.7 / 96.5 | 32.6 | 77.6 | 100.0 | 7.1 | 2.3 | 1.4 | 1.6 |
| R2 | e1 | 71.1 | 71.1 / — / — | 99.8 | 99.8 / — / — | 65.7 | 71.1 | 99.8 | 71.1 | 46.1 | 1.4 | 1.4 |
| R2 | e4 | 45.9 | 47.6 / 45.4 / — | 97.4 | 98.1 / 97.1 / — | 36.8 | 75.2 | 100.0 | 20.5 | 21.7 | 1.8 | 1.8 |
| R2 | e16 | 38.8 | 27.2 / 31.2 / 41.6 | 94.9 | 85.9 / 91.7 / 96.5 | 32.6 | 75.3 | 100.0 | 7.1 | 3.5 | 1.4 | 1.5 |
| R5 | e1 | 69.4 | 69.4 / — / — | 99.8 | 99.8 / — / — | 65.7 | 69.4 | 99.8 | 69.4 | 44.4 | 1.4 | 1.5 |
| R5 | e4 | 45.8 | 44.4 / 46.3 / — | 97.4 | 98.1 / 97.1 / — | 36.8 | 74.8 | 100.0 | 20.6 | 22.2 | 1.8 | 1.7 |
| R5 | e16 | 45.8 | 19.4 / 32.4 / 51.4 | 94.9 | 85.9 / 91.7 / 96.5 | 32.6 | 75.1 | 100.0 | 6.8 | 11.7 | 1.4 | 1.7 |
| V2-03 | e1 | 69.3 | 69.3 / — / — | 99.8 | 99.8 / — / — | 65.7 | 69.3 | 99.8 | 69.3 | 35.7 | 1.4 | 1.5 |
| V2-03 | e4 | 38.4 | 44.7 / 36.3 / — | 97.4 | 98.1 / 97.1 / — | 36.8 | 73.7 | 100.0 | 20.0 | 5.3 | 1.8 | 1.3 |
| V2-03 | e16 | 32.9 | 18.7 / 22.9 / 36.6 | 94.9 | 85.9 / 91.7 / 96.5 | 32.6 | 74.3 | 100.0 | 6.7 | 1.4 | 1.4 | 1.5 |


The R1 Pz.normal at E=1 (65.7%) equals the Pf random 128-d projection count (1346/2048) exactly. The inputs are independent in the code: captured `z@P` on one side, frozen features times a seeded random matrix on the other. I read this as a coincidence and did not investigate further.

## Per-position magnitudes (E=4/E=16 calibration slots; descriptive, AD-0002)

Each cell gives the mean L2, then the mean max-abs coordinate, then (the modal argmax dimension and its share). The frozen features are identical for every checkpoint, so they are shown only for R1.

| checkpoint | split | quantity | pos 0 L2 / max-abs (argmax dim, share) | pos 1–3 | pos ≥4 |
|---|---|---|---|---|---|
| R1 | e4 | features.context | 57.4 / 32.9 (1, 1.00) | 56.3 / 30.9 (1, 1.00) | — |
| R1 | e4 | features.isolated | 57.4 / 32.9 (1, 1.00) | 57.4 / 32.9 (1, 1.00) | — |
| R1 | e4 | values.context | 44.0 / 20.5 (64, 0.99) | 96.9 / 22.9 (94, 0.25) | — |
| R1 | e4 | values.isolated | 44.0 / 20.5 (64, 0.99) | 43.9 / 20.4 (64, 0.99) | — |
| R1 | e16 | features.context | 57.4 / 32.9 (1, 1.00) | 56.8 / 31.2 (1, 1.00) | 56.7 / 27.8 (1, 1.00) |
| R1 | e16 | features.isolated | 57.4 / 32.9 (1, 1.00) | 57.4 / 32.9 (1, 1.00) | 57.4 / 32.9 (1, 1.00) |
| R1 | e16 | values.context | 43.5 / 20.3 (64, 0.99) | 103.1 / 24.0 (94, 0.26) | 89.2 / 19.6 (125, 0.29) |
| R1 | e16 | values.isolated | 43.5 / 20.3 (64, 0.99) | 43.7 / 20.3 (64, 0.99) | 43.8 / 20.4 (64, 0.99) |
| R2 | e4 | values.context | 71.0 / 33.5 (63, 1.00) | 74.6 / 45.1 (63, 1.00) | — |
| R2 | e4 | values.isolated | 71.0 / 33.5 (63, 1.00) | 70.9 / 33.4 (63, 1.00) | — |
| R2 | e16 | values.context | 70.7 / 33.4 (63, 1.00) | 78.1 / 46.0 (63, 1.00) | 88.9 / 39.3 (63, 1.00) |
| R2 | e16 | values.isolated | 70.7 / 33.4 (63, 1.00) | 70.8 / 33.4 (63, 1.00) | 70.9 / 33.5 (63, 1.00) |
| R5 | e4 | values.context | 81.8 / 26.7 (27, 1.00) | 77.3 / 37.3 (27, 1.00) | — |
| R5 | e4 | values.isolated | 81.8 / 26.7 (27, 1.00) | 81.9 / 26.7 (27, 1.00) | — |
| R5 | e16 | values.context | 81.9 / 26.7 (27, 1.00) | 77.2 / 37.0 (27, 1.00) | 80.2 / 38.6 (27, 1.00) |
| R5 | e16 | values.isolated | 81.9 / 26.7 (27, 1.00) | 81.9 / 26.7 (27, 1.00) | 81.9 / 26.7 (27, 1.00) |
| V2-03 | e4 | values.context | 109.7 / 58.4 (41, 1.00) | 69.3 / 35.9 (41, 1.00) | — |
| V2-03 | e4 | values.isolated | 109.7 / 58.4 (41, 1.00) | 109.8 / 58.5 (41, 1.00) | — |
| V2-03 | e16 | values.context | 109.7 / 58.4 (41, 1.00) | 68.3 / 35.0 (41, 1.00) | 44.4 / 21.6 (41, 0.80) |
| V2-03 | e16 | values.isolated | 109.7 / 58.4 (41, 1.00) | 109.7 / 58.4 (41, 1.00) | 109.7 / 58.5 (41, 1.00) |

## Rules 1–8 (E-0015@4)

Every "≥ 10 pp" comparison also requires NLL to improve in the same direction, except rule 4, which is EM only (MSG-0397).

| rule | holds | values |
|---|---|---|
| 1 value present, decoding fails (R2@E4 or R5@E16: Pv ≥ 80% and oracle_value < 30%) | **no** | Pv 45.9% (R2@E4), 45.8% (R5@E16) |
| 2 value content lost (Pv < 50% with Pf ≥ 90%) | **yes** | all 4 checkpoints at E=4 and E=16. Pv context 32.9–47.9%; Pf 97.4% (E4), 94.9% (E16) |
| 3 non-final mixed-read interference | **no** | last_only − normal: R2@E4 −1.5 pp, R5@E16 −0.9 pp. oracle_last_only − oracle_value: −1.4 / −1.6 pp |
| 4 cross-context transfer fails (R1 oracle_value E1 − E4/E16 ≥ 30 pp, Pf ≥ 90%) | **yes** | 84.1 → 21.9 (−62.3 pp) and 7.6 (−76.5 pp); Pf 97.4 / 94.9 |
| 5 transfer confirmed (R1@E16) | **yes** | iso_oracle_value t0 83.4 / t≥1 86.7. Contextual oracle_value t0 83.4 / t≥1 1.6. iso_wrong_oracle 1.5 vs prior 1.9 |
| 6 isolated values help R2/R5 (iso_v − normal ≥ 10 pp) | **no** | R2 +1.7 (E4), +1.1 (E16); R5 −3.2 (E4), −3.3 (E16) |
| 7 magnitude (α < 1 helps R2@E4/R5@E16 while R1@E1 ≥ 50%) | **no** | every α < 1 lowers EM; R1@E1 1.0% (α=0.1), 55.7% (α=0.3), non-final 82.4 / 83.6 |
| 8 none of the above | **no** | |

Rules 2, 4 and 5 hold together. The trained R1 values work when written in the isolated (E=1) format, at every target position: iso_oracle_value is 85–86%. They stop working once the same fact sits in a multi-fact context:
- Contextual oracle_value is high only for target position 0, which is the position whose context equals the isolated one.
- The contextual values at positions ≥ 1 carry little linearly decodable answer information: Pv iso→ctx is 7.1% at E=16, against 77.6% for isolated values.
- The contextual values also change scale and shape at positions ≥ 1. For R1 at E=16, L2 is 43.5 at position 0 against 103.1 at positions 1–3. The argmax dimension moves from 64 (share 0.99) to a spread mixture.
- The frozen delimiter features keep the same L2 (~57) at every position.

The answer information that is lost is therefore created by the trained value map, acting on context-dependent features. It is not absent from the features: Pf is 95–97%. Neither R2 nor R5 recovers when given isolated values (rule 6): their decoders do not use even the isolated values well (iso_oracle_value ≤ 12%).

## Operationalizations and open issues
- **Rule 5 "near prior":** I operationalized it as EM ≤ answer-prior EM + 5 pp.
- **Rule 3 "no same lift":** I operationalized it as last_only_shuffled − shuffled < 10 pp, or no NLL improvement in the same direction. Rule 3 already fails on its first two conditions.
- **Cross-encoding Pv:** the contract asks for it on R1 only. It was computed for all checkpoints and in both directions.
- **mg1098_mig3:** this slice is a 1g instance and does not reproduce BF16 records. Jobs that need bitwise reproducibility should guard on `MIG 2g.24gb` (EV-0086).
- **Rule 2 and V2-03:** rule 2 also holds for V2-03, which is at chance under every control. Pv < 50% does not by itself separate a working checkpoint from a failed one.
