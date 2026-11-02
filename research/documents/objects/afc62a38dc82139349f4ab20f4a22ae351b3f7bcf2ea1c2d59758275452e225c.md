# T-0030 seed smoke (ATT-0304-1 / PBS 3773945 / RES-0281), before any new-seed dev result

Source e8309d7, debug-mig 2g.24gb, 375 scheduler s = 0.104167 MIG-h = 0.026 node-h.
354 deterministic tests passed (incl. seeded no-channel identity for SEG-04/06/07 at seeds 1, 2).
Quality metrics were discarded; no confirmation set was read.

## Registered training orders (first 13,115 batches of each seed's R1 stream)

| Seed | order sha256 | steps | presentations | keep seg1 tokens | seg2 tokens | gold-loss tokens |
|---|---|---:|---:|---:|---:|---:|
| 0 (E-0025) | 33759bc20322d0c9981128dd8574438546fc3439194ab32b7c79e35e0f0fc2bf | 13115 | 26230 | 1,433,313 | 566,756 | 387,653 |
| 1 | 03c16b099e04ed3906b5ec775a46b3c3f2c7bd8e36d94e29043f45789fb8d716 | 13115 | 26230 | 1,435,892 | 569,242 | 390,056 |
| 2 | 0485e356affdd9baf1d122027b6d924f805979dad343de0c68105eabf4acf608 | 13115 | 26230 | 1,435,833 | 566,393 | 387,244 |

Seed 0 under the fixed-step truncation reproduces E-0025's order hash exactly.
All three orders end epoch 1 at step 8722.

## Initial adapter digests (rule 1(e))

SEG-05 = SEG-06 = SEG-07 within each seed (seed1 cfe01ba0cb7b..., seed2 5fe32303dadc...);
SEG-04 and SEG-05 digests differ across seeds 0/1/2. Full values: runs/ATT-0304-1/initial-digests.json.

## Cost table (measured 1% step smoke at seed 1, projected to 13,115 steps)

| Job | step s | full dev s | projected s |
|---|---:|---:|---:|
| SEG-04 train | 0.198 | 160 | 3437 |
| SEG-05 train (+Base_seg, initial eval) | 0.442 | 363 | 8167 |
| SEG-06 train | 0.216 | 200 | 3886 |
| SEG-07 train | 0.204 | 169 | 3569 |
| SEG-05 interventions | — | — | 1712 |

Per seed 1.443 node-h; two seeds 2.885; checks/analysis reserve 0.060; projection 2.945 node-h
plus this smoke 0.026 = 2.971 <= 3.2 cap (E-0025 actuals give 1.40 node-h per seed). Margin ~0.23 node-h:
no room for a repeated training arm. Walltime per job = 1.5x projection, rounded up to 10 min.
Peak CUDA bytes (smoke): SEG-04 2.03e9, SEG-05 2.19e9, SEG-06 2.09e9, SEG-07 2.04e9.
