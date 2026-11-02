# T-0030 formal training launch (E-0027@1, source 054a54a, coverage exception D-0058)

Reviews: RV-0085 pass (e8309d7), recheck RV-0086 pass (054a54a, seed 1/2 order pins).
Exact-source checks ATT-0305-1 / RES-0282: 354 passed, seeded no-channel identity seeds 1, 2.
Smoke ATT-0304-1 / RES-0281 (see T-0030-smoke.md). Coverage exception D-0058 covers
ATT-0306-1..ATT-0313-1 and later SEG-05 interventions / locked analysis at 054a54a only.
No new-seed dev result was read before submission. Notification N-20261102-163033-b187 (8 jobs).

| Attempt | Arm | Seed | PBS job | Queue | Walltime | LR |
|---|---|---:|---|---|---|---:|
| ATT-0306-1 | SEG-04 LoRA_seg r7 | 1 | 3774129.opbs | short-mig | 01:30 | 3e-4 |
| ATT-0307-1 | SEG-05 Keep r128 | 1 | 3774121.opbs | short-mig | 03:30 | 3e-3 |
| ATT-0308-1 | SEG-06 Reset r128 | 1 | 3774134.opbs | short-mig | 01:40 | 3e-3 |
| ATT-0309-1 | SEG-07 T0 r128 | 1 | 3774143.opbs | short-mig | 01:30 | 3e-3 |
| ATT-0310-1 | SEG-04 LoRA_seg r7 | 2 | 3774132.opbs | short-mig | 01:30 | 3e-4 |
| ATT-0311-1 | SEG-05 Keep r128 | 2 | 3774127.opbs | short-mig | 03:30 | 3e-3 |
| ATT-0312-1 | SEG-06 Reset r128 | 2 | 3774136.opbs | short-mig | 01:40 | 3e-3 |
| ATT-0313-1 | SEG-07 T0 r128 | 2 | 3774144.opbs | short-mig | 01:30 | 3e-3 |

Walltime = 1.5x smoke projection rounded up to 10 min. Next: validate manifests (complete,
reload identical, SEG-05 initial==Base_seg, SEG-05/06/07 initial digests equal per seed), then
SEG-05 interventions per seed, then the locked analysis (seg_seed_analysis.py) in PBS.

## Training ended (all 8 complete, 2026-11-02 by 18:38 JST)

RES-0285..0288 (seed 1: SEG-04/05/06/07), RES-0290..0293 (seed 2). All manifests complete,
validator passed, same-job reload identical, 13,115 steps / 26,230 presentations, registered
order hashes, params with grad = stored. SEG-05 initial predictions == Base_seg (both seeds).
Initial adapter digests SEG-05 = SEG-06 = SEG-07 within each seed (seed1 cfe01ba0cb7b...,
seed2 5fe32303dadc...), identical to the smoke. MIG profile metadata (D-0018/D-0019):
ATT-0309-1 and ATT-0312-1 ran on 1g.24gb, the rest on 2g.24gb. Measured 10.150277 MIG-h.

Budget projection before interventions: smoke 0.104167 + checks 0.019722 + training 10.150277
= 10.274166 MIG-h = 2.569 node-h; two SEG-05 intervention jobs (~0.47 MIG-h each, E-0025
measured) + analysis (~0.005) -> ~2.81 node-h <= 3.2.

## SEG-05 interventions submitted

| Attempt | Seed | Training run | PBS job | Walltime |
|---|---:|---|---|---|
| ATT-0319-1 | 1 | ATT-0307-1 | 3775161.opbs | 00:50 |
| ATT-0320-1 | 2 | ATT-0311-1 | 3775164.opbs | 00:50 |
