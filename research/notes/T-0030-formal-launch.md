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
