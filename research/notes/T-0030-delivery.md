# T-0030 delivery: E-0027 BoolQ r128 three-seed replication

**Outcome (locked rules): inconclusive.** Three-seed mean gain G = 88/3000 = **+2.933 pp**, just
below the +3 support gate. Both new-seed gains are positive (+2.6, +1.8), the mean swap drop
W = +6.567 pp exceeds G/2, and the mean margin over segmented LoRA L = +2.067 pp exceeds +1, so
no challenge condition holds either. Per rule 6: descriptive only, no more seeds, widening, tap
change or loops. H-0019 stays inconclusive; H-0020 status awaits conclusion review. No
confirmation set was opened.

Final analysis: ATT-0339-1 / PBS 3775416 / RES-0299,
[analysis.json](/work/xg24i002/x10041/bypass-lora/runs/ATT-0339-1/analysis.json),
[analysis.md](/work/xg24i002/x10041/bypass-lora/runs/ATT-0339-1/analysis.md); input frozen at
research/artifacts/T-0030/analysis-input.json (sha256 593d0eb2...). Seed 0 is read from E-0025's
locked analysis ATT-0303-1 (sha256 a0533c5b..., pinned in code), not rerun.

## BoolQ dev (1000 rows, final checkpoint)

| Seed | SEG-04 LoRA r7 | SEG-05 Keep | SEG-06 Reset | SEG-07 T0 | SEG-05 iv_swap | gain | lora | swap drop |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 (E-0025) | 64.2 | 65.9 | 61.5 | 61.2 | 59.4 | +4.40 | +1.70 | +6.50 |
| 1 | 63.4 | 65.7 | 60.9 | 63.1 | 59.5 | +2.60 [0.1, 5.1] | +2.30 [-0.6, 5.2] | +6.20 [3.1, 9.3] |
| 2 | 63.1 | 65.3 | 61.3 | 63.5 | 58.3 | +1.80 [-0.9, 4.6] | +2.20 [-0.7, 5.1] | +7.00 [3.9, 10.1] |
| mean | | | | | | **G +2.933** | **L +2.067** | **W +6.567** |

Brackets: descriptive per-seed paired task-stratified bootstrap (10,000 draws, seed 20261101),
conditional on each trained model; max control recomputed per draw. Base_seg BoolQ 60.9.

Reading (descriptive): Keep is stable across seeds (65.9 / 65.7 / 65.3) and the swap drop
replicates (6.5 / 6.2 / 7.0). The gain shrinks because the r128 T0 control learned at the new
seeds (61.2 -> 63.1 / 63.5); the seed-0 gain was partly an unlearned-control effect, which is
what the added SEG-04 comparator was meant to detect. Keep exceeds segmented LoRA by +1.7 to
+2.3 pp at every seed, though each per-seed interval includes 0. Reset (61.5 / 60.9 / 61.3)
never clears Base_seg + 1.

| Rule | Outcome | Evidence |
|---|---|---|
| 1 implementation | Pass both seeds: off==Base_seg, initial==Base_seg, seeded no-channel identity, reload identical, SEG-05/06/07 identical init digests | EV-0143 diagnostic |
| 2 per-seed quantities | as table above | EV-0144 |
| 3 means | G +2.933, L +2.067, W +6.567 | EV-0145 |
| 4 supports | Not met (G < 3; all other conditions hold) | EV-0146 |
| 5 challenges | Not met | EV-0147 |
| 6 inconclusive | Yes; no more seeds/width/tap/loop | EV-0148 |
| 7 learnability | Reset unlearned all seeds; T0 unlearned only at seed 0 | EV-0149 |
| 8 PIQA/macro/NLL | Descriptive only | EV-0150 |
| 9 H-0019 / report | H-0019 stays inconclusive; no rule-4/5 stage report trigger | EV-0151 |

## Descriptive (seeds 1 / 2)

| Condition | PIQA % s1 | BoolQ % s1 | PIQA % s2 | BoolQ % s2 |
|---|---:|---:|---:|---:|
| SEG-04 | 67.4 | 63.4 | 67.3 | 63.1 |
| SEG-05 normal | 69.0 | 65.7 | 68.5 | 65.3 |
| SEG-06 | 67.1 | 60.9 | 67.2 | 61.3 |
| SEG-07 | 67.3 | 63.1 | 67.2 | 63.5 |
| SEG-05 iv_off (= Base_seg) | 66.1 | 60.9 | 66.1 | 60.9 |
| SEG-05 iv_swap | 65.2 | 59.5 | 64.2 | 58.3 |
| SEG-05 iv_reset | 66.9 | 62.6 | 66.9 | 61.2 |

PIQA gain/lora/swap: s1 +1.7/+1.6/+3.8, s2 +1.3/+1.2/+4.3. Macro gain s1 +2.15, s2 +1.55.
BoolQ answer NLL: SEG-05 0.644/0.635, SEG-04 0.828/0.809. Intervention normal path matches the
training evaluation with 0 prediction flips at both seeds.

## Implementation and provenance

- Seed path: commit e8309d7 (RV-0085 pass), order pins 054a54a (recheck RV-0086 pass). All formal
  runs, interventions and the analysis execute 054a54a under coverage exception **D-0058**
  (helper files changed by E-0024/T-0027/E-0026 commits, not on the segmented path).
- New seeds train the first 13,115 batches of their own R1 stream; seed 0 under this rule
  reproduces E-0025's order hash. Orders: s1 03c16b09..., s2 0485e356... (registered from smoke
  ATT-0304-1 before training; T-0030-smoke.md). 26,230 presentations each.
- Init derive_seed(20261101,'residual_la_v21','R1','init',s); train order derive_seed(...,'train',s)+epoch.
- Exact-source checks ATT-0305-1: 354 passed. Smoke ATT-0304-1: 354 passed, timing/cost.
- MIG profile metadata only (D-0018/D-0019): ATT-0309-1 and ATT-0312-1 on 1g.24gb, others 2g.24gb.
- Model Qwen/Qwen3-0.6B-Base @ da87bfb6...; uv.lock 31429457...; envs/3c916b51337d-cuda; LR lock
  38b8ee24... (R/T 3e-3, LoRA 3e-4).

## Cost (measured, terminal ledger)

| Attempt | Result | Stage | MIG-h |
|---|---|---|---:|
| ATT-0304-1 | RES-0281 | seed smoke | 0.104167 |
| ATT-0305-1 | RES-0282 | exact-source checks | 0.019722 |
| ATT-0306-1 | RES-0285 | SEG-04 s1 | 0.936111 |
| ATT-0307-1 | RES-0286 | SEG-05 s1 | 2.097778 |
| ATT-0308-1 | RES-0287 | SEG-06 s1 | 1.017500 |
| ATT-0309-1 | RES-0288 | SEG-07 s1 | 1.006944 |
| ATT-0310-1 | RES-0290 | SEG-04 s2 | 1.066944 |
| ATT-0311-1 | RES-0291 | SEG-05 s2 | 2.058889 |
| ATT-0312-1 | RES-0292 | SEG-06 s2 | 0.964444 |
| ATT-0313-1 | RES-0293 | SEG-07 s2 | 1.001667 |
| ATT-0319-1 | RES-0297 | SEG-05 interventions s1 | 0.462500 |
| ATT-0320-1 | RES-0298 | SEG-05 interventions s2 | 0.476944 |
| ATT-0339-1 | RES-0299 | locked analysis | 0.003056 |
| **Total** | | | **11.216666 = 2.804 node-h** (cap 3.2) |

No failed attempts. The analysis validated coverage of the first 12; its own 0.003 MIG-h is added here.

## Limits

Three seeds only; dev both selects and judges; exact point-estimate gates; intervals exclude
seed variation and are not simultaneous. RBc LR not recalibrated for r128 or segmentation.
BoolQ task level only; no confirmation evaluation; no practical, compression or long-context claim.
Checkpoints for all eight arms are retained under runs/ATT-0306-1..0313-1 pending cleanup decisions.
