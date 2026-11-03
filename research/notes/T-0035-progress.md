# T-0035 / E-0032 execution state

Contract: sealed E-0032@1 (rules 1-8). MSG-1022: the contract governs where
DOC-0372 point 5 differs (X screen: delta>=1.5 and gain_X(0)>=133/3000;
selection by largest delta(0), ties W1, W2, X).

## Implementation (640bac7, review RV-0105 by reviewer-3)

- `bypass_model.py`: an F-phase `write_mask` restricts which valid tokens write
  the KernelMatrix state (reads unchanged). None keeps every old path.
- `seg_model.py`: SEG-W1 (`last`), SEG-W2 (`sentence_last`), SEG-X5/X6/X7
  (SEG-05/06/07 with tap `direct_hidden` = input_layernorm output). W arms
  must receive their mask; RES-0281 SEG-05 init digests pinned.
- `seg_data.py`: frozen sentence-end rule, applied literally to each token's
  own decoded text (clean_up_tokenization_spaces=False). Whitespace inside the
  mark's own token does not satisfy "next token starts with whitespace".
- `seg_eval.py` / `seg_train.py`: masks threaded to scoring, training and the
  iv_swap donor (donor written under its own mask). E-0032 jobs skip the
  25/50/75% curve evals (not in the contract; saves ~0.075 node-h per Keep),
  check step-0 and iv_off = same-allocation Base_seg for all five arms, and
  run iv_swap in-job on the reloaded checkpoint for the three Keep arms.
- `seg_w_checks.py`: rule-1 tests, write-mask audit, init digests, 1% timing,
  cost projection, and the same-slice replay of ATT-0379-2.
- `seg_w_rules.py`: rules 2-6 on exact fractions.
- `seg_w_analysis.py`, `seg_w_mechanism.py` (rule 7): reviewed in the RV-0108
  round (MSG-1051); RV-0109 pass at 9aa4d79.

## Runs

- ATT-0402-1 (640bac7, PBS 3786364, 2g): failed, 351 passed / 2 failed. Cause:
  a test forward hook used `captured.setdefault(...)`, which returns a value,
  so PyTorch replaced the side-branch output. Test-only bug. U-0414
  0.018333 MIG-h. RES-0378 (partial).
- ATT-0403-1 (06d191f = 640bac7 + that test fix only, built by plumbing so the
  reviewed worktree files stay unchanged; local ref
  sr/researcher-2/T-0035-testfix): PBS 3786425: tests passed, then timed out
  (30 min) CPU-bound in the mask audit; cause len(tok) in the token-text cache
  key (fixed d3e9630). U-0418 0.515278 MIG-h.
- RV-0105 (reviewer-3): M1 rule-6 X challenge with unlearned new-seed X
  controls -> fixed at 322bee3 (challenge only if X's own controls learned at
  both new seeds); L1 06d191f fast-forwarded; L2 cost gate -> cost table below
  uses ledger usage. Recheck RV-0106 requested.
- RUN-0405 checks at 322bee3: the real PBS job 3786487.opbs (354 passed, 0
  failed/skipped) is recorded as ATT-0405-2 (RES-0380, U-0416 0.017778 MIG-h).
  It ran with SR_ATTEMPT_ID=ATT-0405-1, so its directory and log are named
  research/artifacts/ATT-0405-1. ATT-0405-1 itself never had a job: an
  erroneous pre-submission observation (wrong id 3786622.opbs) was retracted
  and the attempt ended as cancelled (MSG-1040).
- RV-0108 (b0e7a8f): token cache correct; H1 analysis would mark every real
  SEG-X7 run invalid (T0 records 0 writes/reads), L1 rule-1 evidence not bound
  to stage/source, L2 rule-6 split untested -> fixed at 9aa4d79.
- ATT-0408-1 checks at 9aa4d79 (PBS 3786879, 2g): 362 passed, 0 failed/skipped
  (RES-0383, U-0420 0.019167 MIG-h). RV-0109 recheck: pass, 0 findings.
- ATT-0407-1 rule-1 smoke at b0e7a8f (PBS 3786805, 2g mg1101; RES-0384, U-0421
  0.216667 MIG-h): 359 tests passed; plan 13115 steps / 26230 examples, order
  33759bc2; masks train W1 26230 writes (1 per example), W2 58315 (1-39), dev
  W1 2000, W2 5393 (BoolQ 4392, PIQA 1001); W1/W2 init digests = RES-0281
  SEG-05 for seeds 0-2; X5/X6/X7 identical per seed; 1% timing of all five
  arms with same-job reload identity; replay of ATT-0379-2 on 2g: rows
  identical, 0 flips. b0e7a8f..9aa4d79 changes only seg_w_analysis.py and two
  test files, so rule1_source b0e7a8f has no training-path diff to 9aa4d79.

## Cost table before seed-0 training (node-h = MIG-h x 0.25)

E-0032 measured usage to date: U-0414 0.018333 + U-0418 0.515278 + U-0416
0.017778 + U-0420 0.019167 + U-0421 0.216667 = 0.787223 MIG-h = 0.197 node-h
(0.134 of it from the two failed attempts).

| arm | smoke model s | calibrated s | pessimistic s |
|---|---|---|---|
| SEG-W1 | 6911 | 7105 | 7734 |
| SEG-W2 | 6802 | 6981 | 7734 |
| SEG-X5 | 6745 | 6944 | 7734 |
| SEG-X6 | 3306 | 3265 | 3265 |
| SEG-X7 | 3121 | 3128 | 3128 |
| training node-h | 1.867 | 1.904 | 2.055 |

- Smoke model: smoke steady step x 13115 + full-dev evals scaled from 32 rows.
- Calibrated: smoke steps + the measured non-training time of the E-0030
  SEG-05 job ATT-0379-2 (8425 s elapsed: 5664 train, 1116 curves (skipped
  here), 1645 other) + ~450 s in-job iv_swap; Reset/T0 at half the eval time.
- Pessimistic: Keep steps at E-0030's 0.43 s/step instead of 0.37-0.38.
- Remaining seed-0 reserve: rule-7 mechanism (~0.05) + analysis (~0.02).

| | smoke | calibrated | pessimistic | cap |
|---|---|---|---|---|
| seed 0 (usage + training + 0.07) | 2.13 | 2.17 | 2.32 | 2.2 |
| replication W (2 x W arm) | 0.99 | 0.99 | 1.07 | |
| replication X (2 x X5/X6/X7) | 1.83 | 1.86 | 2.01 | |
| total worst case (+0.05 analysis) | 4.01 | 4.08 | 4.38 | 5.0 |

The central projection is within 2.2. The pessimistic case exceeds it by up
to 0.12 (no failures allowed). The total stays within 5.0. Replication cost is
recomputed from the measured seed-0 jobs before seeds 1-2.

## Seed-0 training (9aa4d79, review RV-0109 round; short-mig, notify T0035-seed0)

| arm | run / attempt | PBS | walltime |
|---|---|---|---|
| SEG-W1 | RUN-0415 / ATT-0415-1 | 3787033 | 03:20 |
| SEG-W2 | RUN-0416 / ATT-0416-1 | 3787041 | 03:20 |
| SEG-X5 | RUN-0417 / ATT-0417-1 | 3787042 | 03:20 |
| SEG-X6 | RUN-0418 / ATT-0418-1 | 3787043 | 01:30 |
| SEG-X7 | RUN-0419 / ATT-0419-1 | 3787044 | 01:30 |

rl flagged the code review as stale for shared modules (rb*, rb1_*, v1_lora,
T-0033 runbook) and for seg_data/seg_w_checks/test_seg_w relative to
RV-0106. The shared modules are byte-identical to the E-0030 run source
f741952 (git diff empty), and the seg_* files were reviewed in RV-0108
(token cache correct) with the round closed by RV-0109 pass.
