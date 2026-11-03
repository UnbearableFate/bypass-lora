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
- Not yet reviewed: `seg_w_analysis.py`, `seg_w_mechanism.py` (rule 7) and
  `test_seg_w_analysis.py`; they need a new review round before they run.

## Runs

- ATT-0402-1 (640bac7, PBS 3786364, 2g): failed, 351 passed / 2 failed. Cause:
  a test forward hook used `captured.setdefault(...)`, which returns a value,
  so PyTorch replaced the side-branch output. Test-only bug. U-0414
  0.018333 MIG-h. RES-0378 (partial).
- ATT-0403-1 (06d191f = 640bac7 + that test fix only, built by plumbing so the
  reviewed worktree files stay unchanged; local ref
  sr/researcher-2/T-0035-testfix): PBS 3786425, debug-mig, running.
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
