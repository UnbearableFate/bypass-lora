# T-0003 implementation and preregistered analysis

E-0003 uses the sealed task contract and the E-0002 locked environment (uv.lock blob 3c916b51337de3cba594d645e606a31d4f39c443; cuda/12.8). No official validation is accessed.

Implementation: optional E-0003 train exclusion audit, followed by removal after each task's deterministic shuffle and before alternation. Original split map remains unchanged. Any per-task exclusion fraction strictly above 1% stops the run. Each run archives excluded row task/id/candidate token lengths/reason. The planned full schedule must reproduce 13,077 steps, 2,000,040 input tokens, and dev points 3270/6539/9808. Final checkpoint is the only primary result. Intermediate dev evaluation restores all module modes and Python/NumPy/Torch CPU/CUDA RNG states; it does not touch the optimizer or data iterator.

All eight arms use LR 3e-4, batch 2, microbatch 1. Parameter reporting distinguishes total stored model parameters (including frozen Base), adapter stored parameters, trainable parameters, and parameters with an allocated gradient observed during training (zero-valued gradients count as gradients). Adapter stored/trainable/with-gradient counts must equal the contract. Smoke checks exact zero-exit initialization and exact checkpoint reload for every arm. Evaluation now accepts explicit split names; therefore Base will be rerun, and its dev predictions checked against ATT-0011-2 before using the reference.

Before any formal run: bootstrap uses 10,000 replicates, NumPy default_rng seed 20261029, sampling with replacement within each task (1,000 rows each). Every replicate shares task row indices across all arms and Base. Report two-sided percentile 95% intervals for paired macro-accuracy differences, with raw per-task accuracy and token-weighted task NLL and their macro means. No seed-level uncertainty or confirmatory significance claim is inferred. F* uses the contract's <1 pp equivalence rule with F_out preferred. The exact operational definition of an "effective" C candidate and the equal-parameter C_gate/C_up tie remain for coordinator clarification before formal runs.

Smoke timing is measured after the first 10 of 20 steps for all eight arms. Requested walltime will use measured training/evaluation/save/load durations times 1.5; MIG usage is reported both as instance-hours and node-hours (multiply by .25). E-0003 total ceiling is 10 node-hours including tests and diagnostics.

Status: implementation in progress; no T-0003 PBS checks or formal runs have yet executed.

## E-0003@2 clarification before formal runs (MSG-0063)

Effective means dev macro accuracy at least 1 pp above the rerun Base. Among effective C arms, retain those strictly less than 1 pp below the best effective C arm, then prefer fewer parameters. Equal parameter counts use higher macro accuracy, then lower dev macro NLL, then C_gate. No effective C arm means no C*. Concat wins only when C* exists and the highest scoring concat arm is at least 1 pp above F*. C_dual improves on C_gate only at a difference of at least 1 pp. All decisions use point estimates; bootstrap intervals remain descriptive. This supersedes the unresolved definition above. Formal registrations use E-0003@2.

ATT-0021-1 / RES-0021: 116 tests passed, one new test fixture failed before any smoke training. The fixture supplied one invalid row out of one, correctly triggering the >1% guard while expecting an audit record. Corrected to one invalid row out of 100; the production guard is unchanged. Also corrected the held-out comparison dictionary to key by task and split so both dev and calibration are checked. PBS 3759153.opbs, mg1098, Exit_status 1, 21 s = 0.005833333333333333 MIG instance-h = 0.0014583333333333333 node-h; zero training tokens. Retest pending.
