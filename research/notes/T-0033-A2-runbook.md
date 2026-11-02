# E-0030 A2 execution under D-0072

Read D-0072 in full before opening. The lock must carry
`preopening_checks.opening_protocol_decision = D-0072`. E-0030@1's scientific
comparisons, thresholds, row membership and checkpoint choices remain fixed.

Before reporting the completed lock to the coordinator, register three PBS
`dev-replay` jobs, one per seed, at the lock's evaluation commit. Pass the same
lock path/SHA256 and seed, without `--unseal` and without any confirmation
decision. Each job loads the actual SEG-05 adapter and records dev predictions,
training/allocated MIG profiles, mismatch count and score-row identity. This
mode never calls the A2 reader. Cross-profile discrepancies are descriptive,
including counts above ten; a same-profile discrepancy is an implementation
fault that requires repair, review and a new lock before confirmation. Report
all three outputs and costs with the original, unchanged lock SHA256. Keep the
lock, checkpoints and replay evidence until this task is accepted.

Only the coordinator's later confirmation decision permits the three first
A2 attempts. Register every attempt and use a notification group with early
failure enabled. Pass `SEG_CONFIRMATION_DECISION=D-NNNN`; it must appear in
the receipt, provenance and manifest and match across all successful seeds.
The first attempt for each seed does not set SEG_REQUIRE_TRAINING_PROFILE.

On any failed dev gate, the application raises, exits non-zero, writes a
partial manifest with `opened=false`, and writes no A2 receipt. On notification,
inspect the application manifest and actual torch profile, not just the
scheduler exit or ledger's profile label.

- `cross_profile_dev_gate_failure`: preserve the exact lock, source, scientific
  config, checkpoints, rules and confirmation decision. Register a new attempt
  for this seed and set `SEG_REQUIRE_TRAINING_PROFILE=1`, the pre-registered
  D-0072 execution control. A retry on another profile exits non-zero before
  preparing dev or reading A2, with `failure_reason=retry_profile_miss`.
  Count consecutive attempts that missed the training profile in the records
  (including the initial cross-profile failed attempt). At five consecutive
  misses, submit no further retry; report the exact attempts and costs to the
  coordinator. This retry rule applies even if a peer has already opened A2.
- `same_profile_dev_gate_failure`: through miyabi-mcp, inspect the exact A2
  batch's jobs and cancel only its jobs that have not started. Running peers
  finish. Once the whole batch is terminal, audit every attempt's receipt for
  this lock. With no receipt anywhere, repair, re-review, rewrite the lock and
  obtain a new confirmation decision. If any receipt exists, A2 is consumed;
  do not repair/relock or draw a hypothesis conclusion. Report invalid A2 to
  the coordinator for its user report. This includes a retry that lands on
  the training profile but fails the gate.
- `a2_unencodable_no_model_scored`: the receipt exists but the all-row encoding
  preflight failed before any A2 model was loaded/scored. Stop without repair,
  lock changes or further scoring; the coordinator reports to the user, who
  decides subsequent handling.

Every engineering retry is a new registered attempt with its own persistent
output directory, observed state and measured cost. Never overwrite a prior
attempt or change the lock or confirming decision for a cross-profile retry.
Reconcile notification groups after retries; final analysis must include all
terminal task attempts, including failures, profile misses and the three
preconfirmation replay jobs, in its explicit accounting list. Stop and report
before exceeding E-0030's 5.5-node-hour cap.
