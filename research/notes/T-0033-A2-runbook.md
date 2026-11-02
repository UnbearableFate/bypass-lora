# E-0030 A2 execution under D-0072 and D-0073

D-0073 is the actual confirmation for lock ATT-0393-1/a2-lock.json, SHA256
`cfd2e9b054c26da3d43ed22dbf0c410060599d6599d5fe88c4b8cfe1b78838c5`.
It explicitly overrides the old unrestricted-first/five-miss operational
fields in D-0072 and the unchanged lock. Scientific inputs/rules stay fixed;
evaluation/analysis source stays 4e6e8df, training source f741952.

On EVERY A2 attempt set SEG_REQUIRE_TRAINING_PROFILE=1 and
SEG_CONFIRMATION_DECISION=D-0073; register RunSpec arm A2 exactly. Submit only
seed 4 (requires 1g) initially. Hold seeds 5 and 3 until seed 4 has a complete
manifest, exact dev gate and valid A2 receipt. Tell coordinator that result,
then submit seeds 5 (1g) and 3 (2g) without waiting for its reply. At most one
queued/running attempt per seed. Same-profile failures retain D-0072's
consumed/invalid handling; seed 4 success does not prove seed 5 exactness.

Before EVERY submission compute measured ended-job cost + 80 minutes for
each seed without a completed A2 evaluation (including its queued/running
attempt) + the three-minute analysis request, converted at 1/4 MIG-hour.
If the total exceeds 5.5 node-hours, do not submit; report coordinator.
The initial total is 5.245486 node-hours from 4.232986 measured completed.
After twenty consecutive profile misses for a seed, stop and report even
when budget remains. The unchanged source's failure_action still says five;
the actual D-0073 controller limit is twenty, subject to the earlier budget
stop. Preserve every retry and measured usage for final analysis accounting.

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
Every attempt, including each first attempt, sets
SEG_REQUIRE_TRAINING_PROFILE=1 per D-0073.

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
  (including any initial cross-profile failed attempt). At twenty consecutive
  misses under D-0073, submit no further retry; report the exact attempts and costs to the
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
