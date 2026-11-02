# T-0033 / E-0030 execution state

Implementation source (initial preflight): `e7b1c6c25e08bc0289bc156a61106a8b1ce64e44`.
Branch: `sr/researcher-2/T-0033`. A2 remains sealed.

The new path supports seeds 3, 4, 5 without changing the E-0027 optimizer,
sample/step budget, architecture or LR lock. Keep's dev iv_off check runs
inside its training allocation after final-checkpoint reload; no dev iv_swap
or iv_reset is run. All four arms retain initialization, 25/50/75% and final
dev predictions. The legacy LoRA CUDA initialization is unchanged; every run
records torch's actual MIG name and PBS job ID. This follows MSG-0934.

The E-0030 pre-opening data loader hashes source files but replaces held PIQA
rows with ID-only placeholders before JSON/label conversion, and filters held
BoolQ rows in Arrow before Python row conversion. Its PBS smoke checks all
six order streams, including the previously registered seed 0/1/2 hashes,
and compares the exclusion audit to ATT-0306-1. This follows MSG-0936.

Before registering the new runs, the recorded E-0019 through E-0029 run specs
were inspected through `rl run show`; the only seed values present were 0,
1 and 2. Seeds 3, 4 and 5 are new within the recorded RB/R1 experiments.

PBS preflight: RUN-0374 / ATT-0374-1, job `3779857.opbs`, debug-mig, one MIG,
15-minute walltime based on ATT-0304-1's 358-second predecessor plus additional
loader/tests. It was observed running at submission. Output:
`/work/xg24i002/x10041/bypass-lora/research/artifacts/ATT-0374-1/`.
Its results, actual cost and hardware have not yet been validated. Notification
`N-20261103-024958-7df8` (`T0033-preflight`) targets the coordinator's listener;
notifyd is running. The small source snapshot is read-only. The declared
RunSpec uses the resource's default `runs/` planned location, while the actual
job explicitly writes the session-required `research/artifacts/` location.

Next: validate/import the preflight result and exact measured scheduler usage;
register seed 3/4/5 order hashes; request the full E-0030 changed-code review;
run deterministic checks at the final reviewed source; submit the twelve
formal training runs only after review and the <=5.5 node-hour projection pass.
The A2 reader/evaluator/analysis is implemented but unreviewed and unexecuted.
No A2 job may be submitted until all pre-opening checks pass and the coordinator
confirms the committed A2 lock SHA256 with a decision.

## Failed first preflight: ATT-0374-1

PBS `3779857.opbs` ended with exit 1 after 73 scheduler seconds on mg1101,
actual 2g.24gb. The deterministic suite reported 394 passed and three failed,
all in the new rules tests. It stopped before loading task data or timing.
Measured usage is U-0383: 0.020278 MIG-hour = 0.0050695 node-hour.

The complete `checks.log` and inspected fixture code establish three causes:

- The exact-quantity test asserted a zero lower secondary interval after only
  32 draws. With numpy's linear quantile interpolation that is not guaranteed
  even if one draw is zero. The test now uses the preregistered 10,000 draws
  and explicitly asserts a zero lower bound; the positive point estimate alone
  must still fail the secondary confirmation criterion.
- The fixed-sequence test declared 500/600 correct rows but supplied unrelated
  100/1-row vectors. Production correctly rejected the mismatch. The corrected
  vectors are generated from the declared counts, so it reaches the intended
  fixed-sequence gate with Keep identical to the better control.
- The malformed-length test expected the text `row count`, while the correct
  error is `one-dimensional and match count totals`. It now matches that exact
  validation error.

No production threshold, validation requirement or analysis behavior changed
for these fixes. A fresh PBS preflight is required; no result is assumed.

Replacement preflight RUN-0375 / ATT-0375-1 executes commit
`967f551ede63b8a7f1850bd395f805ada4b9b7ca` as PBS `3779991.opbs`, one
debug-mig instance, 15 minutes. Submission observed it running. Results are
pending under `research/artifacts/ATT-0375-1/`; notification
`N-20261103-030242-b40e` is sealed and notifyd is running. After it ends,
resume the validation and training-source registration/review steps above.

## Validated replacement preflight: ATT-0375-1

The complete manifest and its raw checks.log/XML, plan, initial-digest and cost
outputs were inspected. PBS `3779991.opbs` ran on mg1097_mig2, where the in-job
guard and torch both report 2g.24gb. The suite passed 397 cases, 0 skipped,
0 failures/errors. Four smoke arms reloaded identically after 132 uniform
training steps each, with full parameter gradient coverage and expected state
shapes/accounting. No A2 rows were selected, encoded or scored.

The masking loader reproduces the registered seed 0/1/2 order SHA256 values,
26,230 examples and 13,115 steps each, plus the old exact input-token totals
2,000,069 / 2,005,134 / 2,002,226. Its exclusion audit equals ATT-0306-1;
the pinned 159-byte file's SHA256 is
`b67a489da0a81f8c6c3fcfd112182c9a0e77df5fd468c6d1547166fec83780b5`.

The three new plans are registered in src/seg_data.py before formal training:

| seed | order SHA256 | examples | steps | total full-protocol tokens | segment-2 tokens |
| --- | --- | ---: | ---: | ---: | ---: |
| 3 | 108aec8c1c9684c5e4a58f91da0c5b7aa56e91aab25c7b07d45874e418774b14 | 26230 | 13115 | 2003216 | 567856 |
| 4 | 1c94716a5e449bb711b7135751f6161f062c2456a29065de5e557b144e1f5644 | 26230 | 13115 | 2001319 | 566215 |
| 5 | 6005c84bf0e8610f1dda085736c9fc8f3cdd3aa3d63a0a38211939c5aa22bb23 | 26230 | 13115 | 2003710 | 566084 |

SEG-05/06/07 initial parameter digests agree elementwise within each seed,
with distinct digests across the three seeds. SEG-04 also has distinct
initializations across seeds; its digest remains slice-dependent by design.

Measured usage U-0384 is 0.1 MIG-hour = 0.025 node-hour. Both preflights cost
0.0300695 node-hour. The measured part model projects 4.27048606125148
node-hour before actual smoke usage, or 4.30055556125148 including both
preflights, below the 5.5-node-hour cap. Projected training seconds per arm:
SEG-04 3545.31, SEG-05 7481.36, SEG-06 3523.50, SEG-07 3301.95.
At 1.5x rounded up to 10 minutes, request 3h10m for each Keep job and 1h30m
for each other training job on short-mig. Estimated A2 evaluation is 2356.22s
per seed (future; requires the confirmed lock before submission).

Pinned preflight output SHA256 values:

- manifest.json: ced4505f542913c3ad41cc70d59322af0106317717d6edd37463fa35e2a14203
- seed-plans.json: 6b62b5129e8ddf8676e7dade35aa597f28a1c12f7ef5ef2520927d90ab788f0a
- initial-digests.json: a2a54283da3e1e8ef4c56f0222e465978bc8a2fabfae6703c4cdedff842cfec7
- cost-smoke.json: 265ff9fc95732ba95a056bc1d0eb62a2405365768a45fe5110e68e4f9af1d39f

Formal runs remain unregistered pending the changed-code review and final-source
deterministic checks. A2 remains sealed.

RES-0349 imports the validated successful preflight; U-0384 imports its measured
usage. New order constants are committed at
`61cd97b1881dfe3c6045c425076023f31ca91b03`. Changed-code review RV-0097 has been
requested against E-0027's 054a54a baseline, including the new masked training
path, lock, reader, evaluator, rules, analysis and PBS boundary. Reviewed files
must remain unchanged until its answer.

Candidate-source deterministic checks RUN-0376 / ATT-0376-1 are concurrently
running as PBS `3780105.opbs` (debug-mig, 2-minute walltime from measured
57.5-second suite plus allocation/startup overhead). Outputs:
`research/artifacts/ATT-0376-1/`; notification `N-20261103-031339-bceb`.
The coordinator must resume this role on the review answer or job notification.
Before formal runs, inspect/import these outputs and measured usage, resolve
review findings and rerun affected checks if the candidate source changes.

Candidate checks ATT-0376-1 are now independently validated and imported as
RES-0350. The complete manifest pins `61cd97b` and the correct RunSpec digest;
checks.log and checks.xml agree on 397 passed, zero failures/errors/skips.
The in-job allocation record confirms compute host mg1098 and PBS 3780105;
torch reports the actual 2g.24gb slice. U-0385 records 0.020278 MIG-hour
(73 scheduler seconds), or 0.0050695 node-hour. Total measured task usage
before review/training is 0.035139 node-hour. Adding this check to the earlier
projection conservatively gives 4.30562506125148 node-hour (the earlier
checks/analysis reserve remains included).

ATT-0376-1 manifest SHA256:
`73feb6a959e91be180b05ccf9d47f05bd6892ccea1c89e922488b3ea598cc350`.
checks.xml SHA256:
`2fd857ed4cf97a3e08bc9eaaa9c8928edfdc40eb6dc1342ecbc68c0fa003466a`.
The PBS ledger application verdict is verified_success. RV-0097 is still
pending with reviewer-2; no reviewed file changed, and no formal training or
A2 evaluation has been registered. Resume on that review answer.

## RV-0097 fixes in progress

RV-0097 identified one medium protocol issue and four low findings. MSG-0950
asks the coordinator to choose the protocol for a dev-gate failure after a
peer has opened A2; option (c), an unchanged engineering retry under the same
lock on the training slice, was recommended. No dependent protocol change has
been assumed or implemented while the decision is pending.

Independent fixes are implemented, awaiting PBS checks and review recheck:

- The reader records the confirmation decision D-NNNN. The evaluator passes
  it from PBS; receipt, provenance and manifest must agree, and analysis
  requires the same decision across seeds.
- All A2 rows are encoded before loading/scoring any A2 model. A segmented
  encoding failure records `unencodable_no_model_scored` and stops. The
  earlier E-0013 full-train audit is pinned in the pre-opening lock evidence;
  its only exclusion, PIQA 5528, is checked against the sealed A2 IDs only.
- Lock creation accepts separate training and evaluation/analysis commits,
  while deterministic checks must bind to the evaluation commit.
- PIQA/BoolQ/macro gain, LoRA and swap descriptives, dev gain/LoRA, and a
  pinned E-0027 dev reference comparison now execute inside locked analysis.
- A synthetic orchestration regression verifies that a failed dev gate writes
  a partial manifest and cannot call the A2 reader or write an opening receipt.

Tests and thresholds were not relaxed. No A2 access or formal training occurred.

The independent fixes above are committed at
`909b5f01b7c3e49ff70a6878af42c12927ba8521`. PBS review-fix checks RUN-0377 /
ATT-0377-1 are running as `3780297.opbs`, one debug-mig instance, 2-minute
walltime based on the earlier 73-second checks. They also validate the actual
pinned E-0013 audit and E-0027 dev-reference artifacts in compute. Outputs:
`research/artifacts/ATT-0377-1/`, notification `N-20261103-033130-1992`.
Next: inspect/import these outputs and usage, obtain the coordinator's answer
to MSG-0950, incorporate that exact protocol, and request RV-0097's recheck.

D-0072 has now been read in full and implemented, including its additions.
The runtime classifies cross/same-profile gate failures, exits non-zero without
an A2 receipt, and writes `opened=false`. Cross-profile retries may require the
training profile and exit before dev on another profile. The record-driven
retry limit and same-profile batch cancellation/receipt audit are documented
in `T-0033-A2-runbook.md`; five consecutive profile misses stop for coordinator
action. The lock pins D-0072 and its failure policy. The same evaluator now
supports `--dev-replay-only` without unsealing or confirmation, for all three
actual final checkpoints before the coordinator receives the lock. It reports
cross-profile discrepancies without a new threshold; same-profile discrepancies
remain an implementation failure. Regression tests cover both failure classes,
retry early exit, and dev replay without any A2 read.

The earlier review-fix checks ATT-0377-1 were validated: 410 passed, zero
skipped/failures/errors; actual torch 2g.24gb on mg1103, confirmed compute
allocation. Its pinned E-0013 audit and E-0027 reference also validated.
Measured usage is 0.02 MIG-hour = 0.005 node-hour, making total measured
task usage 0.040139 node-hour before D-0072 final-source checks/training.
The new required real-checkpoint dev replays add an estimated 0.1 node-hour;
the conservative full projection is about 4.41 node-hour before the next check,
within the 5.5 cap. Per MSG-0954, run final-source PBS checks before requesting
RV-0097's recheck; no formal training until that review passes.

The D-0072 implementation is committed at
`f7419525d12ff511845a11fae7ae0fae5100e8f7`. Required final-source PBS checks
RUN-0378 / ATT-0378-1 are running as `3780390.opbs`, debug-mig, 2-minute
walltime (earlier full suite 72-73 seconds). Outputs:
`research/artifacts/ATT-0378-1/`; notification `N-20261103-034013-d893`.
Prior ATT-0377-1 is imported as RES-0351 with measured usage U-0386. The
RV-0097 recheck will be requested only after these D-0072 checks pass and
their actual outputs are inspected, as MSG-0954 requires.

ATT-0378-1 is now validated and imported as RES-0352, with measured usage
U-0387 (0.02 MIG-hour = 0.005 node-hour; 72 scheduler seconds). The exact
candidate f741952 passed 415 cases with zero failures/errors/skips. The
manifest/RunSpec/source pins match, torch reports actual 2g.24gb on mg1097,
and its in-job allocation was confirmed. The E-0013 and E-0027 pinned metadata
checks also passed. Total measured task usage is 0.045139 node-hour; the
conservative full projection including the D-0072 dev replays is about
4.415625 node-hour, within 5.5.

Manifest SHA256:
`a02bf27b61b31fc85c0319d7a6a1c4c0b74f2e1becc4029dcf6ed9b8d5105d63`.
checks.xml SHA256:
`fccdf11ea5e5366f512e28523e3b92c838a31fbadfb5a9e01cbc4850337c2cd8`.

RV-0098 (the second and final review in RV-0097's round) has been requested
and assigned to reviewer-2. It covers all five findings, D-0072 and the A2
runbook. Reviewed files are frozen. Next: inspect that answer, obtain the
coordinator's disposition if medium findings remain, and only after a passing
review register the twelve formal training runs at the covered source.
A2 remains sealed, and no formal training has been registered.

## Formal training registration and analysis-only follow-up

RV-0098 passed at f741952. Its one remaining low finding is being fixed in a
separate analysis-only change under MSG-0962; the training path stays at the
reviewed immutable f741952 snapshot. The new audit examines every accounted
attempt's manifest/receipt metadata, rejects duplicate complete openings for
a seed, permits and lists partial opened engineering attempts, and discovers
lock-matching dev-replay/profile-miss attempts omitted from accounting. Two
required synthetic cases plus accounting regressions are added. This change
requires its own PBS checks and a new-round review before lock creation.

The twelve formal RunSpecs were prepared as RUN-0379 through RUN-0390.
Their initial attempts inherited stale lineage warnings despite the current
files having passed other reviews. The coordinator resolved the exact version
coverage in MSG-0964: add explicit reviews RV-0081, RV-0082, RV-0088,
RV-0092, RV-0094 and RV-0095. Replacement attempts retain the same immutable
RunSpecs and are checked as covered. The initial, unsubmitted preparations
are recorded cancelled with no PBS job/allocation/workload; they are distinct
from executed experiment failures. The authoritative submission table is
`/work/xg24i002/x10041/bypass-lora/research/artifacts/T-0033/training-submissions.json`.
All twelve checked submission plans use short-mig, one instance each, 3h10m
for Keep and 1h30m for the other arms, with one expected-12 notification group.

All twelve covered replacement attempts were submitted at f741952 as jobs
3780692/3780693/3780695/3780697/3780698/3780700/3780702/3780703/
3780705/3780711/3780718/3780721.opbs. Notification
`N-20261103-040731-ab87` (`T0033-formal-training`) is sealed with 12 members
and early failure enabled; notifyd is running and its listener was sent to the
coordinator in MSG-0965. The latest observation at 04:11 JST found eleven
running, with seed-5 SEG-07 (3780721) queued for Qlist resources. Initial and
start observations are imported; running/queued usage remains pending.

The twelve superseded preparations have no PBS job or workload. U-0388
collectively reports their known zero allocation usage as `reported`, distinct
from measured scheduler usage. They have no execution manifests; executable
attempt accounting uses the submitted -2 attempts. All preparation cancellation
records remain in the research records. Exact mapping and immutable RunSpec
digests are in `research/notes/T-0033-training-runs.json` and the central table.

Analysis receipt-audit fix candidate:
`3ae19740b37cd5db0813c5700a818b16c216c4bc`. It changes only the analysis,
its new metadata-only audit/tests, and the test list; training remains exactly
f741952. The audit also accounts for a previous failed analysis attempt
without misclassifying it as a seed opening, so unchanged engineering reruns
can still complete after an analysis failure.

Required PBS checks RUN-0391 / ATT-0391-1 are running as `3780768.opbs`,
debug-mig, 2-minute walltime. Output `research/artifacts/ATT-0391-1/`;
notification `N-20261103-041429-30ec`. After checking/importing these outputs
and measured usage, request the coordinator-authorized new review round with
`--recheck-of RV-0098 --new-round` for the analysis-only fix, then keep those
files frozen. No lock is written until that review passes and all twelve
formal runs have valid final results. Both notification listeners belong to
the coordinator; training notification N-20261103-040731-ab87 remains active.

ATT-0391-1 is validated and imported as RES-0353, usage U-0389 (0.02
measured MIG-hour = 0.005 node-hour; 72 scheduler seconds). At exact source
3ae1974, all 421 tests passed with no failures/errors/skips. The manifest,
RunSpec/source pins and checks.log/XML agree; the in-job allocation confirms
mg1099 and torch reports actual 2g.24gb. Pinned prior audit/reference checks
also passed. Completed diagnostics total 0.050139 node-hour; formal training
usage is still pending. Administrative unsubmitted preparations remain reported
zero under U-0388. The conservative full projection is about 4.42 node-hour.

Manifest SHA256:
`12bc7858103d57faea8b99fee1679de7e3e0d1f96f92d67a05968bd50fe18173`.
checks.xml SHA256:
`512d2f8586fdc93be3dcf5b5887804d7ee2185e19ec872a93758f731ccc74408`.

The coordinator-authorized new review round RV-0099 has been requested with
`--recheck-of RV-0098 --new-round` and assigned to reviewer-2. Its candidate
is 3ae1974; the four analysis/audit/test-list files are frozen. The training,
A2 reader/evaluator, checkpoint validation and lock-construction paths have
an empty diff from the passed training source f741952. Resume on RV-0099's
answer or the formal-training notification. Seed-5 SEG-07 remains under the
coordinator's queue observation (MSG-0968 estimated start 04:56 JST); no queue
or resource change is requested.

## RV-0099 hard-termination fix

RV-0099 identified a concrete medium finding: a walltime/host kill need not
execute Python finally, so a measured attempt can have no manifest, including
an A2 attempt that already wrote its receipt. Rejecting every missing manifest
would prevent a locked, unchanged engineering rerun from completing analysis.

The audit now keeps such declared attempts. Without a manifest or receipt,
it records `no_manifest_unopened`, seed/status unknown and manifest reference
null. A matching receipt without a manifest is recorded as
`partial_opened_no_manifest`; it counts as a non-selected opening, and one
complete selected result is still required for every seed. Known receipt
seed, confirmation decision, D-0072 and time are validated. The last truncated
line in a manifest-less directory is conservatively an opening: only completed
JSON key/value pairs are recovered; missing seed/decision/time remain unknown
and unattributed partial openings are explicitly listed. No record invents
those fields. A matching or unattributable truncated receipt in an unaccounted
directory still rejects the analysis. Complete-manifest receipts and non-final
malformed lines remain strict.

Regression cases replace the incorrect missing-manifest rejection and cover
absent/empty directories, receipt-only partial openings, missing complete
results, undeclared receipts, truncated known/unknown-seed receipts, invalid
known metadata and strict complete/non-final receipt validation. Only the
analysis audit and its tests change; training remains f741952. Next: PBS
checks at the fix commit, then the second review of RV-0099's round before lock.

Hard-termination fix candidate:
`4e6e8dfad40feb35fab60eecf7cf2df428083b1d`. Required full exact-source PBS
checks RUN-0392 / ATT-0392-1 are running as `3780944.opbs`, debug-mig,
2-minute walltime. Outputs `research/artifacts/ATT-0392-1/`; notification
`N-20261103-043626-0ff8` is sealed and notifyd is running. The coordinator
owns the listener and resumes this role on completion. After validating and
importing the outputs and measured usage, request RV-0099's recheck. No
training, evaluation, lock or primary-statistic path has changed in this fix;
the twelve formal runs remain at f741952 under N-20261103-040731-ab87.

ATT-0392-1 has been validated and imported as RES-0354, with usage U-0390
(0.02 measured MIG-hour = 0.005 node-hour; 72 scheduler seconds). Exact
candidate 4e6e8df passed 434 tests, zero failures/errors/skips, actual torch
2g.24gb on confirmed allocation mg1099. Source/RunSpec pins and log/XML agree;
the prior metadata reference checks also pass. Completed diagnostic usage is
0.055139 node-hour; formal-run usage is still pending. The conservative full
projection is about 4.43 node-hour before future measured adjustments.

Manifest SHA256:
`4a05664cece7d733d3c9eba92738f6309bbe241a367ae678284a31c3a5f3992f`.
checks.xml SHA256:
`6e0969ea19c8aff98894a8a744be8476f4845e8fd24132f31d4096b4e080cd4f`.

RV-0100 is requested and assigned to reviewer-2 as the second and final
review in RV-0099's round. The four scoped analysis/audit/test-list files
remain frozen. Resume on its answer or a formal-training notification. A
remaining medium finding after this second review needs the coordinator's
disposition. No lock or A2 evaluation before passing analysis review and valid
completion of all twelve trained checkpoints.

RV-0100 has now passed with zero findings at 4e6e8df. The final fixed sources
are training `f7419525d12ff511845a11fae7ae0fae5100e8f7` and evaluation/analysis
`4e6e8dfad40feb35fab60eecf7cf2df428083b1d`. Exact evaluation-source checks are
ATT-0392-1 / RES-0354 (434 passed); the seed-plan/cost smoke is ATT-0375-1 /
RES-0349. Existing review references RV-0081/82/88/92/94/95 must also be passed
on registrations to include the inherited versions verified in MSG-0964.

Prepared only the lock INPUT, not the lock itself:
`research/artifacts/T-0033/preopening-input.json` (central project), copied
to `research/notes/T-0033-preopening-input.json` for the continuation. It pins
the completed diagnostic/smoke metadata and all twelve expected final run
roots, without claiming their adapters/results are complete. Do not execute
lock creation until all twelve formal outputs have been checked. Then run
stage=lock at 4e6e8df against this input, inspect the generated lock and rule-1
checks, and run three stage=dev-replay jobs (seeds 3/4/5) using the generated
lock path/SHA without unseal. Report that unchanged lock and replay results
to the coordinator for its confirming decision. A2 is still sealed.

2026-11-03 06:38 JST: all twelve formal training jobs ended with PBS 0.
Their complete manifests report 13,115 steps, 26,230 examples, finite final
metrics, the registered seed orders, full gradient coverage and identical
same-job reloads. Keep initial/Base and trained off/Base checks all pass.
Keep/Reset/T0 initial adapter digests match within each new seed. All three
legacy SEG-04 jobs actually ran on 2g.24gb; no repeat or profile change occurred.
Keep seeds 4/5 and T0 seed 5 actually ran on 1g.24gb, despite the ledger's 2g
labels; Keep seed 3 and the other eight jobs ran on 2g.24gb. Use manifest torch
profiles for the D-0072 gates and usage annotations.

Measured training usage is 16.405555 MIG-hour = 4.10138875 node-hour, recorded
as U-0391 through U-0402. Completed training plus diagnostics is 4.15652775
node-hour. Control metadata is in central T-0033/training-completion-jobs.json,
training-observations-ended.jsonl and training-manifest-summary.json.

The exact reviewed-source PBS lock audit RUN-0393 / ATT-0393-1 completed as
3781694.opbs on confirmed allocation mg1103. Application status is complete,
validator E0030_preopening_lock passed, preopening.passed=true and
confirmation_evaluated=false. This runtime audit authenticated all twelve
artifact inventories/checkpoints, finite step ledgers, schedules, parameter
and token budgets, three curve snapshots, initial/reload/off identities and
training/A2-ID disjointness. Lock source is 4e6e8df, training source f741952.
Lock: central research/artifacts/ATT-0393-1/a2-lock.json, SHA256
`cfd2e9b054c26da3d43ed22dbf0c410060599d6599d5fe88c4b8cfe1b78838c5`.
The lock and digest files are now read-only. D-0072 and all fixed rules are
present. Do not report it as ready for confirmation until the three real
Keep checkpoint dev replays complete under this unchanged lock/source.

The updated cost projection is committed separately as
T-0033-cost-projection-after-training.json. It uses the worst measured
eval/reload-to-smoke ratio rounded up to 1.30 for future A2 timing, and actual
maximum Keep replay time plus 60 seconds overhead. Full expected cost is
4.903822 node-hour. Full cost if all remaining requested walltimes are used
is 5.323194 node-hour (dev 3x12 min, A2 3x80 min, lock/analysis 2 min each).
Both are below 5.5, and neither assumes free engineering retries. Recompute
from measured cumulative usage before every subsequent submission/retry.
MSG-0982 reports this interim accounting and actual profiles to coordinator;
A2 remains sealed. No A2 evaluation has been submitted.

All twelve validated training results are imported as RES-0355 through
RES-0366 (arm/seed order in central T-0033/training-imports.json); lock result
is RES-0367. Lock terminal observation is recorded, usage U-0403 is measured
0.004722 MIG-hour = 0.0011805 node-hour (17 seconds). Completed cost including
lock is now 4.15770825 node-hour. Replacing the old lock estimate with this
measurement reduces full expected cost to approximately 4.900142 node-hour,
and the full future requested-walltime bound to 5.316042 node-hour.

D-0072's three actual-checkpoint, dev-only replays are now running at 4e6e8df
under the exact unchanged cfd2e9b0 lock, without unseal, confirmation decision
or training-profile restriction:

| Seed | Attempt | PBS job |
| --- | --- | --- |
| 3 | ATT-0394-1 | 3781728.opbs |
| 4 | ATT-0395-1 | 3781729.opbs |
| 5 | ATT-0396-1 | 3781730.opbs |

All registrations are covered by the inherited reviews plus RV-0100. Each
job requests debug-mig/select=1 for 12 minutes, from 471 expected seconds.
Central T-0033/dev-replay-submissions.json, dev-replay-jobs.json and initial
observations preserve exact spec pins and submissions. One notification group
`N-20261103-064154-f0f6` is sealed, expected=3, early_failure=true; the
coordinator owns its listener. On resumption, check each complete application
manifest, unchanged source/lock/spec and opened=false/no receipt; import all
outputs and measured usage. Report each seed's exact flips and score-row
identity with actual allocated and training torch profiles, explicitly noting
whether the cross-profile <=10 gate would pass (MSG-0983). Cross-profile
replay discrepancies remain descriptive. Same-profile discrepancy needs
repair/review/relock before confirmation. Only then report the completed lock
with the three replay outputs and updated cost to coordinator and await its
confirming decision. Do not submit an A2 evaluation first.

2026-11-03, after MSG-0986/0987: all three dev-only replays were independently
checked against their source/spec/lock pins, confirmed compute allocations,
reported dev check records and task counts. All are complete/unopened, with
no receipt. Seed 3: mg1099, 2g allocated / 2g trained, zero flips, exact rows.
Seed 4: mg1104, 2g / 1g, 18 flips and non-identical rows. Seed 5: mg1104,
2g / 1g, 25 flips and non-identical rows. The latter two exceed the original
cross-profile gate on these replays, but cross-profile replay results are
descriptive under D-0072(5). Actual 1g same-profile checkpoint reproduction
has not yet been shown for either seed.

Results are RES-0368/0369/0370 and measured usages U-0404/0405/0406, respectively
0.091389/0.103333/0.106389 MIG-hour. Total replay cost is 0.07527775 node-hour,
and completed measured cost is now 4.232986 node-hour. A registration mistake
is preserved explicitly: I registered the replay RunSpecs with arm dev-replay
but the source emits arm A2, so RL retains the imports with mismatch warnings
and they cannot serve as pre-specified comparisons. The actual checkpoint,
source, seed, mode, stage and gate checks remain independently pinned
diagnostics. Do not rewrite historical specs or producer outputs; register
future A2 runs with arm A2 exactly. No scientific A2 comparison uses replays.

The completed handoff is T-0033-A2-lock-handoff.md, with the refined accounting
in T-0033-cost-projection-after-replays.json (copies/pinned metadata centrally
under research/artifacts/T-0033). EV-0173 records diagnostic/valid engineering
evidence only, with no support/challenge link to H-0029, and retains the arm
mismatch warnings. The full expected cost without misses is 4.877378;
successful-job requested-walltime bound is 5.245486 (three 80-minute A2 jobs,
analysis 90 expected / 180 requested seconds). Twenty misses per seed at one
to two minutes each adds 0.25–0.5 node-hour across all three seeds; the timing
projection fits but the full requested-walltime upper bound may exceed 5.5.
With all successful-job walltimes reserved, current miss headroom is 61.083
MIG-instance minutes total. Recompute the budget gate before every retry.

MSG-0986 proposes first-attempt profile restrictions and a twenty-miss limit.
The reviewed code/PBS script can do that without source changes, and analysis
accepts many partial unopened retry_profile_miss attempts (all must have
measured terminal accounting, within the global 10,000-directory audit cap).
But D-0072 and the frozen lock say unrestricted first attempt / five misses;
an actual confirming decision must explicitly supersede those operational
fields if retaining the lock hash. Also, only one initial 1g seed job should
be submitted if the intent is to catch its reproducibility failure before
any A2 read; simultaneous seed 4/5 jobs would not ensure that. One successful
1g seed still does not pre-prove the other seed. The later consumed/invalid
rule stays active. No A2 submission before the coordinator's actual decision.

D-0073 has now been read in full and confirms the unchanged cfd2e9b0 lock.
It explicitly supersedes D-0072's first-attempt/five-miss operational rules,
without changing the science or source. Runbook now reflects its every-attempt
profile restriction, twenty-miss reporting point, budget gate and ordering.

Registered RUN-0397 / ATT-0397-1, arm A2 exactly, seed 4, source 4e6e8df and
all required reviews covered. Spec digest:
`blake2b:fb2a1939822a78157e9013ed95034ff4596d2b3e8228111bdc6cc95175eb31ae`.
Submitted only this seed as PBS `3781926.opbs`, short-mig/select=1/group
xg24i002, 80-minute walltime from 3,064 expected seconds. Immutable script
sets SEG_REQUIRE_TRAINING_PROFILE=1 and SEG_CONFIRMATION_DECISION=D-0073.
The pre-submission gate passed: measured ended cost 4.232986 + three 80-minute
uncompleted-seed requests + three-minute analysis request = 5.245486 node-hour.
Gate artifact: central T-0033/a2-budget-ATT-0397-1.json.

One authorized job_wait (45 seconds) returned queued, no outcome or usage;
do not label it running or claim a profile/dev gate pass. End/early-failure
notification `N-20261103-071757-fb32` is sealed for this exact attempt, and
notifyd was restarted successfully on miyabi-g2. Coordinator owns the listener.
Outputs will be central research/artifacts/ATT-0397-1. Central
T-0033/a2-execution-state.json preserves source/decision/lock, logical run,
each attempt, measured cost and consecutive miss counts. Seeds 3 and 5
remain held/unregistered; seed 4's consecutive misses are still zero.

On wake, inspect the actual application outcome and in-job hardware before
importing. A profile miss must be partial/unopened, have no dev/A2/receipt,
and be counted/imported with measured usage; then register another attempt
of RUN-0397 with the exact saved RunSpec, unchanged source/lock/checkpoints/
rules/decision, recomputing the D-0073 budget first. Stop/report at twenty
consecutive misses or an earlier budget failure. For a running/queued
attempt, do not submit another of the same seed. On same-profile dev failure
or A2 encoding failure, follow the D-0072 stop/receipt audit rules. Only when
seed 4 is COMPLETE, dev rows exact and receipt/provenance valid, tell
coordinator and submit seeds 5 (1g) and 3 (2g), without awaiting a reply.

MSG-0993 / seed 4 completion: independently verified metadata only, without
reading A2 condition scores. ATT-0397-1 is complete/validator passed on
confirmed mg1109/3, actual torch 1g.24gb. The dev gate is exact, zero flips,
same training hardware; off equals Base, all seven conditions and 2,000 A2
rows are reported, and encodability passed 1,000 per task. Exactly one receipt
and provenance bind seed 4, job 3781926.opbs, the unchanged lock and
confirmation D-0073 (opening protocol marker D-0072). This also demonstrates
seed 4's 1g same-profile reproduction across training mg1111 and evaluation
mg1109; it does not establish the still-unrun seed 5 gate.

Imported RES-0371 (arm A2 matches the RunSpec, no arm mismatch) and U-0407:
0.606111 measured MIG-hour = 0.15152775 node-hour, from scheduler walltime
00:36:22. Usage is annotated with actual 1g rather than the ledger's 2g label.
Central a2-execution-state.json now selects ATT-0397-1 for seed 4 and releases
seeds 5/3. Completed measured task cost is 4.38451375 node-hour. Gate artifact
T-0033/a2-budget-after-seed4.json reserves 80 minutes each for uncompleted
seeds 3/5 plus three-minute analysis: 5.0636804167 node-hour, passed.
The independent metadata check is ATT-0397-1-verified-metadata.json.

The coordinator is being told the verified seed 4 outcome before the two
authorized next submissions. A2 has now been read; source, lock, checkpoints,
rules and confirmation decision remain frozen. Any subsequent same-profile
failure is handled under D-0072's consumed/invalid rule, without repairs or
relocking. Analysis and score inspection remain for the locked PBS analysis
after all three successful seed outputs are available.

Phase 2 submitted after the verified seed 4 success, as authorized in D-0073:

| Seed | Run / attempt | PBS job | Notification |
| --- | --- | --- | --- |
| 5 | RUN-0398 / ATT-0398-1 | 3782396.opbs | N-20261103-081932-57c6 |
| 3 | RUN-0399 / ATT-0399-1 | 3782400.opbs | N-20261103-081937-c713 |

Both were observed queued at submission, not yet profile/dev-gate checked.
Each is short-mig/select=1, 80-minute walltime, source 4e6e8df, arm A2 exactly,
same unchanged cfd2e9b0 lock and confirmation D-0073, with training-profile
restriction enabled. Required profiles: seed 5 = 1g, seed 3 = 2g. Each has its
own sealed one-job notification with early_failure=true. notifyd is running;
coordinator owns both listeners. MSG-0994 reports independently verified seed
4 completion/import and the passing budget before these submissions.

Both pre-submission budget artifacts reserve the queued/running attempt
within the 80 minutes for its uncompleted seed, without double counting:
4.38451375 measured ended + two A2 requests + three-minute analysis =
5.0636804167 node-hour. The seed 5 job was queried and still queued before
the seed 3 gate/submission. Central a2-phase2-prepared.json,
a2-phase2-submitted.json and per-attempt submission/budget/observation files
preserve exact pins and events. Source-only reusable RunSpecs are
a2-run-RUN-0398-spec.json and a2-run-RUN-0399-spec.json. All three consecutive
profile-miss counts remain zero; no seed has more than one active attempt.

Resume on either notification. Import actual terminal partial/complete
manifests and measured usage, updating a2-execution-state.json before any
further submission. Profile misses alone may retry unchanged after the budget
gate; at twenty misses or a failed gate, stop/report. Same-profile or encoding
failures require the D-0072/D-0073 consumed/invalid handling; cancel only
unstarted peer A2 jobs, let running jobs finish, then audit all receipts.
When all three seeds have valid complete outputs, run the frozen analysis at
4e6e8df under this lock with every executed task attempt and measured cost,
including all diagnostics, replays, lock and any misses/failures. No source,
checkpoint, comparison or rule changes after the first A2 read.
