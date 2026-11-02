# Current checkpoint: eta locked; X-01 interventions running

All 14 initial formal jobs ended successfully and their application manifests
passed. All 13 training arms completed 13,115 steps / 26,230 examples and
same-job export/reload checks. Native terminal observations and usage are
imported once from T-0031-formal-terminal-observations.json. All result imports
are in T-0031-initial-result-imports.json (RES-0303..0317, excluding RES-0312,
which is the eta plan). No Part B dev aggregate was read before the lock.

Calibration-only plan: ATT-0342-1 / PBS3776771 / RES-0312. All 12 calibration
producers passed source/init/budget/row checks. Neither tap triggered an
endpoint extension, so NO additional training is authorized by rule 5.

Final calibration-only lock: ATT-0343-1 / PBS3776823 / RES-0318.
- QK eta*=0.25 (RB1-09).
- X eta*=1 (ETA-X-1).
- Neither is an extended boundary.
- Lock: /work/xg24i002/x10041/bypass-lora/runs/ATT-0343-1/eta-lock.json
- SHA256: e6c1399c683b50cabb05c727e37df84be165109a2ae4dd1ba5bcf93d3349fe60
- The hash and choices are recorded in T-0031-eta-lock-record.json BEFORE
  any Part B dev aggregation. Rule 5 screening evidence has been recorded
  separately; do not create a duplicate rule 5 evidence item later.
- Plan and lock each used 2 scheduler seconds; native usage is recorded in
  T-0031-eta-plan-observations.json and T-0031-lock-intervention-observations.json.

X-01's six frozen interventions are RUN-0344 / ATT-0344-1 / PBS3776824.opbs,
running on verified compute allocation mg1109. Scientific source remains
exactly 0f22cd60f45674ec20b800fdd455f80c2b6d46dd; passing RV-0088 and the
scoped inherited coverage release D-0061 apply. Requested 50 minutes from
the measured ~1,850-second estimate. Output:
 /work/xg24i002/x10041/bypass-lora/runs/ATT-0344-1/
Result and terminal usage are pending, not assumed zero.

One sealed notification, expected 1:
N-20261102-214849-c9e1 / T0031-X01-interventions, notifier miyabi-g2.
The daemon was ensured running (PID3129003). Coordinator MSG-0881 has:
 /work/xg24i002/x10041/miyabi-toolbox/miyabi-mcp/bin/miyabi-mcp notify listen --id N-20261102-214849-c9e1 --notifier miyabi-g2
Coordinator owns waiting and restarts this role. Do not poll in a loop.

On wake:
1. Verify PBS3776824 and the application manifest/conditions/provenance.
   Import native terminal observations and usage into NEW note file
   T-0031-intervention-terminal-observations.json, then import its result.
2. Final analysis input is already prepared in T-0031-analysis-input.json.
   It points to the immutable eta-lock/hash, all 13 training producers,
   baseline ATT-0335-1, X-01 interventions ATT-0344-1, registration/tests
   ATT-0315-1, and all cost exports. It intentionally names the pending
   intervention terminal export; create that file before running analysis.
3. Register a final analysis run before execution with resource miyabi,
   exact source0f22cd6, reviewed T-0031-run.pbs, TAP_JOB=analysis and
   TAP_ANALYSIS_INPUT=<absolute path to T-0031-analysis-input.json>. Explicit
   reviews RV-0076/77/78/79/80/81/82/88 and release D-0061 remain applicable.
   Source files and original launchers have not changed.
4. Execute via PBS. The reviewed analysis validates lock reconstruction,
   final checkpoints/provenance, step0/Base gates, off/Base, reloads, counts,
   budgets/order/LR, curves, RMS, and all exact preregistered thresholds. It
   produces paired10k-bootstrap descriptive intervals and rule outcomes.
5. Record the remaining rule-by-rule evidence (rule1 diagnostic; other
   rules screening; rule5 already recorded), actual total cost including
   final analysis, limitations, and deliver. Do NOT set hypothesis status,
   stop a route or release new seeds: coordinator owns conclusion review.

Before intervention launch, the native ledger reported 15.468888 MIG-hours
for the 17 initial/preformal jobs, well below the 22-MIG-hour equivalent of
the 5.5-node-hour cap. Selection adds only 4 measured seconds. The original
worst-case estimate5.207572029node-hours reserved TWO endpoint trainings,
which were not needed. Still include every check, selection, intervention
and final analysis in actual usage. Retain all checkpoints until acceptance.

Older initial-grid/reference details follow for provenance.

# T-0031 continuation: initial formal grid submitted

Task T-0031 owns E-0028@1 (H-0021@1, H-0022@2). Researcher-1 executes;
coordinator owns route/status decisions. Branch sr/researcher-1/T-0031.
Local shell is login/control only; ALL tests/evaluation/analysis run through
miyabi-mcp PBS. No local workload fallback, merge, rebase or push.

Exact scientific source for EVERY formal training, baseline, intervention,
selection and final analysis:
0f22cd60f45674ec20b800fdd455f80c2b6d46dd.
Its read-only snapshot already exists. Do not use later note-only HEAD commits
for scientific jobs: init registration/tests and selection require exact0f22.
Frozen environment envs/3c916b51337d-cuda, cuda/12.8, one committed uv.lock.

RV-0088 passed with zero findings. D-0060 defines hardware-aware strict
reference gates (pinned on accepted2g; same allocation across profiles).
D-0061 explicitly authorizes remaining inherited stale flags: unused sink
diagnostic/test files and test_rb1_artifacts; bypass_model is byte-identical
to RV-0078/0079's accepted version. All remaining registrations name
RV-0076/77/78/79/80/81/82/88. ATT-0321-1 predates adding78/79; D-0061 explicitly
covers it too. Release rationale is recorded per attempt in
T-0031-release-observations.json. No code was changed to silence a flag.

All13initial new training arms PLUS RB1-09 dev/calibration evaluation are
registered and SUBMITTED. Exact attempt/job/spec/log map:
T-0031-formal-jobs.json. Complete registration inventory:
T-0031-formal-registrations.json. Request keys are sr:<attempt>; reconcile an
uncertain submission using that key rather than launching a duplicate.

| Arm | Attempt | PBS job |
|---|---|---|
| X-01 | ATT-0321-1 | 3775250.opbs |
| X-02 | ATT-0322-1 | 3775251.opbs |
| X-03 | ATT-0323-1 | 3775254.opbs |
| ETA-X-0.0625 | ATT-0324-1 | 3775255.opbs |
| ETA-X-0.125 | ATT-0325-1 | 3775257.opbs |
| ETA-X-0.5 | ATT-0326-1 | 3775258.opbs |
| ETA-X-1 | ATT-0327-1 | 3775260.opbs |
| ETA-X-2 | ATT-0329-1 | 3775262.opbs |
| ETA-QK-0.0625 | ATT-0330-1 | 3775264.opbs |
| ETA-QK-0.125 | ATT-0331-1 | 3775265.opbs |
| ETA-QK-0.5 | ATT-0332-1 | 3775266.opbs |
| ETA-QK-1 | ATT-0333-1 | 3775267.opbs |
| ETA-QK-2 | ATT-0334-1 | 3775268.opbs |
| RB1-09 eval | ATT-0335-1 | 3775269.opbs |

ATT-0328-1 is NOT part of this task. Initial native scheduler observations
(queued/running only, no metrics) are imported from
T-0031-formal-initial-observations.json. Last export had X-01, X-02 and baseline
running, other trainings queued; status will change. Application results and
terminal usage are pending, never zero by assumption.

ONE notification covers all14jobs, sealed:
N-20261102-185642-d15b, name T0031-initial-grid, notifier miyabi-g2.
Daemon ensure reports running; completion/early failure reaches coordinator.
Coordinator was sent the listener:
 /work/xg24i002/x10041/miyabi-toolbox/miyabi-mcp/bin/miyabi-mcp notify listen --id N-20261102-185642-d15b --notifier miyabi-g2
Coordinator owns waiting and restarts this researcher. Do not poll in a loop.

Cost cap is5.5node-hours. PBS audit ATT-0316-1/RES-0289 passes:
initial grid4.542691585, worst with both endpoint trainings5.207572029nodeh.
Planning uses comparable2g timings from ATT-0314-1, reserves same-allocation
Base for EVERY training, baseline legacy dev, both smokes, interventions,
up to2extensions and120s other checks/selection/analysis. See T-0031-cost.md
and runs/ATT-0316-1/cost.json. Retries and extra work must still fit the cap.
MIG usage converts by native instance-hours*0.25. Import executor usage once
per accounting_ref; no invented per-run split or usage duplication.

Preformal records:
- ATT-0314-1/RES-0283:99tests plus init/timing/full calibration smoke on2g.
  678scheduler seconds, native0.188333MIGh. EV-0139 diagnostic.
- ATT-0315-1/RES-0284 at0f22:112tests, all init/tap/config/count checks and
  32-row same-job accepted/new-source GPU comparison passed on actual1g.
  696seconds, native0.193333MIGh. Ledger generic2g label is incorrect; both
  nvidia-smi and torch reported1g. All4representative init predictions match
  same-allocation Base. No calibration file for X-02/X-03.
- ATT-0316-1/RES-0289: deterministic PBS cost/reference audit passed onmg1097,
  1scheduler second/native0.000278MIGh. EV-0140 diagnostic covers revised
  smoke+audit. All three terminal observations/usage are already imported.

When initial jobs finish:
1. Re-establish context; inspect exact scheduler jobs and application validity,
   import native terminal observations/usage, and import producer manifests.
   Register every new attempt BEFORE execution. Keep engineering failure
   separate from scientific signal. Only explicit nonfinite TRAINING ETA
   candidates are excluded by rule5; OOM/eval/infrastructure failure requires
   resolution, not silent exclusion.
2. NEVER aggregate or read/report PartBdev metrics before eta-lock. Native
   scheduler records contain no dev metrics. Do not run rl run show E-0028
   as an aggregate before lock. Importing manifests into records is permitted;
   selection code reads only calibration fields/predictions.
3. Build calibration-only input JSON in research/notes:
   {manifests:[absolute final manifest paths], training_source_commit:0f22...,
    registration:/work/xg24i002/x10041/bypass-lora/runs/ATT-0315-1/tap-registration.json}.
   Initial12producers are X-01, ten ETA arms and baseline ATT-0335-1.
   X-02/X-03 have no calibration and are excluded from this input.
4. Register/run reviewed T-0031-run.pbs with TAP_JOB=eta-plan and
   TAP_SELECTION_INPUT=<absolute input JSON> in PBS. If rule5 requests an
   endpoint, register/train ONLY that endpoint per tap (at most one, eta1/32
   or4), same source/seed/LR/budget. Check actual cost before releasing it.
   All endpoint init configs/digests were included in registration.
5. Once required endpoints complete, run TAP_JOB=eta-lock with all candidate
   manifests. Writer uses exclusive creation, records hashes and selects
   accuracy on4096calibration rows with exact tie/extension/nonfinite rules.
   Import lock, record its SHA256 and provenance BEFORE any PartBdev aggregate.
6. Run X-01's6frozen interventions via TAP_JOB=intervention,
   TAP_TRAIN_RUN=/work/xg24i002/x10041/bypass-lora/runs/ATT-0321-1 and
   TAP_TRAIN_SOURCE=0f22... . Can overlap endpoint work after producer checks.
   Queue/walltime from measured cost, same0f22 source. RB1-09 interventions
   are inherited (preregistered swap drop1.95pp).
7. Prepare final analysis input for rb_tap_analysis.py at0f22, using:
   eta_lock/eta_lock_sha256, calibration_manifests, training_source_commit,
   training_runs(all13+required endpoints incl explicitly excluded candidates),
   baseline_eval=ATT-0335-1, registration=ATT-0315-1/tap-registration.json,
   smoke_manifest=ATT-0315-1/manifest.json, intervention_run,
   base_run=ATT-0024-1,
   reference_runs={RB1-07:ATT-0245-1,RB1-09:ATT-0247-1},
   observations=[all task native exports incl3preformal jobs].
   Register/run TAP_JOB=analysis with TAP_ANALYSIS_INPUT in PBS.
8. Record one evidence item per sealed rule (implementation diagnostic, others
   screening), honest finite/budget/reload/gate outcomes, bootstrap10k seed
   20261101, limitations and measured total cost. No hypothesis status/route
   change: coordinator owns conclusion review. Deliver only after all required
   results/checks/evidence are complete.

Source scripts and constraints:
- sealed contract /work/xg24i002/x10041/bypass-lora/research/experiments/E-0028.md;
- constraints at project/docs/constraints.md; no A2 confirmation or official
  validation rows may be opened;
- init A=randn/sqrt(fan_in), Cu=0, R=I; only eta scalar differs within tap;
- full13115steps/26230examples/2000069input tokens/387653loss tokens;
- R1 seed0/LR0.001, locked order33759bc20322d0c9981128dd8574438546fc3439194ab32b7c79e35e0f0fc2bf.
