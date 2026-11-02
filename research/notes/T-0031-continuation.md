# T-0031 continuation

## Current checkpoint after RV-0087

The first smoke completed successfully (RES-0283), all99tests and registration
passed, and native terminal usage0.188333MIGh is imported exactly once.
Diagnostic EV-0139 records only this pre-review engineering result. See
`T-0031-cost.md` for observed timings and the initial4.36–5.00node-hour table.

RV-0087 answered issues (1medium,4low). Coordinator decision **D-0060**
authorizes the reviewer-proposed reference interpretation of sealed rule1:
same2g profile keeps strict pinned predictions; another profile requires
same-allocation Base for step0/off and the accepted E-0021 source for RB1-09
compatibility. Cross-job differences on another profile and intervention
normal-vs-training agreement are descriptive. Calibration must be written
before prediction-gate failures. No tolerance or profile restriction.

The code now implements D-0060 and all4low fixes. A new smoke must run on this
commit before requesting RV-0087's one recheck from reviewer-2. Its tests
include cross-profile matching local references, perturbed local references,
strict2g failures, source-bound pytest and excluded-candidate provenance,
full calibration-grid lock reconstruction/membership, and calibration-before-
failure ordering. The smoke additionally runs a32-row old/new-source GPU
compatibility comparison without using its quality scores.

The revised PBS cost estimator reserves a same-allocation Base dev pass for
EVERY training, an extra legacy-source baseline dev pass, both smokes, both
possible endpoint trainings, and analysis. X-02/03 no longer get calibration
passes. Formal release still needs this measured projection<=5.5node-hours
and a passing recheck. No formal attempt has been registered or executed.

Selection inputs now require `{manifests, training_source_commit,
registration}`. The registration is SHA256-bound into the eta-lock and every
candidate's source/init is checked before selection; excluded candidates
also require a complete formal config with no smoke/partial fields.

The older sections below preserve the initial smoke checkpoint and general
execution continuation; current facts above supersede its running/pending
wording and source version.

Task E-0028@1, claimed by researcher-1. Coordinator owns route decisions.
Source branch `sr/researcher-1/T-0031`, implementation commit
`a6afccbec614e594ddf3c882787d86b5a2721b94`.

The initial implementation adds the X-01/02/03 and ETA-QK/X arm registry,
fixed eta plumbing, full final-checkpoint calibration scoring, paired-init and
exact normalized-input checks, step0 vbar/Z RMS, calibration-only eta selection,
strict endpoint/tie/nonfinite handling, and a locked dev-analysis entrypoint.
The existing C_up, FeatureTap and LatentReader operators are reused unchanged.
All new training arms use RBc-C's accepted LR0.001 and R1 seed0 budget/order.

Review RV-0087 was requested against the exact source plus the two PBS scripts;
the coordinator received MSG-0828. Do not edit the review's files until its
answer. No formal training and no new dev quality aggregation has occurred.
The smoke's quality scores are discarded.

Pre-review smoke is RUN-0314 / ATT-0314-1 / PBS3774818.opbs, submitted through
miyabi-mcp with key `sr:ATT-0314-1`, one debug-mig instance, expected600s,
requested900s. Immutable snapshot is the implementation commit under
`/work/xg24i002/x10041/bypass-lora/snapshots/`. Output root is
`/work/xg24i002/x10041/bypass-lora/runs/ATT-0314-1/`.
Spec digest:
`blake2b:5ce7a5130c8058076e159292ac5772b7c40208f756ae4ea5f501e311c959b771`.

The allocation guard returned ok=true on mg1098 with PBS3774818.opbs in R.
`pytest.txt` reports **99 passed in45.35s**. At this checkpoint the all-arm
initialization checks and four-arm80-step smoke / full calibration timing
are still running. Read only timing/implementation summaries before review;
do not inspect or aggregate smoke quality scores.

Notification N-20261102-175801-602b (`T0031-smoke-v1`) is sealed with one job,
notifier miyabi-g2. The daemon is running. Coordinator MSG-0829 has the
listener command and continuation:
`/work/xg24i002/x10041/miyabi-toolbox/miyabi-mcp/bin/miyabi-mcp notify listen --id N-20261102-175801-602b --notifier miyabi-g2`.

Resume steps:

1. Re-establish login/control context. Get PBS3774818 details and application
   status. Import native executor observations/usage exactly once. The initial
   running observation is already recorded as OBS-96a4ccdf30ef51d6.
2. Inspect smoke manifest, pytest, tap-registration and cost-estimate.json;
   import the manifest and add diagnostic evidence. If the smoke failed,
   investigate its bounded logs and coordinate review fixes after RV-0087's
   answer. Never rerun a workload on login.
3. Read RV-0087, fix concrete findings, commit and request its one recheck.
   Re-run changed tests and registration in PBS at the final reviewed source.
4. Update a measured cost table in these task notes before training. Cap is
   **5.5 node-hours**, including smoke, all13initial trainings, up to2endpoint
   trainings, every eval/intervention, checks and analysis. MIG conversion is
   measured instance-seconds/14400. If predicted cost exceeds the cap, ask the
   coordinator to revise the contract before formal submission.
5. Register every formal attempt before execution using the immutable reviewed
   commit, resource miyabi, and explicit inherited reviews RV-0076/0077 and
   RV-0080/0081/0082 plus the current E-0028 passing review. Resolve any stale
   coverage with the coordinator. Launch `T-0031-run.pbs` with TAP_JOB=train
   and TAP_ARM for X-01/02/03 plus10initial ETA arms; final calibration runs in
   place after the same-job dev export/reload check. TAP_JOB=baseline re-evals
   the accepted RB1-09 checkpoint on dev (rule1g) and calibration in one job.
6. After all12initial calibration producers are complete, run eta-plan in
   PBS using a JSON input `{ "manifests": [absolute manifest paths] }`.
   Submit only the triggered endpoint per tap, at most once, then eta-lock.
   The lock contains SHA256 of every calibration producer and prediction file;
   its writer uses exclusive creation. Record its SHA256 BEFORE opening any
   PartB dev aggregate. ETA-grid nonfinite training is excluded explicitly;
   infrastructure/OOM/evaluation failures require engineering resolution.
7. Run all6X-01 frozen interventions with TAP_JOB=intervention, TAP_TRAIN_RUN
   and TAP_TRAIN_SOURCE. Inherit accepted RB1-09 interventions; QK swap drop
   is the preregistered1.95pp. The final analysis entrypoint requires the
   calibration lock and all producer/provenance bindings before opening dev.
8. Register/execute locked analysis in PBS, record one evidence item per
   preregistered rule (implementation diagnostic, outcomes screening), leave
   hypothesis statuses/route decisions to coordinator conclusion review, then
   deliver with source/artifacts/actual cost and limitations.

Accepted reference runs remain retained:
Base ATT-0024-1, RB1-07 ATT-0245-1, RB1-09 ATT-0247-1.
Model/tokenizer revision da87bfb608c14b7cf20ba1ce41287e8de496c0cd;
environment envs/3c916b51337d-cuda, cuda/12.8, torch2.10.0+cu128,
uv.lock SHA25631429457ceece707abf1505af6e6210b44a6e2817d2527241892f0587aca47a3.
The sealed contract at the project root is authoritative; no A2 confirmation
or official validation rows may be read.
