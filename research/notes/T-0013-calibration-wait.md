# T-0013: initial LR calibration in progress

Current continuation; task not delivered. Full job/spec map is committed in
T-0013-calibration-jobs.json. Official validation remains sealed and untouched.

## Completed and verified

- E-0013@3 / H-0008@2 released by D-0011 and coordinator MSG-0314.
- V5 training review **RV-0033 pass**; exact three-arm smoke ATT-0098-2,
  RES-0099, EV-0073:36 tests,39 optimizer steps/arm,exact init and fresh-Base
  dev reload,matching order. Full plan29,165 steps /58,329 presentations /
  **4,173,300 input tokens** (coordinator accepts measured tokens, MSG-0306).
- Failed launcher ATT0098-1 used2MIGseconds (U-0112); corrected smoke181s
  (imported executor usage, exact181/14400 node·h). No duplicate usage imports.
- Latency low null/empty path bug fixed in2250925. **RV-0035 pass** after8
  tests in ATT-0110-1 / RES-0101 / EV-0075, PBS3764470 mg1100 Exit0,10s.
  Invalid EV-0074 had an administrative wrong-result link; EV-0075 supersedes it.
  Usage U-0115 exact10/14400 node·h, already imported: strip usage if reimporting
  its executor export. Its end is recorded from verified job_get history because
  ledger exports can lag job_get until the notifier refreshes history.
- Prepared full latency-smoke ATT-0100-1 was never submitted and is recorded
  cancelled, superseded by the low-fix regression. Full latency is deferred to
  final seed1 checkpoints per coordinator MSG-0314; do not run smoke latency.
- Initial-grid report low null-NLL issue is fixed in
  **87bbb59f877f05f63f603319e069ff9a4274203e**. **RV-0038 pass**;
  ATT-0114-1 / RES-0104 / EV-0076:4 tests passed,including null unflagged success
  rejection and explicit failed-LR partial usage acceptance. PBS3764578 mg1098
  Exit0,10MIGseconds;end and exact10/14400 node·h usage already imported from
  verified job_get. Strip its usage if reimporting a later executor export.

## Running jobs and wake-up

Initial calibration seed0, three LRs per arm {1e-4,3e-4,1e-3},200k tokens:

| Arm | Attempts (ascending LR) | PBS job numbers (.opbs) |
| --- | --- | --- |
| V5-C | ATT-0101-1,0102-1,0103-1 | 3764424,3764425,3764428 |
| V5-F | ATT-0104-1,0105-1,0106-1 | 3764431,3764432,3764433 |
| V5-L | ATT-0107-1,0108-1,0109-1 | 3764434,3764435,3764436 |

They execute immutable03438968c71f055a6222b2d45dcb78cec281b793; register
reported covered by RV-0033 and RV-0034 low-only. Allocation check precedes
every workload. C/F expected640s/request960s;L1120s/request1680s, all debug-mig.
Initial observed running facts already imported from executor exports.
First C job additionally runs the original latency synthetic merge unit; the
new null-path regressions were separately validated in ATT0110.

- Initial9 group: **N-20261101-201840-3a6f**, sealed expected9,early failure.
  Listener: `/work/xg24i002/x10041/miyabi-toolbox/miyabi-mcp/bin/miyabi-mcp notify listen --id N-20261101-201840-3a6f --notifier miyabi-g2`.
- **Base repeated dev**: ATT-0113-1, PBS3764504.opbs, immutable2250925,
  expected320s/request480s. Runs full2000 dev examples twice on a fresh Base,
  saves predictions.jsonl and requires reload.identical. Running fact recorded.
  Group **N-20261101-202714-188a**,expected1;listener:
  `/work/xg24i002/x10041/miyabi-toolbox/miyabi-mcp/bin/miyabi-mcp notify listen --id N-20261101-202714-188a --notifier miyabi-g2`.
- Daemon running on miyabi-g2. Coordinator receives these listeners and owns
  waiting; this researcher does not poll. Resume when either group reports.

## Next actions, in order

1. Inspect completed job history/allocation and application manifests; import
   verified ends and exact usage once (node·h=MIGseconds/14400), then results.
   Never guess a RES identifier: simultaneous sessions allocate them. If an
   executor end export is stale, use verified job_get history explicitly.
2. For Base verify complete,full dev counts and exact repeat; record diagnostic
   evidence only. For initial grid wait for all9 before selection; any engineering
   failure stops for reporting, nonfinite LR candidates remain eliminated.
3. Helper **src/v5_calibration_report.py** and4 tested regressions are in commit
   **87bbb59f877f05f63f603319e069ff9a4274203e**. It validates all initial-grid
   manifests and equal seed0 order/steps/tokens, calls reviewed extension_lr, and
   marks smallest-LR-within1% initial values as provisional. It does not lock LR.
   **RV-0038 pass** covers the fixed helper plus unchanged v5_train.py. Register
   a small PBS report job from that snapshot, run
   test_v5_calibration_report.py then the report CLI on the9 absolute manifests.
   Unit tests passed;the report CLI on actual grid results is not yet run.
   Read required_extensions from initial-grid.json.
4. Run only required endpoint candidates (3e-3 for raw minimum1e-3 or3e-5 for
   raw minimum1e-4), same seed0/200k plan and reviewed trainer. Do not select from
   dev; endpoint expansion happens before the1% tie rule. Register all runs first,
   immutable snapshots,allocation check,notifications,WAIT as needed.
5. Execute `v5_train.py lock --manifests <all initial and extension manifests>
   --output <lr-lock.json>` in PBS. Commit the final lock and selection table in
   this worktree before any seed1–3 run. Put the committed lock in the run's
   immutable source snapshot and pass that path with --lr-lock; no mutable lock.
6. Re-estimate all9 formal jobs with measured calibration train/eval costs and
   the exact29,165-step plan. Budget E-0013≤12 node·h, T0013 planning~7.
   If projected total would exceed budget, stop/report before changing schedule.
   Submit three arms×seeds1,2,3,exact3epochs; use only final checkpoint,3dev
   monitoring records and final JSONL predictions;require fresh-Base exact reload.
   The long-job wakeup skill was read:no Goal was requested,the existing site
   MCP notifications/coordinator continuation own wakeup. Do not add a duplicate
   watcher or implicitly create a Goal.
7. After training run reviewed fixed256 latency on final seed1 adapters using
   corrected src/v5_latency.py (RV0035);construct valid arm->directory map and
   record checkpoint provenance. Base and all adapters use the same256inputs,
   GPU type,batch1,16warmups,3passes;LoRA before/after merge. Merging timing only.
8. Deliver LR table/committed lock,all9 checkpoint+manifest SHA256 and paths,
   dev monitoring/reload/steps/tokens/costs to coordinator and researcher-1
   (T0014 evaluation). Evidence for calibration/dev/cost is diagnostic only,
   no --supports/--challenges. T0014 owns final confirmation/WikiText/science.
   T0014 MSG0344 requires top-level learning_rate_lock FileRef and each entry's
   checkpoint_inventory FileRef for checkpoint.json,plus training_manifest ref.
   It verifies phase=train,contractE0013@3,validator passed,actual58329examples/
   29165steps,metrics.lr_lock.sha256 against lock,adapter dir=manifest parent/
   adapter,inventory listed in manifest artifacts,and exact checkpoint file
   paths/bytes/SHA256. Current producer preserves these fields;confirmed MSG0347.

All workloads go through miyabi-mcp PBS. Login only for bounded reads,edits and
records;unchanged environment3c916b51337d-cuda/cuda12.8,HF_HUB_OFFLINE=1.
