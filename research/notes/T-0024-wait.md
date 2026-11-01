# T-0024 endpoint wait

CURRENT: All 12 initial-grid candidates have been verified and imported once. The four required one-time upper endpoint candidates are running. No LR is locked yet.

Initial verification: ATT-0222-1 / RES-0205 / PBS3768977; all12 complete/finite, correct source/spec/config/representative/phase/seed, same split/order, 1096 steps /2192 presentations /200229 input tokens, exactly2048 calibration examples/task. NLL was recomputed from saved calibration gold-token CE sums/counts. No dev, confirmation, export or reload. Results ATT0210..0221 -> RES0193..0204.

Initial training exact usage:7230 MIG seconds. Verification report:13seconds. Total imported once:7243s /14400 = 0.5029861111111111 node·h. Cap1.5. Running endpoint usage is unknown.

Required endpoint group (source ef1f2d549913ef67fcd4d450d2913b04a97d2829; training/data/core/environment unchanged from RV0062):
- RBc-L@0.003: ATT-0223-1, PBS3768984.opbs, output /work/xg24i002/x10041/bypass-lora/runs/ATT-0223-1
- RBc-T@0.003: ATT-0224-1, PBS3768985.opbs, output /work/xg24i002/x10041/bypass-lora/runs/ATT-0224-1
- RBc-R@0.01: ATT-0225-1, PBS3768987.opbs, output /work/xg24i002/x10041/bypass-lora/runs/ATT-0225-1
- RBc-C@0.01: ATT-0226-1, PBS3768988.opbs, output /work/xg24i002/x10041/bypass-lora/runs/ATT-0226-1

Notification N-20261102-042540-0f14, sealed expected4. Coordinator listen command:

```sh
/work/xg24i002/x10041/miyabi-toolbox/miyabi-mcp/bin/miyabi-mcp notify listen --id N-20261102-042540-0f14 --notifier miyabi-g2
```

Continuation:
1. Read the four ended jobs once via miyabi-mcp job_get; inspect allocation and manifests. Record scheduler end and exact walltime seconds/14400 usage once per accounting ref. Ordinary profiles are metadata only, never analysis or rejection. Notification ACK is coordinator-owned.
2. Verify every endpoint's application output: status complete/validator passed, contractE0020@1/stagecalibration/phaseV0/seed0, source/spec/family/representative/LR, correct split and identical complete200k plan, full2048-row/task calibration only. Explicit failed=true/nonfinite candidates are rejected by selection as registered; implementation problems stop/report.
3. Populate a new final-jobs JSON with the 12 initial verified entries from research/notes/T-0024/verified-initial-jobs.json plus four ended endpoint entries from T-0024-endpoint-jobs.json. Preserve exact registered metadata and attach scheduler_exit_status, used_MIG_seconds, used_node_h. Register artifact-only report before executing.
4. In PBS execute existing src/rbc_report.py --mode final --jobs <finaljobs> --output <unique final report ATT directory>. It exhaustively authenticates all16 source/config/budget/allocation/calibration artifacts, then calls the reviewed rb_train.py lock CLI on that exact inventory. It leaves raw output lr-lock-reviewed.json immutable and adds only the authenticated common plan_order_sha256 field to lr-lock.json. It reports final relative NLL excess, the smallest LR within1%, and registered boundary flags. No new LR rule.
5. Import four candidate results and final report once after application verification; record exact final-report usage once. If already inspected before the report, authenticate with final report before delivery; no duplicates.
6. Copy final lr-lock.json to research/artifacts/T-0024/lr-lock.json and commit its small exact file (git add -f if ignored). Commit candidate table/report and delivery/evidence notes. Include split SHA, plan order SHA, input manifest SHA refs, explicit extension/1%/boundary decisions, all ATT/RES/EV refs, and exact full task usage including both PBS reports.
7. Add one diagnostic EV about E-0020/H-0013 without supports/challenges direction; do not change hypothesis state or claim efficacy. Deliver T-0024 via rl and wait for coordinator acceptance. No RB1 training, dev, confirmation, official validation, slice analysis, or second endpoint expansion.

Detailed registration/import/usage metadata is in research/notes/T-0024-endpoint-jobs.json and the initial-grid report. Scientific source paths are unchanged; the only new src file is rbc_report.py (artifact-only verification invoking unchanged reviewed functions).

Historical bookkeeping: ATT0198..0209 were withdrawn before submission (descriptive metric correction), zero compute; excluded from all candidate/report/lock inputs.
