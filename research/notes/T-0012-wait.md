# T-0012 full measurement run: continuation

Task T-0012 remains owned by researcher-2. Coordinator owns interpretation and
route decisions. The full run has not yet been validated or imported.

- Full run: `RUN-0096`, `ATT-0096-1`, PBS `3763105.opbs`.
- Executed source: `8cfcb1a8c696cec3af2b611bdb33a25a5b51e801` (reviewed core
  unchanged from `fed91c959a4ba3460f3956d98ef24a4e6a089139`). `RV-0027` pass.
- Allocation evidence read: `mg1098`, Miyabi-G, allocation confirmed; live PBS
  state RUNNING at job start. One GH200 MIG instance in debug-mig.
- Expected 1136 s; requested 1704 s, from `ATT-0095-1` full-shape timings.
- Outputs: `/work/xg24i002/x10041/bypass-lora/runs/ATT-0096-1/`.
- Full spec digest:
  `blake2b:729189789e7a89994f917e5c7f1f28b5f346d8ab9db4a4cfa97b683f211ce70d`.
- Notification: `N-20261101-185923-e809`, sealed, all_ended / early failure.
  Daemon started on miyabi-g2. Coordinator received MSG-0260 with listener:
  `/work/xg24i002/x10041/miyabi-toolbox/miyabi-mcp/bin/miyabi-mcp notify listen --id N-20261101-185923-e809 --notifier miyabi-g2`.

On resumption:

1. Inspect job history and application manifest/report; acknowledge notification
   if the coordinator has not. Refresh the project job ledger including history
   before exporting final smooth-research observations (job_get alone does not
   necessarily advance the ledger). Import observations/usage once.
2. Require all six full-mode L0 accuracies >=90%. In full mode a failure aborts
   the pipeline before L1--L4. If failed, inspect position/fitting artifacts and
   repair, with review/recheck where needed; do not interpret downstream smoke.
3. A complete run has 120 probe rows, ten L3 arms, fourteen L4 entries. Calibration
   counts: support probes 32768, question/address probes 2048. L3 train4096 /
   calibration2048. Dev1000 key geometry has120000 within-episode key pairs;
   last/head read histograms have n1000. Quantiles/histograms and all input/source
   hashes are in report.json; report.md contains the main tables.
4. Import manifest, add diagnostic-level valid evidence linked only to E-0012
   (no supports/challenges), write a measurements-only delivery note with full
   L0--L4 tables, references and actual cost, then `rl task deliver T-0012`.
   Do not infer information absence, causal mechanism, or choose E-0013.

Prior results already imported: `ATT-0094-1` / `RES-0095`, smoke and five tests;
`ATT-0095-1` / `RES-0096`, full-shape cost / all-ten geometry validation. Their
scheduler facts and measured usage were imported from executor exports. Total
prior use: 48+66=114 MIG seconds =0.007917 node·h. E-0012 ceiling is0.5 node·h.
The initial smoke L0 failed two four-epoch qknorm controls and is explicitly
uninterpretable; full mode uses up to64 epochs with train-internal selection.
