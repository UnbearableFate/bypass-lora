# T-0024 initial-grid wait

Status: twelve initial candidates submitted and observed running; application results are not verified yet.

Contract E-0020@1, DOC-0231@1. Reviewed code source 6df9e479b229feb3d5edfac588b06264612c7839; no code/environment/data changes. All run review checks are covered by RV-0062 (and its underlying reviews). Exact registrations, configs, spec digests, PBS jobs, logs, and outputs are in [T-0024-initial-jobs.json](T-0024-initial-jobs.json).

Notification N-20261102-040252-5355 is sealed with all twelve PBS jobs, expected=12, all_ended with early failure. The coordinator owns the wake-up. Listen command:

```sh
/work/xg24i002/x10041/miyabi-toolbox/miyabi-mcp/bin/miyabi-mcp notify listen --id N-20261102-040252-5355 --notifier miyabi-g2
```

Continuation:
1. Read the twelve jobs once with miyabi-mcp job_get. Inspect each saved allocation and manifest; require complete/validator passed, contract E-0020@1, source SHA/spec digest, correct representative/family/LR/seed0/phaseV0/stagecalibration. Nonfinite candidates are allowed only with explicit failed=true; implementation failures stop.
2. Verify all successful candidates have identical plan order (expected 053284831520676982857bf9677a9fa3e338b0bbad4d2ce2af1c3f88d20457d3), 1096 steps, 2192 examples, 200229 input tokens; A2 split SHA 34277aa4627271dbf4e0bcb9592e3014c6130d87b1766caad17c2d6b97a8a7cc; full calibration 2048 rows per task; no dev-step, dev metrics, checkpoint export, adapter directory or reload. Confirm the calibration-only evaluation rows. Do not read prediction content from any other split.
3. Record end facts and exact usage once per miyabi-pbs:<job-id>; use scheduler walltime seconds /14400, not rounded ledger hours. Import each result once. Profile is metadata only; no slice comparison or reruns.
4. In PBS, validate common provenance and use reviewed extension_lr on each exact initial family grid. Report raw finite minimum and required one-time endpoint only; stop if a family has no finite candidate. Submit required extension candidates in a second notification group.
5. After extensions, run the reviewed rb_train.py lock CLI on the exact expected list in PBS. Authenticate the common plan order and add that SHA as provenance to the resulting lock while preserving lock_lrs decisions. Commit lock and final candidate/cost table. Add one diagnostic EV about E-0020/H-0013 with no support/challenge direction, then deliver.

Important: ATT-0198-1..ATT-0209-1 were withdrawn before submission due solely to incorrect descriptive metric wording. They consumed no compute and must not enter reports/lock. Active candidates are ATT-0210-1..ATT-0221-1. Their correct metric is macro average of per-task gold-token CE sum divided by gold-token count. This correction changed no executable code or scientific definition.

Budget cap 1.5 node·h. Predicted 12-point usage 0.509314; maximum registered 16-point design estimate 0.679085. Actual running usage remains unknown until completion; no usage has been imported.

No RB1, dev, RB confirmation or official validation evaluation.
