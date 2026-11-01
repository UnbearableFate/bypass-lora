# T-0008: formal M training and learning-rate calibration

In progress. Scope E-0009; no E-0010 formal run in this task. Budget <=1 node-hour.

## Shared schedule preflight

Source 588caddecc9210c578ab31254316e07c7dc70fd3; ATT-0053-1, PBS 3761319.opbs, RES-0053. Sealed ATT-0039-1 train SHA verified inside PBS. Results `/work/xg24i002/x10041/bypass-lora/runs/ATT-0053-1/schedule.json`.

Order hash uses UTF-8 episode_id plus LF for each episode, including the final LF, over all three epochs. SHA256 `0d4b55ed8655d91cf0d0532bdac1820a518b5c5730ef70e48f264da6b4bc9e79`.

| Budget | Steps | Actual input tokens | Warmup steps |
| --- | --- | --- | --- |
| 200000 | 300 | 200170 | 15 |
| 4000000 | 6000 | 4000470 | 300 |

Researcher-3 agreed hash serialization and LR convention MSG-0152. First optimizer update uses target_lr/warmup, warmup endpoint target_lr, last update zero. Independent PBS numeric agreement pending; sent MSG-0154.

## Implementation in progress

Shared `v2_schedule.py`: verified data hashes/count/splits; deterministic three-epoch order; whole batches8 budget stopping; linear warmup and cosine; lower-LR tie rule relative to minimum NLL.

`v2_memory.py`: extends allowed fusion to C_up; optional explicit donor_batch on shuffled evaluation allows a fixed global cyclic donor map independent of evaluation batching. Recipient learned target value remains the read-error reference; target-dependent shuffled addressing metrics remain undefined/suppressed.

All workloads run through Miyabi MCP with a compute allocation guard. Subagents implement trainer and focused tests, without local runtime execution.

Schedule preflight completed exit0, measured PBS walltime2 seconds on mg1098 (0.000138889nodeh), usage U-0066. EV-0049 diagnostic/valid. Initial manual observation accidentally used namespace `pbs:miyabi`; ledger uses `miyabi-pbs` and import consequently conflicts. Same physical job, no reexecution; end state and measured usage recorded manually with original bound namespace. Original ledger JSON retained in `research/artifacts/T-0008/schedule-observations.json`; do not double-count usage. Subsequent jobs use the ledger namespace.

Initial formal implementation source f7c4d2bdde6b2167355e131c757e230d23c80a00. ATT-0055-1 PBS3761353.opbs broad core/model plus v2_memory and v2_train tests underway. New tests cover schedule budget crossing/exhaustion, microbatch mean loss accumulation including uneven split, fixed global donors across a batch16 boundary, nonzero-fusion batch composition, and save/reload tensors/config/predictions. Existing v2_memory protocol tests expanded to all12 arms.

ATT-0055-1 passed146 checks in60.43s, RES-0055/EV-0050; PBS3761353.opbs exit0, measured63s. ATT-0056-1 source f7c4d2b: 12-arm x32step smoke with16dev+16calibration, batch8, initial zero-exit check, all controls, learning curves and dev reload. PBS3761393.opbs submitted with6min walltime (expected240s based on E0006 times plus batch scaling); notification N-20261101-171028-49d8. Need inspect output, import result and usage, compute all12 timings/E0010 forecasts, then request review before six calibration runs (calibration may overlap review per contract).

Continuation: source paths src/v2_schedule.py,v2_train.py,v2_memory.py,test_v2_train.py,test_v2_memory.py. No dependency/core/data changes. Researcher3 copied unchanged schedule source to F branch ece2cbd and is independently verifying in ATT-0054-1/3761331.opbs. Schedule math must match before calibration. Six initial runs backend x LR3e-4,1e-3,3e-3 at200000tokens. Endpoint expansion at most once to1e-4 or1e-2 per E0009. Cal selection only by calibration normal NLL, relative difference<0.5% choose smaller LR. Code review after smoke: exact source f7c4d2b (or fixes if needed); author gpt-6.1 with lower-effort gpt-6.1-sol helper implementation. No H0003 evidence from this task.

MSG-0164 researcher-3 independent PBS ATT-0054-2 confirmed exact match of full3epoch order SHA and both budgets:300/200170/15 and6000/4000470/300. Shared M/F ordering cross-check complete before calibration. ATT-0056-1 still running on MSG-0164 arrival; awaiting its notification.

## Twelve-arm smoke and first calibration grid

ATT-0056-1 / RES-0057 / EV-0052 passed all12 configurations:21357tokens/arm,32optimizer steps, dev16/calibration16, all controls + curves and reload. Every zero-exit max difference0; every reload prediction identical. PBS135s =0.0375MIGh. Checks PBS63s=0.0175MIGh; both usage imported from ledger. Combined measured preflight+tests+smoke200s MIG=0.0138889nodeh.

Stable training step0.1956–0.2047s, throughput3271–3421input tokens/s; peak allocated1.477–1.488GB. Full12 per-arm table still to be extracted in PBS analysis. Small-split six-control final eval1.83–2.01s for32episodes; three curve evals1.57–1.67s for16episodes; reload normal eval+load0.148–0.164s. These measured quantities imply about360s per300step calibration including full dev/calibration evaluation;9min PBS request adds1.5x margin. Formal6000step forecast to report per arm, with evaluation overhead and caveat that smoke data sample is short.

Review RV-0017 requested exact source f7c4d2bdde6b2167355e131c757e230d23c80a00, files v2_train/v2_schedule/v2_memory/tests. Do not mutate reviewed files while review active. Author root exact identifier unknown (recorded honestly); helpers gpt-6.1-sol medium. E0009 explicitly allows calibration alongside review, with reruns if path changes.

Six calibration attempts all same source, qknorm_pre_rope/F_out, batch8,200170tokens/300steps/warmup15; full dev/calibration all six controls and dev curves. No data restrictions changed. No scientific selection from dev.

|Attempt|Backend|LR|PBS|
|---|---|---|---|
|ATT-0061-1|kernel_matrix|0.0003|3761516.opbs|
|ATT-0062-1|kernel_matrix|0.001|3761517.opbs|
|ATT-0063-1|kernel_matrix|0.003|3761519.opbs|
|ATT-0064-1|soft_slot|0.0003|3761520.opbs|
|ATT-0065-1|soft_slot|0.001|3761523.opbs|
|ATT-0066-1|soft_slot|0.003|3761526.opbs|

All submitted, current observations imported. Notification N-20261101-171757-7d81 sealed6jobs. `research/artifacts/T-0008/calibration-grid.json` contains immutable specs/job mapping. Upon completion inspect manifests/app metrics and import results/usage; run PBS analysis to extract table and apply LR selection rule (minnormal calibration NLL, within0.5% choose smaller). Endpoint extension once to1e-4 or1e-2. Do not deliver before review resolution and calibration choice. No E0010 run belongs in T0008.

## Review correction

RV-0017 passes under low-only rule: one low finding, no medium/high. Fixed weighted averaging of `read.reconstruction_max_abs` to true split max, documenting exception to token-mean aggregation. Source d8cc3d4011529fa5c2fcbaa4277d91981a077fed. ATT-0068-1 / RES-0062 PBS3761611 exit0/wall10s: one focused mock test passed7.30s; distinct batch maxima1e-5/3e-5 now report3e-5, existing mean metric/NLL/prediction checks retained. Scoped recheck RV-0019 pending. No training reruns needed per reviewer. Old initial calibration f7c4d2b artifacts retain the weighted-batch-max diagnostic under its legacy name; DO NOT claim it is true split maximum. This field is unused for LR selection, addressing gates, or model logits. Future runs use fixed source once reviewed.

PBS timing report at `/work/xg24i002/x10041/bypass-lora/runs/ATT-0068-1/smoke-cost-report.json`, containing all12 arms. Formal6000step estimate1467.2–1543.6seconds, 1.5x requests2201–2316seconds (~36.7–38.6min, short-mig needed). Includes full dev/calibration six-control final eval, dev curves and reload,10s fixed allowance. Batch-count extrapolation from smoke16rows each, not full-run timing. Initial calibration estimates352–378seconds support9min requests.
