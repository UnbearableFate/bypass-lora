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
