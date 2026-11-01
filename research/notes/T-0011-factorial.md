# T-0011: E-0010 V2 factorial execution

Status: waiting for ten released 4M training jobs. Scientific results and delivery are pending.

Code snapshot: `9145309f16231d52bb121118779c986e875951e5`. The task branch adds only `src/v2_factorial_analysis.py`, `src/test_v2_factorial_analysis.py` and `src/v2_factorial_check.py`; the existing reviewed training/evaluation and dependency files are unchanged.

Analysis review RV-0023: pass, zero findings. Author exact model identity was not observed and is recorded as unknown. Training registrations explicitly name RV-0019; each reported code review covered.

PBS regression ATT-0078-1 / RES-0080 / EV-0066: 21 passed, zero failures/errors/skips, 2.060s application time. Confirmed compute host mg1098, PBS3762004.opbs, exit0, measured3s =0.000833 MIG-h (about0.0002083 node-h). Evidence covers synthetic rules/pairing/input validation; the full ten-arm CLI and10000 real-artifact bootstrap replicates remain unexecuted.

The analysis preserves the literal E-0010@2 rules, exact episode-count EM boundaries, dev-only decisions and shared paired episode bootstrap. It reports oracle numbers and the qualitative localization rule: the contract defines no numerical high/low cutoff. Strata and nonself subsets have point summaries; paired intervals cover full-split EM/NLL contrasts, factorial effects, reader cells and dual probes. An exact unresolved V3 tie stays explicit.

All training jobs use seed0,6000steps,4000470tokens,warmup300, batch8/microbatch8, eval16, final dev+calibration all controls and dev reload, with dev-only25/50/75% curves. Each job checks its compute allocation and live RUNNING state before workloads. Data/model/environment pins follow E-0010@2 Release. Short-mig walltime uses the released T-0008 component forecast times1.5, rounded up to seconds.

| Arm | Attempt | PBS job | Tap | Reader | Fusion | LR | Requested walltime s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| V2-01 | ATT-0080-1 | 3762042.opbs | direct_hidden | kernel_matrix | F_out | 0.01 | 2236 |
| V2-02 | ATT-0081-1 | 3762043.opbs | direct_hidden | kernel_matrix | C_up | 0.01 | 2210 |
| V2-03 | ATT-0082-1 | 3762044.opbs | direct_hidden | soft_slot | F_out | 0.003 | 2304 |
| V2-04 | ATT-0083-1 | 3762045.opbs | direct_hidden | soft_slot | C_up | 0.003 | 2303 |
| V2-05 | ATT-0084-1 | 3762046.opbs | qknorm_pre_rope | kernel_matrix | F_out | 0.01 | 2209 |
| V2-06 | ATT-0085-1 | 3762047.opbs | qknorm_pre_rope | kernel_matrix | C_up | 0.01 | 2207 |
| V2-07 | ATT-0086-1 | 3762048.opbs | qknorm_pre_rope | soft_slot | F_out | 0.003 | 2316 |
| V2-08 | ATT-0087-1 | 3762049.opbs | qknorm_pre_rope | soft_slot | C_up | 0.003 | 2307 |
| V2-09 | ATT-0088-1 | 3762050.opbs | qknorm_pre_rope | kernel_matrix | C_dual | 0.01 | 2201 |
| V2-10 | ATT-0089-1 | 3762051.opbs | qknorm_pre_rope | soft_slot | C_dual | 0.003 | 2316 |

Durable execution map: `/work/xg24i002/x10041/bypass-lora/research/artifacts/T-0011/training-run-map.json`. Ledger exports: `analysis-check-observations.json` and `training-submission-observations.json` in the same directory. Run outputs: `/work/xg24i002/x10041/bypass-lora/runs/<ATT>/`. All ten were QUEUED at the last recorded inspection; no scientific run is claimed complete.

One sealed all-ended/early-failure notification group: `T0011-ten-arm-4M`, id `N-20261101-175131-1cd7`, expected10. Its daemon was running with300s polling. The coordinator must run this listener in its own background shell:

```bash
/work/xg24i002/x10041/miyabi-toolbox/miyabi-mcp/bin/miyabi-mcp notify listen --id N-20261101-175131-1cd7 --notifier miyabi-g2
```

On notification: inspect the exact jobs and acknowledge; resume researcher-2/T-0011. Verify manifests and artifacts, export terminal ledger observations and usage once, import each result, then register and run the reviewed analysis in PBS from this snapshot. Pass all ten `--run V2-ID=/work/xg24i002/x10041/bypass-lora/runs/ATT-...` mappings from the execution map and write to the analysis attempt directory. Record screening evidence with literal rule inputs and deliver. The coordinator owns conclusion review, hypothesis statuses and V3 release.

