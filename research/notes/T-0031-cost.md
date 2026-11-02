# E-0028 measured cost gate

Pre-review smoke ATT-0314-1 / RES-0283 / PBS3774818 completed successfully on
verified allocation mg1098. Scheduler walltime678s, one MIG instance. Native
ledger usage is0.188333 MIG instance-hours; exact-second conversion is
678/14400 node-hours. Native observations, including usage under the unique
accounting reference `miyabi-pbs:3774818.opbs`, are imported once through
`T-0031-smoke-observations.json`; no separate estimated usage is recorded.

All99selected tests and all registration checks passed. Timing and peak
allocated CUDA bytes from the80-step smoke:

| Arm | Steady seconds/step | Peak CUDA bytes | Stored/trainable/with-grad |
|---|---:|---:|---:|
| X-01 | 0.2305620734 | 2730313728 | 794624 |
| X-02 | 0.2200430750 | 2715615744 | 794624 |
| X-03 | 0.2198804268 | 2724004352 | 925696 |
| ETA-QK-1 | 0.2254562829 | 2739750912 | 925696 |

A full calibration evaluation (2048rows per task) took360.3281074s. The
PBS-generated estimate used dev202.3529597s, 13115training steps, six dev
passes per training (step0, quarter/half/three-quarter, final and reload), one
full calibration pass, and explicit baseline/intervention/smoke/analysis
overhead. Its immutable original table is
`/work/xg24i002/x10041/bypass-lora/runs/ATT-0314-1/cost-estimate.json`:

| Coverage | Estimated node-hours | Cap |
|---|---:|---:|
| PartA3 + PartB10initial new trainings and evaluation overhead | 4.363088611 | 5.5 |
| Same, plus both permitted endpoint trainings | 4.999864477 | 5.5 |

This is a planning estimate, not measured final usage. It predates RV-0087's
fixes. Removing unregistered X-02/X-03 calibration passes saves about0.05
node-hours (review finding4); same-allocation compatibility checks may add
dev passes, and a second exact-source smoke also counts. A new PBS smoke must
produce the revised cost table before formal release. If its projection
exceeds5.5node-hours, the coordinator must revise the contract before training.
The requested training walltimes will be approximately1.5times the measured
per-arm estimate; requested walltime never substitutes for charged usage.

No formal training has started. No smoke quality score was read or used for
selection or a scientific decision.
