# E-0028 measured cost gate

## Final measured usage after all jobs ended

All 21 T-0031 jobs ended and passed their application validators. Native used
MIG instance-hours total **16.101389**, converted by the registered factor 0.25
to **4.02534725 node-hours**, below the **5.5 node-hour cap**. This includes
every smoke, audit, training, evaluation, selection, intervention and final
analysis job; no endpoint extension or retry training was triggered.

| Work | Attempt | PBS job | Measured MIG-hours | Node-hours |
|---|---|---|---:|---:|
| Initial smoke | ATT-0314-1 | 3774818.opbs | 0.188333 | 0.04708325 |
| Final smoke | ATT-0315-1 | 3774971.opbs | 0.193333 | 0.04833325 |
| Cost/reference audit | ATT-0316-1 | 3775143.opbs | 0.000278 | 0.00006950 |
| RB1-09 evaluation | ATT-0335-1 | 3775269.opbs | 0.140278 | 0.03506950 |
| X-03 | ATT-0323-1 | 3775254.opbs | 0.996389 | 0.24909725 |
| X-02 | ATT-0322-1 | 3775251.opbs | 1.053889 | 0.26347225 |
| X-01 | ATT-0321-1 | 3775250.opbs | 1.213889 | 0.30347225 |
| ETA-X-0.0625 | ATT-0324-1 | 3775255.opbs | 1.136111 | 0.28402775 |
| ETA-X-0.125 | ATT-0325-1 | 3775257.opbs | 1.145833 | 0.28645825 |
| ETA-X-0.5 | ATT-0326-1 | 3775258.opbs | 1.207222 | 0.30180550 |
| ETA-X-1 | ATT-0327-1 | 3775260.opbs | 1.153611 | 0.28840275 |
| ETA-X-2 | ATT-0329-1 | 3775262.opbs | 1.186389 | 0.29659725 |
| ETA-QK-0.0625 | ATT-0330-1 | 3775264.opbs | 1.146667 | 0.28666675 |
| ETA-QK-0.125 | ATT-0331-1 | 3775265.opbs | 1.195000 | 0.29875000 |
| ETA-QK-0.5 | ATT-0332-1 | 3775266.opbs | 1.189722 | 0.29743050 |
| ETA-QK-2 | ATT-0334-1 | 3775268.opbs | 1.106111 | 0.27652775 |
| ETA-QK-1 | ATT-0333-1 | 3775267.opbs | 1.215833 | 0.30395825 |
| Calibration plan | ATT-0342-1 | 3776771.opbs | 0.000556 | 0.00013900 |
| Eta lock | ATT-0343-1 | 3776823.opbs | 0.000556 | 0.00013900 |
| X-01 interventions | ATT-0344-1 | 3776824.opbs | 0.626389 | 0.15659725 |
| Locked final analysis | ATT-0345-1 | 3777371.opbs | 0.005000 | 0.00125000 |
| **Total** | **21 jobs** | | **16.101389** | **4.02534725** |

The native terminal exports were imported once per accounting reference.
[T-0031-final-cost.json](T-0031-final-cost.json) contains the final native
ledger summary and final-analysis usage. The immutable analysis artifact
contains the preceding 20 jobs' 4.02409725 node-hours; its own 18 scheduler seconds
add 0.00125 node-hours. Quantities retain native six-decimal instance-hour
rounding. Requested walltimes are not measured use.

The pre-release comparable 2g estimate and both-endpoint worst case below are
preserved as planning history. Actual 1g allocations for the final smoke,
X-02, ETA-X-2 and ETA-QK-0.125 are recorded from application hardware logs;
the generic ledger's 2g label is not used to infer their actual slice profile.

## Revised gate after D-0060 (before formal training)

The deterministic PBS audit ATT-0316-1 / PBS3775143 passed and returned
**4.542691585 node-hours** for the initial grid and **5.207572029 node-hours**
with both allowed endpoint trainings. Both are within the unchanged5.5cap.
The audit's immutable calculation and input hashes are in
`/work/xg24i002/x10041/bypass-lora/runs/ATT-0316-1/cost.json`.

Planning uses the **observed2g.24gb measurements from ATT-0314-1**, which are
comparable to the accepted RB1 references. The D-0060 fix does not change the
training operator; its additional passes are reserved separately. The revised
estimate includes one fallback Base dev pass for every one of the up-to15
new training arms, one extra legacy-source dev pass for baseline compatibility,
all X-01 interventions including donor forwarding, both completed smokes at
their native measured cost, and120s for other checks/selection/analysis.
X-02/X-03 no longer get a calibration pass. These are projected costs, with
actual scheduler usage recorded as work completes; they are not a guarantee
against failures. Any retry or revised projection must still fit the5.5cap.

| Per-arm class (2g timing, fallback reserved) | Expected total seconds | Requested walltime planning (about1.5x) |
|---|---:|---|
| X-01 and ETA-X candidates | 4820.620418 | 02:05:00 |
| X-02 | 4322.335647 | 01:50:00 |
| X-03 | 4320.202516 | 01:50:00 |
| ETA-QK candidates | 4753.657976 | 02:00:00 |

The recheck smoke ATT-0315-1 / RES-0284 ran on **MIG1g.24gb**, as recorded by
both nvidia-smi and torch, despite the executor ledger's generic2g hardware
label. Its112tests, full initialization registration and32-row same-job
old/new-source GPU comparison passed. The PBS audit additionally verified
all4representative arms' step0 predictions equal their same-allocation Base
predictions (32rows each). Thus the cross-profile execution branch was
exercised, without a tolerance or slice restriction.

The observed1g short-smoke timings were slightly faster here; no claim that1g
is slower is assumed. They are retained as a descriptive comparison, yielding
4.982964924node-hours under the same worst-case reserve model. The planning
number remains the2g-based5.207572029 above. The two smokes cost678s and696s;
their native measured usage totals0.0954165node-hours (six-decimal ledger
rounding), before the audit's terminal usage. The audit itself used1scheduler
second and is covered by the120s check/analysis reserve.

All scientific files and the training/smoke launchers still exactly match
`0f22cd60f45674ec20b800fdd455f80c2b6d46dd`. Subsequent commits contain only
task notes, native observation exports and the standalone deterministic audit
wrapper; none changes the reviewed training/evaluation/selection paths.

## Original pre-review measurements

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

At this original pre-review checkpoint, formal training had not started.
No smoke quality score was read or used for selection or a scientific decision.
