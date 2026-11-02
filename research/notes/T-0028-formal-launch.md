# T-0028 formal release and wait

RV-0079 passes with zero findings. The formal source is exactly
`7d8b3464b6145504d155268f4c667fa3ecc53b00`, the ATT-0276-1/RES-0253
326-test source. The source snapshot is immutable and already verified by
the reviewer; no training/scoring/state/bootstrap code has changed.

Registration inherited a stale label from E-0021 for the sole path
`src/test_rb1_artifacts.py`. Its only change is the restored 16-line
family-LR rejection test, passed in ATT-0250-1. No SEG/runtime path imports
or executes it. Coordinator decision **D-0049**, received as MSG-0756,
explicitly treats these attempts as covered by RV-0078/RV-0079 despite that
test-only inherited label. Each executor note, label and imported start
observation cites D-0049. No unrelated file was changed and no third review
was requested.

| Arm | Attempt | PBS job | Measured-part expectation s | Requested walltime | LR |
|---|---|---|---:|---|---:|
| SEG-00 | ATT-0277-1 | 3770249.opbs | 189 | 00:10:00 | — |
| SEG-01 | ATT-0278-1 | 3770250.opbs | 3296 | 01:30:00 | 0.003 |
| SEG-02 | ATT-0279-1 | 3770251.opbs | 7527 | 03:10:00 | 0.003 |
| SEG-03 | ATT-0280-1 | 3770252.opbs | 3517 | 01:30:00 | 0.003 |
| SEG-04 | ATT-0281-1 | 3770253.opbs | 3464 | 01:30:00 | 0.0003 |

SEG-00 is evaluation only. SEG-01..04 each retain 13,115 optimizer steps,
26,230 presentations, batch 2/microbatch 1, the exact RB1 initialization and
order, the accepted RBc family LR lock and total 512-token segmented budget.
The original candidate tokens and V1 per-example loss/scoring remain fixed.
Full dev curves are at steps 3279/6558/9837; final dev is 1000 rows/task,
with fresh-base same-job reload. SEG-02 also verifies initialization/Base
identity before training. Only four differentiable M,c,count states cross
segments; original KV never crosses and segment 2 positions restart at zero.

All five jobs passed the site's submission plan and were submitted through
miyabi-mcp at the reviewed source. Native queued/running observations were
imported once, without relabelling queued jobs as running. Workloads require
the site allocation_check before application imports; ordinary jobs record
hardware metadata only under D-0018/D-0019.

The full projection, including both preflights, all initial arms and their
interventions, the possible r128 branch, Base/analysis and review reserve,
remains 2.541927988 node-hours against the 3-node-hour cap. No shortening was
needed; every actual terminal attempt will be charged once. Conditional
SEG-05..07 are not registered or submitted.

Grouped notification: **N-20261102-092336-7dab**, sealed at 5 jobs, with early
failure notification. The existing notifier daemon is confirmed running.
The coordinator owns the listener and wakes this researcher. This is the
standing Smooth Research wait/resume flow; no Codex Goal or duplicate job
watcher was created. Runtime is measured; queue wait is unknown.

Listener command:
`/work/xg24i002/x10041/miyabi-toolbox/miyabi-mcp/bin/miyabi-mcp notify listen --id N-20261102-092336-7dab --notifier miyabi-g2`.

On completion: verify all manifests and final checkpoint inventories, exact
sample/step/LR/order/exclusion pins, all curves/full dev and same-job reload,
then import every result and native terminal usage once. Run SEG-02's
off/swap/reset grid from this same source and validated checkpoint, then
the locked 10,000-sample analysis with complete task cost coverage. Require
the exact-source ATT-0276-1 receipt for the no-channel check. Release r128
once only if the exact primary-task rule fires, subject to updated budget.
Every rule's evidence cites E-0025/D-0044 and D-0049; hypothesis status and
route decisions remain the coordinator's. No confirmation set is evaluated.

