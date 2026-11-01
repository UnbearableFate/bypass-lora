# T-0025 phase-two release and training plan

Coordinator MSG-0650 releases phase two after RV-0064 passed and T-0024 was
accepted/merged in master fd66dac. This branch merged master 60e2d1f as ff050b8.
Numerical training/scoring files remain those covered by RV-0064. Nine runs are
registered and staged before any of the four low-finding fixes are applied, so
their immutable training snapshots stay covered. The fixes affect only reporting,
diagnostic summaries and artifact validation; a focused recheck precedes reading
any RB1 results with the revised analysis.

Accepted LR lock: `research/artifacts/T-0024/lr-lock.json`, 3646 bytes, SHA256
`38b8ee24af3ded8aee398a4cc3dfaa5a1dfece4737a6cde34642fbff7bf1012d`.
All sixteen input manifest hashes and the A2 split hash are verified inside each
allocation before model imports. The lock and split are immutable absolute
snapshot paths passed to the existing `rb_train.py run --stage train` entry.

| Arm | LR | RB0 predicted complete node hours | Expected seconds (rounded up) | Requested walltime |
| --- | ---: | ---: | ---: | --- |
| RB1-01 | .003 | .260091 | 3746 | 01:40:00 |
| RB1-02 | .003 | .260239 | 3748 | 01:40:00 |
| RB1-03 | .003 | .271757 | 3914 | 01:40:00 |
| RB1-04 | .003 | .291758 | 4202 | 01:50:00 |
| RB1-05 | .003 | .282254 | 4065 | 01:50:00 |
| RB1-06 | .003 | .293909 | 4233 | 01:50:00 |
| RB1-07 | .0003 | .258521 | 3723 | 01:40:00 |
| RB1-08 | .001 | .263593 | 3796 | 01:40:00 |
| RB1-09 | .001 | .261838 | 3771 | 01:40:00 |

T remains at the registered calibration grid boundary. S2/S4/fixed-query/eta4
inherit R's LR; C_up inherits C_dual's. No additional calibration or tuning.
All runs use seed0, layers6/13/20/27, the same R1 initialization/order, 2M
nonpadding input tokens at the first complete step (or three epochs first),
per-example mean answer CE, final checkpoint, 25/50/75% dev monitoring and a
same-job fresh reload. There is no confirmation access. Hardware is metadata
only under D-0018/D-0019; only later comparative latency requires 2g.

Walltime is the measured RB0 part-model estimate times1.5, rounded up to ten
minutes. Nine training/monitor/save/reload runs sum to2.443961 predicted node
hours. Frozen interventions are budgeted provisionally at0.5 (RV-0064's estimate,
to be smoke-measured), latency0.1, checks/recheck/analysis overhead0.15. Projected
total is about3.2 node hours, below the4 cap. Measured phase-one tests consumed
0.00798611 node hours. Later releases use actual ledger costs and the measured
intervention estimate, and stop/report engineering failure or budget overrun.

The nine jobs use short-mig, one MIG instance each, one notification group with
expected=9 and early-failure delivery. The coordinator owns the listener and
wakes researcher-1. No scheduler polling loop or local workload is used.
