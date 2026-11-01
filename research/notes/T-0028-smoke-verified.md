# T-0028 pre-review verification and cost gate

ATT-0271-1 / PBS 3770096.opbs / RES-0250 completed at immutable source
752b50c582e3002618aed65449105de2c37fcf09. Native terminal observation and
its measured usage were imported once; the result import has no mismatches.
This note does not inspect or report any SEG dev quality. Formal training has
not started; every confirmation set remains unopened.

The allocation receipt confirms compute execution on mg1098. The timing
guard records the allocated UUID as MIG 2g.24gb. XML and checks.log independently
report **315 passed, zero errors, failures or skips** (41.13 test seconds).
The successful cases cover all six no-channel arms with/without a discarded
first forward, zero-exit and iv_off/Base identities, reset and self-donor
identities, KV/position isolation, first-segment state gradients at all four
layers, paired RB1 initialization, both ranks' parameter counts, unchanged
default paths, data-seam encoding, exact rule boundaries including D-0044,
artifact rejection paths, and synthetic full 10,000-bootstrap analysis.

All seven possible training arms completed the same 132 representative steps
(264 examples), sampled uniformly across the exact 13,115-step RB1 plan. The
order hash is 33759bc20322d0c9981128dd8574438546fc3439194ab32b7c79e35e0f0fc2bf;
the full plan retains 26,230 presentations and RB1's 2,000,069 reference tokens.
The smoke used 19,923 actual forward tokens for each keep arm (14,469 first,
5,454 second), versus 5,454 second-segment tokens for each no-channel arm.
All seven fresh-base reloads matched all 32 dev rows/task and candidate scores.
Quality metrics and predictions were discarded. Both keep arms also completed
off, swap and reset timing. Full arm measurements are in cost-smoke.json.

| Arm | Trainable / stored / with gradient | Steady step s | Dev32/task s | Peak allocated bytes | Cross-segment M+c bytes |
|---|---:|---:|---:|---:|---:|
| SEG-01 T0 r32 | 663,552 | 0.184520 | 5.312809 | 2,016,013,312 | 0 |
| SEG-02 keep r32 | 663,552 | 0.399240 | 11.202915 | 2,548,907,008 | 16,896 |
| SEG-03 reset r32 | 663,552 | 0.196163 | 5.748468 | 2,019,779,072 | 0 |
| SEG-04 LoRA | 630,784 | 0.196839 | 5.352569 | 2,049,803,264 | 0 |
| SEG-05 keep r128, conditional | 2,752,512 | 0.389802 | 10.518135 | 2,712,055,296 | 264,192 |
| SEG-06 reset r128, conditional | 2,752,512 | 0.195090 | 5.534296 | 2,100,445,184 | 0 |
| SEG-07 T0 r128, conditional | 2,752,512 | 0.184030 | 5.290093 | 2,043,554,816 | 0 |

Each keep arm additionally carries **32 count bytes** (four int64 counters).
Per-layer state is FP32 M[1,r,r], c[1,r], and int64 counts[1]. Counts and
per-segment write/read accounting are retained. The measured r128 step being
slightly faster than r32 is a timing observation from this short rehearsal,
not a claim of a computational advantage.

The cost model prices all 13,115 steps, three intermediate full-dev curves,
final full dev and reload, initialization/Base identities, Base_seg, the
complete intervention grid, load/save/preparation, analysis and review reserve.
Requested job walltimes will use 1.5 times the measured part estimate, rounded
up to ten minutes; the experiment ceiling is not the requested walltime.

| Cost component | Node-hours |
|---|---:|
| Initial four arms, including SEG-02 interventions | 1.348577716 |
| Conditional three r128 arms, including SEG-05 interventions | 1.090157028 |
| Base, analysis and review reserve | 0.066248800 |
| Measured ATT-0271-1: 477 MIG seconds / 14,400 | 0.033125000 |
| **Total, including conditional branch and this smoke** | **2.538108543** |
| Experiment cap | 3.000000000 |
| Remaining projected headroom | 0.461891457 |

This is a measured projection, not a guarantee of final usage. The initial
plan is 1.447951515 node-hours including reserve and smoke. No sample/step
shortening is needed before review. Conditional r128 remains held until the
locked primary-task rule fires. All actual usage, including failures and any
reviewer jobs, counts toward 3 node-hours.

The ordinary launcher T-0028-run.pbs records profiles without a slice gate;
only the timing smoke keeps the existing 2g fail-fast guard (D-0019). Both
launchers require allocation_check before model/test imports. Shell syntax
passed locally; workload validation ran exclusively in PBS.

Review scope is the new segmented training/evaluation/intervention and exact
analysis path, the small initial_memory extension to BranchModel, their
dependencies, all deterministic tests and launchers. D-0044 and MSG-0726
are recorded in T-0028-protocol.md before any SEG quality was read. Hypothesis
status is unchanged; no scientific edge is inferred from the smoke.
