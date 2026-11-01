# RB0 implementation evidence: E-0019@1 / H-0017@1

Diagnostic evidence for the implemented invariants, with independent code review.
RV-0060 passed with four lows; RV-0062 passed with zero findings and confirmed
all four fixes. Final executed code is183ad685785b8249d8a1c679a09e8dd5280d8cd5;
the recheck snapshot de4443b130653bfbd6c1418ce69d582ee75b57af adds only notes and
artifacts. Subsequent handoff metadata changes do not alter the reviewed scope.

## Implementation and checks

Implemented causal FP32 KernelMatrix prefixes, T0, default-off fixed_query/impl
wiring, F-line history interventions/current weights, A2 filtering and paired RB
training plans, RB family calibration/lock, checkpoint/export/reload, and latency
and cost tools. Prefix reads reuse the unchanged native LatentReader. Shared
defaults retain reference/False and pass bitwise task-baseline comparisons plus
the existing M/V5 regression suites. No dependencies or locked tolerances changed.

ATT-0197-1/RES-0191 ran all408tests in PBS on allocation-verified mg1098, with
zero failures/errors/skips. It also executed the actual trainer CLI for memory
RB1-02,T0 RB1-01 and LoRA RB1-07. Each smoke has4steps/8examples/744input tokens,
the expected parameters, finite calibration NLL, family/seed/phase/stage metadata,
matching split/order/counts, three monitor files, verified checkpoint inventory
and reload.identical=true. Their stage=smoke/phase=R1 records are not LR candidates.

| Required check | Final observation |
| --- | --- |
| Prefix/reference and same-kernel slots | 192 FP32 combinations pass outputs,M/c/counts and q/k/v/Rq/Ro gradients;T17/64,r8/32,B3,masks and initial states,strict/inclusive,S1/2/4,fixed query,update/readout variants |
| Four-layer zero exits | R_res/C_dual/C_up with memory prefix/reference and T0 all have zero logit error vs Base;one attention/FFN call per layer |
| Residual topology | Exact Y=(X+o+F)+bf16(ZP) at all four layers;U6 bitwise Base;every Ui bitwise layernorm_i(Xi+oi);original block once |
| Two-step gradients/frozen backbone | Exit-only first step,finite nonzero projection/read gradients after update at all layers;frozen original parameters unchanged |
| Causality/masks/cache/ownership | Required same-shape BF16 and fresh FP32 carry/expansion gates pass;true same-B permutation and prefix/reference token/expansion errors zero |
| Write/read counts and fixed query | One write/token and S reads/token;fixed-query S4 increment=4*S1 increment;no host item conversion in prefix autograd path |
| T0/history | T0 formula/history independence;window/strict/self-donor/identity-permutation/off identities,current weights and intervention causality pass |
| Swap/permutation mask handling | Unpadded/right-padded valid prefixes accepted;left padding/gaps explicitly rejected for recipient and swap donor |
| Checkpoint classes | Fresh-base reload logits/predictions identical for T0,memory,fixed-query,eta/4,concat and LoRA representatives |
| Counts/init | Stored/trainable/with-grad match R32/T0=663552,Cdual32=1318912,Cup32=925696,R44=916608,R64=1343488,LoRA4r7/28r1=630784,LoRA28r2=1261568;RB1-01..06 initial params identical |
| Negative controls | Actual patched duplicate-write and ignored-padding builders fail the reference state/output gate;doubled writes fail train accounting;future leakage and side-state-clear variants are rejected |
| Data/plans/LR/CLI | A2 regeneration/filtering and paired complete-step schedules;raw-minimum endpoint expansion and1%choice;null successful NLL/order/extension failures;real trainer CLI manifests pass |

FP32 tolerance was fixed before any run (MSG-0543): each tensor satisfies
max(abs(a-r))<=1e-5+1e-4*max(abs(r)); report relative_inf as
max(abs(a-r))/max(max(abs(r)),1e-8). Maxima below are from final checks and can
come from different cases in each column. Small-tensor gradient equivalence
was tested on CPU FP32; real model topology,gradient startup and carry gates
ran on GPU, including fresh FP32 backbone/TF32-off comparisons.

| Prefix/reference tensor | Max absolute | Max relative_inf |
| --- | ---: | ---: |
| Output | 9.536743e-7 | 5.305607e-7 |
| q gradient | 4.768372e-7 | 2.245134e-6 |
| k gradient | 1.788139e-7 | 3.267611e-7 |
| v gradient | 4.768372e-7 | 1.739042e-7 |
| Rq gradient | 7.152557e-7 | 7.622943e-7 |
| Ro gradient | 7.629395e-6 | 6.635038e-7 |

D-0026/D-0027 conventions are preserved. Strong four-layer S4 full/token BF16
error0.8203125 and batch2-to3 expansion0.857421875 both exceed unchanged0.50002
and match the unchanged reference;they remain diagnostics,not numerical passes.
Same-shape BF16 gates remain enforced. Prefix/reference differences for those
token/expansion cases are zero. Exact topology gates pass;subtraction diagnostics
at every layer are0.5ulp_bf16(Y),including layer27 absolute1.59375.
Decisive fresh FP32 carry max error0.00019454956/relative1.1072281e-5;expanded
prefix0.00017124414/9.745914e-6 and reference0.00024902821/1.4172782e-5 pass the
mixed gate. Both side-state-clear negatives are rejected. RV-0062 independently
confirmed every previously recorded numeric test property is unchanged.

## Timing, costs and A2

ATT-0189-1/RES-0185 and ATT-0191-1/RES-0186 measured21cases,20warmups followed
by100stable optimizer steps each,using identical first120R1batches on verified
2g timing allocations. Seven latency cases each have512-token prefill and128
decode steps (896raw samples),including Base and merged LoRA. Persistent M+c
is16896bytes for four r32 memory layers at batch1;backbone KV is separate.

Slowest measured prefix side-layout0.2438990201s/step is below0.4. R_resS1 prefix
is0.214969264s versus reference2.157638178s. The nine-arm RB1 core forecast is
2.443961088node_h. Updated RBc uses one load and one calibration evaluation,
without checkpoint export/reload:0.509314-0.679085node_h for12-16candidates.
Conditional per-RB5-layout three-epoch forecasts,training/latency tables and
source artifact FileRefs are in T-0022-cost-tables.md and
research/artifacts/T-0022/cost-summary.json,regenerated and validated in PBS
ATT-0197-1/cost-report. JSON SHA256:
5fe146e9c6d7f0e64cc236a3205e9f109d2b92f637bae4372cc7e1896faec56b.

These forecasts exclude frozen interventions,bootstrap/analysis,data planning
and cold-cache setup. Only R_resS1 full dev was timed (185.524491s);other evals
scale32rows/task to that anchor,RBc additionally4096/2000rows. Length differences
are unmeasured. RB5 monitoring/matrix is conditional;unmeasured S2 layouts need
labeled interpolation,S1concatT0/mixed-rank LoRA remain unmeasured. Costs for
future intervention/analysis work must be added before a whole RB1 release.

A2 has1000train IDs/task,24798bytes. Committed manifest
research/artifacts/T-0022/rb-confirmation.json is identical to read-only source
/work/xg24i002/x10041/bypass-lora/runs/ATT-0180-1/rb-confirmation.json.
SHA256:34277aa4627271dbf4e0bcb9592e3014c6130d87b1766caad17c2d6b97a8a7cc.
IDs SHA256:72c91fd9ad6ec077e7aa33ec6381754d6e24d1f3e538a1ae8b88854b74dbcac5.
Filtering precedes encoding;calibration/dev are intact. Excluded PIQA5528 has a
537-token complete candidate per the existing max_length rule. Resulting plans:
RBc1096steps/2192examples/200229tokens;RB113115/26230/2000069;
RB5three epochs26165/52329/3629259,with paired order hashes in the cost JSON.

## Provenance and accounting

All runs were registered before execution,used immutable source snapshots and
verified PBS allocations. Results and exact measured PBS walltime were imported
once per job accounting reference. Full run paths are under
/work/xg24i002/x10041/bypass-lora/runs/<attempt>/.

| Attempt | Result | PBS | MIG seconds | Role |
| --- | --- | --- | ---: | --- |
| ATT-0180-1 | RES-0176 complete | 3767828 | 38 | Initial260checks,A2IDs |
| ATT-0186-1 | RES-0177 partial/failed | 3767931 | 96 |390/393,pre-D0026,benchmark not started |
| ATT-0187-1 | RES-0182 partial/failed | 3767969 | 102 |395/397,pre-D0027,benchmark not started |
| ATT-0188-1 | RES-0183 complete | 3767999 | 221 |397checks,ten20+4startup cases |
| ATT-0189-1 | RES-0185 complete | 3768185 | 868 |Main timing,latency,full dev/save |
| ATT-0191-1 | RES-0186 complete | 3768226 | 430 |Eleven additional RB5layout timings |
| ATT-0193-1 | RES-0188 complete | 3768635 | 1 |Original timing/cost integrity report |
| ATT-0197-1 | RES-0191 complete | 3768755 | 162 |408checks,threeCLI smokes,revisedRBc costs |

Total1918MIGseconds/14400=0.1331944444node_h,within E-0019's1.0ceiling. Earlier
failed attempts remain engineering history and are not positive timing evidence.

No formal RBc/RB1/RB5 training,official validation or RB confirmation evaluation
was performed. Full calibration-stage/real-manifest lock CLI execution remains
for RBc;its component paths and one-eval/no-checkpoint branch are covered by the
named smokes/unit tests. The future RB1 intervention runner,n_eff/entropy metrics
and analysis are outside this package. These diagnostics concern implementation
invariants and measured compute only,not task performance or mechanism benefit.
Coordinator retains scientific release and hypothesis-status decisions.
