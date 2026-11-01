# T-0022 RB0 implementation record

Task baseline:de9da72 on sr/researcher-2/T-0022. Shared core/model default paths
remain reference;new fixed_query defaults False,impl defaults reference.
No dependency,existing tolerance,data split or scoring change is authorized.
All workload execution is in Miyabi-G PBS via miyabi-mcp.

FP32 convention fixed before any run,per coordinator MSG-0543:
for each output or q/k/v/Rq/Ro gradient tensor,pass if
max(abs(actual-reference)) <=1e-5+1e-4*max(abs(reference)). Report max absolute
and normwise relative max(abs(actual-reference))/max(max(abs(reference)),1e-8).
Do not impose these two error thresholds separately or use per-element ratios.
GPU BF16 and existing FP32 tolerances remain src/tolerances.json.

T0 ignores history-only interventions(window,strict,swap,permute);iv_off disables
the branch for every arm and therefore returns Base. F-line history diagnostics
require no-grad evaluation,full-sequence input,and no backbone cache;normal
prefix inference supports chunked prefill/decode through MemoryState. Unsupported
in-sequence reset is rejected explicitly. No new options affect old M/V5 paths.

The A2 manifest samples1000 IDs per task from V1 train using a fresh RNG with
derive_seed(20261101,residual_la_v21,RB5,confirmation) in each task. IDs remain in
source order in the manifest. Filtering happens before any RB training encoding;
the calibration and dev slices remain intact. No confirmation labels/content are
exported or evaluated. Formal RBc/RB1 are not released by this implementation task.

Initial core/split run ATT0180/PBS3767828/RES0176:260 checks passed in28.22s;
measured scheduler38MIGseconds (0.00263889node_h). Max prefix output error
9.536743e-7;all5 gradient comparisons passed the pre-fixed mixed normwise rule.
The A2 manifest has2000 IDs,IDs SHA256
72c91fd9ad6ec077e7aa33ec6381754d6e24d1f3e538a1ae8b88854b74dbcac5;
file SHA25634277aa4627271dbf4e0bcb9592e3014c6130d87b1766caad17c2d6b97a8a7cc,
24798bytes. Durable read-only source: runs/ATT-0180-1/rb-confirmation.json.
RB raw train counts after A2:PIQA12065,BoolQ5379,before encoding exclusions.

After that run,MSG0548 required a single recurrence implementation. Prefix now
flattens B*T causal states and calls the unchanged LatentReader.forward directly;
current-weight tracing observes state.read without changing loop semantics.
researcher3 was notified (MSG0551);their T0021 does not edit either shared module.
The expanded core/model regression will validate this consolidation before timing.

RB0 probe is20warmups+4measurement steps per timing case,not a stable timing
claim. Full benchmark is20+100steps for every RB1 arm plus reference S1.
One R_res S1 full dev pass anchors32rows/task samples for per-arm evaluation-cost
estimates. RBc estimates count2passes;RB1/RB5 count5(monitors,final,reload).
Intervention/bootstrap costs are explicitly excluded and must be added before
RB1 release. RB5 estimates assume the same monitoring;its later contract decides.
Formal train currently enforces the released RB1 recipe;the planner also supports
exact3epochs for RB5 cost estimates and smoke. No RB5 contract is fabricated.

Coordinator D0026/DOC0208 resolves ATT0186's three comparisons without changing
any threshold. Topology gates are the exact rounded output identity at each
adapted layer,U6 bitwise Base,and each Ui bitwise norm_i(Xi+oi). Residual
subtraction is diagnostic in ulp_bf16(Y),not a Base-logit-tolerance gate.
The BF16 chunk1 bound0.50002 is retained and reported as passed/exceeded;
chunk3 still gates. Prefix/reference chunk1 agreement also gates at0.50002.
A separate fresh FP32 backbone(TF32off) now checks full/chunk3/token logits
pairwise and stateM/c under the fixed E0019 normwise mixed bound. Deliberately
clearing only sideM/c/n at token boundaries must fail that same FP32 check.
The strong std0.5 exit fixture,all earlier passing assertions,and tolerance file
are unchanged. T0 now rejects non-KernelMatrix backend to avoid a dead extra
temperature parameter;researcher3 was notified before the shared schema edit.

ATT0187/PBS3767969/RES0182:395/397passed;D0026 topology and FP32 carry gates
passed,the state-clear variants were rejected,and every topology subtraction
was <=0.5ulp_bf16(Y). Maximum full/token FP32 logit error0.00019454956
(relative infinity norm1.10723e-5). Two previously unreached failures concerned
B2->B3 beam expansion(std0.5 exits),identical prefix/reference BF16 error
0.857421875 versus unchanged0.50002. Full timing still did not start.
Scheduler102MIGseconds recorded once;T0022 cumulative236seconds=0.01638889node_h.

Coordinator D0027/MSG0569 explicitly generalizes D0026:BF16 comparisons across
GEMM-shape changes with the same excess in unchanged reference are diagnostic;
the same comparison in FP32(TF32off) is decisive under the existing mixed gate.
Same-shape BF16 gates remain0.50002. A true same-B permutation,FP32B2->B3
expansion/ownership,and BF16prefix/reference expanded-batch agreement are added.
The original BF16expansion comparison is retained in named D0027 diagnostics
with its actual error and whether it meets the unchanged bound. No fixture or
tolerance has been changed. No more per-case rulings are needed for this rule.

ATT0188/PBS3767999/RES0183 complete:397checks passed;all10startup cases completed
20warmups+4measured steps on verified2g mg1098. Required D0026/27 gates passed;
the original BF16shape-change exceedances remain recorded as diagnostics.
Prefix/T0/LoRA startup steps0.209-0.241s;referenceS1 1.56685s. These are only
four measured steps,not the final100-step result. Scheduler221MIGseconds
imported once;T0022 subtotal457s=0.03173611node_h before full timing.

Measured plans after A2/excluding PIQA5528:RBc1096steps/2192examples/200229tokens;
RB1 13115steps/26230examples/2000069tokens;RB5 3epochs26165steps/52329examples/
3629259tokens. All per-arm counts and source hashes remain in ATT0188 outputs.

Full timing expected about900s/request1350s(debug-mig),from120times the sum of
startup step times(~427s),32rows/task sample evaluations(~144s),one complete
dev pass(~183s),seven512/128 latency cases(~65s) and setup/planning(~40s).
Request is1.5times expected. This is a planning estimate,not measured usage.
Full benchmark also reports Base latency. RBc cost estimates now scale the
dev-based pass estimate by4096calibration/2000dev rows,explicitly an approximation
that does not measure calibration length differences. No training/loss/model
path changed after the397passing checks;the new cost branch and Base timing
are exercised in the full benchmark. The experiment ceiling remains1node_h.

RB5 cost coverage:the RB1 LoRA4r7 proxy cannot stand in for full28-layer LoRA,
and r44/r64 matching widths or S4concat/T0 layouts have different compute.
src/rb5_cost.py therefore times11 additional diagnostic layouts(20+100steps,
same R1 examples/order/initialization seed as the main timing run):C_dual/C_upS4;
R_res/C_dual/C_upT0S4;r44/r64R_resS1/S4;LoRA28r1/r2. This changes no shared
model or training code. Actual model configs/counts/grad counts are asserted.
These are conditional RB5 cost proxies only,not formal models or performance
comparisons. S2 future-layout costs can be reported as interpolation,explicitly
estimated. Main RB0 full-dev timing will anchor their32row/task eval estimates.
Estimated extra runtime about450s/request675s,using startup0.21-0.24s/step,
historical full-LoRA0.38-0.41s/step and dev sample costs. Expected totalT0022
remains about0.13node_h versus ceiling1.0;use measured scheduler time for billing.
