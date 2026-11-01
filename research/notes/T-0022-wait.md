# T-0022 continuation: waiting for coordinator on three BF16 checks

Task T0022,branch sr/researcher-2/T-0022. All source changes are committed at
1514f7057e7eb4696285899ec4aa0e1893843bea. No formal RBc/RB1/RB5 training
or confirmation evaluation is authorized or started. Do not deliver yet.

Read research/tasks/T-0022-brief.md and sealed E0019@1 in the main project.
Implementation notes:research/notes/T-0022-implementation.md. All runtime is
PBS via miyabi-mcp,including Python tests/data generation/analysis. Environment
unchanged3c916b51337d-cuda,cuda12.8,Qwen3-0.6B-Base revisionda87bfb...0cd.

## Implemented

- Shared bypass_core:default-preserving process_sequence impl dispatch,FP32
  causal prefix statistics,chunk/decode state,accounting,T0. Prefix recurrence
  now flattens B*T causal states and calls the unchanged LatentReader.forward;
  no duplicate recurrence semantics. Current read-weight tracing wraps state.read.
- bypass_model:appended config fixed_query=False,impl=reference;T0 creation order
  identical to memory;new options only affect requested paths. F-history calls
  require full uncached no-grad F evaluation. All normal M/V5 defaults preserved.
- rb_history.py:off/window1/window8/strict/same-task deranged swap/per-read valid
  history value permutation,current weights;no backbone KV intervention.
- rb_data.py:A2 confirmation IDs and filtering before training encoding,paired
  RB epoch streams and2M/3epoch plans.
- rb_train.py:9arms,parameter checks,seeds,example-mean V1 loss/scoring,
  RBc family grids/lock with null-NLL rejection,dev curves,final reload,
  device-side count accumulation,state bytes. No formal RB5 contract invented.
- rb_latency.py:512prefill/128decode,adapter state/peak memory,merged LoRA.
- rb_benchmark.py:20+100steps per9arms+S1reference,full dev anchor,saves and
  per-arm RBc/RB1/RB5 cost estimates. --probe uses20+4 and does no full timing.
- rb0_run.py orchestration;rb_check.py JUnit/numerical summaries;new core,data,
  train-rule and real-model tests. Existing core/model/M/V5 suites retained.

New source files are not reviewed yet. Required review is after checks/timing.
Do not start full timing until the current failed-check interpretation is resolved.

## Fixed tolerances and coordination

MSG0543 locked the FP32 criterion before any run:
max|actual-reference| <=1e-5+1e-4*max|reference|,per output/gradient tensor.
Report absolute max and relative infinity norm. No componentwise gate and no
independent absolute/relative thresholds. GPU BF16 remains tolerances.json:
atol0.50002,rtol0;zero-exit/causality use2e-5.

MSG0548:do not change existing LatentReader semantics;tell researcher3 before
shared-class edits. MSG0551 notified them. MSG0553 confirms T0021 edits neither
shared module and uses existing process_sequence;their loop diagnostics replay
in v3_loop.py. LatentReader is untouched by T0022. Shared defaults have passed
the task-master bitwise regression in the GPU run.

## Completed records

ATT0180-1 / PBS3767828 / RES0176 complete/pass:
260 core/data checks,including192all-option prefix/reference/slot cases and all
5gradients. Source2f1687b(before the single-reader consolidation). PBS38seconds
=0.00263889node_h,usage imported once. A2 split generated without model scoring.

A2 manifest:
 /work/xg24i002/x10041/bypass-lora/runs/ATT-0180-1/rb-confirmation.json
 fileSHA25634277aa4627271dbf4e0bcb9592e3014c6130d87b1766caad17c2d6b97a8a7cc
 24798bytes;read-only.
 IDs hash72c91fd9ad6ec077e7aa33ec6381754d6e24d1f3e538a1ae8b88854b74dbcac5.
 1000IDs/task;RB raw train after A2:PIQA12065,BoolQ5379(before encoding exclusions).
 Identical committed copy:research/artifacts/T-0022/rb-confirmation.json.

ATT0186-1 / PBS3767931 / RES0177 partial/failed:
393tests,390passed,3failed,0skips. Source1514f7057e7eb4696285899ec4aa0e1893843bea,
allocated mg1098;exact2g timing guard passed. Full core matrix after consolidation,
4layer zero exits,grad startup/frozen Jacobian,all parameter counts,paired init,
history identities/causality,fresh reloads and bitwise task-master defaults pass.
The orchestrator stopped after tests:benchmark did NOT start,so no step timing
or latency result exists yet. PBS96seconds=0.00666667node_h,imported once.
Known T0022 subtotal:134MIGseconds=0.00930556node_h. No live jobs remain.

Output /work/xg24i002/x10041/bypass-lora/runs/ATT-0186-1/.
Read checks/checks.log and checks/checks.xml for failures.
Parent manifest.json is partial with validator false;do not call this a pass.
Notification N-20261102-020337-a6a7 may still be delivered by coordinator's daemon;
it describes this already recorded failure,not a new attempt.

## Pending coordinator question MSG0562

Three failures,with no production or test changes made after observing them:

1. test_residual_topology_uses_unchanged_first_FFN_input:
   Exact torch.equal(Y,(X+attn+FFN)+cast(ZP)) PASSES for each layer.
   Layer6FFN input is bitwise Base. The additional subtraction
   Y.float()-(H+F).float() vs cast(ZP).float() exceeds locked0.50002 at layer27
   (max1.59375),because BF16 addition at large hidden magnitudes cannot recover
   its small addend by subtraction. Existing helper nonzero_exits uses std0.5.
2,3. test_four_layer_cache_causality_padding_and_ownership[prefix/reference]:
   Both pass chunk3 but fail chunk1 full-logit bound identically:
   max0.8203125 >0.50002. Both use four-layer S4 and strong std0.5 exits.
   This also affects unchanged reference;it is not prefix/reference disagreement.
   No threshold,fixture or assertion has been weakened to pass.

MSG0562 asks the coordinator to resolve the correct rounded-topology comparison
and cache condition under E0019 before any retest changes. Wait for that answer;
do not reinterpret the sealed tolerance unilaterally. No H0017 evidence-direction
claim or hypothesis-state change has been recorded.

## Next work after answer

Apply only the approved check convention;keep thresholds locked. Investigate any
remaining failures before source changes. Rerun affected PBS checks. Then run
uniform20+4 startup probe (it never ran),size full20+100timing from measurements,
and run required512/128latency/full dev/save measurements on guard-verified2g.
Use new attempts and immutable snapshots,never overwrite old failed outputs.

A static follow-up to consider before final review:T0 with backend other than
KernelMatrix can currently create an unused temperature parameter through the
generic constructor. Reject unsupported T0 backend explicitly and add one narrow
config regression;all registered arms already use KernelMatrix. This is a schema
gap unrelated to the three observed failures.

If prefix stable step>0.4s,profile before RB1. If projected RB1>4node_h,report
coordinator;do not shorten arms yourself. Cost table explicitly excludes later
intervention/bootstrap costs,which coordinator must add before release.
After all required checks/timing,request independent code review through rl
(targetE0019,exact source/test/config scope);freeze reviewed files. On pass,
record one diagnostic evidence linked supportsH0017 (no task performance claim),
complete check/timing/split/review/usage report and deliver T0022. Coordinator
owns RBc/RB1 release. Do not touch any confirmation/official validation content.
