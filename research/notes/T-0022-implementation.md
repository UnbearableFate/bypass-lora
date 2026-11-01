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
