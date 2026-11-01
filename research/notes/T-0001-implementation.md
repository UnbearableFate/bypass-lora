# T-0001 implementation handoff

Source is new implementation; no old LoopLora code is imported or copied.

## Environment

Python 3.13.13, uv.lock blob c073f466bd0e99a2c677dfca66289d1887f7575a,
extra cuda, module cuda/12.8, torch 2.10.0+cu128, transformers 4.57.1.
Environment: `/work/xg24i002/x10041/bypass-lora/envs/c073f466bd0e-cuda`.
Model and tokenizer: Qwen/Qwen3-0.6B-Base at
`da87bfb608c14b7cf20ba1ce41287e8de496c0cd`; offline HF_HOME set in jobs.
Hardware calibrated: NVIDIA GH200 120GB MIG 2g.24gb, driver 615.71.09.

ATT-0001-1 failed before the workload because the PBS launcher omitted cd.
ATT-0001-2 corrected the launcher, confirmed allocation and outbound HTTPS (RES-0001).
ATT-0002-1 installed from the shared lock on compute and measured Base variation (RES-0002).
The pre-fixed rule gave BF16 atol 0.50002, rtol 0. Full/chunk3 repeats were exact;
tokenwise decode differed by up to .25. The tolerance config was committed before
adapter checks. Zero-exit and valid-prefix causality additionally use 2e-5.

ATT-0003-1: 54/60 pass; all six failures were the causal comparison of padded
prefix logits, after cache comparisons had already passed. The pinned eager
backend uses finite-min masking followed by softmax; fully masked query rows
have no semantic output contract. The corrected test selects valid prefix rows
with unchanged tolerance, and adds a Base-only diagnostic reproducing padding
sensitivity. Production code was not changed by that correction.

## API

Add `src` to PYTHONPATH. `BranchModel(base, AdapterConfig(...))` freezes all original
parameters and inserts adapters. BranchModel mutates the supplied Base once.
Call `adapter_parameters()` to construct an optimizer; do not cast the entire
wrapper to BF16, because the side branch uses FP32 parameters and memory.
Forward returns logits, a BranchCache, and observed per-layer accounting.

V1 defaults: tokenwise_silu, rank32, layers (6,13,20,27). tokenwise_linear is
separate. All six fusion interfaces are available, with zero-initialized exits.

F memory: branch=memory, phase='F'; supply cache plus the complete prefix attention
mask for chunked or tokenwise decode. Side state is returned explicitly. Start a
new sequence with cache=None. Functional MemoryState.reset/reorder handles row
resets and owner/beam maps. Callers must apply the same map to token inputs.

M memory: support=model(ids, phase='support', write_mask=delimiter_mask); query=model(query_ids,
phase='query', memory=support.cache.memory). Support produces no backbone KV.
Each independent query starts from the same supplied memory snapshot. A query
may use its own returned cache for subsequent query/answer decoding. Cross-phase
backbone caches are rejected. Support memory must be recomputed after every
parameter update; never cache trainable M/c/slots across optimizer steps.

The default visible_fact_delimiter scope requires an explicit boolean write_mask,
a subset of valid tokens, constructed from public fact delimiters. Support-only
write masks are rejected in F/query phases. Slot backends reject support counts
above the configured capacity (default16). The explicit all_support_tokens scope
permits omission of write_mask for separately registered variants. Physical slot
columns are padded across batch rows; per-owner counts record effective writes.
Backbone KV caches are consumed by HF in place: copy via cache.reorder before
branching. A stale/reused cache with mismatched prefix length is rejected, as is
query memory missing a configured adapter layer.

The original attention/MLP each execute once per selected layer call. Hooks
capture pre-RoPE Q/K and pre-repeat V without repeating projections. Fusion
hooks inject gate/up terms only while the original MLP executes. Hooks are
removed on exceptions. The wrapper is synchronous and must not be shared by
concurrent forwards. Gradient checkpoint replay is explicitly rejected because
it requires a replay-safe state protocol not implemented here.

`export_adapter(directory)` writes configuration and adapter-only weights.
`BranchModel.load_adapter(fresh_base, directory)` restores them; callers load the
pinned Base revision independently. No optimizer or task-training loop is included.

Core implementations are differentiable reference scans, with explicit mask
synchronizations for accounting; their throughput has not been calibrated for
V2 training. No training or scientific mechanism conclusion is claimed here.

## Final validation and delivery

Reviewed implementation commit: `a40f8ff70f10132edb5e992580227a63bf1b5649`.
RV-0001 reported two medium and seven low findings. All nine were addressed;
Claude recheck **RV-0002 passed**, with two new nonblocking lows below. Reviewed
source/config files remain unchanged after a40f8ff; this handoff update is notes only.

| Attempt | Result / evidence | Observed outcome |
| --- | --- | --- |
| ATT-0001-1 | launch log, no result | PBS working-directory error before probe; corrected launcher |
| ATT-0001-2 | RES-0001 / EV-0001 | allocated hardware/network probe passed |
| ATT-0002-1 | RES-0002 | Base-only numerical calibration completed and tolerances locked |
| ATT-0003-1 | RES-0003 / EV-0002 | 54/60 passed; six padded-prefix assertion failures diagnosed |
| ATT-0004-1 | RES-0004 / EV-0003 | 61/61 passed, including Base padding diagnostic |
| ATT-0005-1 | RES-0005 / EV-0004 | 79/85 passed; six newly added positive controls needed stronger test exits |
| ATT-0006-1 | RES-0006 / EV-0005 | **85/85 passed** (44 core, 41 real-model cases) |

Final checks cover all six fusions with zero exits, both tokenwise proxies and
memory, two-step gradient startup, frozen-backbone input Jacobian, same-kernel
output/QKV-gradient identity with mixed masks/empty rows/reset and ranks32/128,
actual fusion-hook output/P-gradient identity, fixed-query magnitude,
inclusive/strict causality, cache equivalence within locked tolerances,
reset/padding, owner reorder, observed read and write counts, support/query
isolation and explicit delimiter write masks, capacity errors for slot backends,
stale-cache and missing-memory rejection, and adapter export/reload.

Negative controls exercise the real invariant assertions with a full-sequence
future leak, repeated writer, mask-ignoring writer, skipped reader step, and
dropped kernel normalization. Strong nonzero-exit fixtures give side-effect
maxima1.84375–6.890625 and support-owner effects3.53125–3.6484375, exceeding the
unchanged BF16 threshold0.50002. Production exits remain zero initialized.
Actual fusion P-gradient maximum error is7.15e-7; backbone parameter change,
causal valid-prefix error, shared-support/reorder error and reload error are0.

Final validation ran on compute host mg1098, PBS3758212.opbs, GH200 MIG2g.24gb;
earlier checks also used mg1102 with the same hardware profile. All application
execution was in PBS allocations. Environment ref remains lock c073f466bd0e,
extra cuda, module cuda/12.8, torch2.10.0+cu128, transformers4.57.1,
Python3.13.13, model/tokenizer revision da87bfb608c14b7cf20ba1ce41287e8de496c0cd.

Miyabi MCP ledger for all seven task jobs reports **0.045278 MIG instance-hours
used**, 0.366667 requested. This is the executor's MIG billing unit, not an
invented full-node conversion. Training tokens consumed:0. Seven jobs ended:
four succeeded and three diagnostic/engineering failures are retained above.

## Required follow-ups from RV-0002 (coordinator MSG-0019)

1. **Before any experiment using chunked or cached memory paths:** add the FP32
   chunk3 state comparison for M/c (and slots where applicable), using the locked
   FP32 tolerances, and record `cache.chunk3.state.max_abs_err`. Also record
   chunk1 state differences as diagnostics. Rank128/S4 chunk3 logit differences
   of0.171875/0.203125 pass the BF16 tolerance, but their numerical cause has not
   been measured; the review's GEMM-rounding explanation is only an inference.
2. **Before V2 runs:** enforce the support write capacity for kernel_matrix as
   well as slot backends under visible_fact_delimiter, and add kernel_matrix to
   the overflow rejection test. Currently KernelMatrix depends on the caller's
   mask validation for the16-fact limit; it must not silently receive a different
   write count from the slot arms.

These two lows remain open by coordinator instruction. Neither affects the V1
tokenwise, non-memory path. V2 throughput/memory and sdpa/flash backends remain
unvalidated. Coordinator acceptance and scientific route decisions are separate
from this implementation delivery.
