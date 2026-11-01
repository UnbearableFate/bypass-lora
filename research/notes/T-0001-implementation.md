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
