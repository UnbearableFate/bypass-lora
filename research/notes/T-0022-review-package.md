# T-0022 RB0 code review package

Historical first-review package (RV-0060,a5d04f8). Its measured timing and397-test
evidence remain valid. The four low follow-ups and the updated RBc single-pass
cost policy are in T-0022-review-followup.md;that note supersedes the two-pass
RBc forecast below. The revised PBS report is validated in ATT0197/RES0191.

Target: E-0019@1 / H-0017@1, task T-0022. This is an implementation and cost
diagnostic. No RBc, RB1 or RB5 scientific training or confirmation evaluation has
started. The coordinator owns release and hypothesis-state decisions.

## Code and compatibility

- `bypass_core.py`: inclusive/strict causal FP32 prefix states and continuation,
  prefix reads, T0. Prefix flattens B*T states and invokes the existing
  `LatentReader.forward`; its update/readout/fixed-query semantics are unchanged.
- `bypass_model.py`: appended default-off config fields (`fixed_query=False`,
  `impl=reference`), T0 wiring, optional history capture/interventions/read tracing.
  Ordinary F/M defaults retain their prior paths and parameter creation order.
- `rb_history.py`: off/window1/window8/strict/swap/permute and current-read weights.
  History diagnostics require full uncached no-grad inference; T0 ignores history
  changes except iv_off. No intervention edits the original attention or KV state.
- `rb_data.py`: A2 IDs-only split, exclusion before encoding, deterministic paired
  order, whole-step 2M-token and exact three-epoch plans.
- `rb_train.py`: nine RB1 arms, seeds, V1 objective/optimizer/scoring reuse,
  accounting, RBc grids/extension/lock, monitoring, final save and fresh reload.
  Formal train mode currently enforces the released RB1 recipe. RB5 planning and
  diagnostic layouts do not create a formal RB5 training contract.
- `rb_latency.py`, `rb_benchmark.py`, `rb5_cost.py`: synchronized timing on a
  verified 2g slice; all benchmark adapters are diagnostics. Merged LoRA is only
  a deployment timing comparator.
- `rb_check.py`, `rb0_run.py`, `rb_report.py`: PBS validation, terminal manifests,
  artifact verification and combined cost tables.

Baseline is de9da72acc1c9f9e0903ff1e144a87cc4ba4c615. New default-path regression
loads that snapshot's actual core/model modules and checks initial parameters
and logits bitwise for V1 tokenwise, F memory and M memory configurations.
The 397-test suite also includes existing core/model, V1 data/LoRA, V5 trainer
and V2 memory regressions. Researcher-3 confirmed T-0021 edits neither shared
module; there is no second implementation of the native read loop here.

No dependency or tolerance-file change. Environment is Python 3.13.13,
torch 2.10.0+cu128, transformers 4.57.1, cuda/12.8, lock 3c916b51337d.
Qwen3-0.6B-Base model/tokenizer revision is
da87bfb608c14b7cf20ba1ce41287e8de496c0cd; eager BF16 backbone, FP32 new params
and memory. FP32 comparison jobs disable TF32.

## Validation and numeric conventions

ATT-0188-1 / PBS 3767999.opbs / RES-0183: complete, validator passed, 397 passed,
zero failed, zero skipped; then all ten 20+4-step startup cases completed.
Exact executed source: 70a49714777960922fee7d03410faeaffe7acabf.

Runtime evidence (all absolute paths are in the shared main project):

- `/work/xg24i002/x10041/bypass-lora/runs/ATT-0188-1/checks/checks.xml`
- `/work/xg24i002/x10041/bypass-lora/runs/ATT-0188-1/checks/checks.log`
- `/work/xg24i002/x10041/bypass-lora/runs/ATT-0188-1/checks/manifest.json`
- `/work/xg24i002/x10041/bypass-lora/runs/ATT-0188-1/manifest.json`
- Its `allocation.json`, `mig-profile.json` and saved nvidia-smi output establish
  the allocation and the timing startup profile.

The coordinator fixed the FP32 rule before any run (MSG-0543): for **each tensor**,
`max(abs(a-r)) <= 1e-5 + 1e-4*max(abs(r))`. Both maximum absolute error and
`max(abs(a-r))/max(max(abs(r)),1e-8)` are reported. Separate absolute and relative
gates are not imposed. The following maxima are transcribed from RES-0183's
checks manifest; maxima in the two columns can come from different cases.

| FP32 prefix vs reference tensor | Max absolute error | Max relative infinity error |
| --- | ---: | ---: |
| Output | 9.536743e-7 | 5.305607e-7 |
| q gradient | 4.768372e-7 | 2.245134e-6 |
| k gradient | 1.788139e-7 | 3.267611e-7 |
| v gradient | 4.768372e-7 | 1.739042e-7 |
| Rq gradient | 7.152557e-7 | 7.622943e-7 |
| Ro gradient | 7.629395e-6 | 6.635038e-7 |

| Design check | Observed result |
| --- | --- |
| Prefix/reference and same-kernel slots | All 192 combinations pass outputs, M/c/counts and five gradients; B3, T17/64, r8/32, masks/initial state, inclusive/strict, S1/2/4, fixed query, both update/readout modes |
| Zero exits and original calls | Four-layer memory prefix/reference and T0, each R_res/C_dual/C_up: zero logit error; attention/FFN once per layer |
| Residual topology (D-0026) | Exact rounded output identity at all four layers; U6 bitwise Base; every Ui bitwise norm_i(Xi+oi); each block once |
| Residual subtraction diagnostic | All four layers at most 0.5 ulp_bf16(Y); layer27 absolute subtraction discrepancy 1.59375 is retained |
| Two-step gradients and frozen backbone | R_res S1/S4 and C_dual: exit-only first step, all new projections/read params finite nonzero after update; original params unchanged/no grads |
| Causality, masks, continuation and ownership | Required same-shape gates pass; true same-B permutation error zero; prefix reset request explicitly rejected |
| Writes/reads and native loops | Writes = valid tokens, reads = S*valid tokens; no prefix tensor-to-host item; fixed-query S4 increment equals four S1 increments |
| T0 and history | Formula/history independence, within-task donor derangement, strict first read, window/self-donor/identity permutation, current weights pass |
| Model interventions | iv_off equals Base (error zero), T0 unchanged; intervention causal prefix errors zero; identity effects within locked BF16 bound |
| Checkpoint classes | Fresh-base reload logits/predictions identical for T0, memory, fixed-query, eta/4, concat and LoRA representatives |
| Parameter counts/init pairing | R32/T0 663552, C_dual32 1318912, C_up32 925696, R44 916608, R64 1343488, LoRA4r7/28r1 630784, LoRA28r2 1261568; RB1-01..06 identical initialization |
| Deliberate broken controls | Full-sequence future leakage, duplicate-write accounting and padding writes rejected; clearing only side M/c/n across token decode rejected in FP32 |
| Data, planning, LR lock | A2 regeneration/filtering and unchanged calibration/dev, complete-step budget, paired order, family raw-minimum extension, 1% selection, successful null NLL/order mismatch/missing extension rejection pass |
| Shared defaults | Baseline initial parameters/logits bitwise equal; M/V5 regressions pass |

**D-0026 / D-0027 are part of the review scope.** Existing BF16 bound 0.50002
is unchanged. Strong four-layer S4 fixtures retain observed full-vs-token error
0.8203125 and B2-to-B3 expansion error 0.857421875, both identical in prefix and
unchanged reference. These cross-GEMM-shape observations exceed the BF16 bound
and are reported as diagnostics under the coordinator's rulings; they are not
claimed as passing that numerical bound. Prefix/reference token and expansion
logit differences are zero. Chunk3, true same-B permutation and other same-shape
BF16 comparisons remain gated. Fresh FP32 backbone/TF32-off comparisons are the
decisive carry/expansion gates: prefix full/token max 0.00019454956, relative
1.1072281e-5; expanded prefix max 0.00017124414, relative 9.745914e-6; expanded
reference max 0.00024902821, relative 1.4172782e-5. All pass the per-tensor mixed
rule. Both side-state-clear negative variants are rejected. No exit fixture,
locked threshold, data split or metric was changed to obtain those results.

Earlier failed attempts remain in the record: ATT-0186-1/RES-0177 (390/393) and
ATT-0187-1/RES-0182 (395/397). They stopped before their benchmark portions and
are not timing evidence. The rulings and regression changes are documented in
`T-0022-implementation.md`, D-0026 and D-0027.

## Timing and combined costs

ATT-0189-1 / PBS 3768185.opbs / RES-0185: complete; ten 20+100-step cases,
required latency cases, Base and merged LoRA, one complete dev pass and saves.
Source f493cec9bd36798a6706592f7d5f248dc45b1558, outputs
`/work/xg24i002/x10041/bypass-lora/runs/ATT-0189-1/benchmark/`.

ATT-0191-1 / PBS 3768226.opbs / RES-0186: complete; eleven additional conditional
RB5 width/depth/concat/T0/full-28-layer-LoRA layouts, each 20+100 steps.
Source b4229ed05a753f44056a3d3ec0bb79b752d7cca0, outputs
`/work/xg24i002/x10041/bypass-lora/runs/ATT-0191-1/`.

Both timing allocations pass their 2g.24gb fail-fast guard with nonempty allocated
MIG identities. Profile is metadata for ordinary jobs, per D-0019.

ATT-0193-1 / PBS 3768635.opbs / RES-0188 independently verifies the saved artifacts
and constructs the combined tables in PBS; source
1b607ae5b21f46b34fc46be6f1ab24317ce2ace3. It checks all 21 stable step logs,
parameter/gradient counts, writes/reads/original calls, identical input-token
sequences across cases/jobs, schedule agreement, numeric/training/dependency
hash equality to the 397-test snapshot, and seven latency cases (896 raw decode
samples, each 512-token prefill and 128 decode). Application validator passed.

Combined output:
`/work/xg24i002/x10041/bypass-lora/runs/ATT-0193-1/{manifest.json,cost-summary.json,cost-tables.md}`.
Small committed copies: `research/artifacts/T-0022/cost-summary.json` and
`research/notes/T-0022-cost-tables.md`. The JSON includes input paths/bytes/SHA256
and the exact source-file hash comparison. Raw step logs and latency samples
remain in their originating runs.

Slowest measured prefix side-layout step is 0.2438990201 s (C_dual32 S4), below
the 0.4 s profiling trigger. Reference R_res S1 is 2.157638178 s/step versus
prefix S1 0.214969264 s. RB1 nine-arm core total is **2.443961088 node_h**, below
4 for the scope measured. RBc estimate is 0.820830-1.094440 node_h for 12-16
candidates; conditional RB5 per-run costs are in the table.

These are forecasts, not billed usage. They use the first 120 paired R1 batches,
20 warmups/100 stable effective-batch2 microbatch1 steps; two loads plus planned
steps, sampled/scaled evals and save. The full R_res S1 dev pass took 185.524491 s;
other evals scale 32 rows/task against that anchor. RBc also scales the 4096/2000
row ratio, without measuring calibration length differences. RB1/RB5 estimates
assume five eval passes; RB5 monitoring/matrix remain conditional on its later
contract. Frozen interventions, bootstrap/analysis and data-planning overhead
must be added before the coordinator releases RB1. Unmeasured S2 layouts require
labeled interpolation; S1 concat T0 and mixed-rank LoRA are unmeasured (the latter
was explicitly deferred by E-0019 Design). No whole-release cost claim is made.

## A2 split and measured usage

Read-only split source:
`/work/xg24i002/x10041/bypass-lora/runs/ATT-0180-1/rb-confirmation.json`.
Committed identical copy: `research/artifacts/T-0022/rb-confirmation.json`.
1000 source-order IDs per task, 24798 bytes; SHA256
`34277aa4627271dbf4e0bcb9592e3014c6130d87b1766caad17c2d6b97a8a7cc`.
IDs SHA256 `72c91fd9ad6ec077e7aa33ec6381754d6e24d1f3e538a1ae8b88854b74dbcac5`.
No confirmation contents or official validation have been evaluated.

Measured schedule after A2 and the complete-candidate max_length exclusion:
RBc 1096 steps/2192 examples/200229 tokens; RB1 13115/26230/2000069;
RB5 three epochs 26165/52329/3629259. Hashes and epoch boundaries are preserved
in the cost JSON. PIQA row5528 is excluded because a complete candidate is537
tokens, following the existing V1 exclusion rule.

| Attempt | Result | PBS | Measured MIG seconds |
| --- | --- | --- | ---: |
| ATT-0180-1 | RES-0176 complete | 3767828 | 38 |
| ATT-0186-1 | RES-0177 partial/failed | 3767931 | 96 |
| ATT-0187-1 | RES-0182 partial/failed | 3767969 | 102 |
| ATT-0188-1 | RES-0183 complete | 3767999 | 221 |
| ATT-0189-1 | RES-0185 complete | 3768185 | 868 |
| ATT-0191-1 | RES-0186 complete | 3768226 | 430 |
| ATT-0193-1 | RES-0188 complete | 3768635 | 1 |

Each exact PBS walltime was imported once with its own `miyabi-pbs:<job>`
accounting reference. Total is **1756 MIG seconds / 14400 = 0.1219444444 node_h**,
within E-0019's 1.0 node_h ceiling. The generated cost table stops its usage
subtotal before the one-second tabulation job; this paragraph supplies that
final accounting increment without altering the executed output.

## Review request and completion boundary

Review the implementation against E-0019 plus MSG-0543, D-0026 and D-0027;
distinguish the explicit numeric diagnostics from required pass gates. Check the
new training/lock/split/export/history/latency paths for concrete result-integrity
risks and confirm shared defaults retain existing semantics. Runtime proof is
limited to the named checks and diagnostic benchmarks, not a full RBc/RB1 run.
After the independent answer, fix/test/recheck findings as required. On passing
review, record one diagnostic evidence item for H-0017's implementation scope
and deliver; the coordinator decides scientific release and hypothesis state.
