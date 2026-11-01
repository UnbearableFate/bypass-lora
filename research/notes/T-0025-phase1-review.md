# T-0025 RB1 phase-one review package

Target: E-0021@1 (DOC-0235@1), single-seed screening. Phase two requires the
coordinator's notice that T-0024's RBc LR lock is merged, and a passing review.
No formal RB1 training or result input and no RB confirmation evaluation has
been performed by phase one. Only synthetic outcomes and synthetic text are
used in the new checks. Base ATT-0024-1 is an existing V1 dev result.

Initial implementation/test snapshot: a62cf30d9459ca913e3da569c475214910e57884.
PBS checks: ATT-0228-1 / 3769016.opbs. Runtime artifacts are in the shared main
project at `runs/ATT-0228-1/{manifest.json,checks.xml,checks.log,allocation.json}`.
The failed result is retained as RES-0207. Final numerical source is
5d301e0ca1c6685ffd292ce9cbf908c16b15b693; its ATT-0230-1 / PBS 3769041.opbs
replacement passed **336 tests, zero failures, zero skips** in 54.65 pytest
seconds. Application manifest is complete and validator-passed. The fresh PBS
record reports Exit0, 57 seconds on mg1098, and the allocation record confirms
that exact compute host/job before execution. All ten planned test modules are
present in JUnit (six new modules, the four existing core/data/training suites).
The replacement also compares default prefix outputs, final state and all input/
reader gradients bitwise against the actual reviewed pre-observer 6df9e479
source for inclusive S1, strict S4 and fixed-query S4. Model scoring tests cover
all requested four-layer intervention targets with synthetic text only.

Both attempts' native scheduler observations and measured usage are imported
once from `research/artifacts/T-0025/ATT-0228-1-observations.json` and
`ATT-0230-1-observations.json` in the shared main project. Total phase-one test
usage is 115 MIG seconds = 0.00798611 node hours (0.007986 using rounded native
MIG-hour exports), well within the phase-one and E-0021 allowances. No additional
successful tests are planned without a review finding or another material change.

## Implementation and unchanged contracts

- `bypass_core.read_kernel_prefix`: optional `observe_read=None` callback around
  the existing MemoryState.read invocation. The native LatentReader is called
  once, with its existing Rq/Ro/update/readout/fixed-query logic. No replacement
  loop or numerical change when the callback is omitted.
- `rb_history.run_history`: only trace_reads=True creates normalized read
  diagnostics. First and last native queries are observed, with the existing
  current-weight diagnostic preserved. Ordinary training and calibration do
  not request tracing. Researcher-2 was notified (MSG-0620), and confirmed no
  shared-file conflict (MSG-0622). No rb_train/bypass_model/data/dependency edit.
- `rb_read_diagnostics`: explicit eligible positive-key dot products, normalized
  like v2_memory.read_metrics for n_eff and entropy, plus normalized current
  weight and n_eff/readable positions. Empty reads are zero; padding is excluded.
  Window, strict, permuted and swapped-prefix eligible sets match rb_history.
  Swap includes only the donor past plus recipient current; no donor future.
  First/last diagnostics are token-weighted, per task/layer/step, both for all
  valid gold-candidate input positions and positions predicting gold answer
  tokens. Gold is used for reporting after identical two-candidate scoring;
  it is never supplied as extra model input or used to choose a donor.
- `rb1_intervene`: delegates encoding/masking/scoring/ties/NLL to the unchanged
  v1_run.evaluate and token_losses through a forward proxy. Donors are a fixed
  R1 within-task derangement, from a normal forward of the same checkpoint and
  candidate index. Four layers are intervened simultaneously. No backbone KV
  cache is used. Independent Base scoring temporarily unwraps the original
  layers in the same allocation; wrappers are restored in finally. iv_off must
  exactly match this Base; T0 iv_swap must exactly match normal. Cross-job bitwise
  comparisons are not added; producer reload identity is already required.
- `rb1_artifacts`: binds complete E-0021 train/R1/seed0 manifests to actual LR
  lock, A2 split, checkpoint inventory/config/weights; checks completed sample
  budget, first whole-step stopping, original warmup/cosine rates, finite steps
  and parameter accounting. Paths/hashes/bytes are retained as references.
- `rb1_rules`: exact Fraction thresholds for rules 1-11. S2 wins only when
  abs(S2-S4)<0.5 pp; C* candidates must be effective and strictly <1 pp from the
  best concat arm before parameter/macro/NLL tie ordering. GR's -1 pp gate is
  separate from H-0013's +1 pp screening direction. GH drops use the intervention
  job's own normal condition when present. D-0032 fixes rule10's band as
  Base-1 <= arm < Base+1, so an exact +1 pp effective arm is outside the band.
  If raw GC holds without an effective concat, C* remains undefined and GC
  does not release runs; routing treats GC as failed while Rule11 directions
  retain the raw GC gate. Raw gates and resolved actions are both visible.
  No hypothesis state or job is changed.
- `rb1_analysis`: select mode authenticates all nine training runs/curves and
  reports the pre-registered target selection. Final mode additionally requires
  the full intervention grid, paired dev IDs/golds/token counts, same-allocation
  controls, complete read strata, checkpoint-bound latency and measured costs.
  The paired task-stratified row bootstrap uses default_rng(20261101), 10000
  samples and shared indices across all arms/conditions. Intervals are
  descriptive, conditional on this seed and without selection adjustment.
  Cost imports deduplicate accounting refs; required train/intervention/latency
  attempts must all have measured usage. The current analysis job is recorded
  after its scheduler end. Reports include plan2 16.5 and analytic dense side
  MAC components, with omitted non-MAC operations identified.
- `rb1_latency`: required five trained arms plus merged LoRA, optional Base.
  Uses rb_latency.measure unchanged: batch1, 512-token prefill, 4 warmup/5
  measured passes, 128 timed decode steps after separate warmup. The existing
  live UUID-bound 2g guard runs before loading a model. Full raw samples and
  recomputed summaries/throughput/state sizes are validated. Merged LoRA is a
  timing reference only. Ordinary jobs record hardware metadata under D-0019.

## Test scope

`rb1_check.py` runs the existing core/RB core/data/training suites and all six
new test modules. New checks include explicit independent slot distributions,
native observer output/gradient identity, first/last and fixed-query reads,
all exact threshold boundaries and advancement/evidence distinctions, invalid
or swapped checkpoints, budget/LR failure paths, timing completeness/guard order,
paired membership/tie/NLL failures, complete synthetic 1000-row-per-task final
analysis with the actual 10000-sample bootstrap, and real pinned Qwen synthetic
text scoring for T0/R_res S1/S4/C_dual/C_up with all required interventions.

The PBS launcher verifies the live allocation before project Python and uses
the existing CUDA12.8 environment at envs/3c916b51337d-cuda. Dependencies and
tolerances are unchanged. Model/tokenizer revision remains
da87bfb608c14b7cf20ba1ce41287e8de496c0cd. Tests record the MIG listing as metadata.

Initial ATT-0228-1 completed 330 tests: 327 passed, 3 failed, zero skipped.
The three failures were incorrect synthetic test setup/expectations: omitted
branch=memory in a fixed-query configuration; a GR -1 pp fixture that had not
also met effectiveness; and a bootstrap lookup with reversed comparison order.
Real-model scoring/intervention tests and full synthetic final analysis passed.
The initial result remains a failed diagnostic and is not a review pass.
D-0032 was recorded during this job, before any actual RB1 result input, and is
implemented before the next PBS regression and review.

## Phase-two entry points (not yet released)

- Training remains `rb_train.py run --stage train --arm RB1-XX --phase R1
  --seed 0 --lr <locked family LR> --lr-lock <immutable absolute lock>
  --split-manifest <immutable absolute A2 manifest> --data-dir <V1> --output <ATT>`.
  All nine share a notify group. No changes to the reviewed calibration path or
  lock format. Formal registration will name the final reviewed source.
- `rb1_analysis.py --input <JSON> --stage select --output <ATT>` authenticates
  the nine final runs and selects S_F*/C*. It does not release another run.
- `rb1_intervene.py --arm <01/02/S_F*/08/09> --training-run <formal ATT>
  --data-dir <V1> --output <ATT>`; 01 runs swap only, other targets all six modes.
- `rb1_latency.py --checkpoints <JSON map of five arms to formal ATT roots>
  --output <ATT> [--include-base]` runs on 2g only.
- `rb1_analysis.py --input <JSON> --stage final --output <ATT>` requires
  training_runs, base_run, intervention_runs, latency_run and observations.

## Cost and limits

T-0022 forecasts the nine training/monitor/reload runs at 2.443961 node hours.
Frozen evaluations, diagnostic tracing, fresh Base checks, permutation and
donor forwards are additional; measure their smoke cost before submitting the
phase-two intervention grid. The E-0021 cap is 4 node hours including phase-one
tests/review regressions, interventions, latency and analysis. Phase two has not
started. No confirmatory claim, same-depth T0 inference, parameter-efficiency
claim, novel confirmation release or cross-line numerical pooling is permitted.
