# T-0023 implementation: E-0022@2 V3 training matrix (common start + plan §9.2 ten arms)

## Scope
- New files only: `src/v3_train.py` and `src/test_v3_train.py`. No existing source file changes: `bypass_core.py`, `bypass_model.py`, `v3_loop.py` and the reviewed evaluators are unchanged (MSG-0549). Nothing in researcher-2's shared code is touched.
- Arm options are default-off. `Variant()` with the start configuration is the stock D5 model. A test checks that V3-01 built through `arm_model` gives logits bitwise equal to `adapted(checkpoint=...)`.

## Data and order (T2, ATT-0077-1, metadata `2ad8cc35…`)
- Data comes from `v3_loop.load_t2`, which accepts released train and dev only and pins the metadata digest. Hops 4/6/8, calibration and confirmation are never opened by the training or evaluation commands.
- Start stage: T2 train rows with hop 1 (6,144 rows). Ten arms: all 18,432 train rows, hops 1–3.
- `v3_schedule(rows, stage)`:
  - base = `derive_seed(20261029, "concat_loop_v2", "V3", stage, 0)` with stage = `start` or `train`, as the contract specifies.
  - Epoch e is `random.Random(base + e)` shuffling all row indices. This is the same pattern as the D5 order rule `make_arm_schedule`, but it is not limited to three epochs.
  - Epochs are concatenated until 6,000 × 8 episodes. The start stage needs 8 epochs of the hop-1 rows; the ten arms need 3 epochs.
  - Warmup is ceil(0.05 × 6000) = 300, followed by cosine (`v2_train.train`).
  - Every arm uses the identical `train` schedule, so the samples and their order match across arms. `order_sha256` is recorded in each manifest.
- Recipe: `v2_train.train` with LR 3e-3, a fresh AdamW (weight decay 0), 8 episodes per step, gradient clipping at 1.0 as in D5, and all side parameters trainable. The training forward is `v2r16_memory.train_forward(support_fn=isolated_support)`, the D5 isolated write, unchanged.

## Start stage (`v3_train.py start`, arm V3-S0)
1. Provenance of ATT-0159-1 comes from `v3_loop.adapter_provenance`: a completed formal isolated-write D5 run, manifest files matching, and `adapter.pt` equal to `c4f3f9f8…`, checked again after loading.
2. Train for 6,000 steps on hop-1 questions. The memory is still built from all 16 isolated facts.
3. Curves at 25, 50 and 75% are run on dev hop 1 with the reviewed evaluator (normal, zero, shuffled). They are descriptive and are not used to select a checkpoint. 100% is the final evaluation.
4. The final evaluation is the reviewed E-0016 evaluator (`v2r16_memory.evaluate`, isolated write, all six controls including oracle_key and oracle_value) on the full T2 dev set. Results are grouped by hop with `v3_loop.reviewed_by_hop`.
5. Gate: `v3_loop.hop1_gate`, which is `v2r16_isolated.gate`, the exact-fraction E-0010@2 H3 on hop 1 at S=1.
6. Export, reload with `load_arm`, and re-evaluate. The full per-example records and summaries must be identical.
7. The manifest carries `metrics.hop1_gate`. The arm runs accept only a formal V3-S0 run with `hop1_gate.passed`, and pin its `adapter.pt` digest (`start_provenance`).

## Arms (`v3_train.py arm --arm V3-xx`)
| Arm | How it is built |
| --- | --- |
| V3-01/02/03 | `steps` = 1/2/4 (AdapterConfig field) |
| V3-04 | `steps=4, eta=0.0625`; reader.eta is set from the config and checked on reload |
| V3-05 | `FixedQueryReader`: a LatentReader subclass whose forward calls `LatentReader.forward(..., fixed_query=True)`. SideBranch does not pass `fixed_query`, so the subclass only sets the flag; the loop is the shared one. Test: equal to the stock reader called with `fixed_query=True` at S4, and Z = 4·(S=1 increment) on the core reader. |
| V3-06 | `TokenwiseKernelSide` (new): in the question phase z⁰ = q̄, a = SiLU(N(z)Rq + N(k̄_q)) ⊙ v̄_q, z ← z + η a Ro, Z = z^S − z⁰, with k̄_q, v̄_q from the question token's own projections. The support phase is SideBranch's own, so the write protocol and accounting are unchanged; memory reads are 0. It uses the common start's A_q/A_k/A_v/Rq/Ro/P. Tested against a float64 reference at S = 1/2/4/8. |
| V3-07 | rank 152 (AdapterConfig field), widened from the common start by `widen` (below) |
| V3-08 | `UntiedReader` (new): Rq/Ro for step 1 plus `untied_Rq/Ro.{0,1,2}` for steps 2–4, initialized as copies. The loop is LatentReader's expression by expression. S_eval is limited to 1/2/4, and S > 4 is refused. |
| V3-09 | `steps=4, update='anchored'` |
| V3-10 | `steps=4, readout='full'` |

- Variants are installed by `install_variant` on the fresh layer-14 side (class swap plus new ParameterLists). Each checkpoint holds the stock `config.json` and `adapter.pt` plus `variant.json`.
- `load_arm` rebuilds the stock config, installs the recorded variant, then loads strictly.
- `adapted()` restores the base layers on exit; a test checks this.

### V3-07 width (E-0022@2, MSG-0572/0573)
Per-question-token MAC for the direct_hidden F_out SoftmaxSlot M line with n = 16 written slots:
- Projections A_q/A_k/A_v: 3·1024r. The question token computes all three in this implementation.
- P: 1024r.
- Per step: Rq r², scores n·r, weighted values n·r, Ro r².

MAC(r, S) = 4096r + S(2r² + 2nr)
- r128/S4: 524,288 + 4 · 36,864 = 671,744.
- r'/S1: 4128r' + 2r'² = 671,744 gives r' = 151.6, so r' = 152: 673,664 (+0.29%). A test checks that 152 is the nearest integer.
- Support-side projection cost per support token is 3·1024r, which grows by 152/128 = +18.75%. This is reported separately.
- Reported MACs are executed per question token (`macs_per_question_token`, RV-0058). The fixed-query arm computes N(z⁰)Rq once, giving 4096r + r² + S(r² + 2nr) = 622,592 for V3-05. The tokenwise kernel is 4096r + 2Sr² = 655,360 for V3-06. The other arms follow 4096r + S(2r² + 2nr).
- No latency-matching claim: the 2g step time is dominated by the token loop. The measured stable step time of V3-07 is recorded (`train.stable_step_seconds`).

Why exact initial identity is impossible:
- Keys are φ(N_r(k̄)) with no matrix after the RMS norm. For 128 < r' < 256, preserving ‖k̄‖²/r for all inputs needs BᵀB = c·I₁₂₈ with rank(B) < 128, which cannot hold.
- At r' = 256, a copy keeps N, but the fixed 1/√r' score scale multiplies every score by √2.

Initialization (`widen`):
- The first 128 dims are the common start.
- The new A_q, A_k, A_v columns are drawn as randn/√d from a CPU generator seeded with `derive_seed(20261029, "concat_loop_v2", "V3", "V3-07", "widen", 0)`, in the order A_q, A_k, A_v.
- Rq' = blockdiag(Rq, I₂₄), Ro' = blockdiag(Ro, I₂₄), and the new P rows are 0. Every new parameter has a nonzero gradient (test).
- Recorded step-0 gap (descriptive): max|Δlogit| on the first 16 dev rows (normal memory) against the common start at S=1, and dev hop-1 EM before training.
- Rule 4(c) does not bind when the V3-07 primary is below V3-01; `decide` implements this and the report states the limitation.

### Initialization checks
- V3-08: on the first 16 dev rows (normal memory), the untied model's logits must equal bitwise those of the same weights through the stock shared reader at S4, which is V3-03 from the common start. Otherwise the job fails.
- V3-07: the gap is recorded, as described above.

## Evaluation (each arm job)
`evaluate_arm` generalizes `v3_loop.evaluate_loop_batch` to the variants:
- It uses the native forward at S_eval through the scoped `eval_steps`, and the reviewed control memories (`v3_loop._snapshots`: normal, zero, shuffled with the global (i−1) donor, permuted) in batches of 16.
- Asserted: memory unchanged, writes = 0, reads = S × valid question tokens (0 for V3-06).
- An explicit replay (`replay`) gives the per-step diagnostics and must reproduce the native output within ‖ΔZ‖/‖Z‖ ≤ 1e-3 per valid position (`v3_loop.RECONSTRUCTION_TOLERANCE`). The replay's slot read uses `read_vectors`' expressions.
- Recorded per step: state norm, read norm, the adjacent-step query cosine `query_cosine_prev_*` (query(s) against query(s−1); None at step 1, as RV-0058 requested), the cosine to step 1 `query_cosine_*`, weight on the path's j-th hop fact (all positions and last), n_eff, and last argmax. For V3-06 the query is the pre-activation N(z)Rq. `_all` is the mean over valid question positions, token-weighted in summaries; `_last` is the last valid position. Path weights are None for shuffled and V3-06.
- Main evaluation is at S_eval = S_train with four controls. The scan adds the normal control at S ∈ {1, 2, 4, 8}, or {1, 2, 4} for V3-08.
- S1 arms (V3-01, V3-07): the S=1 records must equal the reviewed E-0016 evaluator example by example (`compare_to_reviewed`), as in E-0018.
- Reload: export, `load_arm`, re-evaluate at S_train with four controls. The full records must be identical.
- The E-0011 prior and the structural-shortcut baselines come from the T2 metadata (`data.baselines`). Parameter counts, stable step seconds, peak memory, MACs, reads per question token and node·h are recorded.

## Interventions (`v3_train.py intervene`, S_M* arm only, after analysis picks it)
On the normal memory, z^(s) for 1 ≤ s < S is replaced at every valid question position. Steps s+1..S then read the recipient's own memory. Z = z^S − z⁰ is injected through a side forward hook.
- random: a Gaussian with the per-position norm of z^(s). The generator is seeded with `derive_seed(…, "V3", "intervention", s)`.
- unrelated: z^(s) of the global donor episode (i−1) computed on its own memory, so it carries another episode's entities. Positions are right-aligned (recipient t ← donor t + len_d − len_r, clamped).
- matched: z^(s) of the nearest previous dev row with the same hop and the same question length, on its own memory. Positions align exactly. Rows without such a donor are counted and excluded, and `none` is also reported on the matched subset.
- none: the same replay-and-inject path without a replacement. Its predictions are checked against the native forward (mismatch count recorded).
Results are EM and NLL by hop. Following plan §9.4, decoding an entity is not evidence of a causal hop-by-hop chain.

## Analysis (`v3_train.py analyze`, written and reviewed before any ten-arm dev result is read)
- Inputs (RV-0058): the passed formal common start (`--start-run`, `--start-sha256`, `start_provenance`) and ten arm runs, one per arm. `validate_arm_run` checks each run against `analysis_reference()`, which is built from the released T2 data:
  - metadata, train and dev digests;
  - the shared `train` schedule's order digest and token count, recomputed;
  - the ordered dev identities (episode, hop, answer token).
- Each arm run must also satisfy:
  - completed, validator passed, formal, not smoke, command `arm`, contract E-0022@2;
  - `metrics.json` and the main `dev.normal.jsonl` equal to the producer manifest's sha256 and size;
  - config and variant equal to the arm definition;
  - the same start run, adapter digest and checkpoint files, with `formal_start`;
  - 6,000 scheduled and executed steps on the reference order and token count, with no smoke trim;
  - the reference data digests and dev row count;
  - reload identical at S_train;
  - records exactly the released dev identities in order, each with `correct == (prediction == target)`;
  - stored per-hop count, correct_count and EM equal to those recomputed from the records.
- `decide` and the bootstrap use counts rebuilt from the validated records, not the stored summaries. Tests reject each corruption: denominator (the RV-0058 counterexample), correct count, stored EM, start digest, non-formal start, order, steps, smoke trim, dev digest, config, reload, identity, hop, target, correctness flag, truncation, an unrecorded file edit, and matrices that agree with each other on a truncated or wrong dev set.
- `decide` applies rules 2–8 on exact fractions (pp, thresholds inclusive):
  - primary = (EM_hop2 + EM_hop3)/2;
  - S_M* = V3-03 if V3-03 − V3-02 ≥ 0.5 pp, else V3-02;
  - checks (a)–(e), with (c) non-binding when V3-07 < V3-01 and (e) deferred to V4 when S_M* = S2;
  - outcome per rules 6/7/8;
  - V3-04/08/09/10 minus S_M* reported as descriptive.
- `bootstrap`: hop-stratified, paired, 10,000 resamples. A CPU generator seeded 20261029 draws one index matrix per hop (hops 1, 2, 3 in that order), shared by every pair. It reports 95% percentile intervals in pp per hop and for the primary. These are descriptive only.

## Pre-review tests and smoke (not evidence)
- **ATT-0190-1** (30e754b, PBS 3768211): 67 passed, 2 failed, and the intervention smoke failed. The application failed although PBS reported success, so the run is not imported.
  1. The V3-05 test recorded the flag SideBranch passes (False) instead of the forced one. The logits did match.
  2. Two recipients in a batch can share one matched donor, and `collate` refuses duplicate episodes. Fix in 3f90737: each donor is encoded once and its state gathered per recipient.
- **ATT-0192-1** (3f90737, PBS 3768267, RES-0184): `test_v3_train.py`, `test_v3_loop.py` and `test_v2r16_memory.py` all pass, 69 tests.
  - Every arm trains (all parameter groups get finite nonzero gradients, no backbone gradient), evaluates at S_train and over its scan, and reloads with identical records.
  - V3-05 equals LatentReader's own `fixed_query=True` and differs from V3-03.
  - V3-06 controls equal the normal predictions.
  - The `none` intervention equals the native predictions (0 mismatches).
  - The widened S=1 arm and V3-01 equal the reviewed evaluator.
  - V3-08 is identical to the shared S4 forward.
  - Hop-1 train has 6,144 rows: 8 start epochs and 3 train epochs, about 800 tokens per step.
- Smokes (5 start steps, 3 arm steps, 4 dev rows per hop) all complete:
  | Arm | Stable step (s, short smoke) | Stored parameters |
  | --- | --- | --- |
  | V3-01 | 0.18 | 557,056 |
  | V3-03 | 0.30 | 557,056 |
  | V3-06 | 0.14 | 557,056 |
  | V3-07 | 0.18 | 668,800 |
  | V3-08 | 0.29 | 655,360 |
  - Maximum replay ratio 2.8e-5. V3-06's replay is exact.
  - V3-07 step-0 gap (smoke start): max|Δlogit| 0.41.
- ATT-0195-1 (e1f3000, PBS 3768662, the RV-0058 fixes) passes 89 tests. These include the 17 analysis-corruption cases, the truncated and wrong dev matrices, adjacent cosine with padding and a fixed query, the executed MACs, and the real analysis reference: 1000 dev rows at 334/333/333 and the same train order digest as the arm schedule. All smokes complete.
