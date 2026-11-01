# T-0021 implementation: E-0018 T2 isolated-write loop evaluation and timing

## Scope
- New files only: `src/v3_loop.py` and `src/test_v3_loop.py`.
- No existing source file changes: `git diff master -- src` adds these two files and nothing else.
- With the S option off, every existing path is master by construction. `bypass_core.py` and `bypass_model.py` are untouched (MSG-0549; researcher-2 informed in MSG-0553).

## S>1 accumulating loop (plan §3.5)
`loop_steps(model, S)` is a scoped context manager:
- It sets `steps=S` on the BranchModel config and on each SideBranch config, using `dataclasses.replace`.
- It restores the original config objects on exit.
- It is allowed only on the single-read layer-14 SoftmaxSlot model with `update='accumulate'` and `readout='increment'`. It refuses nesting and non-int or non-positive S.

The loop itself is the existing `LatentReader.forward`:
- z⁽⁰⁾ = q̄, and z⁽ˢ⁺¹⁾ = z⁽ˢ⁾ + η·g·a⁽ˢ⁾·Ro.
- Z = z⁽ˢ⁾ − z⁽⁰⁾.
- Rq/Ro are shared across steps; η = 0.25 and g = 1.

The question phase never writes, so every step reads the same fixed memory M. Each evaluation batch asserts this: the returned state equals the control memory in every field, and the accounting shows `reads = S × valid question tokens` and `writes = 0`.

## T2 data
`load_t2(tok, split)` loads the data through `v3_eval.load_released_data`:
- The metadata sha256 is pinned to E-0011's `2ad8cc35…`.
- The shortcut gates, the split digest, the identities and the in-distribution hop check are all verified there.
- Only train, calibration and dev are accepted. Sealed splits are refused before any file is opened.

Every row's path is then checked:
- `facts[path.fact_indices[j]]` joins entity j to entity j+1 under the queried relation.
- Hop 1 is `query.target_fact_index`.
- The answer is the chain end.

Rows are annotated with the E=1 isolated encoding. The number of T2 facts whose context tokens differ from their isolated tokens is 0 on dev (ATT-0181-1). Writes use `v2_memory.isolated_support`, unchanged from E-0016/E-0017.

## Evaluation
- **S_eval = 1, reviewed path.** The reviewed E-0016 evaluator `v2r16_memory.evaluate(..., support_fn=isolated_support)` runs all six controls on T2 dev. The hop-1 H3 gate uses `v2r16_isolated.gate`, the exact-fraction `h3_gate`, on the hop-1 rows.
- **Any S, loop evaluator: `evaluate_loop`.**
  - Controls are normal, zero, shuffled and permuted, built from the reviewed control memories (`compact_snapshot`, `control_snapshot`).
  - The global donor rule is unchanged: recipient i gets support (i−1) mod N. Batches are 16.
  - Logits come from the native S-step forward.
  - Per-step diagnostics come from an explicit replay `loop_trajectory` on the same memory. Its read is `v2_memory.read_vectors`, so step 1 is the reviewed S=1 diagnostic exactly.
  - Per step it records: the weight on the path's j-th hop fact (all positions, token-weighted, and last position), the last-position argmax slot and whether it hits path fact j, n_eff, query cosine to step 1, and state and read norms. It also records the final increment norm.
  - Shuffled path weights are None: the recipient's path is undefined in donor memory.
- **Replay check.** For each valid question position the replay must satisfy ‖Z_replay − Z_native‖ / ‖Z_native‖ ≤ 1e-3. Absolute and state-relative maxima are recorded alongside.
  - Why the replay differs at all: the native read scores each position separately with [rows, r] einsums, while the replay scores all positions at once with [B, T, r]. Rounding of the scores passes through the softmax into the read.
  - ATT-0177-1 and ATT-0178-1 failed my earlier absolute and state-relative bounds. In ATT-0181-1 the maximum ratio is 5.5e-5 at S=8 and 3.3e-5 at S=1.
- **Implementation check inside the formal job.**
  - The S=1 loop records must equal the reviewed evaluator's records example by example for normal, zero, shuffled and permuted. Compared fields: prediction, correct, nll, all/last target weight, all/last n_eff and last argmax. Any difference fails the job.
  - With `--t1-check` (P1 adapter ATT-0172-1 only), the same equality is required on T1 E=16 dev in the same job.
  - The S=1 loop on T1 is also compared with P1's recorded isolated predictions. That comparison is recorded only (D-0018/D-0019).
- **Oracle controls** (oracle_key, oracle_value) come from the reviewed S=1 path. They are reported for hop 1; they are not defined for S>1 in E-0018.

## Timing
`v3_loop.py timing` loads the seed-0 D5 adapter (ATT-0159-1) for each S in {1, 2, 4} and trains about 100 T2 train steps:
- `v2_train.train` with `train_forward(support_fn=isolated_support)` under `loop_steps`.
- LR 0.003, batch 8, `make_step_schedule(phase='V3')` order.
- It records the stable step time, peak allocated memory and tokens per step. This is a smoke run, not evidence.
- Per the brief, this compares timing, so the formal timing job requires a 2g slice; its guard is not record-only.

## Provenance
`adapter_provenance` checks the adapter run:
- It must be a completed formal isolated-write D5 run, eligible as evidence.
- Its adapter file records must equal the manifest's `checkpoint_files`.
- `adapter.pt` must match the digest pinned in the job. The check is repeated after loading.

## Tests and runtime evidence (pre-review, not evidence)
ATT-0181-1 (PBS 3767840, 7f3b5a0, MIG 2g): 38 passed (`test_v3_loop.py`, plus the unchanged `test_v2r16_isolated.py` and `test_v2r16_memory.py`). Covered:
- Core loop vs an independent double-precision plan §3.5 reference at S = 1, 2, 4, 8.
- Gradients of a 4-step loop vs that reference, which itself passes gradcheck.
- Default-off identity: logits and all six gradients are bitwise equal at S=1, with the config objects restored after S=4. S=2 gradients reach all six groups, are finite, and nothing reaches the backbone.
- S=1 loop vs the reviewed evaluator on T2 and T1: exact.
- Per-step record invariants; the zero control is independent of S.
- Path, sealed-split, gate, provenance and CLI guards.

Smokes in the same job:
- Eval smoke: P1 adapter, 32 rows, S_eval 1/2/4/8, `--t1-check`, both S=1 identities exact.
- Timing smoke: 3 steps per S; stable step 0.213 s / 0.234 s / 0.306 s at S = 1/2/4 in ATT-0177-1.

ATT-0177-1 and ATT-0178-1 failed only on the replay bounds described above.
