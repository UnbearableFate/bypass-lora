# T-0018 implementation: E-0016 D4/D5 isolated-encoding write path

## Commits
- 7e457e6: shared isolated-encoding support (`isolated_support`), annotation module, tests, and the isolated controls in the default-off regression.
- c8477d8: test fix. The Base reference logits are computed before the adapter is installed.
- 2f1a95f: D4/D5 runner `src/v2r16_isolated.py` and its tests.
- ac2610c: merge of T-0017's reviewed hooks (6da33b3, RV-0045), as instructed in MSG-0431/MSG-0454.
- 415cac3: RV-0046 fix. The isolated gate reuses the reviewed exact-fraction `v2r16_analysis.h3_gate`, with boundary tests.

Reviews:
- RV-0046 (reviewer-3) raised one low finding: the float EM margin failed at exactly 10 pp. It was fixed in 415cac3.
- The recheck, RV-0048, passed.
- The shared T-0017 parts are covered by RV-0045.
- D-0017 accepted the merged `v2_train.py` after the combined suite ATT-0160-1 (67 passed, default identity exact against e1660c9 and master 8cccbb1).
- The formal D4/D5 runs ran at 415cac3.

## `src/v2_isolated.py` (new)
`fact_text(fact)` gives the E=1 format `"Head relation Tail ;\n"`. `annotate_isolated(tok, rows)`:
- stores per fact `{'ids', 'delimiter'}` in `row['_isolated']`;
- checks that `support_text` is the concatenation of these texts and that the delimiter is the last token;
- returns the number of facts whose context tokens differ from the isolated tokens. This was 0 on train and dev in both arms.

`v2r_diagnostics.annotate` now calls it, so E-0015 and E-0016 share one isolated format.

## `src/v2_memory.py`
- **`isolated_batch` and `place_isolated`.** These are the shared pieces of the isolated encoding:
  - all B×E single-fact sequences go through one support-phase forward, right-padded with pad id 0 (`ISOLATED_PAD_ID`);
  - the one valid slot of each sequence is placed at (row, fact index) by `index_put`.
- **`isolated_writes`** is the E-0015 evaluation path. It is now built from these pieces and is bitwise unchanged.
- **`isolated_support(model, batch)`** is the differentiable training and evaluation path:
  - it requires owners == arange, so there is one question per support;
  - it runs the support forward with the write mask;
  - it requires exactly one valid slot per single-fact sequence;
  - it places keys and values in support order and rebuilds the state with `memory_from_slots`, so keys are not transformed twice;
  - it checks that the per-row slot count equals the write-mask count;
  - accounting is writes = number of facts, reads = 0;
  - the question side is unchanged.

  A test asserts that the stored slots equal `isolated_writes` bitwise.

## `src/v2r16_isolated.py` (new runner)
- **Arms.**
  - D4 starts from scratch.
  - D5 starts from the R1 adapter (`runs/ATT-0122-1`):
    - `r1_provenance` checks that the source is the completed formal E=1 6000-step seed-0 run with the same config and an adapter-only checkpoint ({adapter.pt, config.json});
    - the R1 contextual dev predictions are replayed with the legacy evaluation, and the mismatches must be 0;
    - the schedule comes from T-0017's `make_arm_schedule`: a fresh AdamW, a fresh 300-step warmup then cosine, and the data order from `derive_seed(20261029,"concat_loop_v2","V2R16","D5",0)`.
- **Training.**
  - It uses `v2_train.train(..., forward_fn=partial(train_forward, support_fn=isolated_support), curve_evaluator=...)`, with answer CE only.
  - D4 checks that the zero-initialized exit reproduces the Base logits (max abs 0.0).
- **Evaluation.** It uses T-0017's `evaluate(..., support_fn=isolated_support)`:
  - the six standard controls run under the isolated write;
  - contextual `normal` and `oracle_value` are evaluated as descriptive comparisons;
  - learning curves at 25/50/75% use the isolated normal/zero/shuffled.
- **Gate.**
  - `gate` = `v2r16_analysis.h3_gate` on the isolated controls, plus the last-position target weight;
  - `gate_by_target_position` applies the same gate per stratum (target position 0 vs ≥1).
- **Reload.** The adapter is exported, reloaded and re-evaluated for both protocols, and the full records and summaries must be identical.

## Default off
None of the new options change the existing paths. `v2r_default_regression.py` dumps the soft_slot arm before and after, including the E-0015 isolated controls. The result was identical against master: ATT-0151-1 (regression part), then ATT-0160-1 for the merged tree.
