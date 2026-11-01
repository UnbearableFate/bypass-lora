# T-0026 implementation: E-0023 readiness, staged start, diagnostics

Review package for the E-0023@1 code paths that are new since RV-0059 (diff base eea4fb8). Nothing here changes a shared class. All new behaviour is in my own modules.

## 1. Stage 0: readiness rule R (`src/v3_ready.py`, `src/test_v3_ready.py`)
- `ready` evaluates the final checkpoint of ATT-0196-1 on the released T2 calibration split (2,048 rows, 683/683/682). It uses the reviewed E-0016 evaluator (`v2r16_memory.evaluate`, isolated write, six controls, S=1).
- `start_record` binds the start to its manifest: the formal V3-S0 run under E-0022@2, the checkpoint files equal to the manifest, and the pinned digest `f70ea08e…`. The files are checked again after loading and again after evaluation.
- `readiness(records, rows)` judges R from the per-example records:
  - (1) hop-1 EM margin, as an exact fraction, ≥ 1/10;
  - (2) hop-1 NLL normal < min(zero, shuffled);
  - (3) for each of hops 1, 2, 3, the mean answer-position weight on the first path fact (`last_target_weight`; the record's `target_fact_position` must equal `path.fact_indices[0]`) ≥ 1/8, computed as an exact mean of the floats;
  - (4) on hop 1, the count of the most common answer-position argmax slot divided by n ≤ 1/4.
- The recomputed weights must match `reviewed_by_hop` to within 1e-12.
- The all-position weight, argmax = first fact, and the per-control EM are descriptive only.
- Run: ATT-0227-1. The regression jobs (ATT-0229/0231/0232-1) recompute R independently from the saved records and the released rows; the recompute matches.

## 2. Staged start for the arms (`src/v3_train.py`)
- `CONTRACT = 'E-0023@1'` and `EXPERIMENT = 'E-0023'`. E-0022 takes no more runs (D-0031).
- `START_CONTRACTS` accepts a V3-S0 start from E-0022@2 (ATT-0196-1) or from an E-0023 second start.
- `readiness_provenance(ready_run, files)` requires the readiness run to be:
  - complete, with the validator passed;
  - under contract E-0023@1, command `ready`;
  - formal, not a smoke run;
  - `readiness_passed` true;
  - run on the same start checkpoint files.
- `start_provenance(..., ready_run=None)` makes a start formal only if it is eligible and has such a readiness run (E-0023 rule 1). The old H3 result is recorded as `hop1_gate_passed` metadata only; a start that passed H3 but has no readiness run is rejected (RV-0065). The CLI option `--ready-run` applies to `arm` and `analyze`.
- `validate_arm_run` requires the analysis start to carry a readiness record, and each arm's recorded `start.readiness` (readiness run path, manifest digest, criteria) to equal it (RV-0065).
- Fix: `TokenwiseKernelSide.forward` takes SideBranch's new `runtime` argument. Researcher-1's RB change 2f1687b added it after RV-0059, and V3-06 failed with a TypeError in ATT-0229-1. A non-empty runtime is rejected, because the RB options are not defined for this arm.
- Tests:
  - `test_old_gate_failure_is_a_formal_start_only_with_a_passed_readiness_run` covers seven bad readiness manifests and shows a smoke start stays non-formal.
  - The arm smokes for V3-02 and V3-06 run from the real ATT-0196-1 start with `--ready-run ATT-0227-1`.

## 3. Diagnostics (`src/v3_diag.py`, `src/test_v3_diag.py`)
All of these are descriptive; none enters a rule.

### Value oracles
- `oracle_replay` runs the arm's own loop with every step's read, at every valid question position, replaced by `values[fact] * read_gain`.
  - The update (accumulate or anchored), the readout (increment or full), eta and Ro all come from the arm. Untied arms use the per-step Ro.
  - The query is irrelevant.
- `oracle_batch` captures q̄ from the native forward. It substitutes Z through the same side-output hook as the interventions (`v3_train._with_increment`).
- Oracles:
  - `first_edge_value` uses `path.fact_indices[0]` (the old oracle_value);
  - `end_edge_value` uses `path.fact_indices[-1]`.
- On hop-1 rows the two oracles are the same fact; the runner asserts that their records are equal.
- On the stock S=1 reader (V3-01), `first_edge_value` must equal the reviewed `oracle_value` records exactly (prediction, correct, NLL). The runner asserts this, and a GPU test checks it.
- Not defined for V3-06 (tokenwise, no slot read).
- Closed forms are tested per arm type:
  - accumulate: S·η·v·Ro;
  - anchored: η·v·Ro;
  - full readout: q̄ + S·η·v·Ro;
  - untied: Σ_s η·v·Ro_s.

### Per-step path weights
- `path_weight_table(records, S)` reads the arm's verified main records (`eval-s{S}/dev.normal.jsonl`, `path_weight_last`).
- `table[hop][s][j]` is the mean answer-position weight of step s on path fact j. `diagonal` is s = j, the contract's statistic.
- `secondary_mechanism` reports the rule-5 quantities against 1/16 and 1/8:
  - hop 2, step 2 on fact 2;
  - with S4, also hop 3, step 3 on fact 3.

### EM by answer-name frequency
- The count for a name is the number of released T2 train rows with that name as the answer (`answer_text`).
- Evaluated rows are split at the median of their answers' counts: high is count > median, low is count ≤ median.
- The stratum sizes and count ranges are reported.

### Stage A (rule 3)
- `stage_a(main)` takes exact fractions from the verified V3-01/02/03 counts. S_M* follows E-0022 rule 3.
- The cases:
  - A1: gain ≥ +3 and hop-1 difference ≥ −1;
  - A2: gain ≥ +3 and hop-1 difference < −1;
  - A3: 0 < gain < +3;
  - A4: gain ≤ 0.
- `analyze_stage_a` reuses `validate_arm_run` (RV-0059) for every input. It requires exactly the three arms and adds paired hop-stratified bootstrap intervals (descriptive) for V3-02−V3-01, V3-03−V3-01 and V3-03−V3-02.
- Boundary tests cover every case at exact equality, including the 0.5 pp S_M* tie-break.

### Runner
`v3_diag.py diag` takes one arm. It validates the arm run against the formal readiness start and the released dev set, checks the pinned adapter digest before and after loading, and writes:
- `metrics.json` (oracles by hop, path-weight tables, frequency strata);
- `dev.{first,end}_edge_value.jsonl`.

`v3_diag.py stage-a` writes the rule.

## Not yet implemented
The rule-4 frozen-transfer diagnostic and V3-05b. Each is needed only if its case arises (A3/A4, or A1 with S_M* = 2), and each will be reviewed before it runs.
