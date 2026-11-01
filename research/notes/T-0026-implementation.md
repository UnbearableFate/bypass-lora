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

## 4. Rule 4: frozen-transfer diagnostic (`src/v3_transfer.py`, `src/test_v3_transfer.py`; D-0034)
Stage A gave case A3 with S_M* = V3-02 (ATT-0251-1), so rule 4 applies. The diagnostic trains nothing. It reads the frozen S_M* checkpoint on the T2 calibration rows of hops 2 and 3 (pooled). All reads are scored at S_train, labelled with the chain end, and measured at the answer position (the last valid question token).

### Reads
- `native`: the arm's own loop. The replay must match the native output within the loop tolerance, and the native logits are scored.
- `forced_first` (descriptive only): z^(1) is set to q̄ + η·(v[path fact 1]·g)·Ro, using the step-0 Ro if the arm is untied. Step 2 onward queries and reads natively, and the result goes back through the reviewed side-output hook (`_with_increment`). Defined only for S ≥ 2.
- `bridge`: the question template is re-encoded with the gold bridge entity (`path.entities[1]`) as the subject. Relation and template are unchanged.
- `wrong_bridge`: the subject is the head of the first same-relation fact that is off the path and whose head is not a path entity. Rows with no such fact are left out of this read; the count is reported per read.
- `donor_bridge`: the subject is the gold bridge of the nearest earlier evaluated row (cyclic) whose bridge entity appears nowhere in this episode (no fact head or tail, no path entity).
- Also on the same rows:
  - the reviewed first-edge and end-edge value oracles (`v3_diag.evaluate_oracles`);
  - the zero and shuffled memory controls (`v3_train.evaluate_arm`; global donor rule within the hops 2-3 subset).

### D-0034 mapping (`mapping`)
Every quantity is an exact Fraction, computed from the per-record floats and reported next to its boolean.
- (a) end-edge oracle EM − max(EM_zero, EM_shuffled) ≥ 10 pp.
- (b) Holds when all three of these hold:
  - the bridge read's mean step-1 weight on path fact 2 is ≥ 1/8;
  - the wrong-bridge mean on that fact is < 1/8;
  - the donor-bridge mean on that fact is < 1/8.
- (c) The native read's mean step-2 weight on path fact 2 is < 1/8.

| Criteria | Repair |
|---|---|
| (a) fails | none |
| (a), (b) and (c) hold | successor-slot CE S4 pilot with the permuted-label control |
| (a) holds, (b) fails | linear-start S4 |
| (a) and (b) hold, (c) fails | none; describe and go to the conclusion |

### Tests
- CPU:
  - `wrong_fact` and `donor_bridge` choices, including the absent-name rule;
  - every boundary at equality: a 10 pp gap holds (a); a bridge weight of exactly 1/8 holds; a control at exactly 1/8 fails (b); a native weight of exactly 1/8 fails (c);
  - the summaries.
- GPU (D5 stand-in):
  - the rewritten questions contain the new subject;
  - `native` records equal `evaluate_arm`'s normal records;
  - `forced_first`'s step-2 weights equal the closed form within 1e-5;
  - V3-06 (tokenwise) raises.
- Pre-review check ATT-0252-1 (not evidence):
  - 95 tests passed;
  - a 4-rows-per-hop smoke on ATT-0234-1 completed and wrote all reads, oracles, controls and the mapping.

## 5. Rule 4 repair: successor-slot CE S4 pilot (`src/v3_succ.py`, `src/test_v3_succ.py`; D-0035)
The diagnostic (ATT-0253-1, EV-0116) met (a), (b) and (c), so D-0034 selects this repair. All definitions follow D-0035.

### Arms
- `succ --labels correct` and `succ --labels permuted`.
- Both start from ATT-0196-1, with readiness ATT-0227-1, in the V3-03 configuration (S = 4).
- Both use `v3_schedule(train, 'train', steps=2000)`: the first 16,000 episodes of the stage-A order, 100 warmup steps, LR 3e-3, and the shared schedule shape over 2,000 steps.
- Init and order are identical. The only difference is the labels.

### Successor CE (`succ_forward`)
- The answer CE is `train_forward`'s standard path under the isolated write. With no labels, the loss and logits are exactly equal to it (test).
- The answer-position loop is replayed with gradients through `v3_train.replay`, on the compact snapshot of the same support graph. Every step checks that it matches the native side output at that position within the loop tolerance; a mismatch raises.
- Each step's slot logits are `slot_read`'s logits; their softmax is the read weight. Labels (`successor_labels`) cover steps s = 2..min(4, h) with `path.fact_indices[s-1]`. Step 1 and steps beyond h are unlabelled.
- The loss is answer CE + 1 × the mean −log_softmax over the batch's labelled pairs; a batch without labels adds 0.
- Arms other than the shared slot loop are rejected.
- Permuted labels: pi(label), where pi is a seeded uniform derangement of the episode's slots, `derive_seed(SEED, 'V3', 'succ', 'permuted-successor', 0, episode_id)`. The label count and positions are unchanged.
- The training tally is written to `successor.jsonl`, and the label count seen in training must equal the schedule's count.

### Evaluation and reload
- T2 dev at S = 4 with normal, zero, shuffled and permuted reads (`evaluate_arm`). The checkpoint is saved and reloaded, and the records must be identical.
- `pilot` holds the dev hop counts and the exact hop-2 mean of `path_weight_last[1][1]`, the step-2 weight on path fact 2.

### Comparison (`compare`)
- Both pilots are validated:
  - complete and formal;
  - the right labels on each side;
  - unchanged artifacts;
  - the same start;
  - the pilot summary recomputed from the saved normal records;
  - equal schedule, config, variant, data and label count.
- `decide` then applies D-0035 with exact fractions:
  - (i) primary EM, correct − permuted, ≥ +3 pp;
  - (ii) hop-2 step-2 weight on fact 2 ≥ 1/8 for correct and < 1/8 for permuted.
- Hop 1 is reported descriptively.

### Tests
- CPU:
  - labels per hop;
  - the derangement (no fixed points, seeded, per episode);
  - label counts;
  - every D-0035 boundary at equality;
  - the pilot summary;
  - `compare` on synthetic runs: it accepts a valid pair and rejects swapped roles, another start, or a changed artifact.
- GPU (D5 stand-in):
  - the pilot schedule is the first 2,000 batches of the stage-A schedule;
  - with no labels, the forward equals `train_forward`;
  - the successor CE equals the CE computed from the evaluator's own per-step weights;
  - the CE reaches Rq;
  - permuted labels have the same count but a different loss;
  - V3-06 is rejected.
- Pre-review job ATT-0254-1/RES-0226 (not evidence):
  - 13 tests passed;
  - 4-step smokes of both arms trained, evaluated and reloaded with identical records;
  - stable step 0.32–0.34 s;
  - the answer-position replay error was at most 1e-6 relative.
- Known cosmetic issue: a PyTorch warning about `float()` on a tensor that requires grad, at the replay check. The value is only compared against the tolerance.

## Not yet implemented
- V3-05b: not needed under A3.
