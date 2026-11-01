# T-0019 implementation: E-0017 seed options

## Commits
- 64331a4: seed options. Adds `v2_train --seed`, the D5 continuation seed, the start-run seed check and the E=16 calibration descriptive evaluation, with tests.
- f01b036: RV-0049 fixes.
  - The start-run replay now records numerical differences instead of failing (D-0018, D-0019). It still fails when the episode identities or their order differ.
  - The initialization test now separates the random projections from the identity Rq/Ro and the zero P.
- 7d53d5d: RV-0050 fix. The start adapter.pt is bound to an expected producer digest and checked before and after loading.

## Reviews
- **RV-0049** (reviewer-1), first round: one medium and one low finding, both fixed in f01b036.
  - The reviewer accepted appending the seed to the stage-A order derivation as a documented, backward-compatible reading of the contract.
- **RV-0050**, recheck: both RV-0049 findings resolved; one new medium finding. Without the strict replay, nothing tied the start weights to their producer.
  - The new round RV-0054 passed with no findings.
- **D-0020:** the coordinator let the stage-A runs start under RV-0049's coverage of v2_train/v2_schedule. Those files are byte-identical between 64331a4 and f01b036.

## `src/v2_schedule.py`
- `train_order(rows, phase, seed=0)` and `make_step_schedule(..., seed=0)`:
  - Seed 0 is the recorded rule: `derive_seed(20261029, "concat_loop_v2", phase, "train-order-<epoch>")`.
  - Seed s ≥ 1 appends s to the same parts (`SEED_ORDER_RULE`), and the schedule records `order_seed` and the rule.
  - At seed 0 the output is unchanged.

## `src/v2_train.py`
- `adapted(..., seed=0)` seeds random, torch and CUDA with the seed.
- `--seed` sets both the adapter initialization and the step-budget order. It is allowed only for plain `--step-budget` runs, with no curriculum and no prior path.
- The manifest records the actual seed.
- Only the projections A_q, A_k and A_v are random. Rq and Ro start as identity and P starts at zero at every seed.

## `src/v2r16_train.py`
- `make_arm_schedule(rows, arm, steps, seed=0)`: the resumed arms (D2a, D2b, D5) use `derive_seed(20261029, "concat_loop_v2", "V2R16", arm, seed)`.
- A non-zero seed is rejected for the from-scratch arms.
- The schedule records `continuation_seed` and the rule with the seed filled in.

## `src/v2r16_isolated.py`
- **CLI.** `--experiment {E-0016, E-0017}` sets the manifest's experiment and contract.
  - E-0016 is fixed to seed 0 from ATT-0122-1.
  - E-0017 runs D5 only, with continuation seed 1 or 2, from a start run at seed 0 or at the same seed.
- **Start run.** `--r1-run` and `--r1-seed` name the start run, which must be a run of the R1 recipe. `r1_provenance` checks:
  - the run is complete, formal, E=1, 6000 steps, reload-identical and has the fixed configuration;
  - the recorded seed equals `--r1-seed`;
  - the checkpoint holds only adapter.pt and config.json;
  - adapter.pt has the expected sha256.
- **Expected digest.** It never comes from the source directory itself:
  - For ATT-0122-1 it is `R1_ADAPTER_SHA256`, the digest that E-0016 D5 (ATT-0159-1) recorded while its replay was still exact.
  - For any other start run the job must pass `--r1-adapter-sha256`, or the CLI errors.
  - For the stage-A runs, the producer job wrote `adapter.sha256` when it ended. The run directory was then made read-only and the value was copied literally into the stage-B job script.
  - The files are hashed again after loading and must equal the inventory that was checked.
- **Replay** (`replay_record`). The start run's recorded contextual dev predictions are replayed:
  - prediction and NLL differences are recorded only;
  - a different episode set or order is an error;
  - the same-job save/reload check stays exact.
- **E-0017 evaluation.** E-0017 adds the E=16 calibration split (2,048 episodes) under the isolated write with normal, zero and shuffled. It is descriptive and is included in the reload identity check. The gate still uses isolated dev only.

## Validation
- **ATT-0164-1:** 54 passed and 1 failed, the initialization-test defect that RV-0049 found. The default identity against master 6355d32 was exact, and the seed-1 stage-A and stage-B smokes ran. The manifest is marked failed, so it was not imported.
- **ATT-0165-1 (RES-0158):**
  - 56 passed; default identity exact.
  - Smokes ran.
  - Replay boundary: the aligned altered copy trains and records 4 mismatches with NLL difference 0.300; the reordered copy fails.
- **ATT-0168-2 (RES-0164):**
  - 58 passed; default identity exact.
  - Cases that train: the original R1 run and a byte-identical copy; stage-A ATT-0166-1 with its own digest; the aligned replay copy.
  - Cases that fail before training: a copy carrying ATT-0159-1's adapter.pt; ATT-0166-1 with a wrong digest.
  - The reordered replay copy fails.
- **ATT-0168-1:** the same job as ATT-0168-2, but a missing `cd` stopped the cases from running. Tests and default identity passed; recorded as failed.
