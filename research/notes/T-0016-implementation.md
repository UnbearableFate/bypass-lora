# T-0016 implementation: E-0015@4 diagnostics

## Commits
- 160fde7: controls, isolated encoding, probes, entry script.
- 521cf95: the config is compared as JSON, and the isolated-encoding test uses a relative BF16 tolerance.
- 7576c2d: magnitudes, added in @4.
- 74d7c19: the RV-0042 fixes and `v2r_reload_check.py`.

Reviews: RV-0042 → RV-0043 (pass).

## `src/v2_memory.py`
**New read modes.** The control name gives the read mode. `parse_control` turns it into (read, α, scope).
- `last_only`, `oracle_last` and `oracle_last_only` substitute the read only at the last question position.
- `uniform` gives every slot equal address weight.
- `last_only_shuffled` reads shuffled donor memory at the last position.
- `wrong_oracle` reads the value of fact (target+1) mod E.
- The sweep modes are `normal@aα`, `oracle_value@aα` and `nonfinal@aα`, for α ∈ {0.1, 0.3}. They scale the increment by α at all question positions, or only at the non-final ones.

**Isolated encoding.** `isolated_writes` encodes each fact alone, in the E=1 format, then places the results in support order. `isolated_snapshot` rebuilds the memory state with `memory_from_slots`, in one of two ways:
- `iso_kv` uses the isolated keys and values.
- `iso_v`, `iso_oracle_value` and `iso_wrong_oracle` keep the contextual keys and use the isolated values.

**Captures.** `observe_writes` captures the frozen delimiter features at the projection input, together with the transformed keys and values. `evaluate_batch` exposes `_last_fusion`: the FP32 `z@P` at the last question position, which is the Pz input.

**Regression.** The existing controls (normal, zero, shuffled, oracle_key, oracle_value) are bitwise unchanged against c6037d3 (ATT-0140-1, `runs/ATT-0140-1/regression/compare.json`).

## `src/v2_train.py`
`evaluate` gains two arguments: `capture` (the per-mode Pz increments) and `batch_size`. Diagnostics run with batch 64. The reload check keeps the recorded batch of 16.

## `src/v2r_diagnostics.py`
**`run`, one job per checkpoint.**
- Checks the token format: each fact's context tokens must equal its isolated tokens.
- Checks that the train and calibration splits are disjoint.
- Runs the reload check against the recorded dev predictions, and stops on any mismatch.
- Evaluates the calibration controls with target-position strata.
- Fits the probes: closed-form FP64 ridge with λ=1, fit on the first 4096 train rows, with three baselines (answer prior, label shuffle, random 128-d projection for Pf).
- Computes the per-position magnitudes.

**`analyze`.** Applies rules 1–8 to the four summaries:
- Rule 4 is EM-only.
- Rule 6 iterates R2 and R5 × E=4 and E=16.
- Every other ≥ 10 pp comparison requires NLL to move the same way.

## `src/v2r_reload_check.py`
This is the MSG-0402 diagnostic. It reruns the E=16 dev reload in a single process in the order V2-03, R1, V2-03, followed by a Base-only zero, and logs the device and batching settings.

## `src/test_v2r_diagnostics.py`
- The new read substitutions.
- Control-name parsing.
- The isolated snapshot.
- Probes and magnitudes.
- The real-model controls and the isolated encoding.
- Evaluate capture.
- The released-row token format.
- Synthetic tests for rules 4 and 6.

## Run notes
- Every job runs from a read-only `git archive` snapshot in `/work/xg24i002/x10041/bypass-lora/snapshots/<sha>`.
- mg1098_mig3 is a MIG 1g.24gb instance and does not reproduce recorded BF16 results (EV-0086). For that reason ATT-0144-2 checks `nvidia-smi -L` for `MIG 2g.24gb` before it runs.
