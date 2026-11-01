<!-- RV-0043 answer · entry codex:gpt-6-astra@xhigh · observed model unverified · session unknown · received 2026-11-01T13:12:18+00:00 -->

# RV-0043 — pass

Rechecked the clean candidate `74d7c19d93d5e6b5b141c8d7b603267ae4df0f06` against RV-0042's `7576c2d0573acb762ef81352cc6488d60e2cefef`. All three findings are resolved. No new findings in the changed scope, including the standalone reload diagnostic. This completes the second review of the round.

- **RV-0042 high, rule 4: resolved.** `src/v2r_diagnostics.py:458-464` now gates only on an EM drop of at least 0.30 and Pf >= 0.90 in both multi-fact splits. NLL differences remain descriptive. The emitted rule text is corrected. The synthetic `analyze()` regression demonstrates a pass when E=4/E=16 have the required EM drops but lower NLL than E=1. Rules 3, 6 and 7 continue using the unchanged NLL-aware `gain()`.
- **RV-0042 high, rule 6: resolved.** `src/v2r_diagnostics.py:472-475` evaluates R2 and R5 at both e4 and e16, each against its own normal control. Its rule text is corrected. The regression independently activates each of the four cells and asserts that all four are reported. The trained-split restriction remains unchanged for rules 1, 3 and 7, matching coordinator MSG-0397.
- **RV-0042 low, magnitude assertion: resolved.** `src/test_v2r_diagnostics.py:117` now expects the correct mean of 3.0 for norms 2 and 4; the group-count assertion remains. The production magnitude calculation is unchanged.

ATT-0147-1 was executed from the reviewed snapshot on confirmed Miyabi-G compute `mg1099 / 3765540.opbs`. I inspected its allocation/live-state artifacts, job script, application manifest, pytest log and JUnit report: **10 tests passed, zero failures/errors/skips**. Scheduler history confirms Exit_status 0. All three reviewed files byte-match the runtime snapshot. `git diff --check 7576c2d HEAD` passed and the worktree remained clean.

The complete diff changes only analysis predicates/rule text, tests, and the new `src/v2r_reload_check.py`. The main `run()` evaluation path, data/probe/magnitude collection, `v2_memory.py`, `v2_train.py`, and model/core code are unchanged. RV-0042's earlier inspection and recorded old/new regression therefore remain applicable to those paths.

The new reload diagnostic uses the pinned E=16 dev rows, recorded batch size 16/right padding/position IDs, and the same normal/zero evaluator. It reloads V2-03, R1 and V2-03 again in one process, then evaluates the restored Base directly, while recording device and numerical settings. Its completed/validator-passed status means diagnostic execution completed; the mismatch numbers remain in the report. It does not replace or relax the main evaluator's exact recorded-prediction gate.

I inspected ATT-0146-1's supplied results on `mg1098 / 3765539.opbs` (Exit_status 0): the device reports MIG 1g.24gb with 26 SMs, both V2-03 repeats report 51 normal/12 zero mismatches, R1 reports 4/12, and Base-only zero reports 12 mismatches against both records. The in-progress ATT-0144-2 artifacts show a guarded MIG 2g.24gb run with zero prediction and NLL differences for V2-03's normal/zero/shuffled/oracle_value reload checks. This supports the device-profile explanation for the observed reload discrepancy; it does not establish completion of that full diagnostic run or universal reproducibility across hardware.

Limits: this recheck inspects supplied execution artifacts and does not independently rerun tests. The full four-checkpoint, three-split diagnostic study and its final analysis are not accepted as complete by this code review. No confirmation data was opened, and no scientific conclusion or hypothesis-state decision is made.

```json
{
  "verdict": "pass",
  "findings": [],
  "checked": [
    "Exact clean candidate 74d7c19d93d5e6b5b141c8d7b603267ae4df0f06 and complete diff against 7576c2d; all three RV-0042 findings resolved",
    "Rule 4 now uses EM-only drops plus Pf thresholds; rule 6 covers all four R2/R5 by e4/e16 cells; rule text and synthetic analyze regressions updated",
    "Magnitude test expectation corrected to 3.0 without changing the production calculation",
    "ATT-0147-1 supplied compute evidence on mg1099 / 3765540.opbs: 10 tests passed, zero failures/errors/skips, application success and scheduler Exit_status 0",
    "All reviewed files byte-match the runtime snapshot; git diff --check passed; main evaluation and model/control paths unchanged",
    "Standalone reload diagnostic uses recorded evaluator batching/order and Base-only zero comparison, and records device settings without relaxing the main reload gate",
    "ATT-0146-1 supplied mismatch/device report and partial ATT-0144-2 exact reload-match artifacts inspected with their completion limits"
  ],
  "unchecked": [
    "No independent runtime rerun or workload submitted for this recheck",
    "Full four-checkpoint by three-split diagnostics and final analysis have not been reviewed as a completed study",
    "ATT-0144-2 was still in progress in the inspected artifacts; its reload-stage success is not full-run completion",
    "No universal hardware-determinism claim, confirmation-data access or scientific conclusion"
  ]
}
```
