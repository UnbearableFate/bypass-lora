# T-0025 follow-up to RV-0064

RV-0064 passed with four low findings. Coordinator MSG-0650 requests all four
fixes and releases the nine training runs after the accepted T-0024 merge.
Master 60e2d1f was merged as ff050b8. The nine training jobs are covered by
RV-0064 at immutable source **7bf091d4af24c2a9fdc666a5f885c31817b97174**.
Their complete registration/PBS/source/spec/output inventory and pre-result
analysis pins are in [T-0025-training-jobs.json](T-0025-training-jobs.json).
They run the unchanged rb_train path while the following reporting changes are
tested separately. No actual RB1 predictions or metrics have been analyzed.

Updated numerical source: **ffa4c7124939bbd82b1319f56be8e27913e0ff59**.
Operator, LatentReader, training, scoring, data and dependency files are unchanged
from the covered training source. The changes are:

1. `rb1_rules`: a resolved effective GC with G0 and without GL gives
   `C_S1_S4_pilot_then_draft_RB5_and_review_conclusion`. Rule7 and
   resolved_actions mark `RB5_matrix_status=pending_C_S1_S4_pilot` and retain
   the provisional S1 matrix alongside both pilot outcomes. A successful pilot
   means all nine RB5 groups at fixed S4; otherwise concat stays at S1. The
   numerical gates, D-0032 resolution and Rule11 evidence directions are intact.
2. `rb1_analysis`: each intervention arm reports normal-vs-training prediction
   mismatches, maximum absolute candidate-score difference, macro difference
   and both hardware strings. These are descriptive, with no new acceptance
   gate. They appear in JSON and markdown. All pre-existing pairing, likelihood,
   iv_off/Base and T0 no-op checks remain. The synthetic full final analysis is
   exercised for identical and one-flipped normal predictions.
3. `rb1_artifacts` authenticates the **exact accepted LR-lock SHA256** and all
   sixteen input FileRef byte counts/hashes, calls the existing rb_train.lock_lrs,
   checks the plan_order_sha256 augmentation against every calibration input,
   then requires exact JSON data equality. This handles rbc_report's sole
   provenance augmentation without changing any LR choice (reviewer-3 agreed
   in MSG-0662). Formal source must be the covered 7bf091d release. Analysis
   input must contain all three pre-result expected pins below and agree with
   the loaded source/lock/Base. Accepted-lock reconstruction, invalid input
   hashes, internally consistent wrong choices, wrong source and wrong
   expected pins are tested. No calibration producer or lock format is edited.
4. `ReadSummary.finish()` preserves the existing means, which include empty
   reads as zeros, and adds nonempty_means (null when every read is empty).
   Strict-prefix n_eff, entropy and relative n_eff in markdown use the
   nonempty denominator; current-token weight retains the zero convention.
   Final analysis validates both denominators and the complete first/last
   strata. A realistic strict-prefix empty-read fixture tests the correction.

Pre-result pins (also in the committed job inventory):

- Training source: `7bf091d4af24c2a9fdc666a5f885c31817b97174`.
- LR-lock SHA256: `38b8ee24af3ded8aee398a4cc3dfaa5a1dfece4737a6cde34642fbff7bf1012d`.
- ATT-0024-1 Base manifest SHA256:
  `076a1a0595de6620508a7e99810a96baf979483d805428465807d40beadde196`.

## PBS regression record

The previously passed ATT-0230-1 / RES-0208 had336tests, zero failures/skips.
The revised suite has346cases. It includes the real pinned Qwen synthetic-text
scorer/intervention identities and the default-source output/gradient regression,
plus all four new finding-specific checks and full synthetic1000-row-per-task
paired analysis with10000bootstrap samples.

ATT-0248-1 / PBS3769234 / RES-0217 is retained as a failed diagnostic:
343passed,3failed,0skipped,59scheduler seconds. The three failures exposed one
implementation bug: a per-arm accounting variable shadowed the new release-pin
mapping. The mapping, expected-write count and expected latency labels now use
distinct names. No assertions, thresholds, data or metrics were relaxed.

ATT-0249-1 / PBS3769247 runs the same suite at the corrected ffa4c71 snapshot.
It passed **346 tests, zero failures, zero skips** in58.63pytest seconds. Its
application manifest is complete/validator-passed, and the fresh scheduler fact
is Exit0 with62seconds onmg1098. JUnit includes every one of the ten planned
modules; the live allocation record verifies that exact host/job before testing.
Both attempts' terminal observations/results and native measured usage are
imported once before the focused `--recheck-of RV-0064` request. The native
usage exports are0.016389 and0.017222MIGhours, respectively. Including phase-one
ATT0228/0230, all tests so far consumed236MIGseconds =0.01638889node hours
(0.01638875using rounded native exports). Nine ongoing training jobs are
additional and accounted at their scheduler end. Both regressions use allocation-checked
PBS jobs, offline pinned assets and the existing CUDA environment. No real RB1
result or confirmation data is an input.

## Remaining execution

After a passing focused recheck, validate the nine completed producers and run
the locked selection analysis. Measure the frozen-intervention smoke cost
before releasing the four-target grid plus T0 swap, since permutation and
normal donor forwards add overhead. The estimated intervention allowance is
about0.5node hours rather than the original0.3, as RV-0064 noted. Release the
comparative latency on2g, then complete the locked final analysis and four
Rule11 screening evidence items. The coordinator owns hypothesis status and
any RB1p/RB5 contract. All costs, including failed diagnostics and the current
analysis job after it ends, count toward the4node-hour E-0021 cap.
