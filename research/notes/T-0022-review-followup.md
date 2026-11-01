# RV-0060 focused follow-up

RV-0060 passed with four low findings. Coordinator MSG-0599 authorized closing
all four inside T-0022, one PBS regression/three-CLI-smoke job, then one focused
recheck. No RBc/RB1 training is authorized here.

1. Replace tautological write controls with patched broken prefix-state builders:
   duplicated active writes and ignored padding. Run each through the real
   prefix/reference output/M/c/counts gate. A separate test feeds doubled writes
   through the actual TrainingAccounting forward hook and train assertion.
2. Swap/permutation require valid tokens contiguous from position0; swap also
   checks donor masks. Unpadded/right-padded scoring is supported. Left padding
   and holes explicitly raise; other history modes preserve arbitrary masks.
   Tests retain supported-mask identities and add recipient/donor guard cases.
3. Calibration evaluates NLL once and exports/reloads nothing; smoke and formal
   train retain checkpoint/reload checks. A targeted run-level unit regression
   requires one model load/eval, no checkpoint calls/files and unchanged lock
   input fields. rb_report and rb_benchmark cost formulas now charge RBc one
   load and one calibration pass with no save; RB1/RB5 estimates are unchanged.
4. rb_revision_check executes the actual rb_train CLI for RB1-02, RB1-01 and
   RB1-07, smoke stage, four steps, limit2. Smoke final evaluation includes both
   calibration and dev to exercise the calibration.nll manifest field; the
   planned metadata describes the selected four-step smoke, not the full budget.
   The driver checks family/stage/phase/seed, finite NLL/LR, failed=false, split
   and sample-order hashes, matching plan/train counts, parameter counts,
   intermediate-monitor files, checkpoint inventory bytes/hashes and fresh reload
   identity. Smoke manifests remain explicitly ineligible as LR-lock candidates.

Validation is pending in ATT-0197-1/PBS3768755.opbs,source
183ad685785b8249d8a1c679a09e8dd5280d8cd5. This PBS job runs the full regression suite,
the three CLI smokes and rb_report against the original immutable timing inputs
to recalculate only the RBc forecast. It does not repeat 100-step timing or create
scientific training results. Only ordinary record-only profile metadata is
needed because this job measures no new time for comparisons.

Shared bypass_core/bypass_model, native read loops, FP32/BF16 thresholds, strong
model fixtures, A2 split, data/scoring/objective and formal sample budgets remain
unchanged. D-0026/D-0027 tests remain intact. Original review evidence and old
cost output remain historical artifacts; regenerated tables will cite this fix.
