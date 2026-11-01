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

Validation complete:ATT-0197-1/PBS3768755.opbs/RES-0191,executed source
183ad685785b8249d8a1c679a09e8dd5280d8cd5. Allocation verified on mg1098;the
profile was recorded as2g with nonfatal policy. Scheduler exit0,complete parent
manifest and passed validator. All408regressions pass (0failures,0skips) in
96.72s. Both patched broken-state builders are rejected by the actual reference
gate;doubled accounting is rejected by train;unsupported masks raise;the
calibration run-level unit check proves one eval and no export/reload.

The three actual trainer CLI smoke manifests (RB1-02/01/07) are complete,each
with4steps/8examples/744input tokens and reload.identical=true. Family,stage,
phase,seed,failed,LRS/NLL,split/sample-order hashes,plan/train/parameter counts,
three monitoring files and checkpoint file hashes were checked by the PBS
driver. They retain stage=smoke/phase=R1 and are not calibration candidates.
Their shared smoke order hash is
439277d84952473c71ec378d4f6e6acecdc52b6aac9d8d352b46975901314cbb.

The PBS driver also regenerated rb_report against the original immutable timing
inputs. It did not repeat100-step timing or create scientific training results.
RBc now costs0.509314-0.679085node_h for12-16candidates with one load and one
calibration pass,no export/reload. The RB1 core2.443961088node_h forecast and
conditional RB5 costs are unchanged. All prior estimate limitations apply.
The committed JSON/MD are byte-identical to ATT0197/cost-report outputs.

Exact additional use162MIGseconds=0.01125node_h was imported once;all8attempts
now total1918MIGseconds=0.1331944444node_h,within the1.0node_h ceiling. The
generated table's explicitly historical through-timing subtotal excludes both
tabulation/follow-up work;this is the final total through RES0191.

Focused runtime artifacts:
- /work/xg24i002/x10041/bypass-lora/runs/ATT-0197-1/manifest.json
- /work/xg24i002/x10041/bypass-lora/runs/ATT-0197-1/checks/checks.xml
- /work/xg24i002/x10041/bypass-lora/runs/ATT-0197-1/checks/checks.log
- /work/xg24i002/x10041/bypass-lora/runs/ATT-0197-1/RB1-02/manifest.json
- /work/xg24i002/x10041/bypass-lora/runs/ATT-0197-1/RB1-01/manifest.json
- /work/xg24i002/x10041/bypass-lora/runs/ATT-0197-1/RB1-07/manifest.json
- /work/xg24i002/x10041/bypass-lora/runs/ATT-0197-1/cost-report/manifest.json

Updated cost JSON SHA256:
5fe146e9c6d7f0e64cc236a3205e9f109d2b92f637bae4372cc7e1896faec56b.
Updated cost table SHA256:
f58d74ca58da06b5f75e401e2b520ff3741fc48d91ed486102cdac44f01cc616.

One focused RV-0060 recheck is next,before delivery. No RBc/RB1 release.

Shared bypass_core/bypass_model, native read loops, FP32/BF16 thresholds, strong
model fixtures, A2 split, data/scoring/objective and formal sample budgets remain
unchanged. D-0026/D-0027 tests remain intact. Original review evidence and old
cost output remain historical artifacts; regenerated tables will cite this fix.
