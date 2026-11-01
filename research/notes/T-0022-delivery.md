# T-0022 delivery: RB0 implementation and timing

E-0019@1 is complete for this implementation assignment. Independent reviews:
RV-0060 passed with four low findings; RV-0062 passed with zero findings after
all four fixes. Diagnostic evidence **EV-0112** supports H-0017@1 within the
tested implementation scope. Scientific release and hypothesis status remain
the coordinator's decisions.

Final executed code:183ad685785b8249d8a1c679a09e8dd5280d8cd5.
Passing recheck snapshot:de4443b130653bfbd6c1418ce69d582ee75b57af; subsequent
commits add delivery/record notes only. Branch:sr/researcher-2/T-0022.

Implemented prefix KernelMatrix, T0, default-off model wiring, history
interventions and current weights, A2 filtering/paired training plans, RBc LR
selection and RB1 trainer/checkpoint handling, latency and diagnostic cost tools.
The native LatentReader and existing default paths retain their semantics.

Validation: **408 tests passed, no failures/errors/skips**, plus actual trainer
CLI smokes for RB1-02, RB1-01 and RB1-07, each4steps/8examples/744tokens and
reload.identical=true. The PBS driver verified lock-schema fields, counts,
monitor files and checkpoint inventory. RV-0062 independently inspected that
runtime evidence and the fixes. Shared M/V5 regressions and bitwise default
comparisons pass. No further required T-0022 check is pending.

Full check/error table and all eight ATT/RES/PBS records are in
[T-0022-evidence.md](T-0022-evidence.md). The frozen first-review package and
[focused follow-up](T-0022-review-followup.md) document the implementation and
the four fixes. Core runtime result:ATT-0197-1/RES-0191/PBS3768755,under
/work/xg24i002/x10041/bypass-lora/runs/ATT-0197-1/.

D-0026/D-0027 remain explicit: BF16 token/expanded-batch errors0.8203125 and
0.857421875 exceed unchanged0.50002 and match reference,so are diagnostics.
Exact topology, same-shape BF16 and decisive fresh FP32 gates pass. Prefix vs
reference token/expansion errors are zero. No threshold/strong fixture changed.

The [cost tables](T-0022-cost-tables.md) include all21cases with100stable steps,
seven512/128latency cases, RB1 per-arm estimates, RBc per-candidate estimates and
conditional RB5 per-layout estimates. Current JSON:
research/artifacts/T-0022/cost-summary.json, SHA256
5fe146e9c6d7f0e64cc236a3205e9f109d2b92f637bae4372cc7e1896faec56b.

- Maximum measured prefix side-layout: **0.243899 s/step**, below0.4.
- Nine-arm RB1 core estimate: **2.443961 node_h**, below4 for the measured scope.
- RBc12-16candidate estimate: **0.509314-0.679085 node_h**, one load/eval,
  no export/reload. Full formal runs retain export/reload.
- Timing inputs:ATT-0189-1/RES-0185 and ATT-0191-1/RES-0186;original report
  ATT-0193-1/RES-0188;revised cost report in ATT-0197-1/RES-0191.

Forecasts exclude future frozen interventions,bootstrap/analysis,data planning
and cold-cache setup; add those before a whole RB1 release. Evaluation costs
are scaled from one full R_resS1 dev timing and32row/task samples;RBc also uses
the4096/2000row ratio. Conditional RB5 matrix/monitoring, unmeasured S2 layouts,
S1concatT0 and mixed-rank LoRA need later contract-specific cost treatment.

A2 manifest:research/artifacts/T-0022/rb-confirmation.json, identical to durable
read-only /work/xg24i002/x10041/bypass-lora/runs/ATT-0180-1/rb-confirmation.json;
1000IDs/task,24798bytes. SHA256:
34277aa4627271dbf4e0bcb9592e3014c6130d87b1766caad17c2d6b97a8a7cc.

Measured usage for all8attempts,including failed checks and report/revision work:
**1918MIGseconds /14400=0.1331944444node_h**, within1.0node_h. Exact PBS walltime
was imported once per accounting reference. No run or usage remains pending.

No formal RBc/RB1/RB5 training or official/RB confirmation evaluation occurred.
The real calibration-stage/full-manifest lock CLI first executes in RBc; its
components and no-export branch have smoke/unit coverage. Future RB1 intervention
runner, n_eff/entropy metrics and analysis are outside this package. No task
performance or mechanism-benefit claim follows from this diagnostic evidence.

Coordinator next action:inspect and accept this delivery, merge the reviewed
branch, then decide the next assignment and release. The researcher has not
merged, started RBc/RB1 or changed any scientific contract.
