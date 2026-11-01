# T-0024 delivery: RBc LR lock, E-0020@1

All sixteen calibration candidates and both artifact-report jobs are complete, verified and imported once. The reviewed LR rule is locked; no task work or usage import remains. Coordinator acceptance is next.

| Family | Locked LR | Raw final minimum LR | Selection | Grid boundary |
| --- | ---: | ---: | --- | --- |
| RBc-L: LoRA4 r7 | 3e-4 | 1e-3 | 0.836886% above minimum, within1% | no |
| RBc-T: R_res+T0 S1 | 3e-3 | 3e-3 | minimum | yes |
| RBc-R: R_res+KernelMatrix S1 | 3e-3 | 3e-3 | minimum | no |
| RBc-C: C_dual+KernelMatrix S1 | 1e-3 | 3e-3 | 0.505300% above minimum, within1% | no |

All four raw initial minima were upper endpoints. The reviewed extension rule therefore added L/T at3e-3 and R/C at1e-2 exactly once. T's minimum remains at the new upper endpoint: label “grid boundary” and do not expand again. R's1e-2 is1.0273217% above its minimum and outside1%; decisions used unrounded values. Full per-candidate relative excess and extension decisions are in the evidence and final report.

The committed [lr-lock.json](../artifacts/T-0024/lr-lock.json) is3646bytes, SHA256:

`38b8ee24af3ded8aee398a4cc3dfaa5a1dfece4737a6cde34642fbff7bf1012d`

It was committed in c414c7d97b66bedd0f7a5141b2d90992016ad38c. It preserves the reviewed lock CLI choices and all16 input manifest FileRefs, split SHA and boundary flags; its only addition to the raw CLI output is the authenticated common plan-order SHA. Raw output is [lr-lock-reviewed.json](../artifacts/T-0024/lr-lock-reviewed.json).

Split SHA256:34277aa4627271dbf4e0bcb9592e3014c6130d87b1766caad17c2d6b97a8a7cc.
Plan order SHA256:053284831520676982857bf9677a9fa3e338b0bbad4d2ce2af1c3f88d20457d3.

The complete16-row [candidate/cost table](T-0024-candidate-costs.md) contains family,LR,NLL,finite status,steps,tokens,stable step seconds,node·h and ATT. [Final job metadata](T-0024-final-jobs.json) adds every RES/PBS/spec/source/config/log/output/accounting reference. [Full report](../artifacts/T-0024/final-report.json) is byte-identical to ATT-0236-1/report.json, SHA2569fd7c94bcc473ea79760e2f40057efa2fc1ea4e01c4db93bacf16e2bd35e6b93.

Verification: each candidate is complete/passed/finite, effective V0/seed0, with the registered representative/source/spec/LR, expected parameter counts and side accounting. Each has1096steps,2192presentations and200229nonpadding input tokens, with the penultimate step below200000. Full calibration2048rows/task and its token-normalized task-macro NLL were reconstructed from saved gold CE sums/counts. All16 share the split and plan order. All runs passed startup allocation checks; profiles are metadata only. The trainer/core/model/data/scoring/environment remain byte-identical to RV-0062's reviewed snapshot; only artifact-only rbc_report.py was added. Existing RB0 tests were not rerun; the actual full calibration entry and real lock CLI were exercised in PBS.

Records:
- Initial12: ATT-0210-1..ATT-0221-1 -> RES-0193..RES-0204.
- Required4endpoints: ATT-0223-1..ATT-0226-1 -> RES-0209..RES-0212.
- Initial verification/extension report: ATT-0222-1 / RES-0205 / PBS3768977.
- Final verification/reviewed lock: ATT-0236-1 / RES-0214 / PBS3769071, source77f1e9b1f200940d733ba76afd807af4a7516498.
- Evidence **EV-0114**, diagnostic/valid, sealed **DOC-0241@1**: [T-0024-evidence.md](T-0024-evidence.md). Its neutral context is EV-0114 about E-0020, which tests H-0013@1; there is no supports/challenges relation or hypothesis-state change.

Exact total usage is **9677MIG seconds /14400 =0.6720138888888889node·h**, below1.5. This includes9647seconds for16candidates and30seconds for two reports(13+17). All18usage entries were imported once under miyabi-pbs:<job>. Twelve superseded prepared registrations ATT-0198..0209 never submitted a job and consumed zero compute.

Limitations are the registered ones: short-budget seed0 calibration; T grid boundary; S2/S4/fixed-query/eta/4 inherit R's LR and C_up inherits C_dual's LR. No efficacy claim is made. No implementation issue remains. RB1,dev,RB confirmation,official validation and slice analysis were outside execution scope and were not run. RB1 release and branch merge belong to the coordinator.
