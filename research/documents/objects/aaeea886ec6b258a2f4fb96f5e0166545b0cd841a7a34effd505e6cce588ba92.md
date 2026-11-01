# T-0017 delivery: E-0016 decoder-first and marginal-slot screening

All four assigned seed-0 arms are complete. D1 passes its teacher-forced
stage-one point gate. D2a, D2b and D3 do not pass their registered H3 point
gates. These measurements are single-seed screening evidence; hypothesis
and route decisions remain with the coordinator. No confirmation was read.
D4/D5 belong to T-0018 and are outside this delivery.

## Code and validation

Executed source:6da33b39ca749a65140e2dcd0715c8f7b6d3359c, immutable snapshot
/work/xg24i002/x10041/bypass-lora/snapshots/6da33b39ca749a65140e2dcd0715c8f7b6d3359c.
RV-0045 passed with0 findings. Later commits on sr/researcher-1/T-0017 add
notes only; reviewed source/config/dependencies are unchanged. No unresolved
code-review findings remain. D-0016 records the scoped reason for the CLI's
retained stale label: inherited E-0015/T-0018 files are absent from this
snapshot and cannot execute in these runs.

Implemented modules: v2r16_memory.py (teacher, standard, hard, soft and
marginal protocols and diagnostic collector), v2r16_objective.py (log-space
slot marginalization), v2r16_train.py (schedules, provenance and runner),
v2r16_analysis.py (locked reducers and reports), environment/smoke/launch
helpers and PBS script. Legacy v2_train.train gains optional forward_fn and
curve_evaluator callbacks; its omitted-argument behavior stays exact.
v2_memory.py was not edited on this branch.

Final-source ATT-0152-1 / RES-0140 passed50 tests, exact old/new default
loss/logit/gradient/optimizer/control identity, four smoke reloads and the
D3 step-cost gate (ratio0.948018). Earlier ATT-0150-1 / RES-0138 is also
recorded. These smokes are explicitly ineligible as scientific evidence.
RV-0045 independently checked1000-row reducers and the analysis path in
PBS3766072.opbs; all author workloads likewise ran in verified PBS compute
allocations. The login shell was used for control work only.

The locked final analysis is ATT-0169-1 / RES-0161, PBS3767160.opbs:

- /work/xg24i002/x10041/bypass-lora/runs/ATT-0169-1/analysis/report.json
- /work/xg24i002/x10041/bypass-lora/runs/ATT-0169-1/analysis/report.md
- /work/xg24i002/x10041/bypass-lora/runs/ATT-0169-1/closing-audit.json

Its coverage is complete: D1,D2a,D2b,D3, no missing arms. Counts, labels,
episode alignment and EM/NLL were checked from all final raw predictions.
The separate closing audit verifies all final checkpoint inventories, both
continuations' exact D1 provenance,47 full1000-row prediction-file reloads,
and D2a's frozen A_v/Ro/P tensors bitwise equal to D1. All three D2a address
tensors changed; its selected and nonzero-gradient groups are exactly
A_q/A_k/Rq. Every formal run passed full-record and summary reload checks.

## Primary point gates

| Arm | Scored protocol | Normal/zero/shuffled correct of1000 | EM margin | NLL condition | Gated target weight | Point gate |
|---|---|---|---:|---|---:|---|
| D1 | teacher-forced |668/11/13|+65.5pp|not part of D1 gate|not part of D1 gate|pass (>=50pp)|
| D2a | standard all-question |50/11/18|+3.2pp|pass|0.069221|not passed|
| D2b | standard all-question |48/11/18|+3.0pp|pass|0.086935|not passed|
| D3 | hard last-position |15/11/10|+0.4pp|pass|0.061145|not passed|

The H3 EM threshold is10pp and address threshold0.125. D2a/b use the
valid-question-token-weighted all-position address mean. Their last-position
means,0.161943 and0.135142, are separate diagnostics. D3 gates its registered
last-position hard-read protocol. All gates are point estimates, and no
bootstrap interval changes them. Descriptive paired episode intervals use
10000 draws with RNG20261029 and shared within-arm/split indices; they
reflect data uncertainty conditional on this trained seed, not seed
replication, simultaneous guarantees or equivalence tests.

### D1 teacher controls

| Control | Correct/1000 | EM | NLL |
|---|---:|---:|---:|
| tf_normal |668|66.8%|1.141156|
| zero |11|1.1%|9.248186|
| tf_shuffled |13|1.3%|15.625579|
| tf_wrong |8|0.8%|15.593291|

D1 supports the stage-one portion of H-0010 under explicit target-slot
supervision. Standard all-slot D1 is23/1000, with zero11 and shuffled15;
oracle_value is668/1000. D2a/b inherit this supervised prewarm and then use
answer CE with normal all-slot reads. Each continuation loads adapter
parameters only, resets AdamW state and uses fresh300-step warmup/cosine
and a distinct derived-seed order for6000 steps.

D1 final1000-step mean training answer CE is1.103144 nat. Its OLS fitted
decline over steps5001..6000 is0.052743472 nat; the <0.05 plateau rule is
false. The final two500-step means are1.115847 and1.090441. The window's
maximum LR is7.3896% of peak; this schedule-conditioned label does not
establish convergence. All12 raw500-step means and25/50/75% evaluation
curves are retained in the report and formal run metrics, respectively.

### D3 hard and soft last-position controls

| Control | Hard EM | Hard NLL | Soft EM | Soft NLL |
|---|---:|---:|---:|---:|
| normal |1.5%|4.424336|1.6%|4.425512|
| zero |1.1%|9.248186|1.1%|9.248186|
| shuffled |1.0%|4.428258|1.3%|4.429245|
| permuted |1.3%|4.428310|1.3%|4.426536|
| oracle_key |1.6%|4.424475|1.0%|4.426867|
| oracle_value |1.7%|4.421358|1.7%|4.421358|

Marginal normal EM/NLL is0.9%/4.424916; zero1.1%/9.248186,
shuffled1.2%/4.426308 and permuted1.0%/4.425420. These protocols are
reported separately from standard E-0010 statistics. D3 training does not
use target-fact indices. Its final last-position argmax counts are
[775,0,0,0,0,0,0,0,0,0,1,11,14,28,72,99], with mean n_eff7.892364.

## Position strata and diagnostics

These are descriptive final EM/NLL; all controls and all address/neff/norm
diagnostics for both strata are retained in report.json's summaries.

| Arm/protocol normal | Position0 correct/52 | EM | NLL | Position1+ correct/948 | EM | NLL |
|---|---:|---:|---:|---:|---:|---:|
| D1 teacher |19|36.54%|2.747796|649|68.46%|1.053028|
| D1 standard |2|3.85%|7.257225|21|2.22%|6.902111|
| D2a standard |1|1.92%|4.307814|49|5.17%|5.357547|
| D2b standard |4|7.69%|4.244759|44|4.64%|4.354952|
| D3 hard |0|0.00%|4.325027|15|1.58%|4.429783|

D2a E1 normal/zero/shuffled EM is28.9%/0.9%/0.9%; D2b is9.7%/0.9%/1.4%.
Their E16 oracle_value EM is66.7% and17.0%, respectively. Full six-control
tables, E1 NLL, all/last weights, n_eff, argmax distributions, fusion/residual
ratios and their strata are in the final artifacts; these diagnostics do not
replace the registered gates.

## Run and evidence inventory

| Arm | Attempt | Result | Evidence | PBS job | Steps | Input tokens |
|---|---|---|---|---|---:|---:|
| D1 |ATT-0154-1|RES-0154|EV-0097|3766136.opbs|6000|4,000,470|
| D2a |ATT-0162-1|RES-0159|EV-0098|3766763.opbs|6000|4,000,432|
| D2b |ATT-0163-1|RES-0160|EV-0099|3766766.opbs|6000|4,000,402|
| D3 |ATT-0155-1|RES-0155|EV-0100|3766138.opbs|6000|4,000,470|

Each EV is screening/valid and linked to E-0016. As the brief requires,
none adds supports/challenges edges or sets a hypothesis status: D1 records
the stage-one result in its summary, and no D2/D3 H3 point gate passed.
Final raw outputs live under the project runs/<attempt>/arm directories.
Interim D1/D3 analysis ATT-0161-1 / RES-0157 is retained alongside the final
four-arm ATT-0169-1 / RES-0161 analysis. This task did not access confirmation.

## Compute and metadata

| Attempt | Measured PBS s | MIG-instance hours | Node hours |
|---|---:|---:|---:|
| initial smoke ATT-0150-1 |173|0.048056|0.01201400|
| final-source smoke ATT-0152-1 |172|0.047778|0.01194450|
| D1 ATT-0154-1 |1632|0.453333|0.11333325|
| D3 ATT-0155-1 |1707|0.474167|0.11854175|
| interim analysis ATT-0161-1 |3|0.000833|0.00020825|
| D2a ATT-0162-1 |1506|0.418333|0.10458325|
| D2b ATT-0163-1 |1505|0.418056|0.10451400|
| final analysis ATT-0169-1 |7|0.001944|0.00048600|
| author total |6705|1.862500|0.46562500|

Conversion is the user rule: MIG-instance hours x0.25. RV-0045's independent
reviewer check additionally used13s=0.003611 MIG-h=0.00090275 node-h,
confirmed in its separate ledger record (client_ref RV-0045); it is not
double-imported into the author attempts. Including that check, attributable
T-0017 compute is0.46652775 node-h, within the approximate0.5 task budget.
E-0016's0.9-node-hour ceiling also covers T-0018, which is accounted by its
owner. Measured usage, application timing and requested walltime are distinct.

Profiles are metadata only under D-0018/D-0019: D1/D3 recorded2g.24gb on
mg1109; D2a recorded1g.24gb on mg1109; D2b recorded2g.24gb on mg1110.
D2 jobs used the record-only guard wrapper stored in their immutable MCP
submission scripts, with no src changes. No slice-related rerun, analysis
or flag was made. All required same-run reload checks passed.

## Reusable interfaces and remaining scope

The shared v2r16_memory.train_forward/evaluate_batch/evaluate interfaces
accept support_fn=build_support, including recipient and shuffled donors.
v2r16_train.make_arm_schedule supports D4's released order and D5's derived
continuation order. v2_train.adapted(...,checkpoint=...) loads adapter-only;
v2_train.train creates fresh AdamW and accepts forward/curve callbacks.
The T-0017 CLI arms remain D1/D2a/D2b/D3. R1/D5 provenance and actual isolated
support construction are owned by T-0018. Researcher-3 received the passed
review/source and stable API list in MSG-0455.

No implementation or validation remains for T-0017. Replication, hypothesis
decisions, route selection and integration are the coordinator's next work.
There is no claim of seed generalization or of a result beyond this model,
data, recipe and step budget. This branch has not been merged or pushed.
