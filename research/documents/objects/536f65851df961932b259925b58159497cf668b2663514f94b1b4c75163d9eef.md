# T-0031 delivery: E-0028@1

The locked PBS analysis passed. Replacing the original QK/V tap with normalized
layer input X increased dev macro accuracy by **0.65 percentage points** at
eta=0.25, below the preregistered +1 pp support gate for H-0021. Calibration
selected eta=0.25 for QK and eta=1 for X; their dev changes relative to eta=0.25
were **0 pp and -0.25 pp**, respectively. This is a screening challenge to
H-0022@2. All ten rules have evidence records; hypothesis statuses and route
actions remain for the coordinator's conclusion review.

Selected X is 1.05 pp below the accepted LoRA r7 reference, so rule 10's
point-estimate stop condition is triggered. It exceeds that threshold by only
0.05 pp; its descriptive interval includes zero. This delivery records the
trigger without making a route or status decision.

## Main dev results

Each task has 1,000 rows. Accuracy is percent; NLL is the saved per-task answer
NLL. QK denotes the original q_norm/k_norm/v_proj tap; X is the output of that
layer's input_layernorm. All side branches use C_up and r32.

| Arm | History / eta | PIQA | BoolQ | Macro | PIQA NLL | BoolQ NLL |
|---|---|---:|---:|---:|---:|---:|
| Base | none | 72.30 | 68.00 | 70.15 | 2.750282 | 1.924674 |
| RB1-07 LoRA r7 | accepted reference | 74.40 | 83.50 | 78.95 | 2.530501 | 0.584030 |
| RB1-09 QK | S1 / 0.25; selected QK | 73.20 | 81.80 | 77.50 | 2.558323 | 0.555336 |
| X-01 | S1 / 0.25 | 74.00 | 82.30 | 78.15 | 2.553236 | 0.546032 |
| X-02 | T0 / 0.25 | 73.80 | 82.80 | 78.30 | 2.539571 | 0.569671 |
| X-03 QK | T0 / 0.25 | 74.00 | 82.30 | 78.15 | 2.557071 | 0.571113 |
| ETA-X-1 | S1 / 1; selected X | 73.90 | 81.90 | 77.90 | 2.542444 | 0.579743 |

The complete task/NLL outputs for all 13 new training arms, all
25/50/75/100% dev curves, runtime accounting and 253 paired comparisons are
retained in [the locked analysis](/work/xg24i002/x10041/bypass-lora/runs/ATT-0345-1/analysis.json).
Base, LoRA and RB1-09 were reused rather than retrained; RB1-09 was re-evaluated
for compatibility and calibration.

## Preregistered rules and evidence

| Rule | Observed outcome | Evidence |
|---|---|---|
| 1: implementation | All eight grouped checks pass: init/Base, off/Base, reload, parameter counts, paired initialization and exact tap, unchanged RB1-09, source-bound selector tests and initialization digests. | EV-0154 diagnostic |
| 2: learnability | All 13 new arms exceed Base by at least 1 pp; gains range 6.75 to 8.15 pp. | EV-0155 screening |
| 3: H-0021 | X-01 minus RB1-09 = +0.65 pp, descriptive 95% interval [-0.25,+1.60]. Preregistered outcome: descriptive; no support/challenge edge. | EV-0156 screening |
| 4: history | X S1 minus matched T0 = -0.15 pp [-1.20,+0.90]; QK S1 minus matched T0 = -0.65 pp [-1.70,+0.45]. Swap drops are 2.00 pp for X and inherited 1.95 pp for QK. Neither history-gain/swap conjunction fires. | EV-0157 screening |
| 5: eta lock | Calibration-only selection: QK 0.25, X 1. Neither winner is a boundary; no extension is required. Hash recorded before any Part B dev aggregation. | EV-0153 screening |
| 6: H-0022@2 | QK eta gain = 0 pp by identity; X gain = -0.25 pp [-1.15,+0.65]. Both directional outcomes challenge; no new-seed trigger. | EV-0158 screening, challenges H-0022 |
| 7: LoRA | X-01 minus LoRA = -0.80 pp [-1.95,+0.35]; selected X = -1.05 pp [-2.15,+0.10]; selected QK = -1.45 pp [-2.60,-0.30]. No +1 pp advantage/replication trigger. | EV-0159 screening |
| 8: descriptions | X-02 minus X-03 = +0.15 pp; selected X minus selected QK = +0.40 pp. All curves, task metrics, calibration/NLL and RMS retained. | EV-0160 screening |
| 9: scope | Single-seed screening; task/NLL, interventions, curves and bootstrap intervals descriptive. Confirmation and official validation remain unopened. | EV-0161 screening |
| 10: route condition | True because selected X is strictly more than 1 pp below LoRA. The alternative H-0021-challenges / selected-tap conjunction is false. Coordinator decision required after conclusion review. | EV-0162 screening |

The [evidence index](T-0031-evidence-index.json) binds each rule to its attempts
and results. Rule 5 was recorded before dev aggregation and was not duplicated.
No hypothesis status was changed.

## Calibration selection and complete eta curve

Each calibration score is the exact macro accuracy fraction over 2,048 rows
per task. NLL was descriptive and did not select eta. The frozen accuracy tie
rule was nearest log2 distance to 0.25, then smaller eta; neither winner tied.

| Eta | QK calibration fraction | QK dev macro % | X calibration fraction | X dev macro % |
|---:|---:|---:|---:|---:|
| 0.0625 | 3203/4096 | 77.30 | 799/1024 | 77.45 |
| 0.125 | 1597/2048 | 77.25 | 1601/2048 | 77.60 |
| 0.25 | **201/256** | 77.50 | 3201/4096 | 78.15 |
| 0.5 | 1599/2048 | 77.55 | 3189/4096 | 77.50 |
| 1 | 3205/4096 | 77.10 | **801/1024** | 77.90 |
| 2 | 799/1024 | 76.90 | 3187/4096 | 77.55 |

Plan ATT-0342-1 / RES-0312 and lock ATT-0343-1 / RES-0318 ran only on calibration.
The lock includes manifest and calibration-prediction SHA256s for all 12
producers. Its reconstruction was checked again before the final analysis
opened dev. [Lock record](T-0031-eta-lock-record.json); immutable lock:
[/runs/ATT-0343-1/eta-lock.json](/work/xg24i002/x10041/bypass-lora/runs/ATT-0343-1/eta-lock.json).

SHA256: `e6c1399c683b50cabb05c727e37df84be165109a2ae4dd1ba5bcf93d3349fe60`.

## Frozen X-01 interventions

ATT-0344-1 / PBS 3776824.opbs / usable RES-0320 passed the application validator.
Normal evaluation reproduces the training result exactly: zero prediction
mismatches and zero score difference. iv_off equals same-allocation Base
exactly, and also equals the pinned Base. Drops below mean normal minus
intervention; a negative drop means the intervention scored higher.

| Condition | PIQA % | BoolQ % | Macro % | Drop pp | Descriptive 95% interval, pp |
|---|---:|---:|---:|---:|---|
| normal | 74.00 | 82.30 | 78.15 | 0 | [0,0] |
| iv_off | 72.30 | 68.00 | 70.15 | 8.00 | [6.24875,9.75] |
| iv_window1 | 71.90 | 79.40 | 75.65 | 2.50 | [1.10,3.95] |
| iv_window8 | 73.10 | 80.80 | 76.95 | 1.20 | [0.15,2.30] |
| iv_strict | 74.60 | 82.40 | 78.50 | -0.35 | [-0.90,0.20] |
| iv_swap | 72.30 | 80.00 | 76.15 | 2.00 | [0.90,3.10] |
| iv_permute | 71.10 | 81.30 | 76.20 | 1.95 | [0.80,3.10] |

Per-task NLL, prediction rows and read diagnostics are retained in
[conditions.json](/work/xg24i002/x10041/bypass-lora/runs/ATT-0344-1/conditions.json).
Swap sensitivity does not establish a gain over the matched no-history T0.

The first import RES-0319 was rejected for an administrative label mismatch:
the registered evaluation arm was X-01-interventions, while the producer model
arm was correctly X-01. The import-only envelope
[T-0031-intervention-import-manifest.json](T-0031-intervention-import-manifest.json)
leaves the optional registration arm unspecified and preserves producer_arm,
raw manifest, metrics and validator. RES-0320 has no mismatches and is usable.
The raw scientific producer was not changed; the final analysis independently
checks that raw manifest and X-01 checkpoint.

## Validation and reproducibility

Scientific source for every formal train/eval, intervention, selection and
analysis is exactly `0f22cd60f45674ec20b800fdd455f80c2b6d46dd`.
The normalized-X tap, matched T0 paths, eta-only configuration, calibration
selector and locked analysis were implemented and reviewed there. RV-0088
passed with zero findings. Inherited path coverage is recorded under D-0061
with RV-0076/77/78/79/80/81/82/88; stale auxiliary paths do not execute in
E-0028. No scientific file or launcher differs in later note-only commits.

Final smoke ATT-0315-1 / RES-0284 passed 112 tests, tap/init/configuration/count
registration and a 32-row accepted/new-source GPU comparison. The final
locked analysis ATT-0345-1 / RES-0321 / PBS 3777371.opbs passed
`e0028_locked_analysis`, including source, checkpoint inventory, row identities,
full budgets, order/LR, reloads, parameter gradients, tap, eta lock and all
implementation gates. Both application validators and native terminal records
were inspected; scheduler exit zero alone was not used as success evidence.

Frozen Qwen3-0.6B-Base uses model/tokenizer revision
`da87bfb608c14b7cf20ba1ce41287e8de496c0cd`, layers [6,13,20,27], r32,
A=randn/sqrt(fan_in), Cu=0 and identity Rq/Ro. R1 seed 0 and LR=0.001 are fixed.
All 13 new trainings completed 13,115 steps, 26,230 examples, 2,000,069 input
tokens and 387,653 loss tokens. Order SHA256:
`33759bc20322d0c9981128dd8574438546fc3439194ab32b7c79e35e0f0fc2bf`.
The split/provenance files and prediction-row hashes are bound in the saved
producer inventories and final analysis sources.

Environment: `envs/3c916b51337d-cuda`, cuda/12.8, torch 2.10.0+cu128,
uv.lock SHA256
`31429457ceece707abf1505af6e6210b44a6e2817d2527241892f0587aca47a3`.
All workloads ran through PBS after compute-allocation checks. The login shell
was used only for control work, inspection and records.

Stored/trainable/with-gradient counts agree: normalized-X arms 794,624;
original-tap arms, including X-03 T0, 925,696; accepted LoRA r7 630,784.
Each S1 layer retains FP32 M[1,32,32] and c[1,32]: 4,224 bytes plus an
8-byte count; four layers total 16,896 M+c bytes plus 32 count bytes.
T0 retains no history. X-01 steady step time was 0.215286 s and peak allocated
CUDA memory 3,330,547,712 bytes; selected X eta1 was 0.204875 s and the same
peak. Full per-arm measurements and state/read/write counts are in
analysis.json.runtime; timings from different MIG profiles are descriptive.

Actual 1g allocations were the final smoke, X-02, ETA-X-2 and ETA-QK-0.125;
the generic ledger hardware label sometimes says 2g. The application logged
the actual nvidia-smi/torch profile. Under D-0060, those training init gates
use a same-allocation Base; cross-job pinned comparisons are descriptive.
They also happened to match pinned predictions exactly. Other accepted-2g
jobs use strict pinned-reference gates. RB1-09's full dev re-evaluation on
2g matches its accepted source; the 1g smoke exercised the same-job
old/new-source fallback. No tolerance was introduced.

Step-0 RMS is descriptive, on the same first R1 candidate's 51 valid tokens,
eta=0.25, before any optimizer update:

| Layer | X vbar | X Z | QK vbar | QK Z |
|---:|---:|---:|---:|---:|
| 6 | 0.496035 | 0.066177 | 0.351588 | 0.042915 |
| 13 | 1.544267 | 0.257529 | 1.760445 | 0.147661 |
| 20 | 6.286733 | 1.063111 | 3.332197 | 0.435684 |
| 27 | 13.528601 | 2.069826 | 9.586698 | 1.214471 |

## Cost, scope and handoff

Native measured usage is **16.101389 MIG instance-hours = 4.02534725 node-hours**
across **21 jobs**, below the 5.5 node-hour cap. This includes both smokes, the
cost audit, 13 trainings, reference evaluation, plan, lock, interventions and
final analysis. No endpoint or retry training occurred. Native accounting
references were imported once. The final analysis contains the preceding
20 jobs' 4.02409725 node-hours; its own 18 scheduler seconds add 0.00125.
See [final cost](T-0031-final-cost.json) and the updated [cost table](T-0031-cost.md).
Planning used comparable 2g smoke timings; requested walltimes are not usage.

All intervals use 10,000 shared task-stratified paired bootstrap draws,
seed 20261101, conditional on these seed-0 trained checkpoints. They exclude
training-seed and eta-selection uncertainty and are not simultaneous or
multiplicity-adjusted guarantees. Dev also supplies decision thresholds.
LR was inherited rather than recalibrated for tap/T0/eta interactions; T0
eta remained 0.25; the side branch had an eta search in the LoRA comparison.
The conclusions are restricted to this model, four layers, R1 and fixed LR.
No A2 confirmation or official validation was opened.

All required execution, imports and rule evidence are complete. There are no
pending T-0031 jobs or further runs initiated. Retain all checkpoints and raw
artifacts for acceptance/review. The coordinator should perform conclusion
review before changing H-0021/H-0022 status or deciding rule 10's route action.
