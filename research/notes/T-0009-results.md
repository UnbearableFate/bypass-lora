# T-0009 / E-0008: trained F-line positive control

## Result and scope

The final 4M-input-token F-line LoRA passes the E-0008 positive-control gate:
calibration EM 2047/2048 = **99.951171875%**, compared with the train-frequency
constant prior 1.904296875% and required threshold **21.904296875%**.
Dev EM is **1000/1000 = 100%**. The question-subject-excluded subset is likewise
2020/2021 correct on calibration and 987/987 on dev; nonself subject-copy rate
is zero on both. This demonstrates learnability with facts retained in context
under this fixed synthetic task and training recipe. It neither supports nor
challenges H-0003's memory-only mechanism. E-0010's joint release also requires
finite learning-rate choices for both E-0009 read operators; that check and route
decision belong to the coordinator. Confirmation was never read.

These are single-seed diagnostic results, not statistical confirmation. The
final checkpoint is the main result; intermediate dev measurements were used
only for the learning curve. Calibration NLL is worse after full training than
at the selected short calibration checkpoint, even though calibration EM is the
same; do not claim monotonic likelihood improvement or perfect calibration.

## Reproducible implementation

Executed training source: `170bc52014a7dd67b08727006360ad7eceaf62a0`,
covered by **RV-0018 pass, zero findings**. New F entry point `src/v2_fline.py`
uses unchanged `src/v1_lora.py`: layers6/13/20/27, all seven projections,
rank10/alpha10, 901120 FP32 adapter parameters, frozen BF16 Base, eager attention,
no dropout, seed0, AdamW default betas/eps, weight decay0, clip norm1.
Model/tokenizer: Qwen/Qwen3-0.6B-Base revision
`da87bfb608c14b7cf20ba1ce41287e8de496c0cd`. Environment lock3c916b51337d,
Python3.13.13, torch2.10.0+cu128, transformers4.57.1, cuda12.8,
GH200 MIG2g.24gb. No dependency, old V1, generator or memory-module edits.

F prompt uses the shared prompt_tokens with explicit train/calibration/dev;
default dev is preserved and confirmation rejected. Loss is the eight-episode
mean of single-answer-token CE at the last prompt position; no support labels
or diagnostic fields enter forward. Evaluation batch is always16. Full final
calibration/dev per-row records, including NLL/rank, exactly match a fresh
Base plus reloaded adapter. Checkpoint is ATT-0069-1/adapter/.

Shared schedule copied unchanged from researcher-2's588cadd. Independent PBS
ATT-0054-2 calculation matches researcher-2's ATT-0053-1 (MSG-0164/0166):
three-epoch episode_id sequence SHA256
`0d4b55ed8655d91cf0d0532bdac1820a518b5c5730ef70e48f264da6b4bc9e79`.
Encoding: UTF-8 episode_id followed by LF, including final LF, all three epochs.
200000 budget ->300steps/200170tokens/warmup15;
4000000 budget ->6000steps/4000470tokens/warmup300.
Input count excludes answer and padding. The penultimate full step has3999804
input tokens, so the final6000th step is the first complete batch crossing4M.
Final learning rate is zero. Order/batches and all6000 step logs are retained.

## Learning-rate selection

All four calibrations are300steps/200170tokens; all losses and NLL finite,
2048 calibration rows, exact fresh-checkpoint reload, no dev selection.

| LR | Calibration EM | Calibration NLL | Attempt | Result |
|---|---:|---:|---|---|
|1e-4|99.4140625%|0.0250884891|ATT-0058-1|RES-0059|
|3e-4|99.8046875%|0.00909726144|ATT-0059-1|RES-0060|
|1e-3|99.951171875%|0.00160201389|ATT-0060-1|RES-0061|
|3e-3|99.8046875%|0.0106951695|ATT-0067-1|RES-0063|

Initial best1e-3 is the upper endpoint, requiring the single3e-3 extension.
The reviewed selector ran inside ATT-0067-1, both before extension (guarding the
requested rate) and afterward. Final choice is **1e-3**; no further extension.
The relative0.5% tie rule does not affect this selection. Artifacts:
`runs/ATT-0067-1/initial-selection.json` and `final-selection.json`.
Evidence EV-0053 for initial grid and EV-0055 for extension/final selection.

## Final and learning-curve metrics

Final run ATT-0069-1, result RES-0075. All run paths below are relative to
`/work/xg24i002/x10041/bypass-lora/`.

| Metric | Calibration | Dev |
|---|---:|---:|
|Examples|2048|1000|
|Correct|2047|1000|
|All EM|99.951171875%|100%|
|All NLL|0.0155656118|0.00000123547761|
|Nonself examples|2021|987|
|Nonself EM|99.9505195448%|100%|
|Nonself NLL|0.0157735455|0.00000123351272|
|All subject copy|1.318359375%|1.3%|
|Nonself subject copy|0%|0%|
|Mean correct-answer competition rank|1.23681640625|1|
|Constant prior EM|1.904296875%|1.5%|

Subject copying on all rows reflects the27/13 valid self-edge answers, not a
nonself copying failure. Rank is1+number of vocabulary logits strictly higher
than the target. The audit identifies one nonself calibration error, episode
84b4525aaac41599f3d4146c2abf2d05ea077eae4b3e811e6d9b4af4e76d48ac:
target41187, prediction6798, subject12375, NLL31.875, target rank486. This
single confident error accounts for almost all aggregate calibration NLL;
the other2047 rows are correct. No claim of broad retrieval failure follows
from mean rank1.2368. Dev has no errors.

| Optimizer step | Dev EM | Dev NLL | Use |
|---|---:|---:|---|
|1500|100%|0.00000491865088|curve only|
|3000|99.9%|0.00962693879|curve only|
|4500|100%|0.00000347776488|curve only|
|6000|100%|0.00000123547761|final checkpoint|

Artifacts: `runs/ATT-0069-1/metrics.json`, `learning.jsonl`, `curve.jsonl`,
`dev-step-1500.jsonl`, `dev-step-3000.jsonl`, `dev-step-4500.jsonl`,
`calibration.jsonl`, `dev.jsonl`, and corresponding `*-reloaded.jsonl`.
No intermediate checkpoint selection took place.

## Validation, review and cost

ATT-0054-1 failed before tests because PBS initial cwd was not the snapshot;
fixed explicit cd in shell without changing source. RES-0054 is partial.
ATT-0054-2:11tests passed9.39s, real4steps2681tokens,32-row calibration/dev,
exact reload; RES-0056/EV-0051. RV-0016 found one medium reload-nonfinite
classification issue and one low divergent-smoke validation issue. Both fixed
with stage-aware failure handling. ATT-0057-1:20tests passed9.71s, including
fault injection and selector boundaries, plus unchanged finite real-smoke
trajectory and exact reload; RES-0058. RV-0018 passed the code recheck.

The saved-output audit `src/v2_fline_audit.py` (commitc09e6a7) independently
recomputes schedule/counts, every finite training record, original/reloaded
identity/target alignment, summary EM/NLL/copy/ranks, curve EM/NLL, and records
the error/outlier. ATT-0076-1 audit passed, result RES-0076; final diagnostic
evidence EV-0062 links the final run and audit. Final output review **RV-0021
passed with zero findings**, supporting the F prerequisite and task delivery.
This audit is read-only and does not perform fresh inference.

| Attempt | PBS job | Wall seconds | node-hours |
|---|---|---:|---:|
|ATT-0054-1|3761331.opbs|3|0.000208333|
|ATT-0054-2|3761373.opbs|24|0.001666667|
|ATT-0057-1|3761486.opbs|24|0.001666667|
|ATT-0058-1|3761511.opbs|60|0.004166667|
|ATT-0059-1|3761513.opbs|57|0.003958333|
|ATT-0060-1|3761514.opbs|57|0.003958333|
|ATT-0067-1|3761606.opbs|61|0.004236111|
|ATT-0069-1|3761643.opbs|713|0.049513889|
|ATT-0076-1|3761896.opbs|2|0.000138889|

All above usage imported from executor ledger, including failed work.
Final total including output audit:1001MIGseconds = **0.069513889node-hours**,
versus E-0008 budget1node-hour. Formal application703.882s, training region681.332s,
throughput5871.54 counted input tokens/s; this region includes three dev curve
evaluations, so it is not a pure optimizer-only benchmark. Peak CUDA allocated
memory2395061248bytes. Scheduler wall713s includes startup/exit overhead.
