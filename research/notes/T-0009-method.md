# T-0009 / E-0008 F-line positive control

Implementation uses unchanged `v1_lora.LoRAModel` default four-layer rank-10,
alpha-10 all-seven-module layout (901,120 FP32 trainable parameters). Base is
pinned Qwen3-0.6B-Base BF16/eager; load_base's diagnostic initialization seed is
reset to zero before LoRA initialization. No dependencies changed.

`v2_eval.prompt_tokens` now accepts explicit train in addition to dev/calibration;
default remains dev, and confirmation always rejects. F-line uses full context
and computes CE only from the final prompt position against the answer token.
Labels and fact diagnostics never enter model forward. Training batches contain
eight episodes, loss is their mean, AdamW defaults with zero weight decay,
gradient clip norm one. Evaluation batches are fixed at sixteen.

Shared `v2_schedule.py` is copied byte-for-byte from researcher-2 commit 588cadd.
Order digest serializes all three epochs as UTF-8 episode_id plus LF, including
the final LF. The zero-based learning-rate function gives first update 1/warmup,
warmup endpoint one, final update zero. Stop at first complete eight-episode
step crossing the budget. Count support_input_ids plus question_input_ids,
excluding answer. Full schedule is saved with each run. Independent PBS check
against researcher-2's schedule is pending.

Calibration runs use calibration only; final 4M run evaluates dev at ceil of
25%, 50%, 75% of total steps solely for curves, then final calibration/dev.
Final adapter is saved and reloaded onto a fresh Base, requiring complete
per-row records (predictions, NLL and ranks) to match exactly at batch16.
All/nonself EM/NLL, subject copying, competition rank (1 + number strictly
greater), and train-frequency constant prior are reported. Ties in train
frequency choose smallest token ID. Gate uses all-row calibration EM minus
calibration prior >= 0.20, not nonself accuracy. E-0010 also requires E-0009's
finite LR choices; coordinator owns that joint release decision.

Three initial LRs are 1e-4, 3e-4, 1e-3 at 200,000 tokens, with one endpoint
extension /3 or *3 if indicated, and smaller LR preferred within relative
0.5% of best finite NLL. Nonfinite training/evaluation outcomes are failed LRs;
other exceptions remain engineering failures. Full run uses 4,000,000 tokens.

Smoke ATT-0054-1 uses four training steps, first32 calibration/dev, and runs
new objective/split/schedule tests, existing diagnostic split tests and real
Qwen LoRA initialization/frozen-weight/gradient/checkpoint tests. Smoke metrics
are not the scientific positive-control result. Budget <=1 node-hour total,
MIG usage multiplied by .25. All execution is in PBS; no confirmation reads.

## RV-0016 corrections and selection orchestration

RV-0016 found one medium and one low failure-handling issue. Explicit stages now
restrict scientific failed-LR outcomes to non-smoke training or original final
evaluation. Reload, save/load, curve evaluation, setup and all smoke failures
propagate with partial manifest and failed validation; completed training metrics
are retained for reload failures. Smoke success requires all four steps and exact
reload. Fault-injection tests cover NaN and Inf reload, finite reload mismatch,
scientific calibration divergence, smoke divergence, and finite smoke.

`v2_fline_select.py` runs in PBS after three initial calibrations. It validates
300 steps/200170 tokens, 2048 calibration rows, batch16 and exact reload for finite
candidates, excludes failed LRs, and uses the unchanged shared selector. Only
calibration NLL enters selection. If the selected initial point is an endpoint,
it emits the single required rate /3 or *3; after that run it reselects from all
four without further extension. The recorded selection JSON determines the final
run rate. Three failed candidates raise for implementation investigation.

Verified ATT-0054-2: eleven tests passed, four steps/2681 tokens, exact reload,
32-row calibration/dev smoke. Independent order SHA256
0d4b55ed8655d91cf0d0532bdac1820a518b5c5730ef70e48f264da6b4bc9e79 matches
researcher-2; budgets 200000/4000000 give 300/6000 steps, 200170/4000470 tokens,
warmup15/300. Evidence EV-0051, result RES-0056. Initial cwd failure RES-0054
cost3s; successful smoke24s; both usage imports total0.001875node-hours.
