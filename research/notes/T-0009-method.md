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
