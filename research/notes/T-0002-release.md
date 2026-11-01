# V1 calibration release — before calibration results

E-0002; F context retained, tokenwise only. No official validation file is downloaded or opened. No cached/chunked memory path is used.

Data: PIQA official ybisk/ybisk.github.io revision 21edab439af693b961be2f069e8690a88d3b4e37, piqa/data/train.jsonl and train-labels.lst. BoolQ google/boolq revision 35b264d03638db9f4ce671b711558bf7ff0f80d5, data/train-00000-of-00001.parquet (official GCS URL returned 403). Raw SHA256 values:
- PIQA train: f805f9c96518d7b59f2f870d90f712457cda949979469d9fa3fd706ce22e0a1a
- PIQA labels: df8ae080c41f02f88e81f6932d425f4b7ee72f433ad552e6a30d45e358e8f8bb
- BoolQ train: 4f028e992c0bd4df30b9f056f4946b64f5c23028034ff0ed5ea467d8538cc623

`derive_seed(*args)` SHA256-hashes UTF-8 JSON of args, ensure_ascii=False, sort_keys=True, separators comma/colon; first eight digest bytes big-endian unsigned. Split seed is derive_seed(20261029,"concat_loop_v2","V1",task+"-split"). Shuffle official train indices with Python Random; first1000 dev, next2048 calibration, remainder train. Save exact mapping and hash in each run; reference smoke mapping before calibration.

Train order: Random(derive_seed(20261029,"concat_loop_v2","V1","train",0)+epoch) shuffles PIQA then BoolQ training lists. Alternate PIQA/BoolQ until one ends, then remaining rows of the other. Every row appears once per epoch. All arms use identical order. Effective batch2, microbatch1, accumulated CE sums divided by total supervised tokens in the optimization step. At least100 steps per calibration, checked before training. Stop at first complete step reaching200000 nonpadding input tokens, or three epochs. Record any threshold overshoot. E-0003 reuses batch and order with2M threshold/three epochs.

Templates: PIQA `Goal: {goal}\nSolution:` then ` {solution}`. BoolQ passage `Passage: {passage}\n` followed by `Question: {question}\nAnswer:` then ` yes`/` no`. Prompt segments and completion encoded separately with add_special_tokens=False; no BOS/EOS appended. Reserve longest candidate tokens so both choices share identical prompt. Max512: PIQA retains prompt suffix; BoolQ retains passage prefix and entire question/answer cue. Reject candidate too long or question exceeding prompt budget. Labels mask prompt and actual padding positions only (PAD=EOS allowed). Train correct completion only. PIQA chooses highest mean completion token log-probability; BoolQ highest summed label log-probability; exact tie chooses candidate0. Calibration NLL: gold CE summed over tokens per task, then arithmetic mean of the two task means. Accuracy reported separately. No chat templates.

AdamW beta(.9,.999), eps1e-8, matrix decay.01, no dropout, FP32 adapters, BF16 frozen Base, eager attention. Clip global norm1, record preclip norm. Warmup ceil(5% actual steps), linear rise followed by cosine to zero on last step. Base Qwen/Qwen3-0.6B-Base and tokenizer revision da87bfb608c14b7cf20ba1ce41287e8de496c0cd. LoRA alpha=rank (scale1), A Kaiming uniform, Bzero. Side exits zero; reuse V0 BranchModel unchanged. Reset init seed via derive_seed(...,"init",0) after Base loading.

Final checkpoint only. Save adapters/config; reload onto fresh Base and repeat evaluation with exact per-example score/prediction equality. Base gets a repeat evaluation to establish same-device noise. Record complete step times; steady mean excludes first10, plus load/save/eval/reload time, peak allocation, input/loss/padded tokens, examples, last epoch and order hash.

Learning-rate selection fixed by E-0002: per family lower calibration NLL among1e-4/3e-4; relative difference below1% chooses1e-4. Nonfinite loss or calibration NLL above Base invalidates that LR; both failures go to coordinator, no added LR. Slice gate uses Base dev accuracy below90% each task and at least one LoRA calibration NLL below Base; these are diagnostic, not interface evidence.

Environment: uv.lock git blob3c916b51337de3cba594d645e606a31d4f39c443, cuda extra, environment /work/xg24i002/x10041/bypass-lora/envs/3c916b51337d-cuda; PyArrow23.0.1 added for train parquet. Torch2.10.0+cu128/transformers4.57.1/Python3.13.13 unchanged; module cuda/12.8. Hardware GH200 MIG2g.24gb. New environment requires V0 regressions before calibration.

Pending: smoke split mapping hash, measured costs, code review, calibration results, selected LRs and E-0003 costed release. No calibration has run yet.

## Pre-calibration correction from coordinator MSG-0025

This paragraph supersedes the training loss reduction stated above: each example's supervised-token CE is averaged first, then example means are averaged across the optimization step. With microbatch1, each example mean is divided by step sample count before backward. Calibration NLL remains the sealed per-task token-weighted NLL followed by two-task macro mean.

The first complete step reaching200000 tokens is included: the measured schedule is1103 steps,200280 input tokens,280 overshoot per calibration run (identical across all six; total1201680). Per coordinator, overshoot counts against the authorized V0 calibration2M cap, not reserve, and the threshold is unchanged. Apply the same rule to E-0003's2M threshold. Manifests record threshold, planned/actual tokens and overshoot separately.

Nonfinite loss/gradient is a completed failed-LR outcome (`train.diverged=true`) and writes measured consumed tokens. It returns normally so sequential launches can continue other candidates. Engineering exceptions still fail the process. Every microbatch persists usage before backward, including a failed nonfinite presentation.

Initial smoke ATT-0007-1, commit f7e6b4f:96/96 checks passed; all three20-step train/eval/reload paths completed,11586 training tokens,132s/.036667 MIG instance-hours. Those losses used the superseded token-weighted reduction; no calibration or scientific comparison is based on them. Each family's calibration plan is200280 tokens/1103steps. Measured steady steps .232s LoRA/.200s F_out/.210s C_dual,32rows per task/split evaluation about10–11s, repeat reload about10–11s. Full calibration evaluation plus repeat estimated about1000s, training about260s; request35min per calibration with1.5margin. Next smoke uses corrected loss and only5steps/family; initial timing supplies the cost baseline.

Locked split mapping SHA256:47cf5c777d07863b8def6903487697eafc970b0cd5b5d80babb7f20ded98571b (ATT-0007-1/lora/split-map.json). Subsequent jobs check all three raw checksums and mapping hash before model loading. The small bounds fix rejects empty completions or any candidate leaving no causal prompt token.

## Updated resource policy and corrected smoke

Coordinator MSG-0028, confirmed in main docs/constraints.md: token resource caps and the separate job duration/concurrency caps are removed. Tokens and overshoot remain recorded facts. E-0002's200000-token/arm experimental design is unchanged; E-0002 retains5node·h and V0+V1 retains200node·h ceilings, queue/maintenance limits and measured walltime×1.5. This supersedes earlier token-cap wording in this note.

Corrected smoke ATT-0008-1 / RES-0008 / EV-0007 (8d6a400):98/98 checks passed, including equal-example gradients and completed failed-LR continuation. Five steps in each family consumed981tokens,2943total. All exact reload checks passed.116s/.032222MIG instance-hours. Cumulative T-0002 smoke usage:14529tokens and.068889MIG instance-hours. Source author identifier is unknown (parent exact runtime model not exposed); helper used gpt-6-luna for data/LoRA modules and tests; parent inspected and integrated all changes.

## RV-0003 fixes before calibration

Review RV-0003 reported two failure-path findings; finite ATT-0008-1 evidence remains valid. Evaluation now checks every candidate (including nongold) before scoring/aggregation and checks sums for overflow. Adapted evaluation nonfinites produce a completed failed-LR result with stage/reason and existing training usage retained; Base nonfinites remain errors. Finite checkpoint mismatches remain errors, with no tolerance change. Smoke now requires all requested steps, nondivergence, finite per-task and macro evaluation metrics, and reload.identical=true.

Accounting correction from coordinator MSG-0032: MIG node·h equals ledger mig_instance_hours×0.25. Report both raw and converted values; apply5node·h E-0002 and200node·h V0+V1 ceilings to converted values. T-0002's first two smokes total.068889MIG instance-hours=.01722225node·h. This supersedes treating MIG instance-hours directly as node·h.

RV-0003 fix validation: ATT-0009-1 / RES-0009 / EV-0008, source cbc253a.104/104 checks passed, including injected nongold NaN/+Inf, adapted evaluation failed-LR continuation with usage retained, Base failure, finite reload mismatch, and strict smoke acceptance. All three families completed5steps and exact reload;2943training tokens.114s=.031667MIG instance-hours=.00791675node·h. T-0002 cumulative17472tokens,.100556MIG instance-hours=.025139node·h. No calibration run started. Requesting RV-0003 recheck; reviewed sources stay unchanged pending its answer.
