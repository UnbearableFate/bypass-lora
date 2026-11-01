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
