# T-0002 delivery: calibration complete

All seven full runs completed with finite likelihoods and exact fresh-Base checkpoint reload/repeated Base results. All six calibration runs used the same1103optimization steps,2206examples,200280nonpadding input tokens (280above the200000threshold),26124loss tokens and no padded training tokens (microbatch1/effective batch2). The calibration/dev samples and training-order hash match across all arms. Official validation remains unopened.

## Results and fixed LR selection

Accuracy columns are PIQA / BoolQ. Calibration NLL is the sealed arithmetic mean of per-task token-weighted gold NLL. Training uses equal-example means, as corrected before calibration.

| Arm | ATT | RES | EV | Calibration NLL | Calibration accuracy | Dev accuracy | MIG instance-h | node·h |
|---|---|---|---|---:|---|---|---:|---:|
| Base |0011-2|0011|0010|2.331250|72.803% /68.945%|72.3% /68.0%|.236944|.059236|
| LoRA1e-4 |0012-2|0012|0011|1.625332|72.559% /78.955%|72.3% /76.6%|.329167|.082292|
| LoRA3e-4 |0013-2|0013|0012|1.602104|72.852% /80.420%|73.3% /78.0%|.317500|.079375|
| F_out1e-4 |0014-2|0014|0013|1.660083|71.924% /79.590%|72.3% /77.7%|.333611|.083403|
| F_out3e-4 |0015-2|0015|0014|1.615797|72.119% /79.834%|72.5% /78.0%|.334444|.083611|
| C_dual1e-4 |0016-2|0016|0015|1.639295|72.607% /79.199%|72.2% /76.7%|.331389|.082847|
| C_dual3e-4 |0017-2|0017|0016|1.602429|72.705% /79.883%|72.8% /77.7%|.341944|.085486|

IDs in the table have their standard ATT-/RES-/EV- prefixes. Raw manifests and predictions are under `/work/xg24i002/x10041/bypass-lora/runs/ATT-<id>/`. Base calibration task NLLs: PIQA2.761511, BoolQ1.900988; dev task NLLs: PIQA2.750282, BoolQ1.924674. Base trained zero tokens; its `plan.*` fields describe the shared prospective schedule, not training performed.

**Lock3e-4 for all three families.** Relative NLL gaps are1.449875% LoRA,2.740840% additive,2.300626% concat (difference divided by lower NLL). Each exceeds the1% tie threshold; all candidates improve on Base and none failed. Apply additive LR to F_out/F_pre; concat LR to C_gate/C_up/C_dual/linear; LoRA LR to both layouts. No additional LR candidate is justified by this task.

The pre-fixed slice gate passes: Base dev accuracies are both below90%, and both LoRA calibration NLLs are below Base. This is a diagnostic learnability/non-ceiling check only. It neither ranks interfaces nor supports the H-0002 interface claim. Deterministic selection and prediction-identity audit: ATT-0019-1 / RES-0018 / EV-0017.

## Validation, provenance and costs

Training/evaluation source snapshot:28917cfb9a2dcdf27dd4900b80ead5ce7fd761d8. RV-0003/0004 issues were repaired; RV-0005 and full-scope RV-0006 pass with zero findings. Administrative RV-0004 approval closed its old round; no unresolved medium was carried into calibration. Final smoke ATT-0010-1 / RES-0010 / EV-0009 passed106checks and all three real train/eval/reload paths. Exact equality remains unchanged. Later `src/v1_report.py` is a deterministic audit only, not part of the calibrated trainer.

Environment: lock blob3c916b51337de3cba594d645e606a31d4f39c443, cuda extra; `/work/xg24i002/x10041/bypass-lora/envs/3c916b51337d-cuda`; Python3.13.13, torch2.10.0+cu128, transformers4.57.1, PyArrow23.0.1, module cuda/12.8; GH200 MIG2g.24gb; eager attention. Qwen/Qwen3-0.6B-Base and tokenizer revision da87bfb608c14b7cf20ba1ce41287e8de496c0cd. PIQA source revision21edab439af693b961be2f069e8690a88d3b4e37; BoolQ35b264d03638db9f4ce671b711558bf7ff0f80d5. Raw checksums, templates, initialization and seed definitions remain in T-0002-release.md. Split map SHA25647cf5c777d07863b8def6903487697eafc970b0cd5b5d80babb7f20ded98571b, saved identically in each run.

Seven full jobs:2.224999MIG instance-h=.55624975node·h. Four smokes:.133612MIG instance-h=.033403node·h. Three diagnostic jobs:17sfailed audit +13sboundary audit +14sschedule estimate=.012222MIG instance-h=.0030555node·h. **T-0002 total2.370833MIG instance-h=.59270825node·h**, below E-0002's5node·h. Full calibration training1201680tokens plus20415smoke tokens = **1222095training input tokens**. Analysis jobs and Base trained zero. Seven unused attempt1 registrations were cancelled before submission with zero compute; attempt2 carried refreshed RV-0006 coverage.

Measured steady step times: LoRA.198–.207s; F_out.188–.191s; C_dual.191–.195s. Throughput876–912,948–964,928–947input tokens/s respectively. Model load1.61–1.90s; save.0085–.0152s; full evaluation450–504s and reload+repeat455–499s. Base evaluation417s/repeat418s. Peak training allocation2.69–2.82GB. The five-step corrected smokes are too short for steady timing; the above figures come from1103-step calibration runs.

## Overlength diagnostic and handoff

ATT-0018-1 failed while estimating the larger E-0003 schedule: the locked complete-candidate encoder rejected a later training row. Its traceback is preserved in RES-0019 (partial; manually transcribed failure manifest). No calibration result is affected. ATT-0019-1 audited every train row: exactly **one PIQA row, official train line index/id5528**, candidate lengths **537/537tokens**; original first-epoch interleaved stream index5956. PIQA has13065train rows, so exclusion is1/13065=.007654%; BoolQ has0excluded of6379. This is below the coordinator's1%per-task stop threshold. Calibration had consumed only2206examples, before this boundary.

Coordinator MSG-0058 approves exclusion after per-task deterministic shuffle and before alternating the two tasks, preserving other rows' relative order and applying identically to all arms. No dev/calibration change. **Production filtering and its review belong to the next task and are not implemented here.** The E-0003 draft below uses this rule only for a diagnostic schedule estimate, not training.

## E-0003 release draft

Reuse all locked E-0002 definitions and3e-4LRs, with coordinator's one-row training exclusion. Effective batch2/microbatch1. Stop at the complete step reaching2M input tokens or3epochs, whichever first. Diagnostic estimate ATT-0020-1 / RES-0020 / EV-0018 gives **13077steps,2000040tokens,40overshoot,26154examples**, stopping in zero-based epoch1. Evaluate dev at steps**3270,6539,9808** (ceil25/50/75%); final checkpoint at13077is the main result. Interim checkpoints do not select the winner. Warmup ceil(.05×13077)=654steps, then cosine. Next task must verify its actual filtered implementation reproduces this schedule before launching.

| Arm | Parameters | Expected MIG-h | Expected node·h | Walltime with1.5margin | Requested MIG-h /node·h |
|---|---:|---:|---:|---:|---:|
| V1-01 F_out |262144|1.102|.2755|100min|1.667/.4167|
| V1-02 F_pre |262144|1.102|.2755|100min|1.667/.4167|
| V1-03 C_gate |524288|1.130|.2824|102min|1.700/.4250|
| V1-04 C_up |524288|1.130|.2824|102min|1.700/.4250|
| V1-05 C_dual |917504|1.130|.2824|102min|1.700/.4250|
| V1-06 linear C_dual |917504|1.130|.2824|102min|1.700/.4250|
| V1-07 four-layer LoRA rank10 |901120|1.144|.2860|103min|1.717/.4292|
| V1-08 28-layer LoRA rank2 |1261568|7.982*|1.9956*|719min*|11.983/2.9958*|

These are draft estimates, not E-0003 measurements. Use the slower measured step/eval cost within each calibrated family; count3interim dev evaluations (2000/6096of measured full evaluation), final full evaluation and repeat reload, plus15sload/save overhead. F_pre/C_gate/C_up/linear use their family's proxy, not measurements of those exact arms. The final full-evaluation term conservatively includes calibration rows, although E-0003 needs only dev. **The starred28-layer LoRA estimate is an intentionally loose7×four-layer LoRA planning bound, not a measured cost or a recommended job submission.** Its actual throughput must be measured with the next task's smoke before sizing; use an eligible queue if walltime exceeds short-mig limits. All eight require the next task's filter/evaluation-point implementation and review before execution. No E-0003 training was run in T-0002.
