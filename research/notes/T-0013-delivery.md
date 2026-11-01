# T-0013 delivery: V5 training, checkpoints and measured costs

All nine E-0013@3 training runs and the final seed-1 latency benchmark completed
successfully. All three arm LRs are **0.001**, locked before seeds1–3. Each run
completed exactly three epochs: **29165 optimizer steps,58329 presentations,
4173300 nonpadding input tokens**. Every final saved checkpoint reloaded with
identical dev scores and predictions within its training job. Per-seed sample
orders match across C,F and LoRA. Official validation has not been accessed by
this worker and remains sealed pending the coordinator's release.

This is a training/dev/engineering delivery. It makes no H-0008 outcome claim.
Confirmation scoring,WikiText and scientific analysis belong to the coordinator's
later task. D0018/D0019 are applied:the original LR lock is final;no slice
comparison,alternate lock or retraining was performed. Hardware profiles are
metadata. The latency comparison used the existing fail-fast2g guard.

## Locked inputs and checkpoint handoff

Formal training source:2b5c809f6101b5ba132d5565ebbf73ef91e105da.
The V1 example-mean objective,batch2/micro1,AdamW,warmup/cosine schedule and
epoch-boundary monitoring follow E0013@3. Monitor steps are9722/19443/29165;
only the last checkpoint is exported. Original LR grid and all12 manifest pins
are recorded in [T-0013-lr-selection.md](T-0013-lr-selection.md) and the LR lock.

The confirmation-lock owner can copy `learning_rate_lock` and `checkpoints`
from the nine-checkpoint inventory. It contains all9 arm/seed entries,standard
adapter directories,expected parameter counts,learning rates,config/weight
FileRefs,training_manifest FileRefs and checkpoint_inventory FileRefs.
Each FileRef has an absolute durable path,positive bytes and SHA256. This
inventory is a training handoff,not an authorization to open validation.

| Artifact | Durable path | SHA256 | Bytes |
| --- | --- | --- | ---: |
| Final LR lock | snapshots/2b5c809f6101b5ba132d5565ebbf73ef91e105da/research/artifacts/T-0013/lr-lock.json | 70ba5356203afc128cbf3e4c14a7b4474ece54a55d28b8401b852d957069e01a | 2859 |
| Nine-checkpoint inventory | runs/ATT-0174-1/checkpoint-inventory.json | 2886d4981d2fe954bfc022b2d5b6641af638d1d0529d0012755538bd60619619 | 10830 |
| Full training/dev summary | runs/ATT-0174-1/training-summary.json | 7d2c35cef7a5c8816d653b291e73ccfa3397989cf22f39ad72611aedf694f377 | 41028 |
| Cost summary through latency | runs/ATT-0175-1/cost-summary.json | cc36b0911cb2d1ee274534b95acbeee5be6c577ec13d0431f1ab0f8c7683d695 | 9011 |
| Raw latency | runs/ATT-0174-1/latency.json | 00b0a4a2aa255ae074c3b12b5059920c1b9015e60b645bbec6eddb651eac2d48 | 105942 |

Paths in that table are relative to /work/xg24i002/x10041/bypass-lora/.
The small inventory and training summary also exist byte-identically in the
read-only snapshot3e7c90c8f7f696561b04a71f96a054a5da65e7b6 under
research/artifacts/T-0013/. Final billing is in
[usage-summary.json](../artifacts/T-0013/usage-summary.json),which adds the
one-second tabulation job to the cost summary's through-latency total.

## Final runs and dev monitoring

C/F each have524288 trainable FP32 parameters. LoRA has630784,all28layers/all7
projections,rank1/alpha1. Every entry below is complete/passed with
reload.identical=true. Adapter directory is always
/work/xg24i002/x10041/bypass-lora/runs/<attempt>/adapter/.
The full summary preserves all27 epoch records,including per-task NLL/accuracy,
steps,monitor time and hashes of the source dev-epoch files.

| Arm | Seed | Attempt | PBS | Result | Final dev PIQA acc | BoolQ acc | Macro NLL |
| --- | ---: | --- | --- | --- | ---: | ---: | ---: |
| V5-C | 1 | ATT-0129-1 | 3765067 | RES-0147 | .733 | .830 | 1.617507 |
| V5-C | 2 | ATT-0130-1 | 3765068 | RES-0148 | .731 | .826 | 1.633767 |
| V5-C | 3 | ATT-0131-1 | 3765069 | RES-0149 | .722 | .817 | 1.626274 |
| V5-F | 1 | ATT-0132-1 | 3765071 | RES-0150 | .731 | .820 | 1.613381 |
| V5-F | 2 | ATT-0133-1 | 3765072 | RES-0151 | .736 | .828 | 1.605135 |
| V5-F | 3 | ATT-0134-1 | 3765073 | RES-0152 | .733 | .819 | 1.624803 |
| V5-L | 1 | ATT-0135-1 | 3765074 | RES-0165 | .746 | .836 | 1.616375 |
| V5-L | 2 | ATT-0136-1 | 3765075 | RES-0166 | .741 | .827 | 1.643579 |
| V5-L | 3 | ATT-0137-1 | 3765077 | RES-0167 | .740 | .827 | 1.642620 |

Dev is monitoring only:these values did not select a checkpoint or change LR,
steps,method or confirmation rules. The Base dev repeat was previously imported
as ATT0113/RES0116/EV0078. Calibration and training readiness evidence remains
diagnostic (EV0077/EV0079,EV0101).

## Latency and cost validation

ATT-0174-1/PBS3767640 ran reviewed v5_latency.py from read-only snapshot
388f2f82cfa91253830343c299db52537bbfffd5 on mg1098. Allocation and the exact
allocated MIG2g.24gb UUID were verified before models loaded. The parent
checkpoint-map-provenance.json binds C/F/L seed1 to ATT0129/0132/0135 after
authenticating all9 manifests/specs/budgets/LR/configs/inventories/bytes/hashes
and all three paired sample orders. The manifest is complete/passed,
RES0172;PBS Exit0,179 seconds.

| Model | Count | Mean ms | Median ms | p95 ms |
| --- | ---: | ---: | ---: | ---: |
| Base | 768 | 36.651 | 35.837 | 42.621 |
| V5-C | 768 | 37.803 | 37.782 | 38.541 |
| V5-F | 768 | 38.090 | 38.070 | 39.005 |
| V5-L | 768 | 55.761 | 55.736 | 56.867 |
| V5-L merged | 768 | 35.930 | 35.916 | 36.759 |

All models used the same256 dev candidate inputs,batch1,16 warmups and3timed
passes. Input SHA256:53ebd2072ca5140cb533feb25d9d4f3aebd1d4376097b6eff39f92f83c270faf.
Timing is synchronized host forward walltime,including dispatch/hooks and
excluding encoding. These are descriptive measurements from one GPU and fixed
input order. Merged LoRA is timing only;unmerged checkpoints remain the task
scoring inputs. The merge's first-input logit maximum difference0.875 is
preserved in the latency metadata.

ATT-0175-1/PBS3767690 independently checked the hash-pinned output files,
all3840 positive finite raw samples and agreement of mean/median/p95 with the
reported values,then tabulated costs. Complete/passed,RES0173,Exit0,1second,
verified mg1098 compute allocation. This job accessed no datasets or models.
EV0106 is diagnostic/valid.

[Cost tables](T-0013-cost-tables.md) report every run's load,train,steady-step,
monitor,save,reload-eval and peak CUDA allocation values;the cost JSON retains
full precision and hardware metadata. Nine formal training jobs used
**73357 MIG seconds =5.09423611 node_h**. All T0013 work,including calibration,
smokes,reviews' runtime checks,Base,plain diagnostics,latency and final
tabulation,used **83684 MIG seconds =5.81138889 node_h**. This is the T0013
subtotal;other E0013 workers' confirmation/WikiText/analysis costs are separate.
Every completed job's measured scheduler usage has been recorded once and no
T0013 jobs remain pending. The shared E0013 ceiling is12 node_h.

## Reviewed source and completion boundary

RV0033 passed the training path with36 tests and all3 real smoke arms;
RV0035 passed the latency fix with8 tests;RV0038 passed the calibration report
with4 tests. The existing guard had7 tests and a live allocation check. Trainer,
latency and their numerical dependencies are byte-identical to those reviewed
versions. The registered latency snapshot reports covered review status. No
new model or scoring path was introduced for this delivery.

T0013 is ready for coordinator acceptance and merge. The checkpoint inventory
and costs can feed the later locked confirmation task. No official-validation
opening,WikiText result or H0008 scientific conclusion is included here.
