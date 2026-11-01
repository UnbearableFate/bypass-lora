# T-0013: nine formal training runs in progress

Historical full-nine handoff. **Current continuation is T-0013-main-wait.md**.
Six C/F runs are imported;only three LoRA runs remain. D0018/D0019 supersede
all slice-sensitivity work;do not execute the unused sensitivity helper.

Current continuation. Task owned by researcher-2,not delivered. Coordinator owns
waiting and scientific conclusions. Official validation remains sealed/untouched.

## Locked before any seed1–3 training

- Contract **E-0013@3**. All12 seed0 calibrations finite,paired1101steps/
  2202examples/200103input tokens. Required3e-3 endpoints completed once.
- Reviewed **v5_train.py lock** ran in ATT0128/PBS3764966,Exit0,14s;RES0122,
  EV0079. All3 locked LRs are **0.001**. Full12-point table in
  research/notes/T-0013-lr-selection.md;initial JSON also committed.
- Immutable formal source **2b5c809f6101b5ba132d5565ebbf73ef91e105da** contains
  the lock before training. Every formal registration reports covered.
  Exact LR FileRef:
  `/work/xg24i002/x10041/bypass-lora/snapshots/2b5c809f6101b5ba132d5565ebbf73ef91e105da/research/artifacts/T-0013/lr-lock.json`,
  SHA256 `70ba5356203afc128cbf3e4c14a7b4474ece54a55d28b8401b852d957069e01a`,
  **2859 bytes**. Staged snapshot hash independently matched the lock output.
- Trainer/dependencies byte-identical RV0033;latency RV0035;grid report RV0038.
  Coordinator MSG0363 explicitly authorized master merge5cad3e,which added
  accepted T0014 files and restored clean coverage;no rebase/history rewrite.

## Running jobs and wake-up

All use LR.001,three exact epochs and only the final checkpoint;shared read-only
source above,output under /work/xg24i002/x10041/bypass-lora/runs/<attempt>/.
Full specs,FileRef,allocation evidence in **T-0013-formal-jobs.json**.

| Arm | Seed | Attempt | PBS job (.opbs) | Walltime |
| --- | ---: | --- | --- | --- |
| V5-C | 1 | ATT-0129-1 | 3765067 | 02:44:00 |
| V5-C | 2 | ATT-0130-1 | 3765068 | 02:44:00 |
| V5-C | 3 | ATT-0131-1 | 3765069 | 02:44:00 |
| V5-F | 1 | ATT-0132-1 | 3765071 | 02:40:00 |
| V5-F | 2 | ATT-0133-1 | 3765072 | 02:40:00 |
| V5-F | 3 | ATT-0134-1 | 3765073 | 02:40:00 |
| V5-L | 1 | ATT-0135-1 | 3765074 | 05:26:00 |
| V5-L | 2 | ATT-0136-1 | 3765075 | 05:26:00 |
| V5-L | 3 | ATT-0137-1 | 3765077 | 05:26:00 |

All9 transitioned queued→**running**,observations already imported. All9
allocation.json files independently verified ok,allocation_confirmed and
runtime_execution_allowed true,matching exact PBS_JOBID/nodefile and liveR
before workloads. Hosts mg1109/mg1110/mg1111/mg1112,Miyabi-G aarch64.

Notification **N-20261101-211800-4e24**,sealed expected9,early failure;daemon
running on miyabi-g2. Coordinator MSG0377/0380 has listener:
`/work/xg24i002/x10041/miyabi-toolbox/miyabi-mcp/bin/miyabi-mcp notify listen --id N-20261101-211800-4e24 --notifier miyabi-g2`.
Coordinator owns wake-up. The long-job wakeup skill was read;use this existing
site notification,not an unsolicited Codex Goal or duplicate polling watcher.
End turn on WAIT;do not poll jobs/inbox or sleep in the researcher turn.

## What is complete and what remains

- Completed:36 V1/V5 regression tests+3-arm smoke(RV0033 pass),8 latency tests
  (RV0035 pass),4 report tests(RV0038 pass);all calibrations and LR selection.
- Completed Base same-device repeat:ATT0113/RES0116/EV0078,full dev1000/task,
  scores/predictions and metrics exactly identical on fresh Base. PIQA acc.723,
  BoolQ.680;NLL2.750281812879043/1.9246741668730973.344MIGs already imported.
- Before formal training:actual **8614MIGseconds=0.59819444node·h**,all imported
  once. Includes initial5996,endpoint2047,Base344,launcher2,smoke181,latency
  tests10,grid tests10,report10,lock14. Formal and future latency usage unknown.
- Cost plan research/artifacts/T-0013/formal-cost-plan.json:9 expected
  **5.404792node·h**,requested reservation8.125node·h;prior+expected formal
  **6.002986node·h** before latency. Estimates are not measured usage.
  E0013 ceiling12node·h. Same frozen environment3c916b51337d-cuda/cuda12.8.
- Formal checkpoints,full dev/reload outputs and final seed1 latency remain
  unverified/incomplete. No final delivery or confirmation opening yet.

## Resume after notification

1. Inspect exact job histories and application manifests,not just PBS exit0.
   On an early engineering failure,stop/report before proceeding;cancel only
   owned remaining jobs through miyabi-mcp dry-run/commit if required by the
   contract. Do not weaken reload equality or silently retry with changed rules.
2. For each complete run require phase=train,contractE0013@3,validator passed,
   **train.steps29165,train.examples58329,train.input_tokens4173300**,
   params trainable524288(C/F) or630784(L),allFP32,lr_lock.sha256 above,
   all3dev epoch records,final predictions.jsonl and **reload.identical true**.
   Compare plan.order_sha256 across arms within each seed;they must match.
3. Import verified final executor observations and **exact usedMIGseconds/14400
   node·h** once,then manifests. Native export quantity is rounded MIG-h;use
   job_get resources_used.walltime for exact conversion. Ledger end export can
   lag job_get:record verified history explicitly if needed. Capture returned
   RES IDs instead of guessing across concurrent sessions. Preserve partial
   usage/results on failures. Add diagnostic evidence only,no supports/challenges.
4. Run **reviewed src/v5_latency.py** in a new PBS job from reviewed source
   (2b5c809 contains corrected validator). Map final **seed1** adapters:
   C→runs/ATT0129-1/adapter,F→ATT0132-1/adapter,L→ATT0135-1/adapter. Verify
   complete/reload provenance before constructing the map;record its hashes.
   CLI requires --checkpoints <JSON arm->directory>,--data-dir /work/.../data/v1,
   --output <latency-attempt>. It uses fixed256dev candidates,batch1,16warmups,
   3timing passes on one GPU type;Base,C,F,LoRA before/after static merge.
   BF16 merge rounding is recorded;merged weights are timing only,never scoring.
   This full256/BF16 job has not run yet. Size from measured dev forward timing,
   about a few minutes,then inspect count768 permodel and finite timings.
5. Produce final9-run table:LR,steps/tokens/order,each epoch dev metrics,exact
   reload,checkpoint/inventory/manifest paths+SHA256,training/load/save/eval/
   steady-step/peak-memory costs,latency and total measured node·h. Do not
   interpret dev results as confirmation. Commit task report before delivery.
6. T0014/T0015 lock needs top-level **learning_rate_lock FileRef** above,each
   **training_manifest FileRef** and **checkpoint_inventory FileRef** for
   final_run/checkpoint.json,plus config.json+adapter.pt FileRefs. FileRef is
   {path,sha256,bytes}. It verifies actual full counts,phase/contract/validator,
   LRlock hash,adapter dir=manifest parent/adapter,inventory referenced in
   manifest artifacts,and exact inventory paths/bytes/hashes. Producer already
   preserves this schema. Use durable runs/snapshots paths,not worktree paths.
7. Send finalrefs/report to coordinator/researcher1 and `rl task deliver T0013`.
   Coordinator owns acceptance/merge and T0015 confirmation release. This task
   never reads official validation. No external publishing/push/force history.

All runtime/tests/analysis remain in PBS. Login only for control records,edits,
bounded source/metadata reads and staging. Ignored research/artifacts requires
git add -f for the specific small lock/selection/cost files;never checkpoints or
the whole directory. Do not change dependencies or source without a justified
engineering fix/review;do not change frozen comparison or sample budget.
