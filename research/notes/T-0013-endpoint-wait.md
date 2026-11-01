# T-0013: required upper endpoint calibration in progress

Current continuation;task not delivered. Official validation untouched. Current
task owner researcher-2;coordinator owns scientific conclusions and waiting.

## Verified initial results and source integration

- E-0013@3. Initial9 all complete/finite,matched **1101 steps,2202 examples,
  200103 input tokens**,order SHA256
  `e149a10e8f02f8aa10e134dcfac428b8159969a86f59c228a06dfe88c0a94df4`.
- Attempts ATT0101..0109 imported as RES0107..0115,exact usage5996MIGseconds
  =0.41638889 node·h. All end observations and usage already imported;do not
  import their usage again. Per-job map in T-0013-calibration-jobs.json.
- Reviewed report ATT0117 / PBS3764705 mg1098,Exit0,10s;RES0117,EV0077.
  It verified all9 and requires **3e-3 for every arm**. Committed copy:
  research/artifacts/T-0013/initial-grid.json. Initial1e-3 choices are provisional.
  NLL table (LR columns1e-4,3e-4,1e-3):

| Arm | 1e-4 | 3e-4 | 1e-3 |
| --- | ---: | ---: | ---: |
| V5-C | 1.7013356424961341 | 1.6323481719859676 | 1.5977210394597021 |
| V5-F | 1.65651495608446 | 1.6021932411588116 | 1.5675299750501892 |
| V5-L | 1.664488090351574 | 1.5867400373288816 | 1.553388156312454 |

- ATT0117 retains its original stale review flag. EV0077 records coordinator
  MSG0363 rationale:actual report/trainer exactly match RV0038/RV0033;stale
  paths were unrelated T0014 files absent from old87bbb59 branch,never imported.
- Coordinator explicitly authorized merging master before endpoint/formal
  registrations,overriding the original no-merge work-pack rule (MSG0363).
  Done with merge commit **5cad3ebb5ec40b1a5990c1d3f92a2cea940b24bc**,no rebase.
  Three git diff --exit-code checks proved byte identity:training/shared files
  against f4b2c19(RV0033),latency against2250925(RV0035),report against87bbb59
  (RV0038). All3 endpoint registrations report **covered**,including T0014
  RV0039 files now present. Added T0014 files are not executed in this task.
- Base ATT0113 / PBS3764504 mg1100,Exit0,344s;RES0116,EV0078. Dev PIQA1000
  acc.723 NLL2.750281812879043;BoolQ1000 acc.680 NLL1.9246741668730973;
  task-mean NLL2.3374779898760703. **reload.identical true**,all per-example
  scores/predictions and metrics exactly equal on fresh Base. Output predictions:
  /work/xg24i002/x10041/bypass-lora/runs/ATT-0113-1/predictions.jsonl.
  Usage344/14400=0.02388889 node·h already imported. Base is finished.
- Previous diagnostics:RV0033 training pass,36tests+3arm smoke;RV0035 latency
  pass,8tests;RV0038 report pass,4tests. No code changes since these reviews.
  Before endpoints,all T0013 measured compute6553MIGseconds=0.45506944 node·h
  (includes initial5996,Base344,launcher2,smoke181,latency10,gridtests10,report10).

## Running endpoint jobs and notification

All LR3e-3,seed0,identical200k token-plan,reviewed trainer on immutable5cad3e:

| Arm | Attempt | PBS | Expected s | Walltime |
| --- | --- | --- | ---: | --- |
| V5-C | ATT-0118-1 | 3764753.opbs | 555 | 00:14:00 |
| V5-F | ATT-0119-1 | 3764755.opbs | 548 | 00:14:00 |
| V5-L | ATT-0120-1 | 3764756.opbs | 981 | 00:25:00 |

Expected costs use slowest measured same-arm initial calibration;walltime~1.5x.
Allocation check precedes each workload. All observed R at submission;start
facts imported. Full specs and result mappings in T-0013-endpoint-jobs.json.

Notification **N-20261101-204459-d502**,sealed expected3,early failure;
coordinator has listener:
`/work/xg24i002/x10041/miyabi-toolbox/miyabi-mcp/bin/miyabi-mcp notify listen --id N-20261101-204459-d502 --notifier miyabi-g2`.
Daemon running on miyabi-g2. Do not poll;resume when this group reports.

## Resume and complete

1. Verify endpoint job histories/allocation and manifests;finite runs must have
   exact1101steps/2202examples/200103tokens/order above. Handle nonfinite LR as
   eliminated per contract;any engineering failure stops for reporting. Import
   end observations/exact usage once and manifests;capture actual RES IDs.
2. Register a small PBS lock job on the integrated source. Execute existing
   **src/v5_train.py lock --manifests** on all12 initial+endpoint manifests and
   --output /work/.../runs/<lock-attempt>/lr-lock.json. It verifies required
   extension set and chooses smallest finite LR within1% of final raw minimum.
   No second upper extension allowed. No selection on dev.
3. Commit lock to **research/artifacts/T-0013/lr-lock.json** in this worktree,
   plus a selection table with all12 NLLs and chosen LRs. Do not start any seed1–3
   until committed. Stage that commit read-only and pass **its snapshot path**
   to --lr-lock in every formal run;never a mutable worktree lock path.
4. Size9 formal jobs from measured steps and dev eval costs,about1.5x expected.
   Exact3epochs=29165steps/58329examples/4173300tokens per run. Use slowest
   measured same-arm calibration/smoke stable step cost and4full-dev evaluations
   (3epoch monitors+fresh reload),load/save/prep overhead. Per-arm initial maxima:
   stable C.18299448/F.18680969/L.39720290s;smoke C.19903338/F.19499615/
   L.40831688s. Full calibration4096 eval maxima330.14/328.32/522.12s;dev2000
   roughly scales by2000/4096. Endpoints supply another measured check.
   E0013≤12node·h,T0013 planning~7;report before any budget-driven schedule change.
5. Register9 training runs,three arms×seeds1,2,3,from the committed-lock snapshot;
   coverage must report covered (if stale,send exact paths to coordinator).
   `v5_train.py run --phase train --arm ... --seed ... --lr ... --lr-lock <snapshot>/research/artifacts/T-0013/lr-lock.json --data-dir /work/.../data/v1 --output /work/.../runs/<ATT>`.
   Submit with notification group expected9;record facts and `rl task wait`.
   Long-job wakeup skill was read:the site MCP/coordinator notifications own
   continuation,no unsolicited Goal or duplicate watcher is needed.
6. After completion require full counts,3epoch dev records,standard adapter
   config.json+adapter.pt,final JSONL dev predictions and exact fresh reload.
   Use only final checkpoint. No official validation access by this task.
7. Run reviewed **v5_latency.py** on **final seed1** checkpoint map:fixed256dev
   candidates,batch1,16warmups,3passes,sameGPU type;Base,C,F,LoRA before/after
   merge. Corrected checkpoint validator must be present (RV0035). Record map
   provenance/hashes;merge is timing only,unmerged scoring remains T0014/T0015.
8. Final deliver LRtable/lock,nine checkpoint inventories+SHA256,manifest refs,
   per-epoch dev/reload/counts,latency/cost tables,actual node·h and review IDs.
   T0014 MSG0344 requires top-level learning_rate_lock FileRef,and each entry's
   training_manifest + checkpoint_inventory FileRefs. It checks phase=train,
   contractE0013@3,validator passed,actual58329examples/29165steps,lr_lock.sha256,
   adapter dir=manifest parent/adapter,inventory listed in artifacts and exact
   paths/bytes/hashes. Current producer preserves all fields;confirmed MSG0347.
   All calibration/dev/cost evidence is diagnostic only,no supports/challenges.

Login/control only for records,edits,bounded reads/staging;all runtime/analysis
through miyabi-mcp PBS. Environment unchanged3c916b51337d-cuda,cuda12.8,offline.
