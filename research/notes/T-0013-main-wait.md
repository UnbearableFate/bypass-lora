# T-0013: main-path continuation, three LoRA runs pending

Current instructions. Task remains owned by researcher-2,not delivered. The
coordinator owns wake-up,acceptance and official-validation opening. This session
has not read official validation. Do not reopen any slice-sensitivity work.

## Latest user/coordinator decisions override earlier notes

- **D-0018 / MSG0472**:ignore BF16 rounding differences between MIG profiles.
  The slice-sensitivity comparison and all-2g lock recomputation are **canceled**.
  Actual-data comparison/helper execution was never registered or started.
  Do not run src/v5_slice_check.py or produce lr-lock-2g-check.json,despite its
  historical source/tests/review. The two finished repeats are plain diagnostics
  only. Original lr-lock.json is final,with no sensitivity caveat. No retraining,
  extra analysis or flags because of a slice. Profiles are metadata only.
- **D-0019 / MSG0481**:guards on ordinary jobs are record-only and cannot fail
  the job. Do not cancel queued jobs. If a job failed only on a guard,re-submit
  as a new attempt at the same reviewed source with the launcher guard made
  record-only;no src changes. The explicit exception is **latency/throughput
  comparison**,which retains fail-fast on2g. Existing guard source is
  src/mig_profile_guard.py at fbf97b4;for ordinary jobs use it nonfatally in the
  shell,for the future latency benchmark keep its original fail-fast behavior.
- A cross-job bitwise mismatch explained by different slices/rounding is not
  a failure:note it and continue. Existing within-job saved-checkpoint checks
  have passed in all six completed runs;do not invent a new tolerance or check.
- No actual comparison/all-2g lock was run after these instructions. Unit tests
  for the abandoned helper finished earlier and were recorded as-is,not used.

## Fixed formal input and source

E-0013@3. All3 formal LRs **0.001**,fixed before seeds1–3. Formal source:
**2b5c809f6101b5ba132d5565ebbf73ef91e105da**,covered reviews. Exact LR FileRef:
`/work/xg24i002/x10041/bypass-lora/snapshots/2b5c809f6101b5ba132d5565ebbf73ef91e105da/research/artifacts/T-0013/lr-lock.json`,
SHA256`70ba5356203afc128cbf3e4c14a7b4474ece54a55d28b8401b852d957069e01a`,
2859bytes. Never overwrite or replace this original artifact.

Code reviews:RV0033 trainer pass;RV0035 latency pass;RV0038 initial report
pass. Trainer/latency/dependencies remain unchanged. fbf97b4 adds only the
standalone guard,validated in ATT0149/RES0139/EV0092:7tests+liveinventory,2s.
The abandoned helper source db0fe06 was reviewed RV0047 and15tests passed
ATT0157/RES0146,but it must not process actual results.

## Completed C/F training,already imported

| Arm | Seed | Attempt | PBS | Result | Used MIG s |
| --- | ---: | --- | --- | --- | ---: |
| V5-C | 1 | ATT-0129-1 | 3765067.opbs | RES-0147 | 5961 |
| V5-C | 2 | ATT-0130-1 | 3765068.opbs | RES-0148 | 6177 |
| V5-C | 3 | ATT-0131-1 | 3765069.opbs | RES-0149 | 6023 |
| V5-F | 1 | ATT-0132-1 | 3765071.opbs | RES-0150 | 6287 |
| V5-F | 2 | ATT-0133-1 | 3765072.opbs | RES-0151 | 6002 |
| V5-F | 3 | ATT-0134-1 | 3765073.opbs | RES-0152 | 5680 |

All6 Exit0/application complete+validator passed,phase=train,contractE0013@3,
**29165steps/58329examples/4173300tokens each**,all3epoch dev records,final
checkpoint inventory/predictions,and **reload.identical true**. LRlock hash
matches. C/F plan.order_sha256 matches within each seed. EV0094 diagnostic only.
Profiles retained as metadata,not flagged. All6 ended observations and exact
usage **36130MIGseconds=2.50902778node·h** already imported once;do not import
their usage again when processing the eventual full-nine ledger export.

Outputs:/work/xg24i002/x10041/bypass-lora/runs/<attempt>/,standard adapter/
config.json+adapter.pt,checkpoint.json,manifest.json,predictions.jsonl and
dev-epoch-{1,2,3}.json. Durable current result/status map:T-0013-main-state.json.

## Still pending

| Arm | Seed | Attempt | PBS | Requested walltime |
| --- | ---: | --- | --- | --- |
| V5-L | 1 | ATT-0135-1 | 3765074.opbs | 05:26:00 |
| V5-L | 2 | ATT-0136-1 | 3765075.opbs | 05:26:00 |
| V5-L | 3 | ATT-0137-1 | 3765077.opbs | 05:26:00 |

Last observed running. Original group **N-20261101-211800-4e24** still covers
all9;coordinator has daemon/listener:
`/work/xg24i002/x10041/miyabi-toolbox/miyabi-mcp/bin/miyabi-mcp notify listen --id N-20261101-211800-4e24 --notifier miyabi-g2`.
Do not poll or sleep in the researcher turn. Coordinator resumes on completion.

## Ancillary records are closed;no further work on them

- Original calibrations/lock and Base already imported;see prior notes. Base
  dev repeat exact,ATT0113/RES0116/EV0078. Original LR lock ATT0128/RES0122,
  EV0079;validity EV0077/0079 restored per D0015 and remains valid under D0018.
- Finished duplicates ATT0103-2/RES0144 and ATT0107-2/RES0145:plain diagnostic
  imports only,EV0093,no comparison. They preserve original RunSpec digests,
  matched budget/order,guard artifacts as metadata. Used584+934=1518MIGs.
- Unused-helper regression ATT0157/RES0146:15tests,13MIGs,recorded as-is.
- All non-formal completed T0013 use **10147MIGseconds=0.70465278node·h**
  (original pre-formal8614+guard2+duplicates1518+unusedtests13).
  Known completed total with C/F:46277MIGseconds=3.21368056node·h.
  Three running LoRA jobs have unreported usage,**not zero**. Ceiling E0013≤12.

## Resume the main path

1. Verify each LoRA application's full counts,LR hash,3epoch dev artifacts,
   standard checkpoint/inventory and saved-checkpoint reload. Record profiles
   only. Do not rerun/flag because of profile or cross-job rounding. Report a
   genuine engineering issue independently;do not silently weaken assertions.
2. Import only newly completed LoRA ends/exactMIGseconds/14400 usage and results.
   Existing C/F/ancillary usage is already recorded. Capture actual returned RES
   IDs. Compare full9 order hashes within each seed;this checks sample pairing,
   not numeric slice effects. All training/dev evidence stays diagnostic.
3. Once all9 checkpoints are verified,run the already-reviewed **v5_latency.py**
   on final **seed1** adapter map C=ATT0129-1,F=ATT0132-1,L=ATT0135-1. Verify
   parent complete/reload provenance before constructing map. No checkpoint
   substitution or LR change. Latency is the D0019 timing exception:call the
   existing guard fail-fast before model/timing work,save profile/raw inventory.
   No source changes are needed. Do not cancel queued jobs.
4. Benchmark uses fixed256dev candidate inputs,batch1,16warmups,3passes;
   Base,C,F,LoRA before/after merge. Same256inputs/GPU in one job. Check768
   timings/model,finite results. BF16 merge logit difference is metadata,merged
   LoRA is timing only. Time request from measured eval-forward costs,roughly
   expected240s/request360s (about6min);revise only if measurements require.
5. Final delivery report:LRtable/original lock,nine manifest+inventory+adapter
   file refs/SHA256,steps/tokens/order,each epoch dev/reload,training/load/save/
   evaluation/steady-step/peak-memory costs,latency and measured node·h. Do not
   include a slice-sensitivity caveat or compare the two diagnostic repeats.
6. T0014/T0015 expect top-level learning_rate_lock FileRef above,per-entry
   training_manifest and checkpoint_inventory FileRefs plus adapter config/
   weight refs. FileRef={path,sha256,bytes}. Requirements:phase=train,
   contractE0013@3,validator passed,full actualcounts,LRhash,adapter dir=manifest
   parent/adapter,inventory listed in artifacts,and exact paths/bytes/hashes.
   Use durable runs/snapshots paths,not removable worktree paths.
7. Commit final task report,send refs/report to coordinator/researcher1 and
   `rl task deliver T-0013`. Coordinator owns acceptance/merge and T0015 opening.
   This worker continues to avoid official validation;all main-path runtime,
   tests and data analysis stay in Miyabi PBS through miyabi-mcp.

Local shell is login/control only. Environment unchanged3c916b51337d-cuda/
cuda12.8,HFoffline. Never force-push/rebase or alter the sealed comparison.
