# T-0013: all nine training runs verified; final latency running

All nine formal runs completed with scheduler Exit0 and complete/passed application
manifests, exact within-job reload, 29165 steps / 58329 examples / 4173300 input
tokens each. LoRA imports: ATT0135/0136/0137 -> RES0165/0166/0167. Existing C/F
imports remain RES0147..0152. All ended observations and exact usage are imported
once; do not duplicate them. EV0101 is diagnostic/valid, not a scientific claim.

The immutable formal source remains 2b5c809f6101b5ba132d5565ebbf73ef91e105da.
The original lr-lock.json is final, with all three LRs 0.001 and SHA256
70ba5356203afc128cbf3e4c14a7b4474ece54a55d28b8401b852d957069e01a.
D0018/D0019 apply: no slice comparisons, no alternate lock or retraining, profiles
are metadata only. Timing comparisons retain the existing fail-fast 2g guard.
This worker has not read official validation; the coordinator owns its release.

ATT-0174-1 / PBS 3767640.opbs runs reviewed src/v5_latency.py from immutable
snapshot 388f2f82cfa91253830343c299db52537bbfffd5. Reviewed trainer/latency/
dependency files are unchanged. The CLI reports covered reviews RV0033/RV0035/
RV0038/RV0039/RV0047. Only a control launcher and measured accounting input were
added. PBS allocation verified on mg1098; the allocated MIG UUID resolved to
2g.24gb and passed the existing timing guard.

Before loading models, the job authenticated all nine training specs, arm/seed,
actual budgets, LR hash, parameter counts, epoch records, adapter configs and
weight/config paths/bytes/SHA256 against each final checkpoint.json. It also
verified matching sample order across arms within all three seeds. Preflight
passed, then the fixed256-dev seed1 benchmark began (Base,C,F,LoRA unmerged and
merged;16 warmups,3 passes,768 timings per model).

Output root: /work/xg24i002/x10041/bypass-lora/runs/ATT-0174-1/.
checkpoint-inventory.json (10830 bytes) SHA256
2886d4981d2fe954bfc022b2d5b6641af638d1d0529d0012755538bd60619619.
training-summary.json (41028 bytes) SHA256
7d2c35cef7a5c8816d653b291e73ccfa3397989cf22f39ad72611aedf694f377.
Small identical copies are under research/artifacts/T-0013/ for delivery.
checkpoint-map-provenance.json binds the exact final seed1 C/F/L adapters.

Notification N-20261102-011115-6426 is sealed for this one job;daemon running.
Coordinator listener:
/work/xg24i002/x10041/miyabi-toolbox/miyabi-mcp/bin/miyabi-mcp notify listen --id N-20261102-011115-6426 --notifier miyabi-g2

On completion:verify manifest/768 finite samples per model and timing guard,
import ended observation plus exact seconds/14400 node_h and result once,copy
small latency/cost artifacts,finish research/notes/T-0013-delivery.md,commit and
deliver T0013 to the coordinator. Do not open validation or run the abandoned
src/v5_slice_check.py. All runtime/analysis stays in PBS via miyabi-mcp.

Known completed use:formal73357 MIG seconds + ancillary10147 =83504 seconds
(5.79888889 node_h),excluding the latency job whose usage is pending.
E0013 ceiling12 node_h. All9 result IDs are in T-0013-main-state.json and
T-0013-training-accounting.json.
