# T-0013 completion handoff

All requested T0013 work is complete. No T0013 jobs remain queued or running.
Read [T-0013-delivery.md](T-0013-delivery.md) for the final nine-checkpoint
inventory,original LR lock,training/dev results,review coverage,latency and costs.

Nine training runs complete/passed with exact within-job reload and identical
per-seed sample orders across arms. Every run used LR0.001 and completed
29165 steps/58329 presentations/4173300 input tokens. C/F results RES0147..0152;
LoRA RES0165..0167. Full nine-inventory preflight passed in ATT0174.

Final latency ATT0174-1/PBS3767640/RES0172 completed on verified2g mg1098.
All5 models have768timings. Independent artifact tabulation
ATT0175-1/PBS3767690/RES0173 checked all3840 positive finite raw samples and
mean/median/p95 agreement. EV0101 and EV0106 are diagnostic/valid.

All measured usage imported once:formal73357 MIG seconds,other prior10147,
latency179,tabulation1,total83684 =5.81138889 node_h. This is T0013 only;
other E0013 workers' costs are separate. Final authoritative billing file:
research/artifacts/T-0013/usage-summary.json.

D0018/D0019 still apply:original LR lock final,no slice comparison,alternate lock
or retraining. Profiles are metadata;timing used the existing2g guard. Official
validation has not been read by this worker and remains sealed until the
coordinator releases it. No H0008 conclusion is made here.

All9 manifest/inventory/config/weight FileRefs:
 /work/xg24i002/x10041/bypass-lora/runs/ATT-0174-1/checkpoint-inventory.json
 SHA2562886d4981d2fe954bfc022b2d5b6641af638d1d0529d0012755538bd60619619
 10830bytes.
Copy-compatible fields for the later confirmation lock are learning_rate_lock
and checkpoints. This artifact is also committed byte-identically in read-only
snapshot3e7c90c8f7f696561b04a71f96a054a5da65e7b6/research/artifacts/T-0013/.

Current branch sr/researcher-2/T-0013. Next action is coordinator acceptance and
merge,then the coordinator's separately authorized confirmation workflow.
Do not rerun training/latency or revive src/v5_slice_check.py.
