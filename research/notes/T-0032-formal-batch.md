# E-0029 formal batch continuation

Runtime source remains dc82cfc6c1f642f9380596a6be3244dafd4147a4; later
commits change only notes. Reviews RV-0090/RV-0092/RV-0094/RV-0095 pass and
all8 formal registrations report coverage. Source-matched126 tests, all8
80-step train-only timings, and all3 full256-example I-G checks passed.
D-0064/D-0065 authorize this exact sealed E-0029@1 batch after LR locking.

All4 fresh calibration candidates used identical V0 initialization SHA
b70142c4659891f51bcf7e73877414bc5328e3d19784850f888835e5f84767bf,
the same source/environment/split/order,1096steps,2192examples,200229tokens
and4096 paired calibration rows. No dev was read for calibration or selection.
PBS planner ATT-0360 required the single3e-3 extension; it was completed as
ATT-0361 /3778478.opbs, RES-0335, U-0368=.151389MIG-h (545s2g onmg1097).
Endpoint NLL1.7544837926876862 was finite and worse than the grid minimum.
No further extension is permitted.

PBS final lock ATT-0362 /3778583.opbs passed all4-candidate artifact checks:
RES-0336, U-0369=.003889MIG-h (14s2g onmg1098). Lock selected1e-3 as the
smallest finite LR within1percent of the finite minimum; boundary_minimum=false.
Its file is /work/xg24i002/x10041/bypass-lora/runs/ATT-0362-1/lr-lock.json,
SHA256 dcd6488206ce249b5aa4233c1517d1aede7fcda31c34df3927d67f4467975062.
It binds source, split, environment, common init, order, calibration membership,
all4 manifests and all4 config/predictions/steps inventories. Every formal
job reconstructs/verifies this lock before dev access. Both up-only arms
use1e-3; the6 side arms retain their inherited1e-3 side lock and also check
the new up lock. Old f46 calibration candidates remain excluded entirely.

Exactly8 jobs submitted00:38JST2026-11-03, immutable source snapshot,
short-mig select1:mpiprocs=1, groupxg24i002; no slice pinning:

| Arm | Attempt | PBS job | Expected seconds | Requested walltime |
|---|---|---|---:|---|
| G-T | ATT-0363-1 |3778633.opbs|4449|01:52:00|
| G-QN | ATT-0364-1 |3778634.opbs|4331|01:49:00|
| G-N | ATT-0365-1 |3778640.opbs|4342|01:49:00|
| Q-N | ATT-0366-1 |3778641.opbs|4306|01:48:00|
| P-T | ATT-0367-1 |3778642.opbs|4329|01:49:00|
| P-N | ATT-0368-1 |3778643.opbs|4240|01:46:00|
| L-56 | ATT-0369-1 |3778644.opbs|4091|01:43:00|
| L-48 | ATT-0370-1 |3778645.opbs|3928|01:39:00|

Submitted states were observed queued and recorded. All jobs check live
allocation before workload. Walltimes round approximately1.5x measured
source-smoke forecasts, including7 dev passes/fallbackBase/startup/read costs.
Queues have unknown wait; no extra concurrency cap is imposed by constraints.
Measured prior cost is.447916node-h, including every failed/diagnostic run,
old invalid calibration grid, fresh4 calibration candidates, planner and lock.
Current v4 projection including8 formal arms, reference reads,180s checks/
analysis and360s scheduler reserve is2.8496410718node-h against3.5cap.
Formal usage is unreported until terminal accounting; add actual runtime
hardware/seconds, not requested walltime. Generic MCP2g labels can disagree
with actual1g runtime; retain actual nvidia-smi/manifest hardware evidence.

Source-matched I-G inputs reused (no further collection planned):
G-T runs/ATT-0354-1/ig-G-T, G-QN runs/ATT-0355-1, G-N runs/ATT-0356-1,
all under /work/xg24i002/x10041/bypass-lora. Runner checks manifest/source/
sample and checkpoint inventory, then confirms only intended A_v changed.

Notification N-20261103-003817-9e73 /T0032-formal-eight-dc82cfc is sealed8,
all-ended plus early-failure reporting. The Miyabi notifier daemon is running;
MSG-0921 sent the coordinator its id and listener command:
/work/xg24i002/x10041/miyabi-toolbox/miyabi-mcp/bin/miyabi-mcp notify listen --id N-20261103-003817-9e73 --notifier miyabi-g2
Site-owned notifications wake the coordinator, who resumes researcher-2;
do not register a second watcher or implicitly create a Codex Goal. Expected
runtime exceeds1h; turn ends with task WAIT per researcher brief, without polling.

On completion, inspect application outputs, never infer success from PBS0.
Require complete validator,failedfalse, pinned source/environment/split,
R1seed0, common order,13115steps26230examples2000069inputtokens, expected
stored/trainable/gradient counts, initial identity to Base under D-0060's
runtime hardware rule, curves0/3279/6558/9837/final, exact final reload,
side64-train-example step0/final read diagnostics, IG and LR lock provenance.
Import results, terminal observations, exact measured usage; update cost.
Engineering/OOM/nonfinite/reload/identity failure stops for investigation.

After all8 pass, run reference reads and final analysis in PBS only. Verify
current E-0028 reviewed validation report for valid X-01 rather than guessing
its paths (earlier candidate: runs/ATT-0345-1/analysis.json, X-01 ATT-0321-1).
Reference RB1-09 ATT-0247-1, Base ATT-0024-1; verify RB1-07 run from locked
E-0021 analysis instead of assuming a run id. Use exact Fraction decisions
and preregistered descriptive task-stratified bootstrap10000 seed20261101.
Record evidence as diagnostic/screening appropriately, send delivery and
remaining limits to coordinator; it handles conclusion review and states.
Known RV-0092 conservative partial-X-01 low is inactive with valid fullX-01;
raise to coordinator if reference validation is withdrawn.

A2 confirmation and official validation remain sealed. D-0065 is bounded to
these8 arms: no new ordinary tap/source variants and no follow-up seeds.
If a replication trigger fires, report it to coordinator for a fresh rule10
decision; do not register the follow-up yourself.
