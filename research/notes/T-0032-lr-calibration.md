# E-0029 LR calibration and current release

Runtime source is now `dc82cfc6c1f642f9380596a6be3244dafd4147a4`.
The old f46a13d grid below is diagnostic-only and must never be used for LR
selection: ATT-0349/0350 shared initialization digest4b9e9951..., but
ATT-0351 had65ccff56... across different MIG shapes. ATT-0353-1's reviewed
planner correctly rejected that cohort before producing a lock. No LR was
selected, and no new dev was read. EV-0164 records the failure.

MSG-0907 authorized CPU initialization only for up-only LoRA, preserving
legacy LoRA. RV-0095 passed with zero findings at dc82cfc. PBS tests report
126 passed, including exact CPU/CUDA initial adapter equality for ranks48/56.
The source-matched smoke ATT-0354-1 /3778174.opbs passed (RES-0330); full256
G-T orth/RMS gates and all8 finite80step train-only timings passed. Full256
G-QN and G-N checks ATT-0355/0356 passed (RES-0328/0329). EV-0165 records
implementation validation only. N-20261102-235003-e198 was acknowledged by
the coordinator. Details: T-0032-portable-init-repair.md.

Fresh portable-init L-56 calibration grid submitted2026-11-03 at00:06JST:

| LR | Attempt | PBS job |
|---|---|---|
| 1e-4 | ATT-0357-1 |3778348.opbs|
| 3e-4 | ATT-0358-1 |3778349.opbs|
| 1e-3 | ATT-0359-1 |3778351.opbs|

All three run the immutable dc82cfc snapshot, reviewed by RV-0095. No MIG
slice pinning. Expected570s, wall14:15 from measured source-matched timing.
Notification N-20261103-000608-66da /T0032-up-lora-portable-grid is sealed3,
earlyfailure enabled. Daemon is running; coordinator was sent its listener
command in MSG-0915. The NEW input list is
T-0032-up-lr-grid-portable-input.json. Never use the historical f46 input list
or any candidate from it. Current results/usage are unreported until completion.

RV-0090/RV-0092 passed. RV-0092's remaining low is accepted for this run:
without X-01 the implementation conservatively defers some already decidable
outcomes. The current full-X-01 path is unaffected; E-0028 rule1 is valid.
If that validation is withdrawn, raise the partial-outcome limitation to the
coordinator before any affected conclusion rather than silently claiming it
was fixed.

D-0064 explicitly makes the two inherited test-file stale paths non-gating.
Their registration record is retained. Registrations were repeated with
`--review RV-0090,RV-0092` and idempotently returned the existing attempts.
RV-0094 is a separate supplemental inherited-test review: pass, zero findings,
for record completeness. Per MSG-0897, ATT-0352-1's optional two-file test job
was withdrawn before submission (only its dry-run plan existed); no workload
ran. Do not run it unless reviewer-2 specifically requests it.

Initial L-56 calibration grid submitted through Miyabi MCP:

| LR | Attempt | PBS job |
|---|---|---|
| 1e-4 | ATT-0349-1 | 3777881.opbs |
| 3e-4 | ATT-0350-1 | 3777882.opbs |
| 1e-3 | ATT-0351-1 | 3777883.opbs |

All three ended with complete finite calibration output, but the shared-init
gate failed as described above. Imported RES-0325/0326/0327 and measured
usage U-0357/0358/0359 remain diagnostics, excluded from LR selection.
V0 init/train seed0, first complete batch-2 step reaching200k input tokens:
the fixed plan is1096 steps,2192 examples,200229 input tokens, order SHA256
053284831520676982857bf9677a9fa3e338b0bbad4d2ce2af1c3f88d20457d3.
Each run evaluates the full4096-row calibration set and opens no dev data.
Measured projection per candidate is about512seconds; walltime13minutes
rounds approximately1.5x. Allocation is checked inside each job.

Notification N-20261102-232403-3ef2 / T0032-up-lora-calibration-grid is sealed
at3 jobs. Its message was updated after MSG-0900 to remove the former formal
hold. The notifier daemon was ensured running; the coordinator owns the
listener and wakes this researcher. The initial input list is
T-0032-up-lr-grid-input.json; freeze/cite its digest when registering lr-plan.

For the fresh source-matched grid, verify manifests, actual allocations, source,
initialization, order, first-step token stop and full calibration membership;
import results, terminal observations and exact usage. Typed non-finite
calibration candidates are exclusions. OOM/untyped/engineering failures stop.
Run lr-plan in PBS on the three manifests. Extend once to3e-5 if the raw
finite minimum is at1e-4, or3e-3 if at1e-3; a minimum at3e-4 needs no extension.
Then run lr-lock in PBS, choosing the smallest finite LR within1percent of
the finite minimum. Bind all candidate manifest/config/prediction/step hashes.
No new dev before that lock.

**D-0065 / MSG-0900 lifts MSG-0888's formal hold.** After the LR lock,
all eight sealed E-0029 arms are authorized exactly as registered: G-T,
G-QN,G-N,Q-N,P-T,P-N,L-56,L-48, within3.5node hours. Cite D-0064 and D-0065
in run notes. The old v2 cost projection2.4743083792node hours is historical;
the dc82cfc v3 table supersedes it. Measured usage before the dc82cfc checks
was0.24152725node hours, including the invalid calibration grid and failed
planner. U-0361/0362 add0.008125node hours for G-QN/G-N; U-0363 adds0.04770825
for full smoke. Measured total0.2973605node hours. Current projection includes
all8 future arms, worst-case4 fresh LR candidates, reference diagnostics,
180s checks/analysis and360s scheduler reserve:2.8555888954node hours<=3.5.

The exception is bounded to E-0029@1. Propose no new ordinary-setting tap
or source variants. Any follow-up on a tap-arm outcome, including a triggered
two-new-seed replication, requires a fresh coordinator decision addressing
E-0028 rule10; report the trigger rather than registering the follow-up.
