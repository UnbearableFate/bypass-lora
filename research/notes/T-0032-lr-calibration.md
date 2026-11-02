# E-0029 LR calibration and current release

Runtime source remains `f46a13dc4c65f6bf1292e3fafa4198af4fbd59f2`.
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

All three were observed queued at submission and recorded accordingly.
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

After all three jobs end, verify manifests, actual allocations, source,
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
in run notes. Current projection is2.4743083792node hours; measured usage
through the pre-calibration validation is0.12701325node hours. Calibration
usage is still unreported and must be added after completion.

The exception is bounded to E-0029@1. Propose no new ordinary-setting tap
or source variants. Any follow-up on a tap-arm outcome, including a triggered
two-new-seed replication, requires a fresh coordinator decision addressing
E-0028 rule10; report the trigger rather than registering the follow-up.
