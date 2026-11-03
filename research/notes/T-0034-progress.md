# T-0034 implementation and first smoke

Assignment DOC-0371@1 and E-0031@1 are read and task claimed. Code is committed
on sr/researcher-1/T-0034 at 6f85da7aa8a8b540c47e8a386e3f37dc6714f785.
No merge, rebase or push was performed. Snapshot is staged read-only under the
main project's snapshots/<commit>. Local commands are control work only;
the shell is login/control on miyabi-g2, aarch64, without an allocation.

Implementation: C_qvu uses the same X input_layernorm tensor for the side read
before the original attention, injects q_proj before q_norm/RoPE and v_proj
before GQA, and reuses z at up_proj. All three exits start at zero. H-QVU and
T0-QVU are r17/629000 parameters; conditional U-25/T0-U-25 are r25/619400.
The dedicated runner fixes R1 seed0, LR1e-3, 13115 steps/26230 examples,
the accepted data/order hashes and CPU initialization. It evaluates step0,
same-allocation Base, quarter curves, full dev/calibration, exact reload and
iv_off. Conditional formal training requires a rule6 analysis release.
New evaluation/analysis entrypoints preserve the V1 scorer and same-task
deranged history donor intervention. Analysis joins retained dev with fresh
calibration, uses rational thresholds and common split/task bootstrap indices.
A2 and official validation remain unopened.

Static control checks: git diff --check and bash -n launcher passed. Runtime
checks have not yet passed; no scientific scores or conclusions are available.
The test suite and actual model/gradient checks run inside the registered PBS
smoke. The complete source including tests and launcher is under RV-0104,
assigned by the coordinator to reviewer-2 (S-0022). Keep reviewed paths unchanged
until its answer. RV-0104 author model is unknown; coordinator MSG-1026 says
future requests must use --author-model gpt-6.1-sol.

First submission ATT-0401-1 / PBS3786303.opbs on mg1103 received MIG1g.24gb.
The allocation check passed but the explicit 2g guard exited3 before any test
or model import. Scheduler walltime2s; native usage0.000556 MIG-instance-hours
=0.000139 node-hours (native six-decimal rounding). Observations/usage imported
once from T-0034-smoke-v1-rejected-observations.json; guard artifacts imported
as partial RES-0377. The ledger's generic GH200 MIG2g label does not identify
the actual slice: nvidia-smi and mig-profile.json show1g. Include this rejected
attempt in all future cumulative costs. This is diagnostic, not scientific
evidence against either hypothesis.

Unchanged second attempt ATT-0401-2 / PBS3786323.opbs was submitted through
the MCP dry-run/commit flow with request key sr:ATT-0401-2. Full metadata is
T-0034-smoke-v1-job.json. Notification N-20261103-150310-7299 is sealed;
notifyd is running on miyabi-g2. Coordinator must hold the new listener:

`/work/xg24i002/x10041/miyabi-toolbox/miyabi-mcp/bin/miyabi-mcp notify listen --id N-20261103-150310-7299 --notifier miyabi-g2`

On completion, inspect the actual manifest, pytest output, real registration,
each child smoke, hardware, cost forecast and retained-checkpoint compatibility.
Refresh/export native terminal observations and measured usage once per job;
import the result and diagnostic evidence. No formal training before a passing
RV-0104 round and a measured-cost forecast within initial1.3 and total2.0
node-hours including failures. Update the cost note before formal submission.
If the smoke fails, investigate the exact error; never weaken gates. Fix reviewed
code only after the reviewer answers, commit and request the one recheck.
The 2g guard may reject another scheduler-selected1g slice before workload.

Formal training, four full checkpoint reevaluations, H-QVU swap, locked analysis,
any rule6 conditional controls, per-rule evidence and task delivery remain.
Coordinator owns hypothesis-state/route decisions and conclusion review.

## Resume after coordinator MSG-1030

ATT-0401-2 also exited at the profile guard on the mislabeled mg1103_mig3
1g slice, before tests or model imports. Native1s =0.000278 MIG-instance-hours
=0.0000695 node-hours was imported through
T-0034-smoke-v1b-rejected-observations.json; partial RES-0379 contains the
actual profile evidence. Both rejections total3s,0.000834 native MIG-hours,
0.0002085 node-hours. The coordinator already acknowledged both notifications.

MSG-1030 permits correctness/timing smoke on actual1g, while checkpoint
reproduction and formal arms retain the contract's profile requirements.
A separate T-0034-smoke.pbs accepts/records actual1g or2g for this bounded
pre-review workload. It calls the unchanged rb_qvu_smoke.py; none of the
RV-0104 reviewed paths changed. Reviewer-2 was notified via MSG-1032.
This smoke's cross-profile legacy comparisons are compatibility/timing
diagnostics, not substitutes for full evaluation at the respective historical
profiles (RB1-01/07/09:2g;RB1-02:1g). Do not claim a guaranteed relative1g/2g
speed from these records; the cost forecast uses the measured actual profile
and must be checked against the task cap including all previous failures.

Correctness smoke is registered as RUN-0404/ATT-0404-1 at source
aedd0436ad549f4b773c21c91206ebd995215ebd and submitted as PBS3786440.opbs
through the MCP dry-run/commit flow. Notification N-20261103-150955-8b0c is
sealed, notifyd is running, and coordinator owns the listener named in
T-0034-correctness-smoke-job.json. Static bash -n/diff checks passed for the
new launcher; runtime results and code-review findings are still pending.

## RV-0104 passed; smoke tests passed, other outputs pending

Reviewer-2 returned pass with two low findings (MSG-1034). The76 selected
tests passed in28.20s inside ATT-0404-1 on actual1g.24gb (mg1102,
PBS3786440); remaining real-model registration/timing/compatibility outputs
are still running. No formal training has started.

Reviewer-2 MSG-1037 confirmed src/ is unchanged between6f85da7 and aedd043.
L1 is closed by freezing aedd043 as every scientific execution's source and
adding earlier failures in the manual cost table (T-0034-cost.md). L2's
conditional handling is accepted: no recheck needed if diagnostic checkpoints
load; if one truly cannot load, add unavailable-reason fallback and recheck.
Do not edit reviewed scientific files or move runs to current HEAD merely
because later notes have been committed. The coordinator owns the existing
N-20261103-150955-8b0c notification and will resume this role on completion.

## MSG-1043: completed diagnostic smoke and required final revision

ATT-0404-1 is ended/success, all76tests, real count/gradient/initialization
checks,4training smoke children and4retained checkpoint children complete.
Actual MIG1g.24gb on mg1102; native807s/0.224167 MIG-hours imported once;
RES-0381 complete and EV-0175 diagnostic/valid. No scientific effect scores
were used. The historical RB1-07 full-dev comparison had18cross-profile
flips; do not substitute it for a2g reproduction. All other bounded legacy
checks had0flips, including RB1-02 at its actual1g profile.

Coordinator MSG-1043 now explicitly requires L2 fixed (supersedes the earlier
conditional deferral), followed by a new review round and final-source smoke.
The revised analysis accepts diagnostic unavailable-reason entries without
blocking rules3/4, while RB1-07 remains mandatory. New regression tests run
the analysis with an absent RB1-02 and check the primary mandatory guard.
The smoke forecast now takes mandatory prior measured node-hours; initial
prior is0.05625025 including all3ended attempts. Formal historical reevaluation
launchers now guard2g for RB1-01/07/09 and1g for RB1-02; formal new arms stay2g.
The final snapshot must be committed before its counted smoke and remain
fixed for all subsequent scientific executions. Note-only HEAD changes are
not execution-source changes. No formal training has started.

Final-source revision49f70094b78474ef0158dd631e3b25ea6ebdd174 is committed
and staged read-only. RV-0107 is the requested new review round (reviewer-2,
author-model gpt-6.1-sol); leave its paths unchanged until the answer. Static
diff and both launcher bash syntax checks passed; new runtime checks pending.
Exact-source counted smoke RUN-0406/ATT-0406-1 is submitted as PBS3786715.
It carries prior_node_hours0.05625025 and notification N-20261103-153445-e123.
Coordinator must hold its listener from T-0034-final-smoke-job.json; notifyd
is running. On completion import outputs and native observations/usage,
require RV-0107 pass and updated cumulative cost <=1.3initial/2.0total,
then train and reevaluate on49f7009 (not later note-only HEAD).

## RV-0107 passed; execution conditions for its two low findings

RV-0107 passed with no medium/high findings; RV-0104 L1/L2 are both fixed.
The final-source90tests passed in28.03s inside ATT-0406-1 on actual2g.24gb,
mg1099/PBS3786715. Remaining smoke outputs are still running. Do not infer
the final cost or full-check completion from pytest alone.

Reviewer-2 MSG-1048 accepted procedural handling of the two new lows without
another code review or source change. For EVERY analysis spec, including the
first rule6-release analysis and final conditional-control analysis:

- Bind legacy reevaluations only to their separate registered reevaluate
  attempts executed with matched-profile guards. Never bind a smoke's
  reference-RB1-07 directory, even if its source/stage otherwise match.
- Check all four available legacy entries in analysis.json under
  rules.1_implementation.checkpoint_compatibility: same_profile must be true,
  flips must be0. Match RB1-01/07/09 to2g and RB1-02 to1g using each run's
  actual nvidia-smi evidence and mig-profile.json (passed true with expected
  and actual profile equal). Keep each profile artifact path and digest in
  the analysis input/binding note. Conclusion review must see these records.
- If an optional diagnostic checkpoint cannot load, bind its unavailable
  reason to the failed attempt, manifest and log with the actual load error.
  A same-profile prediction mismatch, OOM or other runtime failure must not
  be relabeled unavailable. RB1-07 remains mandatory. Preserve the evidence
  in the task note and analysis input for conclusion review.

These conditions concern actual producer selection; scientific source49f7009
stays fixed. Before every analysis run, inspect the bound producer manifests
and guards; after it runs inspect the compatibility output again before
importing scientific evidence or releasing rule6. Cite this note together
with the binding artifact in the diagnostic evidence provided to the
coordinator/conclusion reviewer.

## Formal batch and guarded retries (current continuation)

Final smoke imported as RES-0382/EV-0176. All90tests, real counts/initial
pairing, child reload/identity checks and frozen-source cost gate passed.
Formal cost release is T-0034-formal-cost-release.json: initial1.0101741017,
with conditionals1.6694988604node-hours before subsequent guard failures.
All scientific execution remains49f70094b78474ef0158dd631e3b25ea6ebdd174.

Explicit supplementary passing RV-0094/RV-0095 scopes cover three inherited
files reported stale by the first training registrations; exact git diffs
were empty. See T-0034-inherited-review-coverage.md. ATT-0409-1/0410-1 were
prepared only, retired before any submission, then replaced by covered
ATT-0409-2/0410-2 with the identical immutable RunSpecs. No duplicate training
was launched. All actual execution attempts report code review covered.

Initial batch metadata and notification are T-0034-formal-jobs.json,
N-20261103-160419-5dfa (coordinator holds its listener). Initial H-QVU,
RB1-02 and RB1-09 got wrong profiles and exited at their guards in2s each:
ATT-0409-2/PBS3786995 actual1g instead2g (RES-0386), ATT-0412-1/3787000
actual2g instead1g (RES-0385), ATT-0414-1/3787005 actual1g instead2g
(RES-0387). Native observations and usage have been imported once; these
are pre-model scheduling diagnostics, not model-load or scientific failures.

The afterok H intervention ATT-0420-1/PBS3787063 is ended/cancelled,
FINISH no-start: PBS automatically terminated it because its parent3786995
had failed before the dependency was submitted. No application ran and no
native usage was exported. Do not resubmit RUN-0420 unchanged: its immutable
producer metadata names failed ATT-0409-2. Register a fresh intervention
spec naming the successful current H retry when available.

Coordinator MSG-1059 authorized guarded retries, no vnode resource (qsub
rejects it), and an explicit diagnostic-only RB1-02 exception after10
consecutive wrong-profile rejections, with every attempt/log attached. That
administrative exception is not a fabricated model-load failure; keep its
coordinator provenance visible to conclusion review if it becomes necessary.
Only1RB1-02 wrong-profile failure has been recorded so far. H and T0 must
both actually run on2g. RB1-09 must match its historical2g profile.

Latest retry metadata: T-0034-formal-retry-jobs.json. Notification
N-20261103-161637-a3a1 is sealed3, notifyd running; coordinator must hold:

`/work/xg24i002/x10041/miyabi-toolbox/miyabi-mcp/bin/miyabi-mcp notify listen --id N-20261103-161637-a3a1 --notifier miyabi-g2`

| Arm | Current attempt | PBS | Last observed state |
| --- | --- | --- | --- |
| H-QVU | ATT-0409-3 | 3787112.opbs | queued retry; must pass2g guard |
| T0-QVU | ATT-0410-2 | 3786996.opbs | running, actual2g mg1112 |
| RB1-01 | ATT-0411-1 | 3786999.opbs | complete/success, imported RES-0388 |
| RB1-02 | ATT-0412-2 | 3787111.opbs | queued retry; must pass1g guard |
| RB1-07 | ATT-0413-1 | 3787002.opbs | complete/success, imported RES-0389 |
| RB1-09 | ATT-0414-2 | 3787113.opbs | queued retry; must pass2g guard |

Completed RB1-01/RB1-07 evaluation used full1000dev and2048calibration rows
per task, frozen49f7009 and correct historical checkpoint producers. Both
same_profile=true/flips0; each guard passed actual/expected2g. Native489s
and541s usage imported. EV-0177 records these checks only, not a hypothesis
effect. T-0034-analysis-profile-bindings.json lists each guard/manifest/
provenance path and is deliberately partial until remaining arms complete.
Do not rerun these successful original evaluations or the running T0.

All ended charged work currently totals2613scheduler seconds and0.725836
native MIG-instance hours =0.181459node-hours. Running/queued usage remains
unreported, not zero. Conservative initial/conditional forecasts with the
three additional2s failures are1.0105911017/1.6699158604, still within1.3/2.0.
On notification inspect latest retry guards/application outputs, import
native usage/results, retry wrong profiles within MSG-1059/budget, then
intervene on successful H and run initial locked analysis. Apply MSG-1048
input checks to EVERY analysis, including the first rule6 release. No
conditional U25 arm has been released or trained formally. No hypothesis
state or route decision has been changed.
