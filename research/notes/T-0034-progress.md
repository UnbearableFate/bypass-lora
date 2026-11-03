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
