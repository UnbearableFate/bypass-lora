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
