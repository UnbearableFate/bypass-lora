# Post-RV-0078 deterministic regression

ATT-0276-1 / PBS 3770216.opbs / RES-0253 completed at exact source
7d8b3464b6145504d155268f4c667fa3ecc53b00. XML, checks.log and the complete
validator-passed manifest agree: **326 passed, zero failures, errors or skips**.
The test run took 44.92 seconds. Allocation_check confirms mg1098 compute;
profile is retained as metadata, without a slice gate or slice-related work.

All three RV-0078 low findings are covered by the revised rules/source/cost
failure tests. The deterministic suite also replays all no-channel identities,
state carry/gradient, initialization, operator and synthetic artifact checks,
and the full synthetic 10,000-bootstrap analysis. The original training,
tokenization, evaluation, model and checkpoint paths are unchanged from the
seven-arm ATT-0271-1 cost smoke. No formal SEG run or SEG dev quality was read.
All confirmation sets remain unopened; no hypothesis status changed.

Scheduler F/Exit0 and actual walltime **55 seconds** were independently read
through job_get. The notification refresh then emitted the native terminal
observation; that observation and measured usage were imported once. The
result was separately imported once, without mismatches. Site-native usage
is 0.015278 MIG-hours, or 0.0038195 node-hours at its six-decimal precision;
the exact scheduler-seconds conversion is 55/14,400 = 0.0038194444 node-hours.

Total measured T-0028 usage is now 532 MIG seconds = 0.0369444444 node-hours
(smoke 477 plus regression 55). Adding it to the complete conditional-branch
part model gives **2.541927988 node-hours**, below the 3-node-hour cap, with
0.458072012 projected headroom. The prior review/analysis reserve remains in
that estimate; future actual usage is charged once and the final analysis
requires every declared terminal attempt, including failures and this check.

The site job_wait cache still said running after the job ended. Fresh job_get,
the completed artifacts and the subsequent native terminal export resolved
the state; no execution was repeated.

The focused recheck uses exactly source 7d8b346 and the revised launcher,
against RV-0078. If it passes, formal Base/train/intervention/analysis runs
will register this same immutable source. The final analysis will require
this check manifest's source_commit to equal that training source.
