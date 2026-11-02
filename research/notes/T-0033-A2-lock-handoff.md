# T-0033 completed-lock handoff, before A2 confirmation

A2 remains sealed. No A2 evaluation has been submitted. This is an execution
readiness report, with no conclusion about H-0029.

The lock is
`/work/xg24i002/x10041/bypass-lora/research/artifacts/ATT-0393-1/a2-lock.json`,
SHA256 `cfd2e9b054c26da3d43ed22dbf0c410060599d6599d5fe88c4b8cfe1b78838c5`.
The file is read-only and its hash remains unchanged after the replays.
Training source is `f7419525d12ff511845a11fae7ae0fae5100e8f7` (RV-0098),
evaluation/analysis source is `4e6e8dfad40feb35fab60eecf7cf2df428083b1d`
(RV-0100, zero findings, exact-source 434 PBS tests). The inherited review
references RV-0081/82/88/92/94/95 also cover the registered sources. No source
file has changed since 4e6e8df.

All twelve formal runs passed the PBS lock validator, including final
13,115 steps / 26,230 examples, registered order hashes, finite step ledgers,
LR and parameter/token budgets, three paired curve snapshots, all artifact
and checkpoint hash/size bindings, reload identity, matched Keep/Reset/T0
initial digests and Keep initial/off-to-Base identities. Training IDs are
disjoint from the sealed A2 IDs; the prior full-train encodability exclusion
audit is pinned. Training results are RES-0355..RES-0366; lock is RES-0367.
All three legacy LoRA runs used actual 2g.24gb, without repeats. Keep seed 3
used 2g; Keep seeds 4/5 used 1g. The lock fixes the model/tokenizer revision,
split SHA, seven conditions, donor seed/method, 10,000 shared BoolQ bootstrap
draws and exact primary/secondary rules. Coordinator's independent audit in
MSG-0986 found no pin/rule problem and no A2 receipt.

## Actual-checkpoint dev replays

Each job used the same unchanged lock and reviewed evaluation source,
without unseal, confirmation decision or profile restriction. Allocations
were confirmed compute, and all manifests are complete with opened=false
and no A2 receipt. Reported checks in dev-reproduction.json agree with the
manifest; every replay reports 1,000 dev examples per task.

| Seed | Attempt / result | Host | Allocated / training MIG | Flips / 2,000 | Exact rows | Rule 2(a) |
| --- | --- | --- | --- | --- | --- | --- |
| 3 | ATT-0394-1 / RES-0368 | mg1099 | 2g / 2g | 0 | yes | pass |
| 4 | ATT-0395-1 / RES-0369 | mg1104 | 2g / 1g | 18 | no | fail on observed 2g replay |
| 5 | ATT-0396-1 / RES-0370 | mg1104 | 2g / 1g | 25 | no | fail on observed 2g replay |

These cross-profile failures are descriptive under D-0072(5), not new
scientific gates or a reason to change thresholds. Same-profile exactness
across nodes is demonstrated only for seed 3 on 2g. Neither seed 4 nor seed 5
has yet had an actual-checkpoint replay on 1g. A future run on another node
remains an observation to check, not a guaranteed reproduction.

Registration limitation: I used RunSpec arm `dev-replay`, while the existing
evaluator emits arm `A2` also in dev_replay mode. RL retained RES-0368..370
with an arm mismatch and marks them unusable as pre-specified comparisons.
They are retained as execution diagnostics; no confirmatory comparison uses
them. Source, seed, spec digest, checkpoint, mode, stage, lock and gate pins
were checked independently. Immutable RunSpecs and producer artifacts were
not rewritten. Future A2 attempts must register arm `A2` exactly.

## Proposed execution controls in MSG-0986/0987

The existing code and PBS script support the proposed profile restriction
without source changes: setting `SEG_REQUIRE_TRAINING_PROFILE=1` passes
`--require-training-profile`, which compares actual torch hardware with that
seed's trained Keep hardware. On a miss it exits non-zero before tokenizer,
dev evaluation or read_a2, and writes a partial unopened manifest with
failure_reason=retry_profile_miss. The program does not distinguish a first
attempt from a retry and does not enforce the consecutive-miss limit; the
controller/researcher owns counting and ordering. The matched-profile job
still must pass exact dev rows before reading A2. The code's failure_action
text continues to say five misses, as does the existing runbook.

The reviewed analysis accepts many profile-miss attempts: every unique
registered ATT must be in accounted_attempts and have terminal measured
usage; profile misses must be partial, unopened, stage=a2_evaluation. There
is no five- or twenty-attempt cardinality limit. The metadata scan has a
10,000-directory global bound. It still requires exactly one complete opened
selected attempt per seed. Replays, all misses, failures, lock, training and
diagnostics stay in the accounting. This is code inspection backed by the
existing regression suite, not a newly executed 60-miss stress test.

Three points need to be explicit in the confirming decision:

1. First-attempt profile restriction and a twenty-miss reporting limit
   supersede D-0072's unrestricted first attempt / five-miss rule and the
   unchanged lock's first_attempt_slice_restricted=false and
   consecutive_profile_miss_limit=5.
   The current lock has not been edited. Retaining its hash requires an
   explicit operational addendum; do not describe the old fields as the new
   execution policy. Scientific rows/checkpoints/thresholds/rules stay fixed.
2. If the aim is to catch the first 1g reproducibility failure before any A2
   read, initially submit only one seed's A2 attempt, with all other seeds
   held. Concurrent seed 4/5 attempts can let one read A2 before the other
   fails. Even a successful first 1g A2 job does not establish seed 5's
   reproducibility before that job's own gate; D-0072's consumed/invalid rule
   still applies to a later same-profile failure. Establishing both 1g
   replays before any A2 read would require additional 1g dev-only checks.
3. Twenty consecutive misses is a reporting ceiling, not permission to
   exceed 5.5 node-hours. The budget gate can require stopping earlier.

No A2 job will be registered/submitted before the actual confirming decision.

## Measured cost and misses

Training: 4.10138875 node-hour. Diagnostics: 0.055139. Lock: 0.0011805.
Replays: 0.301111 MIG-hour = 0.07527775 node-hour (U-0404..U-0406).
Completed measured total: **4.232986 node-hour**.

For the remaining three successful A2 jobs, the timing projection is
3,063.080 seconds each (the smoke estimate inflated by the worst observed
eval/reload ratio, rounded up to 1.30); each requests 80 minutes. Analysis
reserves 90 expected / 180 requested seconds (rounding up 1.5 times the
estimate). The full expected total without
misses is **4.877378**, and using all remaining requested walltimes would be
**5.245486** node-hour. Future costs are estimates, not measurements.

| Total profile misses across seeds | Full expected cost, 1–2 min/miss | Full cost using successful-job requested walltimes, 1–2 min/miss |
| --- | --- | --- |
| 0 | 4.877378 | 5.245486 |
| 20 | 4.960711–5.044044 | 5.328819–5.412153 |
| 40 | 5.044044–5.210711 | 5.412153–5.578819 |
| 60 | 5.127378–5.377378 | 5.495486–5.745486 |

Thus twenty misses per seed fits the timing estimate but is not guaranteed
to fit the cap if every successful job consumes its requested walltime.
With those walltimes reserved, the current miss budget is 0.254514 node-hour,
or 61.083 MIG-instance minutes total. Recompute from measured usage and all
remaining/running requests before every attempt; stop and report before
exceeding the cap. No independent or uniform scheduler-placement assumption
is made. Source data for these calculations and replay metadata/hash pins
are in central T-0033/cost-projection-after-replays.json and
dev-replay-verified-metadata.json.
