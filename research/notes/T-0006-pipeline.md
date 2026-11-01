# T-0006 M-line pipeline implementation and smoke

Status: implementation, fixture smoke, E-0005 integration and focused review-fix
regression complete; re-check RV-0012 pending. This is not a delivery or V2 release.

## Implementation

Candidate code commit `94fb78b37d9a8c6ec8c426a88d78093db95469f8` adds:

- `src/v2_memory.py`: collate public delimiter writes, support-owner indexing,
  answer-only CE, offline projected-fact observation, read diagnostics, zero,
  cross-episode derangement, permuted key/value association, and key/value oracles.
- `src/v2_run.py`: bounded 1..64-step smoke CLI, eight fixed arms, zero-exit Base
  checks, gradient/clip statistics, step timing, peak allocated memory,
  checkpoint tensor roundtrip, full-vocabulary NLL/EM and relation/write-age slices.
- `src/v2_probe.py`: independent clean/read ridge probes, trained only on train
  vectors and evaluated on disjoint calibration/dev identities and canonical
  contents. These diagnostic heads never enter ordinary predictions.
- `src/v2_report.py`: training-throughput cost projection, with explicit exclusions.
- `src/test_v2_memory.py`: protocol/gradient/ownership/control/metric/probe checks.

No edits to `bypass_core.py`, `bypass_model.py`, `v1_*.py`, `v2_data.py`, locked
numerical tolerances or dependencies. Ordinary training invokes the existing
BranchModel support/query paths without oracle hooks. Short-lived hooks exist
only under no-grad evaluation; target facts do not enter `train_forward`.

All eight arms are layer14/r128/S1, gain1, default fixed softmax scale, F_out or
C_dual, direct_hidden or qknorm_pre_rope, kernel_matrix or soft_slot. Matrix
fact slots retained for offline diagnostics are compact projected delimiter
features; production memory remains M/c. Permuted association rebuilds M/c
from original keys and cyclically permuted values. Shuffled control deranges
support owners before the question-owner selection. Query memory is never
mutated, and support backbone KV is discarded.

The copy fixture is deliberately easy: two episodes with identical questions,
contrasting names, and repeated same-name facts. It tests that memory can carry
an answer signal; it does not test single-hop addressing or generalization.

## Evidence so far

- ATT-0036-1 / RES-0028 / EV-0025: commit e6c956f, 123 tests passed in 46.37s.
  Real GH200 MIG 2g.24gb, PBS 3760226.opbs on mg1098, exit0, 49s walltime.
  All eight arms pass Base identity, zero/nonzero gradient startup, immutable
  snapshot reuse, owner reordering, metadata exclusion, and oracle/control
  checks; numerical diagnostics match constructed hand calculations.
- ATT-0038-1 / RES-0030 / EV-0027: two-update timing pilot, eight arms, 672
  forwarded input tokens. Application elapsed9.26s. Positive-control comparison
  not yet met after two updates; this was a timing pilot, not the acceptance run.
- ATT-0040-1 / RES-0031 / EV-0028: 32-update copy fixture, all eight arms have
  normal NLL below both zero and shuffled. 10,752 forwarded input tokens,
  application elapsed42.53s, throughput303–317 tokens/s. No H-0003 support claim.
- ATT-0041-1: exact candidate integration, pending. Checks plus four updates per
  arm on two E=16 episodes, independent train/calibration probes and cost report.

## Data interface

Researcher-3 confirmed delimiter indices use separately tokenized support with
add_special_tokens=False; spans cover semicolons and may share whitespace tokens.
`identity` is nested, facts are in rendered support order, and target_fact_index
indexes that order. Training only accesses delimiter positions and answer labels.

Coordinator MSG-0106 permits provisional integration with read-only ATT-0037-1
smoke artifacts before T-0005 merge. Source train SHA256:
`602d79131873d05ef22465022d2347854c68725cba8ab819cfbcffcacdb2c6fe`;
calibration SHA256:
`437f85e99be5fdd01b5869ba005e40fa3d42c538107b2fc4370627f7c30837ec`.
Both checksums are asserted in the integration PBS wrapper. No generator import
or copy, and no confirmation data read. If the accepted schema changes, repeat
integration using the accepted artifacts as directed by the coordinator.

## Limits and review history

RV-0011 reviewer correctly identified that shuffled address metrics cannot be
interpreted using the recipient target index in donor memory. The correction
below omits those target-dependent metrics with an explicit unavailable marker,
while retaining full-vocab NLL/EM and original-recipient learned-value read error.

Throughput counts actual support plus question tokens forwarded; the single-token
answer is a CE label rather than another forwarded input. Larger-study contracts
must state whether their 4M budget counts labels as well. Cost reports use actual
forwarded inputs and are planning extrapolations, not authorization. Calibration
feature/fusion choice is not set by this task: report the range across the four
choices per backend for the two LR runs. Evaluation/checkpoint schedules and
startup overhead must be added to the training-only estimate.

No full V2 training or learning-rate calibration has been run.

## Exact-candidate integration result

ATT-0041-1 / RES-0033: 124 checks pass. All eight E=16 arms completed four
updates (5,248 forwarded training tokens total); the protocol-gradient tests
separately report672 forwarded tokens. Probe path fitted on8 train and evaluated
on8 calibration episodes; clean/read probe accuracies0.125/0.0 are tiny-smoke
outputs only and cannot support a representation conclusion. PBS3760269.opbs
exit0, measured77s. All scheduler observations and usage for attempts36/38/40/41
are imported from `research/artifacts/T-0006/scheduler-through-integration.json`.

Measured stable step and throughput on the two E=16 episodes (three timed steps
per arm after warmup):

| Feature | Read | Fusion | sec/step | input tokens/sec | peak allocated bytes |
|---|---|---|---:|---:|---:|
| direct_hidden | kernel_matrix | F_out |0.18029|909.63|1318665216|
| direct_hidden | kernel_matrix | C_dual |0.17266|949.83|1326362624|
| direct_hidden | soft_slot | F_out |0.17649|929.25|1318727168|
| direct_hidden | soft_slot | C_dual |0.20492|800.30|1326424576|
| qknorm_pre_rope | kernel_matrix | F_out |0.17193|953.90|1320238080|
| qknorm_pre_rope | kernel_matrix | C_dual |0.17093|959.48|1327935488|
| qknorm_pre_rope | soft_slot | F_out |0.17602|931.72|1320300032|
| qknorm_pre_rope | soft_slot | C_dual |0.17784|922.19|1327997440|

Training-only projection uses tokens/throughput/3600×0.25 for the MIG conversion:

- V2 eight arms ×4M forwarded tokens: **2.42391 node·h** total.
- KernelMatrix two LRs ×0.2M: **0.02895–0.03054 node·h** across feature/fusion.
- SoftSlot two LRs ×0.2M: **0.02981–0.03471 node·h**.
- Combined calibration: **0.05876–0.06525 node·h**.

Each six-control evaluation on the two episodes measured0.71–0.81s;
checkpoint export/reload measured0.0064–0.0092s. The report does not extrapolate
those without an evaluation/checkpoint schedule. Allocation peak can exceed
PyTorch allocated bytes; no whole-job GPU-memory claim is made. These estimates
come from three steady steps on two fixed episodes and are provisional.

## RV-0011 correction

RV-0011 found one medium issue: target-dependent shuffled addressing scores
used an unrelated donor ordinal. Fixed in5464b98: shuffled output now has
`address.target_available=false` and a reason, and omits `address.top1`,
`address.margin`, `address.target_weight`. Entropy and n_eff remain, as do
recipient-target read errors and answer logits/NLL/EM. No production forward,
contract, data, tolerance or metric formula changed.

Focused regression uses distinct recipient/donor facts, differing target
indices0/1, repeated/reordered owners[1,0,1], both backends, and nonzero fusion.
It compares diagnostic forward with unhooked deployed logits, NLL/EM, and
original-target errors computed from core memory reads. Normal/permuted target
addressing is checked separately.

ATT-0044-1 / RES-0036 / EV-0033: five checks passed, one test-reference error:
SoftSlot NLL differed4.77e-7 because the test used fused mean CE while evaluation
uses per-example CE followed by mean; logits were bit-identical. The test now
uses the declared per-example aggregation, retaining exact equality. No
production change or tolerance relaxation. Rerun commit9de4127, ATT-0045-1.

ATT-0045-1 / RES-0037 / EV-0034: all six focused regressions passed in9.83s;
PBS3760300.opbs on mg1098 exit0 and12s measured walltime. Diagnostic/unhooked
logits are bit-identical for normal, permuted and shuffled under both backends.
NLL/EM match exactly under the declared aggregation. Read-error discrepancy is
at most1.27e-7 overall and4.47e-8 for shuffled. Re-check requested asRV-0012.

Measured cost through the six runs: first four jobs total0.054444 MIG-instance
hours (196s), plus12s for each focused regression =220s, about0.015278 node·h.
Final two scheduler usage imports will be confirmed when the ledger exports end
observations; no additional training occurred during these two regressions.
