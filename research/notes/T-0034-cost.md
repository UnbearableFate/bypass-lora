# E-0031 cumulative cost and execution source

Fixed execution source: aedd0436ad549f4b773c21c91206ebd995215ebd.
Use this exact commit for the counted smoke, both formal QVU arms, retained
checkpoint reevaluations, H-QVU intervention, analysis and any rule6 controls.
Later note-only commits do not change this registered execution source.
RV-0104 passed; reviewer-2 MSG-1037 confirmed that src/ is identical between
the reviewed6f85da7 and aedd043. The added smoke-only launcher is allowed on
actual1g/2g for pre-review correctness/timing (coordinator MSG-1030).

## Measured completed use

| Attempt | PBS | Outcome | Scheduler seconds | Native MIG-instance h | Node h |
| --- | --- | --- | ---: | ---: | ---: |
| ATT-0401-1 | 3786303.opbs | 1g rejected before workload | 2 | 0.000556 | 0.000139 |
| ATT-0401-2 | 3786323.opbs | 1g rejected before workload | 1 | 0.000278 | 0.0000695 |
| Total completed | | | 3 | 0.000834 | 0.0002085 |

Native observations and usage were imported once for each accounting_ref;
partial result records are RES-0377 and RES-0379. The ledger's generic2g
hardware label is not used as actual hardware evidence: both were1g slices.
MIG-instance hours multiplied by0.25 give node-hours; six-decimal native
rounding is retained. Requested walltimes are not measured usage.

## Running and forecast

ATT-0404-1/PBS3786440 on mg1102 is the counted correctness/timing smoke.
Actual profile1g.24gb is recorded in its mig-profile.json. The76 selected
tests passed in28.20s. The real initialization/count checks,80-step timing,
checkpoint compatibility and full RB1-07 timing pass are not yet complete.
No formal training is released by this note.

After completion, inspect and import the actual outputs and native usage.
The smoke's cost-estimate.json includes that smoke's own application elapsed,
both planned QVU trainings, four planned complete reevaluations, intervention
and120s analysis reserve. Its forecast omits all previous attempts. Therefore
the release table must add0.0002085 node-hours for both earlier rejections,
and account for the counted smoke's measured scheduler/application difference
and every further failure or rerun, without double counting its application
elapsed already present in the forecast. The initial forecast must stay
<=1.3 node-hours, and the forecast including both rule6 arms <=2.0. Recompute
from measured values before submission; do not substitute requested walltime.

Formal H-QVU and T0-QVU must both pass the2g guard. Historical prediction
reproduction is separate: RB1-01/07/09 require2g; RB1-02 requires1g. The
1g smoke's cross-profile checks do not substitute for those evaluations.

## Review handling

RV-0104 has no medium/high finding. L1 is closed by fixed execution source
plus this manual cumulative accounting, confirmed in reviewer MSG-1037.
L2 remains an explicitly accepted conditional limitation: current analysis
requires complete diagnostic checkpoints. If the smoke loads all checkpoints,
no recheck is needed. If RB1-01/02/09 genuinely cannot load, implement the
contract's unavailable-reason fallback and obtain a recheck before analysis;
RB1-07 is mandatory. Same-profile prediction mismatches remain implementation
failures and must not be silently treated as unavailable checkpoints.
