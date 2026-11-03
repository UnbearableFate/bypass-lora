# E-0031 cumulative cost and execution source

Current status after coordinator MSG-1043: aedd043 is historical diagnostic
evidence, not the final execution source. The coordinator requested the rule8
fallback now. A new source snapshot, code-review round and counted smoke are
required before formal training. Freeze that replacement snapshot for every
subsequent scientific execution. All use below remains charged.

Historical first passed-smoke execution source: aedd0436ad549f4b773c21c91206ebd995215ebd.
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
| ATT-0404-1 | 3786440.opbs | passed correctness/timing smoke; actual1g | 807 | 0.224167 | 0.05604175 |
| Total completed | | | 810 | 0.225001 | 0.05625025 |

Native observations and usage were imported once for each accounting_ref;
partial result records are RES-0377 and RES-0379. The ledger's generic2g
hardware label is not used as actual hardware evidence: both were1g slices.
MIG-instance hours multiplied by0.25 give node-hours; six-decimal native
rounding is retained. Requested walltimes are not measured usage.

## Running and forecast

ATT-0404-1/PBS3786440 on mg1102 completed as a historical correctness/timing smoke.
Actual profile1g.24gb is recorded in its mig-profile.json. The76 selected
tests passed in28.20s. All4real count/initialization checks, all4eightystep
children and all4legacy load/scoring children passed. H/T0 QVU counts were
629000 and U25 counts619400; paired initial SHA values matched; all exits
received nonzero first-step gradients and backbone stayed frozen. H/T0 QVU
steady step seconds were0.242406968/0.222244712. RB1-07 full dev on1g had18
prediction flips against its2g training record, so historical profile-matched
reproduction is still required. Result RES-0381 and diagnostic EV-0175 are
imported. No formal training is released by this note.

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
This historical resolution is superseded by MSG-1043's required code change:
the replacement counted smoke accepts --prior-node-hours, which must include
the full0.05625025 measured above. The forecast now also explicitly reserves
10% over measured LoRA dev time for side-branch scoring. Scheduler overhead
of the replacement smoke still must be added once after completion.

L2 was initially an explicitly accepted conditional limitation: the analysis
requires complete diagnostic checkpoints. If the smoke loads all checkpoints,
no recheck is needed. If RB1-01/02/09 genuinely cannot load, implement the
contract's unavailable-reason fallback and obtain a recheck before analysis;
RB1-07 is mandatory. Same-profile prediction mismatches remain implementation
failures and must not be silently treated as unavailable checkpoints.

MSG-1043 supersedes deferral of L2: the new analysis accepts an explicit
unavailable-reason entry for diagnostic RB1-01/02/09, omits it from merged
scores, and records missing rule8 comparisons without blocking rules3/4.
RB1-07 remains mandatory. A full-file analysis regression with RB1-02 absent
and a separate RB1-07 mandatory failure test will run in the replacement PBS
smoke. Reviewed-code changes receive the requested new review round.
