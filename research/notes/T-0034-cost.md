# E-0031 cumulative cost and execution source

Current execution source is49f70094b78474ef0158dd631e3b25ea6ebdd174,
passed RV-0107 and counted smoke ATT-0406-1/PBS3786715. Freeze this snapshot
for every subsequent scientific execution. The coordinator requested the
rule8 fallback in MSG-1043; all use below remains charged. aedd043 is now
historical diagnostic evidence and is not accepted as final-source smoke.

Historical first passed-smoke execution source: aedd0436ad549f4b773c21c91206ebd995215ebd.
The original plan to retain aedd043 was superseded by the required rule8 fix.
Use49f7009 for counted smoke, formal QVU arms, checkpoint reevaluations,
intervention, analysis and any rule6 controls. Later note-only commits do not
change this registered execution source.
RV-0104 passed; reviewer-2 MSG-1037 confirmed that src/ is identical between
the reviewed6f85da7 and aedd043. The added smoke-only launcher is allowed on
actual1g/2g for pre-review correctness/timing (coordinator MSG-1030).

## Measured completed use

| Attempt | PBS | Outcome | Scheduler seconds | Native MIG-instance h | Node h |
| --- | --- | --- | ---: | ---: | ---: |
| ATT-0401-1 | 3786303.opbs | 1g rejected before workload | 2 | 0.000556 | 0.000139 |
| ATT-0401-2 | 3786323.opbs | 1g rejected before workload | 1 | 0.000278 | 0.0000695 |
| ATT-0404-1 | 3786440.opbs | passed correctness/timing smoke; actual1g | 807 | 0.224167 | 0.05604175 |
| ATT-0406-1 | 3786715.opbs | passed final-source smoke; actual2g | 767 | 0.213056 | 0.053264 |
| Total completed | | | 1577 | 0.438057 | 0.10951425 |

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

The revision is committed as49f7009. New RV-0107 (round1of2) is assigned to
reviewer-2 with author-model gpt-6.1-sol. The exact-source smoke ATT-0406-1
is registered/submitted with prior_node_hours0.05625025. Its walltime request
is21min =1.5x840s (previous807s plus33s regression allowance). On completion
import measured scheduler usage; add only current scheduler/application
overhead beyond the application elapsed already included in its forecast.
Do not add0.05625025 again: the revised forecast explicitly includes it.

RV-0107 passed. The90final-source tests have passed on actual2g.24gb in
ATT-0406-1; other smoke work and its cost estimate are still pending.
Reviewer MSG-1048 accepts closing its two new lows by explicit actual-input
checks on every analysis (initial rule6 release included). See the current
T-0034-progress.md section for the mandatory matched-profile producer binding,
zero-flip verification and failed-load provenance needed by conclusion review.
No new scientific-source change is needed for these procedural conditions.

## Final smoke verified and formal cost release

ATT-0406-1 has completed with90tests passed, including the complete-analysis
RB1-02-unavailable regression and mandatory RB1-07 failure test. All4real
registration checks and all8children are complete/passed. Actual2g.24gb,
mg1099. H/T0 QVU stored/trainable/with-grad629000; U25 pair619400; paired CPU
SHA values match; all exits received nonzero first-step gradients; frozen
backbone and finite train/reload/iv_off checks passed. H/T0 QVU steady steps
0.247453769/0.211776897seconds. Same-profile full RB1-07 dev has0flips.
The bounded RB1-02 cross-profile check has1flip, so its formal1g reevaluation
remains required. Imported complete RES-0382 and diagnostic EV-0176.

Formal release receipt is T-0034-formal-cost-release.json. It starts from
the PBS-produced cost-estimate.json (initial0.9798966535, conditional1.6392214122),
already including prior0.05625025 and the final smoke's754.060463application
seconds. Add only the final smoke's native scheduler/application difference
0.0008986900node-hours. Increase historical reevaluation reserves to610seconds
each for RB1-01/02/09 and500seconds for RB1-07; increase intervention reserve
to600seconds using reviewed comparable2g timings. This adds0.0293787582.
The resulting forecasts are **1.0101741017 initial** and **1.6694988604 with
conditionals**, below1.3/2.0 including all completed use. These are forecasts,
not guaranteed runtime; retain actual usage for all further attempts.

Submit H-QVU expected5194seconds/walltime2h10m and T0-QVU4726seconds/2h,
both with2g guard on short-mig. Submit historical reevaluations with respective
training-profile guards (RB1-02:1g, others2g), walltimes approximately1.5x
the above per-arm reservations. A wrong-profile allocation exits before model
work and is charged; check existing jobs before any retry. Freeze source49f7009
for all these jobs. Coordinator MSG-1053 authorizes continuation on passing
smoke/review and this within-cap cost update. No rule6 controls are released yet.

## Formal work accounted before waiting on retries

| Attempt | PBS | Actual profile/outcome | Seconds | Native MIG h | Node h |
| --- | --- | --- | ---: | ---: | ---: |
| ATT-0409-2 | 3786995 | actual1g rejected; expected2g | 2 | 0.000556 | 0.000139 |
| ATT-0412-1 | 3787000 | actual2g rejected; expected1g | 2 | 0.000556 | 0.000139 |
| ATT-0414-1 | 3787005 | actual1g rejected; expected2g | 2 | 0.000556 | 0.000139 |
| ATT-0411-1 | 3786999 | actual2g; complete RB1-01 eval | 489 | 0.135833 | 0.03395825 |
| ATT-0413-1 | 3787002 | actual2g; complete RB1-07 eval | 541 | 0.150278 | 0.0375695 |
| All ended work incl prior table | | | 2613 | 0.725836 | 0.181459 |

Every native accounting_ref is imported exactly once. The cancelled afterok
child ATT-0420-1/PBS3787063 has FINISH no-start and no exported usage;
no workload ran. Prepared-only ATT-0409-1/0410-1 never had a PBS allocation.
Keep running/queued work unreported until the native terminal observation.

T0-QVU remains running on2g; original RB1-01/RB1-07 results are imported
RES-0388/0389 and profile checks are EV-0177. H/RB1-02/RB1-09 retries are
ATT-0409-3/0412-2/0414-2; details/notification in formal-retry-jobs.json.
The last release projection plus0.000417 for3guard failures is
1.0105911017initial/1.6699158604with-conditionals. It retains the original
conservative reservations for already-completed reevaluations rather than
using their small savings to expand scope. Reconcile all later actual use.

## Both formal trains complete; current evaluation-wave release

| Attempt | PBS | Outcome | Seconds | Native MIG h | Node h |
| --- | --- | --- | ---: | ---: | ---: |
| ATT-0409-3 | 3787112 | H-QVU complete, actual2g | 5252 | 1.458889 | 0.36472225 |
| ATT-0410-2 | 3786996 | T0-QVU complete, actual2g | 4852 | 1.347778 | 0.3369445 |
| ATT-0414-2 | 3787113 | RB1-09 full eval complete, actual2g | 573 | 0.159167 | 0.03979175 |
| ATT-0412-2 | 3787111 | wrong2g rejected, expected1g | 1 | 0.000278 | 0.0000695 |
| All ended work incl previous tables | | | 13291 | 3.691948 | 0.922987 |

Native terminal usage and results RES-0390/0391/0392/0393 are imported once.
No formal training or successful legacy evaluation needs repetition.
Current wave reserves all3parallel RB1-02 attempts authorized by MSG-1061,
so even3successes (two duplicate evaluations) remain within the cap. Receipt
T-0034-evaluation-wave-cost.json uses actual ended0.922987 +600s intervention
+3x610s RB1-02 +120s analysis +20s guard reserve, yielding1.1014592222initial.
Both conditional controls'9494.276525seconds with5%timing reserve add
0.6922909966, total1.7937502189. U25 is still conditional on the locked gate;
the reservation is not release. Count every future failure and duplicate.

Intervention now correctly names producer ATT-0409-3 as new RUN-0421,
ATT-0421-1/PBS3788305; old cancelled RUN-0420 is not reused. RB1-02
ATT-0412-3/4/5 are PBS3788306/7/8. Required profiles remain2g/1g respectively;
minimum numeric successful job ID selects the RB1-02 producer. Notification
N-20261103-180705-6699 covers all4; see evaluation-wave-jobs.json.

## Final measured total

| Attempt | PBS | Outcome | Seconds | Native MIG h | Node h |
| --- | --- | --- | ---: | ---: | ---: |
| ATT-0421-1 | 3788305 | H intervention complete,2g | 593 | 0.164722 | 0.0411805 |
| ATT-0412-3 | 3788306 | selected RB1-02 complete,1g | 591 | 0.164167 | 0.04104175 |
| ATT-0412-4 | 3788307 | wrong2g guard rejection | 1 | 0.000278 | 0.0000695 |
| ATT-0412-5 | 3788308 | duplicate RB1-02 complete,1g | 565 | 0.156944 | 0.039236 |
| ATT-0424-1 | 3788665 | final locked analysis complete | 10 | 0.002778 | 0.0006945 |
| All ended charged work incl prior tables | | | 15051 | 4.180837 | 1.04520925 |

Native filtered ledger summary in T-0034-final-cost.json confirms19submitted
job references all ended:11success,7guard failures,1cancelled FINISH no-start.
All18charged jobs' native accounting_refs are imported once. The cancelled
dependency never allocated/ran and has no usage export; two prepared-only
registrations had no PBS jobs. This is below the1.3initial and2.0total caps.
No future-work forecast is charged as usage. Conditional controls were not
triggered, so there is no required further computation or resource request.
