# E-0029 results: no prescribed improvement or replication trigger

All eight sealed E-0029@1 arms completed and the locked PBS analysis passed.
The preregistered point-estimate rules give screening support for H-0023,
inconclusive evidence for H-0024, and screening challenges to H-0025/H-0026.
No new side arm exceeded RB1-07 by1pp, so no replication or LoRA-GA follow-up
trigger fired. No follow-up was registered under the bounded D-0065 release.
These are seed0 screening results; hypothesis states and route decisions
remain with the coordinator after independent conclusion review.

The authoritative result is
`/work/xg24i002/x10041/bypass-lora/runs/ATT-0373-1/analysis.json`
(RES-0347; SHA256
8c503660024f012254b34d534dff1429ac89d73016a0dd0b7ad872b64dd8bb6f).
The input is T-0032-final-analysis-input.json, SHA256
99991e360b4aa30df01976b2c3af144bb24922ce16754a363c02a6cf50191ffc,
committed563f386. Runtime source is dc82cfc6c1f642f9380596a6be3244dafd4147a4;
later branch commits only update notes. RV-0090/RV-0092/RV-0094/RV-0095
cover this source. D-0064 and D-0065 are cited in registrations.

| Arm | PIQA accuracy % | BoolQ accuracy % | Macro accuracy % | Macro answer NLL |
|---|---:|---:|---:|---:|
| Base |72.30|68.00|70.15|2.337478|
| RB1-07, all7 LoRA |74.40|83.50|78.95|1.557266|
| RB1-09, T random side |73.20|81.80|77.50|1.556830|
| X-01, N random side |74.00|82.30|78.15|1.549634|
| G-T |72.80|80.90|76.85|1.550767|
| G-QN |73.30|81.30|77.30|1.544538|
| G-N |74.30|81.70|78.00|1.552939|
| Q-N |73.10|81.00|77.05|1.556017|
| P-T |73.00|81.30|77.15|1.553995|
| P-N |73.90|82.00|77.95|1.548987|
| L-56 |73.60|81.60|77.60|1.573123|
| L-48 |73.70|81.20|77.45|1.574364|

All new arms exceed Base by6.70–7.85pp, passing rule2's learnability check.
Rules3–6 use exact Fraction point differences, not the descriptive intervals.

| Rule/contrast | Macro difference, pp | Descriptive paired95% interval, pp |
|---|---:|---|
| H-0023: L-56 minus RB1-09 |+0.10|[-1.25,+1.45]|
| H-0023: L-48 minus X-01 |-0.70|[-2.05,+0.65]|
| H-0024: G-T minus RB1-09 |-0.65|[-1.45,+0.15]|
| H-0024: G-QN minus Q-N |+0.25|[-0.60,+1.10]|
| H-0024: G-N minus X-01 |-0.15|[-0.90,+0.60]|
| H-0025: P-T minus RB1-09 |-0.35|[-1.15,+0.45]|
| H-0025: P-N minus X-01 |-0.20|[-0.80125,+0.40]|
| H-0026: Q-N minus RB1-09 |-0.45|[-1.30,+0.40]|

Both up-only contrasts are below+1pp, satisfying rule3's screening-support
condition for H-0023; H-0027 activation is false. This does not establish
equivalence: the L-56 interval includes effects above1pp. I-G has no+1pp
gain, but G-QN is positive, so rule4 is inconclusive rather than challenging
H-0024; H-0028 activation is false. Both paired contrasts are<=0, satisfying
rule5's challenge condition. Q-N is<=0 against RB1-09, satisfying rule6's
challenge condition. The intervals include zero, so negative point rules
do not establish confirmed harm. X-01 minus Q-N=+1.10pp and G-QN minus
G-T=+0.45pp are descriptive source contrasts only.

Rule7 side-arm differences from RB1-07 are G-T=-2.10pp, G-QN=-1.65pp,
G-N=-0.95pp, Q-N=-1.90pp, P-T=-1.80pp and P-N=-1.00pp. Both replication
lists are empty. No extra seeds, LoRA-GA run, tap/source variant, A2 read,
or official-validation read was submitted.

Implementation rule1 passed in the PBS analysis: complete source-matched
manifests and validators; all eight R1 seed0 runs share order SHA
33759bc20322d0c9981128dd8574438546fc3439194ab32b7c79e35e0f0fc2bf,
13115steps/26230examples/2000069inputtokens; all registered parameters
receive gradients; step0 Base gates and exact final checkpoint reloads pass.
G-T ran on1g and passed the D-0060 same-allocation Base gate; the other
seven ran on2g and passed the pinned Base gate. All three I-G producers
passed their full256-example stored-basis orthogonality and actual-forward
RMS gates. The analysis checks their source/sample/checkpoint bindings.

All curves at3279/6558/9837/final and side step0/final read diagnostics are
validated and retained in analysis.json. Read diagnostics cover the fixed
first64 training-side I-G samples, tasks/layers/scopes and nonempty prefix
bins. Anchor replays ATT-0371/0372 (RES-0345/0346) bind exactly to RB1-09
ATT-0247 and X-01 ATT-0321. X-01 is the valid reference in E-0028 analysis
ATT-0345, whose rule1 passed; RB1-07 is ATT-0245 and Base is ATT-0024.
The known RV-0092 conservative partial-X-01 low is inactive because these
comparisons are complete. Raise it to coordinator if X-01 validity changes.

The CPU-only random initialization repair made the four fresh L-56 V0
candidates share exact initial SHA b70142c4... across1g/2g. Their full4096
calibration rows,1096steps/200229tokens and paired membership passed PBS
planner/lock checks. The prescribed single3e-3 extension was run; its NLL
1.754484 was worse than1e-3's1.584567. The smallest finite LR within1percent
of the minimum is1e-3, locked before formal dev access as ATT-0362/RES-0336,
SHA dcd6488206ce249b5aa4233c1517d1aede7fcda31c34df3927d67f4467975062.
The older mismatched-initialization grid remains diagnostic-only and excluded.
The source passed126 PBS tests, including exact CPU/CUDA initialization for
both up-only ranks; no source changed after that passing review/validation.

| Formal arm | Attempt / result | PBS job | Measured MIG-h |
|---|---|---|---:|
| G-T |ATT-0363-1 /RES-0337|3778633.opbs|1.150000|
| G-QN |ATT-0364-1 /RES-0338|3778634.opbs|1.010556|
| G-N |ATT-0365-1 /RES-0339|3778640.opbs|1.023889|
| Q-N |ATT-0366-1 /RES-0340|3778641.opbs|1.117222|
| P-T |ATT-0367-1 /RES-0341|3778642.opbs|1.095556|
| P-N |ATT-0368-1 /RES-0342|3778643.opbs|1.033056|
| L-56 |ATT-0369-1 /RES-0343|3778644.opbs|0.948889|
| L-48 |ATT-0370-1 /RES-0344|3778645.opbs|0.882778|

The eight formal runs consumed8.261946MIG-h. All35 executed attempts,
including old failures, invalid calibration, source regressions, final
reference reads and10-second PBS analysis, total2.517847node-h against
the3.5cap. Accounting uses measured MIG-h/4 and deduplicates accounting
references; revised hardware records do not add cost twice. The frozen
analysis usage input contains34 attempts totaling2.5171525node-h; the
analysis job adds.002778MIG-h=.0006945node-h. Final cost is recorded in
T-0032-final-cost.json. Steady training steps span.17335–.21602seconds,
and peak allocated CUDA memory spans3.302–3.342billion bytes; full per-arm
runtime, forward/token and parameter accounting is retained in the analysis.

Evidence: EV-0167 implementation; EV-0168 H-0023 screening support;
EV-0169 H-0024 inconclusive; EV-0170 H-0025 screening challenge; EV-0171
H-0026 screening challenge; EV-0172 no replication trigger. No hypothesis
state was changed. Coordinator conclusion review and route decisions remain.

Intervals use paired task-stratified resampling,10000 draws, common indices,
seed20261101. They are descriptive and exclude seed and selection uncertainty.
Other limits: dev1000/task was used for decisions; multiple taps were screened;
LoRA received no LoRA-GA initialization; side eta=.25 and inherited LR=.001
are fixed. Results apply to this frozen0.6B four-layer setting and seed only.
