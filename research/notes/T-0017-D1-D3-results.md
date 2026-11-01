# E-0016 seed-0 D1/D3 interim results

Contract E-0016@1, source6da33b39ca749a65140e2dcd0715c8f7b6d3359c,
code review RV-0045 pass. These are per-arm screening point estimates;
hypothesis and route decisions belong to the coordinator. No confirmation
was accessed. D2a/D2b have been released but are not included in this report.

D1 ATT-0154-1 / RES-0154 / PBS3766136.opbs and D3 ATT-0155-1 /
RES-0155 / PBS3766138.opbs both produced complete application manifests,
passed validators,6000 training steps and4,000,470 input tokens. Every final
control has1000 dev predictions; the runner's full per-example records and
summaries are exactly identical after same-run checkpoint save/reload.
Curves at25/50/75% are retained. The locked analysis independently checked
all final record alignment, counts and point summaries: ATT-0161-1 /
RES-0157 / PBS3766677.opbs, application complete, explicitly missing D2a/b.
Reports: /work/xg24i002/x10041/bypass-lora/runs/ATT-0161-1/analysis/report.json
and report.md. Raw artifacts remain under the corresponding attempt roots.

## D1 teacher-forced point gate: pass

| Protocol/control | Correct/1000 | EM | NLL |
|---|---:|---:|---:|
| teacher tf_normal |668|66.8%|1.141156|
| teacher tf_shuffled |13|1.3%|15.625579|
| teacher tf_wrong |8|0.8%|15.593291|
| teacher zero |11|1.1%|9.248186|
| standard normal |23|2.3%|6.920577|
| standard shuffled |15|1.5%|7.905047|
| standard permuted |21|2.1%|6.969748|
| standard oracle_key |70|7.0%|5.190172|
| standard oracle_value |668|66.8%|1.141156|
| E1 normal |288|28.8%|2.725127|
| E1 zero |9|0.9%|9.265308|
| E1 shuffled |9|0.9%|8.518037|

The registered teacher margin is668/1000-max(13/1000,11/1000)=65.5pp,
above the50pp point-estimate gate. This result uses target-slot supervision;
it does not establish unsupervised addressing. Standard all-slot D1 margin
is0.8pp and is descriptive for this stage.

Per500-step mean training answer CE, steps1..6000 in order:
5.327913,4.368536,3.913930,3.341793,2.710771,2.337130,1.929750,
1.697481,1.425270,1.283844,1.115847,1.090441 nat. The final1000-step
mean is1.103144 nat. OLS slope over5001..6000 is-0.0000527962684 nat/step;
the fitted decline over999 steps is0.0527434722 nat, so the registered
plateau label is false (<0.05 rule). The window's LR is0..0.000221688,
at most7.3896% of peak. This schedule-conditioned label does not establish
convergence and is not needed for D1's passing gate.

## D3 hard last-only H3 point gate: not passed

| Hard control | Correct/1000 | EM | NLL |
|---|---:|---:|---:|
| normal |15|1.5%|4.424336|
| zero |11|1.1%|9.248186|
| shuffled |10|1.0%|4.428258|
| permuted |13|1.3%|4.428310|
| oracle_key |16|1.6%|4.424475|
| oracle_value |17|1.7%|4.421358|

Hard EM margin is0.4pp against the10pp point gate. Normal NLL is below both
zero and shuffled, satisfying that strict condition. Last-position target
weight is0.0611445694 against0.125, failing the address condition. Last
argmax counts by public slot are[775,0,0,0,0,0,0,0,0,0,1,11,14,28,72,99].
Mean last-position n_eff is7.892364. This is the newly registered hard,
last-question-position protocol, separate from the standard E-0010 H3.
Soft normal EM/NLL is1.6%/4.425512; marginal normal EM/NLL is0.9%/4.424916.
All soft/marginal controls, position0 versus1+ strata, weights and fusion
norms are retained in the analysis artifacts; no alternative protocol gates
the result. No hypothesis verdict is recorded here.

## Usage and continuation

Measured PBS usage, imported once from terminal observations:

| Attempt | Wall s | MIG-instance hours | Node hours |
|---|---:|---:|---:|
| D1 ATT-0154-1 |1632|0.453333|0.11333325|
| D3 ATT-0155-1 |1707|0.474167|0.11854175|
| analysis ATT-0161-1 |3|0.000833|0.00020825|

Including the two author smokes, T-0017 usage so far is0.25604175 node
hours. Application wall differs from charged PBS wall and is retained in
each manifest. Both formal jobs recorded2g.24gb on mg1109 as metadata only,
per D-0018/D-0019; no slice-related rerun, analysis or flag is made.

The separate analysis preflight verified D1's final adapter inventory:
adapter.pt SHA2566209268349661db898d560097ec3ff0690f0d570e29e26710e98eb217106703b,
2,230,997 bytes; config.json SHA256fa2b3c9bb1ddae1fb548add4c32502fc99f3b458dbe58bd8de1bd57e60941436,
454 bytes. The source manifest and metrics hashes are retained in
runs/ATT-0161-1/D1-resume-provenance.json. Source parent is the completed
6000-step D1, full-record reload passed, adapter-only, no optimizer state.

D2a ATT-0162-1 / PBS3766763.opbs and D2b ATT-0163-1 / PBS3766766.opbs
were submitted after this preflight, from the same reviewed6da33b3 source,
each6000x8 with fresh AdamW,300-step warmup and separate derived arm order.
D2a trains A_q/A_k/Rq only; D2b trains all adapter parameters. Each requests
40min based on measured smoke costs. A job-shell wrapper changes only the
existing guard to record-only under D-0019; it preserves the reviewed
training/evaluation launcher and all src files. Submission records cite
D-0016 for the inherited stale label, which now also lists absent T-0018
files; none are present in or imported by the executed snapshot.

Shared notification N-20261101-234440-f651 covers both D2 jobs. On completion,
validate full controls/E1/reload, D1 provenance and D2a frozen parameters,
import results/usage, then run the locked full four-arm analysis. The final
task delivery remains pending those continuations. Bootstrap intervals are
descriptive data uncertainty conditional on this one trained seed and do
not enter any point gate.
