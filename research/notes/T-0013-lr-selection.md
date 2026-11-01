# V5 LR selection locked before seeds1–3

Current policy D0018/D0019:the original lock is final,with no slice-sensitivity
caveat. The later two completed calibration repeats are plain diagnostics only;
no actual slice comparison or all-2g lock is run. Nothing below is replaced.

E-0013@3. Twelve finite calibration runs used seed0,identical1101 optimizer
steps/2202 presentations/200103 nonpadding input tokens and the locked V1
calibration split. There was no dev or official-validation selection.

| Arm | NLL1e-4 | NLL3e-4 | NLL1e-3 | NLL3e-3 | Locked LR |
| --- | ---: | ---: | ---: | ---: | ---: |
| V5-C | 1.7013356424961341 | 1.6323481719859676 | 1.5977210394597021 | 1.602503256707123 | 0.001 |
| V5-F | 1.65651495608446 | 1.6021932411588116 | 1.5675299750501892 | 1.6106530962317276 | 0.001 |
| V5-L | 1.664488090351574 | 1.5867400373288816 | 1.553388156312454 | 1.6429711629701167 | 0.001 |

The initial raw minimum was1e-3 for each arm,so all required upper endpoint
3e-3 runs were executed once. The existing reviewed `v5_train.py lock` CLI
then chose the smallest LR within1% of the final finite minimum. All3 are1e-3.
No further grid extension is allowed or planned.

Lock was produced in PBS **ATT-0128-1 / 3764966.opbs**,immutable5cad3e,
Exit0,14MIGseconds. Exact file copied without editing to
research/artifacts/T-0013/lr-lock.json;it pins all12 manifest SHA256 values.
Formal training reads this file from its own immutable committed snapshot.

Initial manifests ATT0101..0109 imported RES0107..0115. Endpoint manifests
ATT0118/0119/0120 imported RES0119/0120/0121. Initial report ATT0117/RES0117,
EV0077;Base repeat ATT0113/RES0116,EV0078. Calibration evidence is diagnostic.

Formal cost plan is research/artifacts/T-0013/formal-cost-plan.json. It uses
the slowest measured per-arm stable step across smoke/all4calibrations,and the
larger full-dev estimate from calibration or per-epoch/reload smoke timing;
counts4full dev evaluations per run (3monitors+fresh reload),plus30s setup/save.
Walltime is1.5x expected,rounded to a minute.

| Arm | Expected seconds/run | Requested walltime | Expected node·h/run |
| --- | ---: | --- | ---: |
| V5-C | 6541 | 02:44:00 | 0.454236 |
| V5-F | 6396 | 02:40:00 | 0.444167 |
| V5-L | 13006 | 05:26:00 | 0.903194 |

Nine runs are projected at5.404792 node·h;requested reservation is8.125node·h.
All measured T0013 compute before formal training is8614MIGseconds=
0.59819444node·h,including lock14s. Projected T0013 total before latency is
6.002986node·h,within E0013≤12node·h. These projections are estimates,not usage.
The exact sample budget stays3epochs/29165steps/58329presentations/
4173300input tokens per run for every arm and seed1,2,3.

Code remains byte-identical to RV0033 (training),RV0035 (latency),RV0038 (report).
The coordinator-authorized master merge5cad3e adds accepted T0014 files and
restores clean shared-experiment review coverage. No official validation read.
