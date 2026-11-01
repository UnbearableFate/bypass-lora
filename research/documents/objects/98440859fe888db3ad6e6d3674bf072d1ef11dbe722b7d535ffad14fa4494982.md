# T-0003 submission cost plan

Administrative resource sizing from ATT-0022-1 measured timings; no new scientific analysis. Code and reviewed release remain at 1fe7d3da9c7dc936ecad99d08f832015e1988b91, RV-0008 pass. One MIG instance per independent arm. Expected seconds = ceil(13077 * measured steady step seconds + 5 * max(mean intermediate dev seconds, final dev seconds, reload seconds) * 1000/32 + 30). The 30 seconds covers load, tokenization, save and process setup. All five evaluations are dev-only; source timings measured on each arm. Request = expected *1.5 rounded up to one minute. This replaces T-0002's unmeasured LoRA28 bound with its measured .435291s/step.

| Arm | Expected seconds | Requested walltime | Expected MIG-h | Expected node-h |
| --- | ---: | --- | ---: | ---: |
| V1-01 F_out | 3483 | 01:28:00 | 0.967500 | 0.241875 |
| V1-02 F_pre | 3864 | 01:37:00 | 1.073333 | 0.268333 |
| V1-03 C_gate | 3596 | 01:30:00 | 0.998889 | 0.249722 |
| V1-04 C_up | 3514 | 01:28:00 | 0.976111 | 0.244028 |
| V1-05 C_dual | 3670 | 01:32:00 | 1.019444 | 0.254861 |
| V1-06 linear | 3699 | 01:33:00 | 1.027500 | 0.256875 |
| V1-07 lora | 3781 | 01:35:00 | 1.050278 | 0.262569 |
| V1-08 lora28 | 7062 | 02:57:00 | 1.961667 | 0.490417 |

Eight arms expected 2.268681 node-h; total requested 3.416667 node-h. Full Base dev-only evaluation/reload budget uses prior full Base E-0002 walltime853s as conservative proxy (it included calibration too): request ceil(853*1.5/60)=22min, .091667 node-h reservation. Existing E3 validation used about .034097 node-h. Even total requested plus Base and prior validation is below10node-h; project total remains far below200node-h. MIG instance-h always reported separately; multiply by .25 for node-h. Use short-mig for arms and debug-mig for Base. Queues' own limits and2026-11-25 maintenance remain respected.

Base command uses --arm base --e3 --tokens 2000000 --batch 2, no eval/smoke limits; actual training remains zero. Formal arms use --e3 --tokens 2000000 --batch 2 --lr .0003, no limits; all registrations must show E-0003@2 and RV-0008 coverage. The eight training jobs share one expected8 notification; Base has its own expected1 notification. Coordinator runs returned listeners. No duplicate watcher or Codex Goal is created.
