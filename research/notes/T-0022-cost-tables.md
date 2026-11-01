# RB0 measured timing and conditional costs

Reference scan is a diagnostic comparator and is excluded from the RB1 sum. Node_h uses measured/predicted one-MIG seconds divided by14400.

| Case | Trainable/stored/with-grad | Step s | Tokens/s | Peak CUDA GiB | RB5 3epochs node_h |
| --- | ---: | ---: | ---: | ---: | ---: |
| RB1-01 | 663552 | 0.217694 | 843.6 | 2.532 | 0.457376 |
| RB1-02 | 663552 | 0.214969 | 854.3 | 2.540 | 0.455055 |
| RB1-03 | 663552 | 0.225660 | 813.8 | 2.541 | 0.476262 |
| RB1-04 | 663552 | 0.243274 | 754.9 | 2.544 | 0.512225 |
| RB1-05 | 663552 | 0.233511 | 786.4 | 2.541 | 0.493873 |
| RB1-06 | 663552 | 0.243832 | 753.1 | 2.544 | 0.514882 |
| RB1-07 | 630784 | 0.216349 | 848.8 | 2.623 | 0.454587 |
| RB1-08 | 1318912 | 0.217286 | 845.2 | 2.563 | 0.460509 |
| RB1-09 | 925696 | 0.214911 | 854.5 | 2.546 | 0.456602 |
| RB1-02-reference | 663552 | 2.157638 | 85.1 | 2.554 | 4.489551 |
| Cdual32-S4 | 1318912 | 0.243899 | 752.9 | 2.568 | 0.514800 |
| Cup32-S4 | 925696 | 0.228508 | 803.6 | 2.550 | 0.484653 |
| T0-R32-S4 | 663552 | 0.210353 | 873.0 | 2.536 | 0.445365 |
| T0-Cdual32-S4 | 1318912 | 0.212420 | 864.5 | 2.556 | 0.449226 |
| T0-Cup32-S4 | 925696 | 0.211840 | 866.9 | 2.543 | 0.447745 |
| R44-S1 | 916608 | 0.203823 | 901.0 | 2.551 | 0.432034 |
| R44-S4 | 916608 | 0.231833 | 792.1 | 2.557 | 0.489007 |
| R64-S1 | 1343488 | 0.203864 | 900.8 | 2.572 | 0.432302 |
| R64-S4 | 1343488 | 0.234739 | 782.3 | 2.581 | 0.497194 |
| LoRA28-r1 | 630784 | 0.371851 | 493.9 | 3.404 | 0.758830 |
| LoRA28-r2 | 1261568 | 0.398286 | 461.1 | 3.414 | 0.809076 |

RB1 arms:training,three intermediate dev passes,final dev and fresh reload evaluation.

| Arm | Estimated node_h |
| --- | ---: |
| RB1-01 | 0.260091 |
| RB1-02 | 0.260239 |
| RB1-03 | 0.271757 |
| RB1-04 | 0.291758 |
| RB1-05 | 0.282254 |
| RB1-06 | 0.293909 |
| RB1-07 | 0.258521 |
| RB1-08 | 0.263593 |
| RB1-09 | 0.261838 |

RB1 core total: 2.443961 node_h. Frozen interventions,analysis and planning overhead are additional.

| RBc family | Estimated node_h/candidate |
| --- | ---: |
| RBc-L | 0.066836 |
| RBc-T | 0.067244 |
| RBc-R | 0.069168 |
| RBc-C | 0.070363 |

RBc12-16candidate estimate: 0.820830-1.094440 node_h.

| Latency case | Prefill tok/s | Decode p50 ms | Decode p95 ms | Peak CUDA GiB | Persistent adapter M+c bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base | 12253.7 | 38.582 | 39.513 | 1.382 | 0 |
| RB1-01 | 11579.5 | 40.372 | 41.233 | 1.387 | 0 |
| RB1-02 | 10833.1 | 43.378 | 44.458 | 1.387 | 16896 |
| RB1-04 | 9940.0 | 47.564 | 48.328 | 1.387 | 16896 |
| RB1-07 | 11533.2 | 40.828 | 41.422 | 1.387 | 0 |
| RB1-07-merged | 12258.5 | 37.520 | 39.307 | 1.382 | 0 |
| RB1-08 | 10383.4 | 45.674 | 46.548 | 1.392 | 16896 |

Complete R_resS1 dev pass: 185.524s. All21training cases have100stable steps;all7latency cases have512token prefill and128decode samples.
Measured T0022 usage through timing:1755MIGseconds=0.121875node_h. This tabulation job is additional until its scheduler end.

Estimates omit future frozen interventions,bootstrap/analysis and data planning overhead;they are not billed usage.
Only R_res S1 full dev was timed;other evals scale32rows/task against that full/sample ratio;RBc also scales row count,not measured calibration lengths.
RB5 matrix/monitoring remains conditional;S2 unmeasured layouts need labeled interpolation;S1concatT0 and mixed-rankLoRA are not measured here.
Latency uses synthetic512tokens and128fixed decoder steps;mergedLoRA timing only,unmerged adapters for task scoring.
D0026/D0027 preserve0.50002;reference-matched BF16shape-change exceedances remain diagnostics,decisiveFP32 and same-shapeBF16 gates pass.
