# V5 descriptive cost tables

Sources and full precision values are in cost-summary.json. All nine training runs used LR0.001, exactly three epochs and the final checkpoint. Official validation was not accessed.

| Arm | Seed | PBS MIG s | Train s | Step ms | Peak CUDA GiB | Load s | Monitor s | Save s | Reload load/eval s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V5-C | 1 | 5961 | 5293.83 | 181.427 | 3.071 | 1.356 | 483.91 | 0.0229 | 0.477 / 157.15 |
| V5-C | 2 | 6177 | 5498.74 | 188.456 | 3.077 | 1.395 | 498.64 | 0.0072 | 0.262 / 155.38 |
| V5-C | 3 | 6023 | 5370.17 | 184.047 | 3.072 | 1.376 | 467.99 | 0.0075 | 0.262 / 160.01 |
| V5-F | 1 | 6287 | 5587.52 | 191.502 | 3.069 | 1.427 | 510.30 | 0.0076 | 0.260 / 165.24 |
| V5-F | 2 | 6002 | 5348.12 | 183.295 | 3.075 | 1.324 | 471.88 | 0.0362 | 0.453 / 156.18 |
| V5-F | 3 | 5680 | 5061.46 | 173.463 | 3.076 | 0.855 | 445.47 | 0.0296 | 0.523 / 147.43 |
| V5-L | 1 | 11945 | 10992.86 | 376.850 | 3.976 | 0.612 | 697.97 | 0.0258 | 0.500 / 235.29 |
| V5-L | 2 | 13178 | 12073.92 | 413.923 | 3.981 | 0.597 | 812.26 | 0.0233 | 0.305 / 272.17 |
| V5-L | 3 | 12104 | 11127.22 | 381.448 | 3.985 | 1.178 | 722.49 | 0.0388 | 0.492 / 228.10 |

The step column reports the trainer's train.steady_step_seconds metric. Peak CUDA is the trainer's allocation metric, not total device memory. PBS walltime includes runtime overhead; only that measured walltime contributes to node_h. Hardware strings are retained as metadata in the JSON.

| Model | Timings | Mean ms | Median ms | p95 ms |
| --- | ---: | ---: | ---: | ---: |
| base | 768 | 36.651 | 35.837 | 42.621 |
| V5-C | 768 | 37.803 | 37.782 | 38.541 |
| V5-F | 768 | 38.090 | 38.070 | 39.005 |
| V5-L | 768 | 55.761 | 55.736 | 56.867 |
| V5-L-merged | 768 | 35.930 | 35.916 | 36.759 |

batch1, first64 dev rows per task, both candidates, 16 warmups, 3 timed passes; synchronized host forward walltime.

BF16 merged weights incur one rounding; timing only, unmerged checkpoint used for task scoring.
Merged LoRA was used only for timing. The first-input maximum logit difference is retained as metadata.

Formal training: 73357 MIG s = 5.09423611 node_h.
Other T0013 work before latency: 10147 MIG s. Latency: 179 MIG s.
Total measured through latency: 83683 MIG s = 5.81131944 node_h. Add this tabulation job after its scheduler end. E0013 ceiling:12 node_h.
