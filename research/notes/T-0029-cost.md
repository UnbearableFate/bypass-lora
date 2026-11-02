# T-0029 / E-0026 smoke cost update (before formal training)

Smoke ATT-0289-1 (PBS 3771956, source 44f880c, MIG 2g.24gb, 224 s wall,
0.062222 MIG h = 0.015556 node·h). Status failed only because of one unit test
(an error-message wording change, fixed for the recheck); registration and all
three smoke trainings completed with reload.identical=true.

Implementation registration (width-registration.json): W-01 = W-02 = 2,752,512
stored/trainable/with-grad; LoRA all-7 four-layer counts r28..34 are
90,112 x r; nearest to W-02 is r31 = 2,793,472 (r30: 2,703,360). W-01 and W-02
initial parameters are elementwise identical (24 tensors).

| Arm | Steady step s (80-step smoke) | Peak train CUDA GB |
| --- | ---: | ---: |
| W-01 | 0.2249 | 2.76 |
| W-02 | 0.2274 | 2.90 |
| W-03 | 0.2269 | 2.86 |

r32 E-0021 reference: RB1-02 0.205 s/step, 3.10 GiB, full dev eval 180 s.

Projection per formal arm: 13,115 x 0.227 = 2,980 s train + six full dev
evaluations (step 0, 25/50/75%, final, reload) ~1,080 s + load/save ~ 4,100 s
= 1.14 MIG h = 0.285 node·h. Three arms 0.86 node·h. W-02 interventions
(normal, same-allocation Base, iv_off, iv_swap with donor forwards,
iv_permute) ~1,400 s = 0.10 node·h. Smoke, recheck tests and analysis ~0.04.
Projected total ~1.0 node·h <= 1.5 node·h budget; no contract revision needed.
Walltime request: training 01:45:00 (short-mig), intervention 00:45:00.
