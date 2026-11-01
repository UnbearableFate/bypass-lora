# T-0017 measured smoke and opening cost estimate

ATT-0150-1 / RUN-0150 / RES-0138 used immutable source
7354cdfd97dce795b69cc239c51795533b36bd10, PBS3765791.opbs. The job completed
with Exit0 and every planned stage produced its required output:43 tests
passed in10.66s; identity.json reports identical=true and no differences;
D1/D3/D2a/D2b all completed20 steps and32-dev evaluations, with exact full
records and summaries after adapter save/reload. All arm outputs explicitly
exclude this smoke from scientific evidence. D2a's selected/nonzero-gradient
parameters are A_q/A_k/Rq and the decoder parameters stayed unchanged. D2a/b
record adapter-only D1 provenance and optimizer_state_loaded=false.

The allocated UUID MIG-46edf63e-ba80-5529-8c60-bc2e6d1add55 was matched to
nvidia-smi's MIG2g.24gb line and torch's actual device (32SM), on mg1098_mig0.
The fresh baseline stable step was0.1989532453s; D3 was0.1975220613s, ratio
0.9928064307, passing the registered<=3 cost gate. No confirmation was read.

Terminal observations were imported from
research/artifacts/T-0017/ATT-0150-1-terminal-observations.json. PBS charged
173s =0.048056 MIG-instance hours =0.012014 node hours. Application wall was
171.2635s. Logs and per-arm metrics are in runs/ATT-0150-1 under the project
root; these measured checks were inspected separately from scheduler Exit0.

Opening estimates use each arm's measured stable-step time for6000 steps and
scale its32-example final evaluation and three combined learning-curve
evaluations by1000/32. Final evaluation is counted twice for exact reload;
6s is added for model startup/export overhead. These are estimates, not
measured full-run costs. Different shrink factors for steps and dev size are
explicit here; the cost parts were timed separately in the smoke.

| Arm | Stable s/step | Final eval s/32 | All curve eval s/32 | Estimated full s | PBS request |
|---|---:|---:|---:|---:|---|
| D1 |0.194872801|3.664391|2.927336|1496|00:40:00|
| D3 |0.197522061|4.960789|2.919038|1593|00:40:00|
| D2a |0.207430606|2.774296|3.316478|1528|00:40:00|
| D2b |0.200596204|2.631701|3.073313|1471|00:40:00|

The requests cover approximately1.5 times each estimate, rounded to a ten
minute boundary. Four jobs total about6088 MIG seconds =0.423 node hours;
with the initial smoke, about0.435 node hours. A second bounded smoke of the
coordinator-approved shared hooks and a small statistics job remain within
the task's approximate0.5-node-hour allocation. The full E-0016 ceiling is
0.9 node hours shared with D4/D5; researcher-3 owns those runs and their cost.
Formal jobs wait for code review; D2a/b additionally wait for a complete,
validated final D1 checkpoint. Failed hardware guards are infrastructure
failures; they never trigger a login-node fallback.
