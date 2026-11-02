# E-0029 portable up-only initialization repair

Runtime source: dc82cfc6c1f642f9380596a6be3244dafd4147a4.

The first L-56 grid ATT-0349/0350/0351 completed, but the1g candidate had a
different initial parameter digest from the2g candidates despite the same
V0 seed and order. ATT-0353-1 rejected it at the common-initialization gate.
This is an implementation failure, with no LR selected, no scientific claim,
and no new dev read. All three old candidates are excluded, including the
two2g candidates. EV-0164 records this diagnostic.

The pinned PyTorch CUDA DistributionTemplates.h calc_execution_policy uses
the SM count to cap the RNG launch grid; distribution kernels assign RNG
streams by thread/grid. LoRALinear initialized A directly on base.device.
The sealed shared-initialization requirement therefore failed across MIG
shapes. This is distinct from the accepted BF16 numerical differences in
D-0018/D-0019. MSG-0907 explicitly authorizes the repair and narrow new round.

Only E-0029's up-only wrapper changed: UpLoRALinear draws FP32 A on CPU with
kaiming_uniform_(a=sqrt5), then copies it to the base weight device. B stays
zero, scale1 and rank/count/target contracts are unchanged. Existing
LoRALinear and LoRAModel code remains byte-identical to f46a13d. Side-branch
Aq/Ak/Av already draw CPU torch.randn; reader matrices use CPU identity.
No other fresh E-0029 adapter initialization draws randomness on CUDA; no
set_default_device/set_default_tensor_type occurs in src.

RV-0095 is the MSG-0907-authorized new round, linked with --recheck-of RV-0092:
reviewer-2 passed it with zero findings (MSG-0908). Earlier RV-0090/RV-0092,
RV-0094 and D-0064 coverage/exception remain recorded. The known conservative
partial-X-01 low remains accepted and inactive for the valid full-X-01 path.

Source-matched PBS validation:

| Attempt | PBS | Scope | Current evidence |
|---|---|---|---|
| ATT-0354-1 |3778174.opbs|126 tests, full G-T I-G, all8 train-only timings|passed, RES-0330;687s, U-0363=.190833 MIG-h;2g on mg1104 |
| ATT-0355-1 |3778175.opbs|full256 G-QN I-G|passed, RES-0328;57s, U-0361=.015833 MIG-h; runtime1g on mg1100 |
| ATT-0356-1 |3778176.opbs|full256 G-N I-G|passed, RES-0329;60s, U-0362=.016667 MIG-h; runtime2g on mg1104 |

Both completed I-G jobs verified live compute allocation, no dev, zero
optimizer updates, common sample SHA25fe197b...,16766 tokens,512 forwards
and33532 input exposures. All4 layer orthogonality and actual second-forward
RMS gates passed. G-QN max errors: orth1.863e-8, relativeRMS9.237e-8;
G-N: orth3.267e-8, relativeRMS1.417e-7. G stays FP32; SVD is FP64 gesvd.
MCP's generic usage hardware field said2g for G-QN; runtime nvidia-smi and
manifest establish1g. Usage record uses the actual runtime hardware.

The two new tests reseed the same derived V0 init seed before constructing
CPU-base and CUDA-base up-only models, then compare every A/B and the exact
state digest for ranks48/56. They passed in the PBS job.

N-20261102-235003-e198 /T0032-cpu-init-checks covers these three jobs.
ATT-0354 application success and terminal usage are imported, and the measured
projection is updated from its eight timings (all old failed/diagnostic work
counts). Three NEW L-56 candidates1e-4,3e-4,1e-3 on dc82cfc are submitted as
ATT-0357/0358/0359, PBS3778348/3778349/3778351. Notification
N-20261103-000608-66da is sealed3; MSG-0915 sent the listener to coordinator.
Fresh input list: T-0032-up-lr-grid-portable-input.json.
No slice pinning. Require identical initial digest, source, environment,
split, order,1096 steps/2192 examples/200229 tokens and full4096 calibration
membership. Run lr-plan in PBS. Apply only its one prescribed endpoint,
then artifact-bound lr-lock, and only then dev/formal training.

Measured prior cost before these three checks:0.24152725 node-h, including
0.019097 earlier node-h and every recorded failed/diagnostic attempt. The
two terminal I-G checks add0.008125 node-h; main smoke adds0.04770825.
Measured total is0.2973605 node-h. Current-source v3 cost table includes this
total, all8 future formal arms, worst-case4 fresh calibration candidates,
reference diagnostics,180s check/analysis allowance and360s scheduler reserve:
2.8555888954 node-h within3.5 cap. The three current-source I-G initializations
are reused, with no further collection cost forecast. G-T maximum errors:
orth1.080e-8, relativeRMS3.408e-8; all8 finite80-step timings passed.

D-0065 authorizes exactly all8 sealed E-0029@1 formal arms after LR lock.
No new ordinary tap/source variants or follow-up seeds are authorized; report
any rule trigger to the coordinator for a fresh rule10 decision.
