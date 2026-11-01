# Final-source E-0016 validation before formal training

Source6da33b39ca749a65140e2dcd0715c8f7b6d3359c adds MSG-0430 support-builder
and D4/D5 scheduling hooks. ATT-0152-1 / RUN-0152 / RES-0140 ran the full
pre-review validation in PBS3765958.opbs. Root manifest status is complete,
validator passed, failure null. This note records engineering results only;
the20-step,32-dev arm outputs are explicitly ineligible as scientific evidence.

Evidence directory: /work/xg24i002/x10041/bypass-lora/runs/ATT-0152-1.
The tests.xml/log records50 passed, no errors/failures/skips,11.48s.
identity.json reports identical=true, differences={} for the frozen e1660c9
reference versus the candidate. The new tests separately verify omitted and
explicit support builders preserve exact standard logits/loss/gradients,
full evaluation records/summaries, and recipient plus global shuffled donors;
non-default returned support is consumed by training and evaluation. Schedule
tests cover D4's released order and D5's separate derived-seed namespace.

hardware.json matches CUDA_VISIBLE_DEVICES to UUID
MIG-545f0a46-f182-5af2-b861-afeae7a5fbab in nvidia-smi.txt, profile2g.24gb,
one torch device,32SM, on physical host mg1098 (ledger slice mg1098_mig2).
The live allocation check passed before application execution. Control work
remained on the verified miyabi-g2 login shell; no workload ran there.

All four arm manifests are complete with actual20 steps,32-dev evaluation,
eligible_as_evidence=false. All full per-example records and summaries match
after adapter save/reload for every arm protocol, including D1/D2 E1. D2a
requires-grad and nonzero-gradient names are exactly A_k/A_q/Rq; runtime also
asserts frozen decoder parameters unchanged. D2a/b source checkpoint paths
and inventory hashes bind to this job's D1 adapter, record adapter_only=true
and optimizer_state_loaded=false. No confirmation was read.

Fresh baseline stable step0.2037143974s; D3 stable step0.1931249339s;
ratio0.9480180902 satisfies the<=3 registered gate. D1 stable step was
0.2082788414s, D2a0.1796063309s and D2b0.1845153145s. The longest observed
per-arm component times across this smoke and ATT-0150-1 give estimated
full D1~1602s, D3~1597s, D2a~1528s, D2b~1471s, including twice the final
evaluation, all curve evaluations and6s startup/export allowance. These are
estimates. Use50min for D1 and40min for each other formal job, based on~1.5x
and ten-minute rounding. Total formal estimate is about0.430 node hours.

Measured PBS usage172s=0.047778 MIG-instance hours=0.0119445 node hours
was imported once from terminal observations; application wall170.5063s.
Together with ATT-0150-1, smoke usage is0.0239585 node hours. Observations
are research/artifacts/T-0017/ATT-0152-1-terminal-observations.json in the
project root. Both result and usage are recorded, not inferred from Exit0.

RV-0045 covers the source and shared hooks. Reviewer-1 received RES-0140
and the evidence paths in MSG-0444. Reviewed files remain frozen until its
answer. Formal D1/D3 require passing review; D2a/b also require validated
final D1 provenance. No formal run has started at this note's creation.
