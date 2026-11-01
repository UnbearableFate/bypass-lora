# T-0025 delivery: E-0021 RB1 screening

The nine-arm screen, frozen-history grid, timing and locked analysis are complete.
Raw gates are `G0=true, GR=false, GH=false, GL=false, GC=false`.
Resolved gates are `G0=true, GR=false, GL=false, GC=false`.
S_F*=S2/RB1-03 and C*_RB=C_up/RB1-09. Rule 9 applies; no RB5 matrix
or C-S1/S4 pilot is released by these rules. No rule-9 probe was started.
Hypothesis statuses remain unchanged; the coordinator owns conclusion review
and the next route decision.

The reviewed implementation provides rb1_intervene.py for the frozen grid,
rb_read_diagnostics.py for normalized native read summaries, rb1_analysis.py
for authenticated inputs/exact rules/paired bootstrap/costs, and rb1_latency.py
for the prescribed timing protocol. Artifact binding and pure gate reducers
are in rb1_artifacts.py and rb1_rules.py. The observer defaults to off; the
PBS regressions verify unchanged default outputs and training gradients.

The final compute analysis is ATT-0267-1 / RES-0248, PBS 3769776 on mg1098,
complete/passed in 18 scheduler seconds. It used immutable reviewed source
ffa4c7124939bbd82b1319f56be8e27913e0ff59 (RV-0069), the accepted training,
lock and Base pins, and 10,000 paired task-stratified bootstrap samples with
default_rng(20261101), shared indices across every arm/condition. Full tables,
per-task intervals, both read scopes and provenance are in
[runs/ATT-0267-1/analysis.md](/work/xg24i002/x10041/bypass-lora/runs/ATT-0267-1/analysis.md) and
[analysis.json](/work/xg24i002/x10041/bypass-lora/runs/ATT-0267-1/analysis.json).
The immutable analysis document is DOC-0263@1.

## Main table and paired training

Accuracy is percent; NLL is gold-answer token NLL within each task.

| Arm/layout | PIQA % | BoolQ % | Macro % | PIQA NLL | BoolQ NLL |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base  | 72.30 | 68.00 | 70.15 | 2.750282 | 1.924674 |
| RB1-01 R_res T0 S1 | 73.30 | 81.30 | 77.30 | 2.564870 | 0.462577 |
| RB1-02 R_res KM S1 | 72.40 | 81.80 | 77.10 | 2.560449 | 0.477105 |
| RB1-03 R_res KM S2 | 72.70 | 82.20 | 77.45 | 2.562397 | 0.495133 |
| RB1-04 R_res KM S4 | 72.80 | 81.00 | 76.90 | 2.561025 | 0.538533 |
| RB1-05 S4 fixed first query | 72.90 | 81.30 | 77.10 | 2.562617 | 0.533269 |
| RB1-06 S4 eta/4 | 72.70 | 80.30 | 76.50 | 2.560882 | 0.537300 |
| RB1-07 LoRA4 all7 r7 | 74.40 | 83.50 | 78.95 | 2.530501 | 0.584030 |
| RB1-08 C_dual KM S1 | 73.30 | 81.60 | 77.45 | 2.548022 | 0.573512 |
| RB1-09 C_up KM S1 | 73.20 | 81.80 | 77.50 | 2.558323 | 0.555336 |

All nine runs used seed0, the same R1 initialization seed/training order and first whole step
at or above 2M nonpadding input tokens: 13,115 steps, 26,230 examples and
2,000,069 input tokens. The paired order SHA256 is
`33759bc20322d0c9981128dd8574438546fc3439194ab32b7c79e35e0f0fc2bf`.
Each has full 1,000 dev rows/task, three monitor files and final checkpoint
inventory/reload.identical=true. The final checkpoint was used throughout.
Training source is 7bf091d4af24c2a9fdc666a5f885c31817b97174.
Accepted LR lock SHA256 is
`38b8ee24af3ded8aee398a4cc3dfaa5a1dfece4737a6cde34642fbff7bf1012d`;
its 16 calibration input manifests are authenticated and must be retained.
T/R LR=.003, C=.001, L=.0003; T is the calibration grid boundary. No retuning.
A2 confirmation IDs were excluded from training; no confirmation was evaluated.

The loss is per-example mean answer CE, effective batch2/microbatch1,
AdamW(.9,.999), eps1e-8, decay.01, clip1, dropout0, 656 warmup steps then
cosine. Frozen backbone BF16, side parameters/state FP32, eager attention.
Base reuses accepted ATT-0024-1/EV-0022 with pinned manifest SHA
`076a1a0595de6620508a7e99810a96baf979483d805428465807d40beadde196`.

Macro dev curves below correspond to steps 3279/6558/9837/13115. Per-task
accuracy/NLL and their row-level sources are in analysis.json; no best-checkpoint
selection was performed.

| Arm | 25% | 50% | 75% | Final |
| --- | ---: | ---: | ---: | ---: |
| RB1-01 | 74.95 | 75.70 | 77.65 | 77.30 |
| RB1-02 | 71.55 | 75.30 | 76.65 | 77.10 |
| RB1-03 | 69.90 | 75.05 | 76.05 | 77.45 |
| RB1-04 | 73.35 | 75.05 | 76.25 | 76.90 |
| RB1-05 | 69.40 | 75.75 | 76.50 | 77.10 |
| RB1-06 | 71.10 | 76.15 | 76.30 | 76.50 |
| RB1-07 | 76.45 | 76.90 | 78.45 | 78.95 |
| RB1-08 | 74.90 | 76.95 | 77.20 | 77.45 |
| RB1-09 | 74.80 | 76.65 | 77.45 | 77.50 |

## Exact rules 1-11 and evidence

Point gates use exact correctness fractions. Intervals below are descriptive
95% paired row-bootstrap intervals in pp.

| Rule | Recorded outcome |
| --- | --- |
| 1 G0 | Pass: LoRA minus Base +8.80 pp. |
| 2 Effective | All nine arms are at least Base+1 pp. |
| 3 GR | Fail: RB1-02 minus Base +6.95 [5.25,8.70], but minus LoRA -1.85 [-3.00,-0.70], below the -1 pp allowance. PIQA/BoolQ differences are -2.00/-1.70 pp. |
| 4 GH | Fail: memory minus T0 -0.20 [-1.30,0.90]. Swap/permutation drops are +1.85 [0.75,3.00] / +17.15 [14.70,19.60]; the required trained-arm T0 advantage is absent. |
| 5 GL | Fail: S2 chosen (S2-S4=+0.55), selected minus S1 +0.35 [-0.55,1.30]. S4-fixed_query=-0.20 [-1.10,0.70]; eta/4-S1=-0.60; S4-eta/4=+0.40. The extra S4 fixed-query gate is not required for selected S2. |
| 6 GC | Raw=false, resolved=false: best concat minus residual +0.40 [-0.70,1.50]. Both concat arms are effective and within the strict <1 pp near-best band; fewer parameters chooses C_up. |
| 7 RB5 | No release and matrix=null. |
| 8 RB1p | Raw/resolved trigger=false; no pilot. |
| 9 Diagnostic route | Trigger=true, T0 effective=true. Frozen analyzer leaves gradient_started=null because it does not ingest prior diagnostic test records; the records are supplied below for coordinator assessment. |
| 10 Base band | False under D-0032: Base-1 <= arm < Base+1; effective exact+1 arms are outside the band, and this rule never overrides gates. |
| 11 Evidence | EV-0118 challenges H-0013; EV-0119 challenges H-0014; EV-0120 (H-0015) and EV-0121 (H-0016) are descriptive, without supports/challenges edges. All four are screening/valid. |

This is one trained seed and reused dev selection. Intervals describe row
uncertainty conditional on the trained models, exclude training-seed and
selection uncertainty, and are not simultaneous guarantees. A failed +1 pp
point gate or a rule-11 challenge does not establish equivalence or absence
of every smaller effect. GR is a progression gate; H-0013 has the stronger
+1 pp advantage and per-task -1 pp point conditions.

## Frozen history and read diagnostics

All 35 condition bundles have 1,000 paired rows/task: normal plus independently
unwrapped same-allocation Base for each of five checkpoints, six interventions
for RB1-02/03/08/09, and swap only for T0. Off exactly matches same-allocation
Base for all four memory arms; T0 swap exactly matches normal. All five normal
runs reproduce the same IDs, predictions and candidate scores as their respective training outputs.
Checkpoint paths/hashes, native first/last read grids, candidate pairing and
the supplied latency bundle passed the final analysis checks.

Differences are intervention minus that run's normal, in pp with descriptive
95% intervals. Original attention/KV remain intact; four side layers are
intervened together. Donors use normal same-checkpoint forwards, within-task
derangement and their own causal past at the recipient position.

| Arm | Intervention | Macro difference [95% interval], pp |
| --- | --- | ---: |
| RB1-01 | iv_swap | +0.00 [+0.00, +0.00] |
| RB1-02 | iv_off | -6.95 [-8.70, -5.25] |
| RB1-02 | iv_window1 | -30.05 [-32.90, -27.15] |
| RB1-02 | iv_window8 | -14.00 [-16.40, -11.60] |
| RB1-02 | iv_strict | +0.05 [-0.60, +0.70] |
| RB1-02 | iv_swap | -1.85 [-3.00, -0.75] |
| RB1-02 | iv_permute | -17.15 [-19.60, -14.70] |
| RB1-03 | iv_off | -7.30 [-9.05, -5.55] |
| RB1-03 | iv_window1 | -31.65 [-34.45, -28.75] |
| RB1-03 | iv_window8 | -18.75 [-21.30, -16.20] |
| RB1-03 | iv_strict | +0.15 [-0.40, +0.70] |
| RB1-03 | iv_swap | -1.50 [-2.60, -0.45] |
| RB1-03 | iv_permute | -23.45 [-26.10, -20.75] |
| RB1-08 | iv_off | -7.30 [-9.10, -5.50] |
| RB1-08 | iv_window1 | -4.55 [-6.20, -2.90] |
| RB1-08 | iv_window8 | -1.40 [-2.50, -0.30] |
| RB1-08 | iv_strict | +0.15 [-0.35, +0.65] |
| RB1-08 | iv_swap | -1.30 [-2.45, -0.15] |
| RB1-08 | iv_permute | -0.35 [-1.45, +0.70] |
| RB1-09 | iv_off | -7.35 [-9.15, -5.60] |
| RB1-09 | iv_window1 | -2.90 [-4.35, -1.45] |
| RB1-09 | iv_window8 | -1.10 [-2.15, -0.10] |
| RB1-09 | iv_strict | +0.45 [-0.05, +1.00] |
| RB1-09 | iv_swap | -1.95 [-3.15, -0.75] |
| RB1-09 | iv_permute | -0.65 [-1.50, +0.20] |

Large window/permutation losses show sensitivity to these frozen interventions.
They do not replace the failed trained T0 comparison or rule out distribution
shift. Q/K/V already carry original-model context. T0 is S1 only, so RB1 cannot
exclude a same-depth tokenwise explanation for deeper loops.

For rule 9, RB1-02 normal answer-predicting read summaries are:

| Task | Layer | Query count | Current-token weight | Relative n_eff | n_eff | Entropy (nat) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| boolq | 6 | 1000 | 0.010800 | 0.752721 | 110.999 | 4.734 |
| boolq | 13 | 1000 | 0.005080 | 0.035763 | 4.831 | 2.910 |
| boolq | 20 | 1000 | 0.002808 | 0.022651 | 3.120 | 2.651 |
| boolq | 27 | 1000 | 0.000576 | 0.009989 | 1.189 | 0.682 |
| piqa | 6 | 23183 | 0.007409 | 0.079527 | 1.731 | 0.961 |
| piqa | 13 | 23183 | 0.006523 | 0.066200 | 1.785 | 1.266 |
| piqa | 20 | 23183 | 0.005520 | 0.054857 | 1.523 | 1.062 |
| piqa | 27 | 23183 | 0.001452 | 0.043613 | 1.114 | 0.376 |

These are gold-candidate answer-predicting positions, with padding excluded and
zero empty reads. Relative n_eff is the mean per-read n_eff/readable-count,
not a ratio of two aggregate means. All-valid-input summaries, actual first/last
native queries for S2, and strict-mode empty/nonempty denominator conventions
are retained in analysis.json. The broad BoolQ layer6 answer read (relative
n_eff=.753) differs from later BoolQ layers (.036/.023/.010) and PIQA layers
(.080/.066/.055/.044). Inference from these diagnostics: low current-token
weight alone does not make all layers diffuse; the proposed selectivity
explanation is layer/task dependent and has no pre-fixed numeric cutoff here.
No cross-line numeric pooling or new selectivity probe was performed.

The T0 positive control is valid: RB1-01 macro77.30%, Base70.15%,
+7.15 [5.40,8.90] pp, full budget/finite steps/reload and exact swap no-op.
Gradient-start provenance is ATT-0197-1/RES-0191, the passed GPU test
`test_four_layer_two_step_gradient_and_frozen_Jacobian` for RB1-02/04/08 in
[runs/ATT-0197-1/checks/checks.xml](/work/xg24i002/x10041/bypass-lora/runs/ATT-0197-1/checks/checks.xml).
At all four layers, the first backward has nonzero fusion-exit gradients and
zero internal projection/read gradients; after one update all are finite and
nonzero, while the frozen backbone remains unchanged with no gradients.
For RB1-02 layer6, step1 fusion.P norm=.24857448; step2
Aq/Ak/Av/Rq/Ro/P norms=.00955959/.01250599/.03194028/.00115041/.00528123/.25091928.
These are implementation-test records, not per-family telemetry from the
formal run. Formal RB1-02 steps1/2/3 record finite aggregate preclip norms
1.34958196/1.23721278/2.22363377; T0 records
2.05317116/1.91714823/3.55561113. The final formal manifests have all 663,552
R/T parameters with gradients. Coordinator decides whether this satisfies
the rule-9 prerequisites and whether the one allowed probe should be used.

## Parameters, operations and measured timing

Stored/trainable/with-gradient adapter counts agree for each arm.

| Arm | Each parameter count | M+c bytes at batch1 | Training seconds | Peak training GiB |
| --- | ---: | ---: | ---: | ---: |
| RB1-01 | 663,552 | 0 | 2613.871 | 3.098 |
| RB1-02 | 663,552 | 16,896 | 2687.756 | 3.102 |
| RB1-03 | 663,552 | 16,896 | 2923.790 | 3.104 |
| RB1-04 | 663,552 | 16,896 | 3281.665 | 3.107 |
| RB1-05 | 663,552 | 16,896 | 2955.337 | 3.104 |
| RB1-06 | 663,552 | 16,896 | 3221.561 | 3.107 |
| RB1-07 | 630,784 | 0 | 2981.770 | 3.189 |
| RB1-08 | 1,318,912 | 16,896 | 2809.282 | 3.126 |
| RB1-09 | 925,696 | 16,896 | 2796.442 | 3.113 |

M+c bytes exclude original backbone KV. KernelMatrix has 4,224 FP32 bytes/layer
at r32, hence 16,896 across four layers, plus 32 count bytes separately.
Each memory layer writes once/valid token and reads S times: 2,000,069 writes,
and 2,000,069 / 4,000,138 / 8,000,276 reads for S1/S2/S4. Each original adapted
attention/FFN executes once/example (26,230 each/layer). T0 has no history
writes/matrix reads. All operation accounting was validated against the plan.

The analytical dense-side MAC components per token/layer are retained in JSON:
projection131072; fusion32768 (R/T),98304(C_up),196608(C_dual);
loop query/output2048*S; memory read/denominator1056*S plus write1024.
They exclude normalization, nonlinearities, additions, backbone, prefix-scan
additions and diagnostics. They are operation accounting, not measured speed.

ATT-0261-1/RES-0241 measured the fixed latency protocol on verified 2g:
batch1,prefill512 (4 warmup+5 raw timed passes),128 raw decode steps after
separate warmup, p50/p95 and peak memory. P95 uses the fixed lower order statistic.
All raw samples, checkpoint inventory bindings and state counts are retained.

| Arm | Prefill tokens/s | Decode p50 ms | Decode p95 ms |
| --- | ---: | ---: | ---: |
| Base | 11909.664 | 40.090 | 40.932 |
| T0 RB1-01 | 11034.692 | 43.747 | 45.266 |
| R S1 RB1-02 | 10317.846 | 46.729 | 47.613 |
| R S4 RB1-04 | 9300.631 | 51.257 | 52.131 |
| C_dual RB1-08 | 10158.465 | 46.953 | 47.734 |
| LoRA RB1-07 | 10935.569 | 43.231 | 44.106 |
| Merged LoRA reference | 11423.604 | 39.948 | 40.748 |

Quality uses unmerged LoRA. Selected S2 and C_up are not separately measured
by the fixed latency protocol. Timing is one allocation and descriptive.
All ordinary train/eval profiles are metadata only under D-0018/D-0019; only
the cost/latency timing jobs required2g. No slice-related reruns or probes.

## Records, validation and cost

| Runs | Results | Purpose |
| --- | --- | --- |
| ATT-0239-1..0247-1 | RES-0228..0236 | Nine final trained arms (arm order01..09). |
| ATT-0258-1 | RES-0237 | Complete nine-arm audit and locked selection. |
| ATT-0260-1 | RES-0242 | All-five 32dev/task comparative cost smoke, no retained quality scores. |
| ATT-0261-1 | RES-0241 | Prescribed latency plus Base and merged LoRA. |
| ATT-0262-1/0263-1/0264-1/0265-1/0266-1 | RES-0243/0244/0245/0246/0247 | Frozen02/03/08/09/T0 grid. |
| ATT-0267-1 | RES-0248 | Full validation, exact rules and locked10000 bootstrap. |
| ATT-0228-1/0230-1/0248-1/0249-1/0250-1 | RES-0207/0208/0217/0218/0222 | Phase checks, fixes/recheck and restored LR regression. |

RV-0064 passed with four lows; RV-0069 passed after all four fixes, with a new
coverage-only low resolved by the restored LR-family regression. ATT-0249-1
has346passed/0failed/0skipped; ATT-0250-1 has16passed and confirmed the production
file hashes equal reviewed ffa4c71. No production source was changed for the
formal grid. The two failed check attempts are retained engineering history.
Neither reviewer executed additional jobs; their reused diagnostic costs are
already in the table below. PBS and application success were checked separately.

| Cost class | MIG seconds | MIG hours | node hours |
| --- | ---: | ---: | ---: |
| Five check attempts including failures | 252 | .070000 | .017500 |
| Nine training/monitor/reload runs | 35018 | 9.727222 | 2.431806 |
| Audit/selection | 17 | .004722 | .001181 |
| Intervention cost smoke | 335 | .093056 | .023264 |
| Latency | 59 | .016389 | .004097 |
| Five full frozen runs | 9625 | 2.673611 | .668403 |
| Final analysis | 18 | .005000 | .001250 |
| Total | 45324 | 12.590000 | 3.147500 |

Native rounded ledger quantities total3.14749975node hours; exact scheduler
seconds total3.1475, below4. Usage is imported once per accounting reference;
no duplicate manual usage. The final analysis artifact's cost table excludes
its own18seconds, which are added here after its termination. All retained
raw rows, diagnostics, checkpoints, calibration manifests, source snapshots
and native exports remain available; no cleanup was performed.

No runtime check is outstanding for the assigned screen. Confirmation,
same-depth T0, parameter-matched concat controls, multi-seed uncertainty and
any rule-9 capacity/feature/selectivity probe were not run. No RB5 contract,
pilot, hypothesis status or cross-line conclusion was decided by this task.
