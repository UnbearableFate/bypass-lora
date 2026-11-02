# T-0028 delivery: E-0025 segmented information access

The locked final reducer returns **inconclusive_secondary_only**, with
**eligible_edge = null**. Primary PIQA gain over the better matched no-history
control is +1.6 pp at r32 and +1.4 pp at r128. Both are below the +3 pp support
gate. The challenge rule requiring both gains below +1 pp does not hold.
Secondary BoolQ meets the r128 task-level point gates: gain +4.4 pp and swap
drop 6.5 pp, exceeding half the gain (2.2 pp). This is a BoolQ screening
outcome; the primary hypothesis has no supports/challenges edge or status
change. The sealed rule 8 calls for two new BoolQ seeds before a conclusion.
No replication job or further widening, tap change or loop was launched.

All authorized work is complete: seven full-budget training arms, Base_seg,
both five-condition intervention grids, full dev curves, same-job identities,
strict artifact/provenance checks and the locked 10,000-sample bootstrap.
Total E-0025 usage is **37,382 MIG seconds = 2.595972222 node-hours**, within
the unchanged 3-node-hour cap. Native six-decimal ledger usage is
**10.383888 MIG instance-hours = 2.595972 node-hours**. All 16 compute attempts,
including checks, audits and both analyses, are counted; no failed compute
attempts occurred.

The final result is **ATT-0303-1 / PBS3773604 / RES-0280**. Its compute
allocation was verified; application status is complete and validator passed.
The final PBS validator independently rechecks all eight arm artifacts,
all 19 scored bundles including the accepted full-context reference and both
intervention grids, paired 1000 dev rows per task, source/checkpoint bindings,
the fixed within-task R1 donor derangement, exact off/Base identities and the
15-prior-attempt cost inventory. Its own 18-second terminal cost is appended
in [T-0028-final-cost.json](T-0028-final-cost.json). Raw outputs:
[analysis.json](/work/xg24i002/x10041/bypass-lora/runs/ATT-0303-1/analysis.json),
[analysis.md](/work/xg24i002/x10041/bypass-lora/runs/ATT-0303-1/analysis.md),
[manifest.json](/work/xg24i002/x10041/bypass-lora/runs/ATT-0303-1/manifest.json).

The contract is E-0025@1 / DOC-0273@1 / H-0019@1, with pre-result rule6
clarification D-0044. All formal training, interventions and analyses execute
the immutable reviewed scientific source
`7d8b3464b6145504d155268f4c667fa3ecc53b00`. RV-0078 passed and RV-0079
passed with zero findings after the three low fixes. D-0049 covers the
automatic stale label caused solely by an inherited RB1 test file that no
SEG runtime imports. The later source 8b3ef43 audit wrapper only invokes
reviewed validators and aggregates descriptive seam metadata; it does not
change the scientific source. The final input digest is
`blake2b:1d4d39f6d2399eda0da6ba39f54e35dfe36e8f079c20a9819c864723015b84da`;
its references are retained in
[final-analysis-input.json](/work/xg24i002/x10041/bypass-lora/research/artifacts/T-0028/final-analysis-input.json)
and [the registered job note](T-0028-final-analysis-job.json).

The r32 result was recorded before any width run in ATT-0288-1 / RES-0266:
[preserved analysis](/work/xg24i002/x10041/bypass-lora/runs/ATT-0288-1/analysis.json) and
[T-0028-r32-outcomes.md](T-0028-r32-outcomes.md). At that time the direction
was pending_r128, PIQA gain was exact2/125 (+1.6 pp), and rule6 released the
triplet because 0 < 1.6 < 3. The width runs are the preregistered release.
D-0052's requested exploratory fallback was unused. The r32 files, rules and
thresholds were preserved.

Every rule outcome has exactly one evidence record, valid diagnostic for
implementation and valid screening for the other rules. All nine records
have no hypothesis support/challenge link because the final eligible edge is
null. Full scopes and record IDs are in
[T-0028-evidence-index.json](T-0028-evidence-index.json).

| Rule | Final recorded outcome | Evidence |
|---|---|---|
| 1 implementation | Pass: off/Base, initial/Base, no-channel identity and all same-job reloads; both keep widths included. | EV-0130 diagnostic |
| 2 dynamic range | Both informative: full-context LoRA minus segmented LoRA is +7.0 pp PIQA and +19.3 pp BoolQ; each >=3. | EV-0131 screening |
| 3 primary | PIQA; BoolQ remains secondary. No no-information stop fires. | EV-0132 screening |
| 4 learnability | Primary Base+1 threshold67.1%. SEG01/06 fail; SEG02/03/04/05/07 pass. SEG01 and SEG04 are not both unlearned, so no diagnostic downgrade. | EV-0133 screening |
| 5 r32 | PIQA gain+1.6 pp, swap drop4.2 pp. Gain<3, so descriptive; pre-width release outcome retained. | EV-0134 screening |
| 6 width / D-0044 | Released once and completed. PIQA gain128+1.4 pp, swap drop5.3 pp. D-0044c descriptive/inconclusive: neither support nor challenge. No more width/tap/loop. | EV-0135 screening |
| 7 r32 challenge | False: gain32>0. | EV-0136 screening |
| 8 secondary | BoolQ r128 gain4.4 pp and swap drop6.5 pp meet task-level gates. Overall inconclusive_secondary_only; two new BoolQ seeds needed before a conclusion. | EV-0137 screening |
| 9 macro/NLL | Descriptive only; no hypothesis status change, eligible_edge=null. | EV-0138 screening |

The conditional follow-ups in contract rule10 for a primary supports or
challenges result do not fire. Rule8's replication requirement is reported
for coordinator review; this delivery makes no route or formal-experiment
decision.

Final dev scores use 1000 rows per task and the final checkpoint:

| Arm | PIQA % | BoolQ % | Macro % | PIQA answer NLL | BoolQ answer NLL |
|---|---:|---:|---:|---:|---:|
| SEG-00 Base_seg | 66.10 | 60.90 | 63.50 | 3.524690 | 3.242247 |
| SEG-01 T0 r32 | 66.90 | 63.80 | 65.35 | 3.052494 | 0.641290 |
| SEG-02 Keep r32 | 69.40 | 65.80 | 67.60 | 2.822721 | 0.667019 |
| SEG-03 Reset r32 | 67.80 | 61.20 | 64.50 | 3.056727 | 0.668558 |
| SEG-04 LoRA r7 | 67.40 | 64.20 | 65.80 | 3.014400 | 0.814079 |
| SEG-05 Keep r128 | 69.30 | 65.90 | 67.60 | 2.848474 | 0.663548 |
| SEG-06 Reset r128 | 66.90 | 61.50 | 64.20 | 3.064483 | 0.652607 |
| SEG-07 T0 r128 | 67.90 | 61.20 | 64.55 | 3.053429 | 0.670726 |

Answer NLL is token-normalized within each task; macro NLL is the mean of
the two task NLLs. Final checkpoint selection is fixed; intermediate dev
curves do not select a checkpoint.

The maximum-control estimand is recomputed within every bootstrap draw.
The following are gains in pp with descriptive 95% intervals; the maximum
control can differ by task. The swap column is the drop from the same
intervention job's own normal path.

| Width | Task | Better point-estimate control | Keep − max(control), pp [95%] | Normal − swap, pp |
|---|---|---|---:|---:|
| 32 | piqa | SEG-03 | +1.60 [-0.30, 3.50] | +4.20 |
| 32 | boolq | SEG-01 | +2.00 [-0.80, 4.90] | +5.30 |
| 128 | piqa | SEG-07 | +1.40 [-0.40, 3.20] | +5.30 |
| 128 | boolq | SEG-06 | +4.40 [1.50, 6.80] | +6.50 |

The descriptive macro gains over the better macro control are +2.25 pp
[0.50,3.95] for r32 and +3.05 pp [1.35,4.55] for r128. Macro uses the maximum
of the macro control scores, rather than averaging taskwise maxima.
No gate uses a macro or NLL result.

Both intervention jobs completed Base_seg, normal, off, swap and reset for
all 1000 rows per task. ATT-0302-1 / RES-0279 binds the exact SEG-05 final
adapter through its manifest, checkpoint inventory and config/adapter hashes.
Its normal predictions and scores exactly match the training evaluation.
ATT-0287-1 / RES-0265 reports nine prediction differences between its normal
path and the SEG-02 training evaluation (BoolQ65.5% versus65.8%; PIQA69.4% in
both). The main gain uses final training rows and each intervention effect
uses its own normal reference, as fixed in the reviewed analysis. Off matches
the same-allocation Base_seg exactly for both jobs. The fixed donors use
only another same-task row's first-segment state; candidate and gold tokens
are not donor inputs. The final PBS analysis verifies the complete donor
map against the R1 derangement.

| Condition | PIQA % | BoolQ % | Macro % | PIQA NLL | BoolQ NLL |
|---|---:|---:|---:|---:|---:|
| SEG-02/Base_seg | 66.10 | 60.90 | 63.50 | 3.524690 | 3.242247 |
| SEG-02/normal | 69.40 | 65.50 | 67.45 | 2.822752 | 0.665311 |
| SEG-02/iv_off | 66.10 | 60.90 | 63.50 | 3.524690 | 3.242247 |
| SEG-02/iv_swap | 65.20 | 60.20 | 62.70 | 3.511991 | 0.849706 |
| SEG-02/iv_reset | 66.70 | 54.50 | 60.60 | 3.357356 | 1.135481 |
| SEG-05/Base_seg | 66.10 | 60.90 | 63.50 | 3.524690 | 3.242247 |
| SEG-05/normal | 69.30 | 65.90 | 67.60 | 2.848474 | 0.663548 |
| SEG-05/iv_off | 66.10 | 60.90 | 63.50 | 3.524690 | 3.242247 |
| SEG-05/iv_swap | 64.00 | 59.40 | 61.70 | 3.470065 | 0.798504 |
| SEG-05/iv_reset | 66.50 | 58.40 | 62.45 | 3.419861 | 0.719281 |

Paired drops from each intervention's own normal path, pp [descriptive95%]:

| Comparison | PIQA | BoolQ | Macro |
|---|---:|---:|---:|
| SEG-02, normal − iv_off | 3.30 [0.70, 5.90] | 4.60 [1.60, 7.60] | 3.95 [1.95, 5.95] |
| SEG-02, normal − iv_swap | 4.20 [1.70, 6.60] | 5.30 [2.00, 8.50] | 4.75 [2.75, 6.75] |
| SEG-02, normal − iv_reset | 2.70 [0.60, 4.90] | 11.00 [7.00, 15.00] | 6.85 [4.55, 9.10] |
| SEG-05, normal − iv_off | 3.20 [0.70, 5.70] | 5.00 [2.20, 7.80] | 4.10 [2.25, 6.00] |
| SEG-05, normal − iv_swap | 5.30 [3.00, 7.60] | 6.50 [3.40, 9.50] | 5.90 [4.00, 7.80] |
| SEG-05, normal − iv_reset | 2.80 [0.50, 5.10] | 7.50 [3.70, 11.20] | 5.15 [2.95, 7.35] |

Full-context references come from the accepted E-0021 delivery and use a
different protocol. Only RB1-07 versus SEG-04 enters the dynamic-range gate.
The full-context minus segmented LoRA intervals are PIQA +7.0 [4.8,9.2] pp
and BoolQ +19.3 [16.1,22.5] pp. The other references and the Base differences
are descriptive.

| Full-context reference | PIQA % | BoolQ % | Macro % |
|---|---:|---:|---:|
| Base | 72.30 | 68.00 | 70.15 |
| RB1-01 T0 | 73.30 | 81.30 | 77.30 |
| RB1-02 KM | 72.40 | 81.80 | 77.10 |
| RB1-07 LoRA | 74.40 | 83.50 | 78.95 |

Source: [accepted T-0025 delivery](/work/xg24i002/x10041/bypass-lora/research/notes/T-0025-delivery.md).
The Base minus Base_seg gaps are +6.2 pp PIQA and +7.1 pp BoolQ. These
references do not establish practical superiority of a segmented method.

Full-dev learning curves are retained at steps3279/6558/9837 and final13115.
Each cell is PIQA% / BoolQ%; all checkpoint rows were validated in PBS.

| Arm | 25% / step3279 | 50% / step6558 | 75% / step9837 | Final / step13115 |
|---|---:|---:|---:|---:|
| SEG-01 | 67.40/61.20 | 66.80/63.20 | 67.10/63.00 | 66.90/63.80 |
| SEG-02 | 66.10/61.40 | 67.70/64.20 | 69.20/65.40 | 69.40/65.80 |
| SEG-03 | 67.10/61.20 | 65.60/60.80 | 67.40/60.80 | 67.80/61.20 |
| SEG-04 | 66.70/62.60 | 66.50/62.90 | 66.70/63.70 | 67.40/64.20 |
| SEG-05 | 66.80/61.20 | 65.50/62.40 | 67.80/64.80 | 69.30/65.90 |
| SEG-06 | 66.60/61.20 | 67.00/61.20 | 67.60/61.40 | 66.90/61.50 |
| SEG-07 | 66.10/61.20 | 67.00/60.80 | 67.70/61.20 | 67.90/61.20 |

Encoding was fixed before any SEG score in
[T-0028-protocol.md](T-0028-protocol.md), per MSG-0726: V1's longest-candidate
reserve and a total512-token budget across both segments. BoolQ segment1 is
the retained V1 Passage prefix. PIQA independently encodes Goal and Solution:
and left-crops Goal within the same budget. Candidates remain unchanged.
The expanded PBS audit ATT-0301-1 / RES-0278 verifies identical encodings
across all arms and **zero seam-edited prompts or changed tokens** against
the retained V1 prompt, in training and dev for both tasks.

| Split/task | Presentations | Unique IDs | Segment1 encoded tokens | Segment2 prompt tokens | First-segment cropped rows/tokens |
|---|---:|---:|---:|---:|---:|
| Train PIQA | 16458 | 12064 | 174958 | 32916 | 1 / 7 |
| Train BoolQ | 9772 | 5379 | 1258355 | 146187 | 18 / 1641 |
| Dev PIQA | 1000 | 1000 | 10640 | 2000 | 0 / 0 |
| Dev BoolQ | 1000 | 1000 | 129803 | 14968 | 2 / 298 |

The training counts include repeated presentations. These crops accompany
zero token edits versus the retained V1 prompt; they do not remove extra
information at the seam. The common excluded row is PIQA5528, whose complete
candidates each contain537 tokens. Raw per-row records:
[SEG-05 seams-train](/work/xg24i002/x10041/bypass-lora/runs/ATT-0290-1/seams-train.json),
[SEG-05 dev evaluation audit](/work/xg24i002/x10041/bypass-lora/runs/ATT-0290-1/evaluation-audit.json).
The expanded [audit.json](/work/xg24i002/x10041/bypass-lora/runs/ATT-0301-1/audit.json) contains the
same counters and hashed per-row FileRefs for all eight arms.

All seven training arms complete13,115 steps /26,230 presentations, effective
batch2/microbatch1, dropout0, R1 seed0 initialization/order and the fixed
candidate loss mask. Order SHA256 is
`33759bc20322d0c9981128dd8574438546fc3439194ab32b7c79e35e0f0fc2bf`.
The shared sample budget is fixed; actual executed tokens differ by channel.
Keep arms execute1,433,313 segment1 and566,756 segment2 training tokens;
other arms skip segment1 and execute566,756 segment2 tokens. All training
arms have387,653 gold answer-loss tokens. Dev gold loss tokens are23,183
PIQA and1000 BoolQ.

Every adapter's stored and trainable parameter counts are equal; the gradient
column counts parameters receiving gradients during training. State bytes
below are actual cross-segment M+c plus count bytes. Reset has no state
crossing the boundary even though it uses the same within-segment operator.

| Arm | Stored = trainable | With gradient | Train segment1 tokens | Train segment2 tokens | Cross-segment M+c + counts bytes | Peak allocated CUDA bytes |
|---|---:|---:|---:|---:|---:|---:|
| SEG-00 | 0 | 0 | — | — | 0 + 0 | — |
| SEG-01 | 663,552 | 663,552 | 0 | 566,756 | 0 + 0 | 3472551424 |
| SEG-02 | 663,552 | 663,552 | 1,433,313 | 566,756 | 16896 + 32 | 3482695680 |
| SEG-03 | 663,552 | 663,552 | 0 | 566,756 | 0 + 0 | 3479605248 |
| SEG-04 | 630,784 | 630,784 | 0 | 566,756 | 0 + 0 | 3574985728 |
| SEG-05 | 2,752,512 | 2,752,512 | 1,433,313 | 566,756 | 264192 + 32 | 3641856000 |
| SEG-06 | 2,752,512 | 2,752,512 | 0 | 566,756 | 0 + 0 | 3640517120 |
| SEG-07 | 2,752,512 | 2,752,512 | 0 | 566,756 | 0 + 0 | 3503861248 |

Keep state is four FP32 M[1,r,r], c[1,r] tensors plus int64 counts[1].
At r32 each layer has4224 M+c bytes and8 count bytes (total16896+32).
At r128 each has66048+8 (total264192+32). Per layer6/13/20/27, the keep
training path records26,230 attention and26,230 FFN calls in each segment,
1,433,313 writes and reads in segment1, and566,756 writes and reads in
segment2. Reset records only segment2's26,230 calls and566,756 writes/reads.
T0 records26,230 calls in segment2 and zero memory writes/reads. LoRA has
no side-memory accounting. All counts are per layer, not summed over four
layers; the complete runtime-accounting dictionaries are retained in the
formal manifests and final analysis.

Observed times are descriptive seconds. Forward columns are instrumented
segment1 / segment2 totals; train-total includes backward and optimization,
and final-eval/reload columns cover full dev. The hardware profile is
metadata only under D-0018/D-0019.

| Arm | Train forward s1/s2 s | Train total s | Eval forward s1/s2 s | Final eval s | Reload eval s | MIG profile |
|---|---:|---:|---:|---:|---:|---|
| SEG-00 | — / — | — | 0.00 / 147.05 | 150.29 | — | 2g.24gb |
| SEG-01 | 0.00 / 1416.83 | 2633.89 | 0.00 / 176.44 | 179.95 | 182.69 | 2g.24gb |
| SEG-02 | 1430.83 / 1390.11 | 5140.69 | 172.11 / 169.46 | 345.21 | 342.90 | 1g.24gb |
| SEG-03 | 0.00 / 1514.26 | 2817.34 | 0.00 / 200.93 | 204.52 | 208.46 | 2g.24gb |
| SEG-04 | 0.00 / 1438.07 | 2879.51 | 0.00 / 186.95 | 190.42 | 192.04 | 2g.24gb |
| SEG-05 | 1461.83 / 1410.38 | 5361.66 | 172.36 / 169.79 | 345.71 | 345.21 | 2g.24gb |
| SEG-06 | 0.00 / 1447.78 | 2633.33 | 0.00 / 175.93 | 179.46 | 180.09 | 2g.24gb |
| SEG-07 | 0.00 / 1367.25 | 2515.17 | 0.00 / 167.11 | 170.68 | 169.06 | 1g.24gb |

SEG-05 intervention timings, including swap donor construction, are retained
in [conditions.json](/work/xg24i002/x10041/bypass-lora/runs/ATT-0302-1/conditions.json); the swap donor
forward measurement is88.645s and the full five-condition application
elapsed time1685.525s. The scheduler charges1702s including launch overhead.
Both segment forwards are independent: no original attention KV crosses,
second-segment positions restart at zero, and the original causal mask is
unchanged. Keep transfers only the four M,c,counts tensors with the training
autograd graph; the first segment has no answer loss.

The model/tokenizer pin is Qwen/Qwen3-0.6B-Base revision
`da87bfb608c14b7cf20ba1ce41287e8de496c0cd`. Environment: Python3.13.13,
torch2.10.0+cu128, transformers4.57.1, cuda/12.8, environment
`envs/3c916b51337d-cuda`, uv.lock SHA256
`31429457ceece707abf1505af6e6210b44a6e2817d2527241892f0587aca47a3`.
RBc lock SHA256
`38b8ee24af3ded8aee398a4cc3dfaa5a1dfece4737a6cde34642fbff7bf1012d`
fixes R/T LR0.003 and LoRA LR0.0003. The held-out A2 ID manifest SHA256 is
`34277aa4627271dbf4e0bcb9592e3014c6130d87b1766caad17c2d6b97a8a7cc`;
its excluded confirmation rows are not evaluated. The final validator checks
all pins and checkpoint inventories. All checkpoints and result artifacts
remain retained for coordinator acceptance and potential registered follow-up.

Every cost is a native measured terminal observation imported once; there is
no duplicate manual usage record. Smoke quality was discarded and did not
enter any gate. ATT-0271 includes315 tests and the seven-arm timing smoke;
ATT-0276 passed326 deterministic checks at the exact reviewed source.

| Attempt | Result | PBS job | Stage | Scheduler seconds | Native MIG h |
|---|---|---|---|---:|---:|
| ATT-0271-1 | RES-0250 | 3770096.opbs | Pre-review tests + 7-arm cost smoke | 477 | 0.132500 |
| ATT-0276-1 | RES-0253 | 3770216.opbs | Exact-source regression | 55 | 0.015278 |
| ATT-0277-1 | RES-0259 | 3770249.opbs | SEG-00 Base | 179 | 0.049722 |
| ATT-0278-1 | RES-0260 | 3770250.opbs | SEG-01 train | 3562 | 0.989444 |
| ATT-0279-1 | RES-0261 | 3770251.opbs | SEG-02 train | 7393 | 2.053611 |
| ATT-0280-1 | RES-0262 | 3770252.opbs | SEG-03 train | 3866 | 1.073889 |
| ATT-0281-1 | RES-0263 | 3770253.opbs | SEG-04 train | 3857 | 1.071389 |
| ATT-0286-1 | RES-0264 | 3771591.opbs | Initial artifact audit | 17 | 0.004722 |
| ATT-0287-1 | RES-0265 | 3771652.opbs | SEG-02 interventions | 1576 | 0.437778 |
| ATT-0288-1 | RES-0266 | 3771942.opbs | Preserved r32 analysis | 16 | 0.004444 |
| ATT-0290-1 | RES-0275 | 3772011.opbs | SEG-05 train | 7676 | 2.132222 |
| ATT-0291-1 | RES-0276 | 3772012.opbs | SEG-06 train | 3568 | 0.991111 |
| ATT-0292-1 | RES-0277 | 3772013.opbs | SEG-07 train | 3403 | 0.945278 |
| ATT-0301-1 | RES-0278 | 3773187.opbs | Expanded artifact/seam audit | 17 | 0.004722 |
| ATT-0302-1 | RES-0279 | 3773252.opbs | SEG-05 interventions | 1702 | 0.472778 |
| ATT-0303-1 | RES-0280 | 3773604.opbs | Final locked analysis | 18 | 0.005000 |

The final analysis itself validates terminal cost coverage for the other15
attempts (2.594722 native node-hours); its own0.005 MIG-h adds0.00125
node-hours. The complete16-attempt total above includes all launch overhead,
preflights, audits and the preserved r32 analysis. Small differences between
native six-decimal and exact-second totals are rounding only. The final
accounting sources and every row are in
[T-0028-final-cost.json](T-0028-final-cost.json).

All intervals use10,000 shared task-stratified bootstrap draws with
default_rng seed20261101. They describe dev data uncertainty conditional on
these trained seed0 models, exclude seed and task/width-selection uncertainty,
and are not simultaneous or multiplicity-adjusted intervals. The numerical
gates use exact correctness fractions, not confidence bounds. BoolQ's
task-level support therefore is not a95% guarantee of at least+3 pp gain,
and PIQA's descriptive/inconclusive result is not equivalence or no effect.
The fixed RBc LR may not be optimal for the segmented protocol; the study
does not resolve that limitation. There is no confirmation evaluation,
compression claim, long-context claim or practical superiority claim.
Hypothesis status and any subsequent route remain for the coordinator after
conclusion review.
