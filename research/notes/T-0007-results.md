# T-0007: Base F difficulty diagnostic

Status: full diagnostic and authorized FP32 follow-up complete; independent review
pending. All workloads ran in PBS on GH200 MIG (mg1098), with allocation guards;
login node miyabi-g2 was used for control work only. No training, dev reevaluation,
or confirmation reading. H-0003 remains untested by this diagnostic.

## Outcome and evidence limits

E-0007@2's original implementation gate **failed** on exactly one condition:
F batch1/batch8 per-example NLL mean absolute difference 0.088144 > 0.05.
This remains in ATT-0050-1/metrics.json; it was not retroactively changed to pass.
The requested first action—investigate implementation—was performed under
coordinator MSG-0135, an additional diagnostic selected after seeing this failure.

The first 256 calibration episodes loaded as true FP32 (all parameters asserted
FP32, eager attention, TF32 disabled, highest matmul precision) give M/F batch
NLL mean absolute differences 0.000003979/0.000009557 and 100% top-1 agreement.
On those same 256 episodes the saved BF16 differences were 0.013251/0.102523,
with 98.8281%/98.4375% top-1 agreement. This supports BF16 numerical sensitivity
rather than a scoring/masking error in the tested path; it is not a proof that
all model/evaluator behaviors are error-free. No thresholds or BF16 results were
changed, and the FP32 follow-up is explicitly post-result diagnostic evidence.

Under MSG-0135's follow-up rule, the E16 retrieval-weak branch applies:
F candidate-b unconditional accuracy is **14.2090%**, below 50%, though above its
3.6703% attainable random baseline. F candidate-b falls from 62.5% at E2 to
32.13%/18.31%/14.21% at E4/E8/E16. E1 b is degenerate and not interpreted.
E1 nonself two-choice candidate-a accuracy is 69.3221% (random 50%); full-vocabulary
nonself EM is 66.5512%. E16 nonself EM is 0.2969%. Thus E16 combines weak retrieval
with strong subject copying; the evidence does not justify claiming the correct
entity is reliably found but merely hidden by the full-vocabulary argmax.

Design implication for coordinator review: retain the generator and include a
trained F positive control in any later V2 contract; the E curve is diagnostic
design context, not authorization for V2 training. V2 release remains the user's
decision. These results neither support nor challenge learned-memory H-0003.

## Original path consistency

| Line | Batch1 EM | Batch8 EM | Batch1 mean NLL | Batch8 mean NLL | Top1 agreement | NLL mean absolute difference |
|---|---:|---:|---:|---:|---:|---:|
| M | 0.008301 | 0.008301 | 9.231220 | 9.233152 | 0.994629 | 0.016228 |
| F | 0.014160 | 0.014160 | 5.582497 | 5.579768 | 0.992676 | 0.088144 |

Both top-1 agreements pass 99%; M NLL passes 0.05. E1 F NLL1.421515 < M9.233152
passes. Diagnostic E16 batch8 predictions and NLL match original evaluate exactly
for every episode in both lines. Calibration M/F EM remains near E-0005's dev
1.1%/1.3% (that dev result is historical; no dev file was read here).

F individual NLL carries about 0.09 BF16 batch noise in this comparison. Aggregate
EM is identical across the tested batch sizes; F mean NLL differs by only 0.002730,
M by 0.001932. This is evidence of stable aggregates here, not a guarantee that
batching can never affect an aggregate or a trained model.

## BF16 padding and length association

Full2048 F Pearson correlation of absolute NLL change with padding tokens is
+0.224027, with prompt length -0.216547. All347 unpadded F examples have zero
change. Padding1–7 tokens has group mean changes 0.10985,0.10542,0.10961,0.10048,
0.11655,0.10764,0.09534; padding8–11 gives 0.06347,0.13149,0.08832,0.12183
with only63/24/8/5 examples. The relation is not monotonic and is descriptive,
not a causal estimate. Shorter F prompts typically receive more padding.
Examples with lengths82/83/84 have mean changes0.10375/0.10353/0.09595;
lengths87/88/89 have0.03068/0.02302/0.00614; lengths90/91 have zero.
Exact counts and all length/padding groups are in ATT-0051-1/precision.json.
All M prompts are7 tokens and unpadded, so both M correlations are undefined;
its nonzero0.016228 batch difference also shows padding alone is not the only
possible numerical source. No causal claim about a particular CUDA kernel.

## Validation and resources

| Attempt | Result / Evidence | Check | PBS time | MIG-h |
|---|---|---|---:|---:|
| ATT-0046-1 | RES-0038 / EV-0035 | Post-merge155 tests pass | 49s | 0.013611 |
| ATT-0047-1 | RES-0039 partial / EV-0036 | 13 tests pass; strict original-path check detects CPU/CUDA CE difference | 25s | 0.006944 |
| ATT-0048-1 | RES-0040 / EV-0036 | Identical-logit CPU/CUDA audit | 10s | 0.002778 |
| ATT-0049-1 | RES-0041 / EV-0037 | Fixed13 tests pass; smoke32 complete | 24s | 0.006667 |
| ATT-0050-1 | RES-0042 / EV-0038 | Full2048; six diagnostic groups | 331s | 0.091944 |
| ATT-0051-1 | RES-0043 / EV-0039 | FP32 control256; saved12288-record aggregate audit | 36s | 0.010000 |

Total475 measured MIG-seconds = **0.032986 node-h**, below0.05. ATT0051 measured
36s is from qstat history; final ledger export may lag and is imported separately.
All earlier final ledger usage is imported exactly once. Application full run
320.811s; FP32 probe23.107s. Test durations155 tests46.759s and13 tests7.74s.

Code: inference commit073b1ac86b4ead5b614e5f22104ef05cc85b0ce0; report and FP32
control commit34381e8410c6239051618d19e8a688d4ea70721a. Shared frozen environment
3c916b51337d-cuda; Qwen3-0.6B-Base revision da87bfb608c14b7cf20ba1ce41287e8de496c0cd.
Calibration SHA256 c08753365165216b5a9f4c3d629271b2183e1323973886256867e826c1712f18.
The new split parameter defaults to dev, explicitly opts into calibration, and
always rejects confirmation. No dependency or generator changes.

Tables below are copied from PBS-produced ATT0051/report/tables.md. Its warning
records the original strict gate; the FP32 follow-up above explains the later
interpretation. Empty category keys in the displayed counts mean zero. All
probabilities and ranks are full vocabulary; ranks use competition ties and
categories prioritize correctness before subject copying. See T-0007-method.md.

## Full calibration results (ATT-0050-1)

All scores below use batch 8; failed implementation gate limits interpretation.

| Line / E | n | EM | NLL | Subject copy | Nonself EM | Nonself copy | Candidate a | Candidate b | Candidate c |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| M.E16 | 2048 | 0.008301 | 9.233152 | 0.762695 | 0.000000 | 0.764473 | 0.013184 | 0.044922 | 0.264160 |
| F.E1 | 2048 | 0.669922 | 1.421515 | 0.292969 | 0.665512 | 0.283523 | 0.697266 | 0.986816 | 1.000000 |
| F.E2 | 2048 | 0.069336 | 2.919480 | 0.906738 | 0.058387 | 0.906977 | 0.069336 | 0.625000 | 0.944336 |
| F.E4 | 2048 | 0.021973 | 4.372667 | 0.946777 | 0.009896 | 0.947056 | 0.022949 | 0.321289 | 0.864746 |
| F.E8 | 2048 | 0.013672 | 5.323531 | 0.962891 | 0.001484 | 0.963384 | 0.014160 | 0.183105 | 0.704590 |
| F.E16 | 2048 | 0.014160 | 5.579768 | 0.959473 | 0.002969 | 0.960910 | 0.017578 | 0.142090 | 0.504883 |

27/2048 episodes have answer = subject; nonself denominator is 2021.
E1 nonself candidate-a two-choice accuracy: 0.6932211776348343
E1 candidate-b is degenerate and is not interpreted.

## Candidate denominators and random baselines

| Line / E | Set | Accuracy | Coverage | Empty | Random nonempty | Random attainable | Conditional accuracy |
|---|---|---:|---:|---:|---:|---:|---:|
| M.E16 | a | 0.013184 | 1.000000 | 0.000000 | 0.035871 | 0.035871 | 0.013184 |
| M.E16 | b | 0.044922 | 0.986816 | 0.000000 | 0.037211 | 0.036703 | 0.045522 |
| M.E16 | c | 0.264160 | 1.000000 | 0.000000 | 0.256877 | 0.256877 | 0.264160 |
| F.E1 | a | 0.697266 | 1.000000 | 0.000000 | 0.506592 | 0.506592 | 0.697266 |
| F.E1 | b | 0.986816 | 0.986816 | 27.000000 | 1.000000 | 0.986816 | 1.000000 |
| F.E1 | c | 1.000000 | 1.000000 | 0.000000 | 1.000000 | 1.000000 | 1.000000 |
| F.E2 | a | 0.069336 | 1.000000 | 0.000000 | 0.254883 | 0.254883 | 0.069336 |
| F.E2 | b | 0.625000 | 0.986816 | 0.000000 | 0.343424 | 0.336100 | 0.633350 |
| F.E2 | c | 0.944336 | 1.000000 | 0.000000 | 0.898193 | 0.898193 | 0.944336 |
| F.E4 | a | 0.022949 | 1.000000 | 0.000000 | 0.129622 | 0.129622 | 0.022949 |
| F.E4 | b | 0.321289 | 0.986816 | 0.000000 | 0.149056 | 0.146737 | 0.325581 |
| F.E4 | c | 0.864746 | 1.000000 | 0.000000 | 0.728760 | 0.728760 | 0.864746 |
| F.E8 | a | 0.014160 | 1.000000 | 0.000000 | 0.067053 | 0.067053 | 0.014160 |
| F.E8 | b | 0.183105 | 0.986816 | 0.000000 | 0.071896 | 0.070851 | 0.185552 |
| F.E8 | c | 0.704590 | 1.000000 | 0.000000 | 0.487223 | 0.487223 | 0.704590 |
| F.E16 | a | 0.017578 | 1.000000 | 0.000000 | 0.035871 | 0.035871 | 0.017578 |
| F.E16 | b | 0.142090 | 0.986816 | 0.000000 | 0.037211 | 0.036703 | 0.143988 |
| F.E16 | c | 0.504883 | 1.000000 | 0.000000 | 0.256877 | 0.256877 | 0.504883 |

## Top-1 categories and score distributions

### M.E16

Category counts: {"question_subject": 1545, "non_name_token": 486, "correct": 17}

| Quantity | Mean | Min | P25 | Median | P75 | P90 | P95 | P99 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| answer_rank | 664.6978 | 1 | 174.75 | 389 | 826.5 | 1578.1 | 2281.55 | 3941.23 | 8164 |
| answer_probability | 0.003678346 | 8.076278e-07 | 3.597611e-05 | 9.245523e-05 | 0.0002272409 | 0.0006131874 | 0.001094959 | 0.1709827 | 0.4874672 |
| subject_probability | 0.301733 | 0.06389431 | 0.2149433 | 0.2889439 | 0.392656 | 0.4681621 | 0.5198781 | 0.6078118 | 0.675893 |

### F.E16

Category counts: {"question_subject": 1942, "non_name_token": 55, "correct": 29, "other_head": 17, "other_tail_same_relation": 3, "name_outside_support": 2}

| Quantity | Mean | Min | P25 | Median | P75 | P90 | P95 | P99 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| answer_rank | 40.95752 | 1 | 8 | 19 | 44 | 99 | 157.3 | 313.53 | 657 |
| answer_probability | 0.01596048 | 1.800024e-05 | 0.001245867 | 0.003814147 | 0.01169427 | 0.03099178 | 0.05962872 | 0.2382674 | 0.8188052 |
| subject_probability | 0.4797646 | 0.02605002 | 0.3212415 | 0.4871872 | 0.6447573 | 0.7585671 | 0.8163712 | 0.8946623 | 0.9380321 |

## E16 F strata

Position bins 0/1/2/3 mean indices 0–3/4–7/8–11/12–15.

| Axis | Value | n | EM | NLL | Subject copy | Candidate a | Candidate b | Candidate c |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| position_bin | 0 | 515 | 0.013592 | 5.189529 | 0.959223 | 0.021359 | 0.201942 | 0.570874 |
| position_bin | 1 | 500 | 0.018000 | 5.844367 | 0.946000 | 0.020000 | 0.076000 | 0.424000 |
| position_bin | 2 | 522 | 0.009579 | 5.942487 | 0.971264 | 0.011494 | 0.099617 | 0.425287 |
| position_bin | 3 | 511 | 0.015656 | 5.343631 | 0.960861 | 0.017613 | 0.189824 | 0.598826 |
| relation | emails | 520 | 0.021154 | 5.479278 | 0.961538 | 0.023077 | 0.138462 | 0.534615 |
| relation | photographs | 528 | 0.007576 | 5.578248 | 0.956439 | 0.013258 | 0.132576 | 0.477273 |
| relation | rescues | 530 | 0.015094 | 5.702212 | 0.979245 | 0.018868 | 0.133962 | 0.516981 |
| relation | tutors | 470 | 0.012766 | 5.554581 | 0.938298 | 0.014894 | 0.165957 | 0.489362 |
| support_order | natural | 1024 | 0.010742 | 5.618313 | 0.958984 | 0.014648 | 0.149414 | 0.496094 |
| support_order | shuffled | 1024 | 0.017578 | 5.541222 | 0.959961 | 0.020508 | 0.134766 | 0.513672 |
| answer_frequency_stratum | high | 1565 | 0.014058 | 5.624959 | 0.959744 | 0.018530 | 0.138019 | 0.501597 |
| answer_frequency_stratum | low | 483 | 0.014493 | 5.433341 | 0.958592 | 0.014493 | 0.155280 | 0.515528 |

## Five fixed calibration examples

### Calibration index 0, E=1

```text
Ryan rescues George ;
Who does Ryan rescue?
Answer:
```

Answer: ' George'

Top-5: [{"token_id": 9857, "text": " George", "probability": 0.2162175476551056}, {"token_id": 13646, "text": " Ryan", "probability": 0.14860400557518005}, {"token_id": 758, "text": " In", "probability": 0.13114257156848907}, {"token_id": 576, "text": " The", "probability": 0.048244658857584}, {"token_id": 3070, "text": " **", "probability": 0.029261862859129906}]

### Calibration index 1, E=16

```text
Eric tutors Karen ;
Kathleen rescues Tina ;
Jonathan rescues Kathleen ;
Kelly photographs Ethan ;
Albert rescues Terry ;
Victoria tutors Harry ;
John photographs Paul ;
Ruth emails William ;
Shirley emails Angela ;
Margaret emails Ralph ;
Michael emails Gary ;
Luke tutors Susan ;
Jennifer photographs Angela ;
Mason rescues Anthony ;
Virginia photographs Lauren ;
Charlotte tutors Oscar ;
Who does Luke tutor?
Answer:
```

Answer: ' Susan'

Top-5: [{"token_id": 24453, "text": " Luke", "probability": 0.48943662643432617}, {"token_id": 2014, "text": " To", "probability": 0.05158624053001404}, {"token_id": 20205, "text": " Based", "probability": 0.04552470147609711}, {"token_id": 29650, "text": " Jennifer", "probability": 0.03128863498568535}, {"token_id": 576, "text": " The", "probability": 0.027612125501036644}]

### Calibration index 2, E=1

```text
Dean rescues Patricia ;
Who does Dean rescue?
Answer:
```

Answer: ' Patricia'

Top-5: [{"token_id": 53934, "text": " Patricia", "probability": 0.4520922601222992}, {"token_id": 758, "text": " In", "probability": 0.11430682241916656}, {"token_id": 24061, "text": " Dean", "probability": 0.05399472266435623}, {"token_id": 3070, "text": " **", "probability": 0.03274945542216301}, {"token_id": 576, "text": " The", "probability": 0.025505300611257553}]

### Calibration index 3, E=16

```text
Julia tutors Dean ;
Andrew emails Harry ;
Jerry photographs Stephen ;
Maria tutors Alan ;
Kelly photographs Shirley ;
Gary photographs Terry ;
Elizabeth tutors David ;
Sophia photographs Angela ;
Lucas rescues Elizabeth ;
Susan rescues Walter ;
Nicole emails Joan ;
James emails Joan ;
Luke rescues Justin ;
Ashley emails Kathleen ;
Charlotte rescues Alan ;
Shirley tutors William ;
Who does Susan rescue?
Answer:
```

Answer: ' Walter'

Top-5: [{"token_id": 29933, "text": " Susan", "probability": 0.6859822273254395}, {"token_id": 2014, "text": " To", "probability": 0.02071486786007881}, {"token_id": 20205, "text": " Based", "probability": 0.02071486786007881}, {"token_id": 576, "text": " The", "probability": 0.01613275520503521}, {"token_id": 220, "text": " ", "probability": 0.014237106777727604}]

### Calibration index 4, E=16

```text
Mason rescues Ethan ;
Walter tutors Luke ;
Michael photographs Marie ;
Marilyn emails William ;
Olivia emails Anthony ;
Eva emails Eva ;
Jerry photographs Patricia ;
Luke rescues Justin ;
Bob rescues Emily ;
Virginia photographs Oscar ;
Patrick rescues Caroline ;
Susan tutors William ;
Alan emails William ;
Jennifer tutors Jordan ;
Brian tutors Elizabeth ;
Sarah photographs Richard ;
Who does Virginia photograph?
Answer:
```

Answer: ' Oscar'

Top-5: [{"token_id": 12989, "text": " Virginia", "probability": 0.4634694457054138}, {"token_id": 576, "text": " The", "probability": 0.04884931817650795}, {"token_id": 11867, "text": " Richard", "probability": 0.03804389014840126}, {"token_id": 2014, "text": " To", "probability": 0.029628610238432884}, {"token_id": 20205, "text": " Based", "probability": 0.023074783384799957}]

