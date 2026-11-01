# T-0010 / E-0011@2: V3 T2 data and difficulty checks

Status: full data and Base checks are complete; the 0.2M-token LoRA pilot awaits
scoped code review. These are diagnostic engineering/difficulty observations,
not support for or against H-0005. No M-line training or sealed model evaluation
has run.

## Sources, review and integration

Generator `concat-loop-v3-t2-2`; full generation and Base evaluation used
`d0ce03cbc76eb8bb2ae4c93c059493ec1123fac3`, covered by RV-0022 (pass, zero
findings). RV-0020's sole high finding was the @1 order/distractor coupling;
E-0011@2 and the versioned rotation fix resolve it before any full generation
or model evaluation. The superseded @1 smoke remains separately recorded.
See `T-0010-data-contract.md` for sampling, sealing and independence details.

Coordinator MSG-0231 authorized merging accepted T-0009 from master. Merge
commit `2d985a71ed3d40b1287f75fd1425379e17109b88` preserves run commits;
no rebase or squash occurred. Pilot extension `36c0d1ebc11f09aa6c60e11713e9351dbf84dbf6`
adds `--data-phase V3` to the shared F-line runner, keeps V2 as the default,
and adds only an optional phase argument to the common scheduler. V3 inputs
are pinned to the full metadata hash below and checked before model loading.
The unchanged F-line training, answer-only loss, optimizer, evaluation and
checkpoint reload paths are reused. V3 strata are joined by episode identity
after evaluation; axes do not enter model forward. Only the 0.2M pilot is
allowed by this selector. V2 tests are unchanged.

## Final data

Directory: `/work/xg24i002/x10041/bypass-lora/runs/ATT-0077-1/data/`.
Metadata SHA256: `2ad8cc358ec5f6005e832125e3fa9995a0aa0a7c73f77c8fec6da6b7416a178f`.
Name-pool SHA256: `a6bde954098e6bee7452abba1eabc56e7d0bf811b4ab220a5106c95c216ef57b`,
byte-identical to V2.

| Slice | Episodes | Prompt tokens | SHA256 |
| --- | ---: | ---: | --- |
| train | 18,432 | 1,842,535 | afa63152efe21511be91df0d625444283085383c7448261ffbbde236e3526756 |
| calibration | 2,048 | 204,316 | 49e0787d242645bf17bbdddd37af185c9521e88e784141c139a53beb2f63110b |
| dev | 1,000 | 99,852 | bb288bc46601ebb29fdedaa64be0c12057d3f80a99e42c932cc805e7280baad3 |
| confirmation, sealed | 2,000 | not materialized | 1d5ca2bac2e89368090b29fd43d5575552d15057bbec4b625ed61c2488c6babc |
| hop4 seen, sealed | 1,000 | not materialized | df729a5e751814f0d730e3a2fdf5365c010486407b140423f3de37386fd8bedc |
| hop6 seen, sealed | 1,000 | not materialized | 645c88fe769205eeddf031bcffadd2c55764bc5c12c1833ab5b23bf4c8cab40b |
| hop8 seen, sealed | 1,000 | not materialized | 4e17cd43afe3999f48fbbfa9560a69696c3e85a357fae5d6f1b5cab0c171b2e1 |
| hops1–3 unseen, sealed | 1,000 | not materialized | acb48cd9a92a30425610fae91b4dfeb66e4398942c6d39b5921a6b95642555af |

All 27,480 identities and semantic contents are unique across/within all eight
slices; every written-file hash matches an independent second generation.
Actual V2 hashes and task-normalized readable V2 fact/query hashes have zero
overlap. The V2 sealed-split relation-count proof and pinned old-registry
disjoint-label proof apply as described in the contract note; the latter is
not an empirical enumeration of unavailable legacy facts.

Train provides **5,527,605 support+question tokens over three epochs**, exceeding
the 5.5M requirement. Including answer tokens gives 1,860,967 per epoch and
5,582,901 over three epochs; answers are excluded from the coverage gate and
from the shared training-budget counter.

| Axis | Train | Calibration | Dev |
| --- | --- | --- | --- |
| hops1 / 2 / 3 | 6144 / 6144 / 6144 | 683 / 683 / 682 | 334 / 333 / 333 |
| natural / shuffled | 9216 / 9216 | 1024 / 1024 | 500 / 500 |
| terminal distractor yes / no | 9216 / 9216 | 1024 / 1024 | 500 / 500 |
| emails / photographs / rescues / tutors | 4601 / 4503 / 4697 / 4631 | 525 / 496 / 492 / 535 | 237 / 279 / 243 / 241 |
| answer synthetic high / low stratum | 13486 / 4946 | 1494 / 554 | 733 / 267 |
| answer bare-name tokens1 / 2 / 3 | 9828 / 7539 / 1065 | 1069 / 841 / 138 | 533 / 409 / 58 |
| support token range | 65–93 | 66–92 | 66–93 |
| question tokens | 23 or 25 | 23 or 25 | 23 or 25 |
| full prompt token range | 88–118 | 89–117 | 89–118 |

Each episode has 16 visible delimiter writes and one single-token answer.
In-distribution r graphs have 3–8 chains, lengths1–4 with at least two >=3;
hop changes the chosen start, not the r-graph sampling distribution. Every
twelve indices realize the twelve hop/order/terminal-noise combinations once;
dev's remainder has eight 83-row and four 84-row cells. Full chain-composition,
chain-count, distractor-position, per-name and token histograms are in
metadata.json. Sealed metadata contains only aggregate graph-composition
statistics; its JSONL files contain only identity and hashes.

## Structural shortcuts

All values below are EM percentages, rounded for display. Exact values and
counts are in `ATT-0077-1/metrics.json` and `data/metadata.json`. Constant choice
is frozen from train; random/longest ties use expected scores. The last-endpoint
rule uses the last displayed r-chain endpoint, and the bridge rule uses the
direct r successor. All gate checks pass separately on hops2 and3 for both
calibration and dev: prior<=5%, random endpoint<=30%, untyped traversal<=55%.
No generator adjustment was made after seeing full-size/model results.

| Slice | Hop | Order | Prior | Random endpoint | Untyped | Longest chain | Last endpoint | First bridge |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| calibration | 1 | natural | 0.585 | 17.104 | 50.000 | 19.420 | 15.497 | 100 |
| calibration | 1 | shuffled | 2.639 | 17.153 | 50.147 | 17.498 | 18.182 | 100 |
| calibration | 2 | natural | 2.346 | 17.581 | 49.853 | 30.596 | 17.302 | 0 |
| calibration | 2 | shuffled | 1.462 | 16.977 | 50.000 | 33.626 | 14.912 | 0 |
| calibration | 3 | natural | 3.226 | 17.008 | 50.147 | 50.000 | 18.182 | 0 |
| calibration | 3 | shuffled | 1.173 | 17.207 | 49.853 | 46.188 | 21.114 | 0 |
| dev | 1 | natural | 0.599 | 16.796 | 49.701 | 18.263 | 19.162 | 100 |
| dev | 1 | shuffled | 2.395 | 16.701 | 50.299 | 19.112 | 17.964 | 100 |
| dev | 2 | natural | 0.602 | 17.095 | 50.000 | 33.333 | 12.651 | 0 |
| dev | 2 | shuffled | 0.000 | 17.366 | 49.701 | 29.441 | 16.168 | 0 |
| dev | 3 | natural | 0.599 | 17.382 | 50.299 | 49.202 | 20.359 | 0 |
| dev | 3 | shuffled | 2.410 | 17.162 | 50.000 | 46.586 | 19.277 | 0 |

## Full Base dev

ATT-0090-1, RES-0081, EV-0067. Qwen/Qwen3-0.6B-Base model/tokenizer revision
`da87bfb608c14b7cf20ba1ce41287e8de496c0cd`; frozen BF16, eager attention,
full-vocabulary argmax and FP32 answer-token CE, batch8. The engine was first
runtime-checked on a 60-row prefix (ATT-0079-1 / RES-0079 / EV-0065).

| Hop | Episodes | M EM | M NLL | F EM | F NLL | F subject copy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 334 | 0% | 9.5788551391 | 0.8982% | 5.5410141752 | 6.2874% |
| 2 | 333 | 0% | 9.6137430088 | 0.6006% | 5.5663065334 | 9.3093% |
| 3 | 333 | 0% | 9.5538478785 | 1.2012% | 5.5955799976 | 8.1081% |
| all | 1000 | 0% | 9.5821453819 | 0.9% | 5.5676069493 | 7.9% |

| Hop | Order | M NLL | F EM | F NLL | F subject copy |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | natural | 9.5203595618 | 1.1976% | 5.2036271110 | 7.7844% |
| 1 | shuffled | 9.6373507163 | 0.5988% | 5.8784012395 | 4.7904% |
| 2 | natural | 9.7108506639 | 0.6024% | 5.2878313323 | 7.2289% |
| 2 | shuffled | 9.5172168366 | 0.5988% | 5.8431142183 | 11.3772% |
| 3 | natural | 9.6026878700 | 2.3952% | 5.2983595333 | 8.3832% |
| 3 | shuffled | 9.5047136703 | 0% | 5.8945909466 | 7.8313% |

M EM is 0 in all order strata too, satisfying <=prior+5pp overall and per hop.
The dev prior is 1.1% overall (hops 1/2/3: 1.4970% / 0.3003% / 1.5015%). F results
are difficulty information, without a pass threshold in E-0011. Low Base EM
does not establish what a trained model can learn.

## Pilot readiness

ATT-0091-1 tests the integrated selector and unchanged V2 defaults. All 26 PBS
tests pass in 12.68 s, including the unchanged V2 F-line suite, revised T2 graph
suite and new data-selection/order/stratification boundaries. A real four-step
V3 F-line smoke presents 3158 prompt tokens at LR 1e-3; its 32-row calibration/dev
records reproduce exactly after loading the adapter into a fresh Base.
Training took 1.1920 s and the combined entrypoint 17.4904 s. PBS measured 19 s on
mg1098. These four steps are an engineering smoke, not the 0.2M difficulty pilot.

The full pilot remains pending review. It will use LR 1e-3, batch 8 and the
common complete-step 0.2M prompt-token rule, V3 train-order purposes, full 2048-row
calibration, and exact reload. No intermediate dev selection or sealed data is
part of that pilot.

## Records and measured cost so far

| Attempt | Purpose | Result | Evidence | PBS | MIG seconds |
| --- | --- | --- | --- | --- | ---: |
| ATT-0073-1 | superseded @1 smoke | RES-0071 | EV-0059 | 3761734.opbs | 15 |
| ATT-0075-1 | @2 regression smoke | RES-0077 | EV-0063 | 3761891.opbs | 14 |
| ATT-0077-1 | full data | RES-0078 | EV-0064 | 3761966.opbs | 54 |
| ATT-0079-1 | Base runtime smoke | RES-0079 | EV-0065 | 3762008.opbs | 11 |
| ATT-0090-1 | full Base dev | RES-0081 | EV-0067 | 3762040.opbs | 20 |
| ATT-0091-1 | pilot selector / four-step smoke | RES-0082 | EV-0068 | 3762092.opbs | 19 |

Measured total 133 MIG seconds = 0.0369444 MIG-instance·h = **0.0092361 node·h**,
well below E-0011's 0.3 node·h ceiling. Executor exports own the usage charges;
the quarter-node conversion is reported, not charged again. Some end/usage
exports lag job-get; explicit verified scheduler end observations are retained
and later ledger exports are imported idempotently. Environment remains lock
3c916b51337d, cuda extra, cuda/12.8, torch 2.10.0+cu128, Python 3.13.13.

Open work: scoped pilot review, full 0.2M pilot and reload, final result import
and task delivery. No data or dependency protocol changed beyond the
coordinator's recorded E-0011@2 rotation correction.
