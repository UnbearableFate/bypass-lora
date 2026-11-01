# T-0005 delivery: V2 generator and Base bounds

Implementation and data checks are complete. The frozen Base **does not meet
the F-line upper-bound criterion** under the fixed protocol. This is task
difficulty information, as specified by E-0005; the protocol was not tuned
after seeing results. It is not evidence for or against H-0003's learned-memory
claim and does not authorize V2 training.

## Code, review and interface

- Generator `concat-loop-v2-t1-1`: `src/v2_data.py`.
- Preparation, legacy proof and tests: `src/v2_prepare.py`, `src/test_v2_data.py`.
- Frozen Base evaluator: `src/v2_eval.py`.
- Read-only legacy inventory: `src/v2_legacy_probe.py`.
- Executed generator source commit: `6fa3309b00c0e6d1a715f0032a6a6f297b3e14fb`.
- Executed evaluation commit: `00226b38d59ac5396f4d868eaccd80f255656964` (same source plus smoke notes).
- Review `RV-0010`: pass, zero findings, including the generator, sealing,
  diagnostic-field interface, legacy argument, and evaluator. Both Base run
  registrations reported this exact commit covered by the passing review.
- Eight PBS regression tests passed in both smoke and full preparation runs.
- Exact JSONL interface and source-line proof are in
  `research/notes/T-0005-data-contract.md`; researcher-2 received both smoke and
  final artifacts for T-0006. Diagnostic triples, target indices and answer
  labels are excluded from model inputs. Write positions are text-derived.

Qwen/Qwen3-0.6B-Base model/tokenizer revision:
`da87bfb608c14b7cf20ba1ce41287e8de496c0cd`.
Environment: existing lock `3c916b51337d`, cuda extra, CUDA 12.8,
torch 2.10.0+cu128; frozen BF16 backbone with eager attention. Dependencies and
other researchers' `v1_*` / `bypass_*` files were not modified.

## Final data and hashes

Directory: `/work/xg24i002/x10041/bypass-lora/runs/ATT-0039-1/data/`.

| File | Episodes | SHA256 |
| --- | ---: | --- |
| train.jsonl | 16,384 | a2acb5451bc8aade1c274ebe225b4411bb3063e81bf57ba73c35101d9ac807d2 |
| calibration.jsonl | 2,048 | c08753365165216b5a9f4c3d629271b2183e1323973886256867e826c1712f18 |
| dev.jsonl | 1,000 | c6583cf8f7cd9d5652ed57093432b278b0600c611076fac2fe6c6b7d9e7715f4 |
| confirmation.registry.jsonl | 2,000 | 6e3fe3062e5d28a38b8477359b1898be896be016b427c32cf18b64273542cdfe |
| metadata.json | — | 78731ae9b38b7559757efe385b1b7721ef593429cf2b9cf0fcaed891e30d1796 |

All 21,432 identities and canonical fact-set-plus-question contents are unique
within/across splits. Independent repeat generation and written-file SHA256
agree for every split. Confirmation contains identity and hashes only; its
structural content is transiently computed for hashing but no text, answers,
token IDs or model results are released. No confirmation model evaluation ran.

Train has 1,381,843 support+question+answer input tokens per epoch, giving
4,145,529 across three epochs: enough for each arm's 4M token budget. Eight arms
share the same dataset. Calibration/dev have 172,827/84,332 input tokens.
Multi-hop and unseen-name layouts have reserved identities only; they were not
materialized or evaluated.

## Axis statistics

All released episodes have hop=1, 16 facts, four relations (four facts per
relation), one question, and 15 distractor facts. Every answer is exactly one
token including its leading space. All main splits use seen names. The validated
pool has 92 seen names and 89 disjoint unseen names; Zachary failed candidate
tokenization and was excluded before evaluation.

| Axis | Train | Calibration | Dev |
| --- | --- | --- | --- |
| Natural / shuffled order | 8192 / 8192 | 1024 / 1024 | 500 / 500 |
| Query emails / photographs / rescues / tutors | 4096 / 4095 / 4036 / 4157 | 520 / 528 / 530 / 470 | 237 / 264 / 254 / 245 |
| Answer bare-name length 1 / 2 / 3 tokens | 8753 / 6641 / 990 | 1103 / 819 / 126 | 523 / 419 / 58 |
| Answer synthetic high / low frequency stratum | 12385 / 3999 | 1565 / 483 | 749 / 251 |
| Support token range | 69–85 | 70–84 | 70–83 |
| Question token length | 7 | 7 | 7 |
| Joint full-prompt token range | 76–92 | 77–91 | 77–90 |
| Support character range | 354–414 | 356–410 | 356–411 |
| Question character range | 27–38 | 27–38 | 27–38 |
| Verified delimiter positions | 262144 | 32768 | 16000 |

High/low means controlled 3:1 **tail/answer sampling weights**, not observed
pretraining or corpus frequency. Heads are uniformly sampled, so total support
entity frequencies need not be 3:1. Full histograms, actual per-name support
counts and train answer counts are retained in metadata; tokenization lengths,
answer token IDs and sampling weights for each name are in `name_pools.json`.

## Legacy comparison

`ATT-0035-1` inventoried 169 run registry entries and nine public sample
archives, finding no facts/questions available for empirical enumeration.
Coordinator MSG-0093 authorized the construction proof instead. All old fixed
relation phrases and the numeric fallback are disjoint from new labels
emails/photographs/tutors/rescues. Thus every nonempty new labeled fact set
differs from every fact set under that pinned renderer, regardless of names,
seeds, or support order; adding the question cannot create equality.

The full preparation run verifies pinned source/registry SHA256 values and the
registry data version and records overlap=0 **by this construction argument**.
See `runs/ATT-0039-1/legacy_disjointness.json` and the contract note's source
paths/line numbers. This is not an empirical overlap count or a graph-isomorphism
claim. Legacy code was neither imported nor executed.

## Base dev results and contract disposition

Full 1,000-example dev, batch size 8, full-vocabulary argmax and FP32 one-token
cross entropy. M sees question text only; F sees support plus question.

| Condition | Correct / 1,000 | EM | Answer NLL |
| --- | ---: | ---: | ---: |
| Train-selected constant prior | 15 | 1.5% | not defined by contract |
| M Base | 11 | 1.1% | 9.2494026874 |
| F Base | 13 | 1.3% | 5.7063963076 |

M <= prior +5 pp (6.5%) passes. F >= prior +20 pp (21.5%) fails. The combined
task-suitability criterion therefore did not pass. Per E-0005, retain this as
task-difficulty information for V2 design rather than changing the protocol on
this result. Lower F NLL alone does not satisfy the EM criterion. These Base
results do not establish a learned-memory signal, refute it, or measure a
trained model's attainable ceiling.

Detailed outputs:
`/work/xg24i002/x10041/bypass-lora/runs/ATT-0043-1/metrics.json` and
`predictions.jsonl` (one row per episode per line).
The 32-example prefix smoke was only for timing and execution checks; it had
M/F EM=0 and is not substituted for the full results.

## Records and measured cost

| Attempt | Purpose | Result | Evidence | PBS job | Measured walltime |
| --- | --- | --- | --- | --- | ---: |
| ATT-0035-1 | Legacy inventory | RES-0027 | EV-0024 | 3760163.opbs | 9 s |
| ATT-0037-1 | Data/metric smoke | RES-0029 | EV-0026 | 3760231.opbs | 10 s |
| ATT-0039-1 | Full data preparation | RES-0032 | EV-0029 | 3760256.opbs | 35 s |
| ATT-0042-1 | Base timing smoke | RES-0034 | EV-0031 | 3760273.opbs | 11 s |
| ATT-0043-1 | Full Base M/F dev | RES-0035 | EV-0032 | 3760289.opbs | 19 s |

All work ran through Miyabi MCP on GH200 MIG 2g.24gb, with allocation checks;
login node miyabi-g2 was used only for control work. The five jobs total
84 measured seconds on one MIG instance, approximately **0.005834 node-h**
using the required quarter-node conversion, well below the 1 node-h budget.
Executor usage is imported once per job in MIG-instance-hours; do not add
converted node-hours as a second charge.

No tests were run locally. Remaining limitations: no real-model batch-size
numerical equivalence check; no trained positive control, memory training,
multi-hop materialization or unseen-name evaluation. These are not claimed
validated by this task. Review RV-0010 has no unresolved findings. The
coordinator owns acceptance, merge and subsequent research decisions.
