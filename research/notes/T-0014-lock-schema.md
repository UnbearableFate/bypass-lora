# T-0014: E-0013@3 evaluation and lock

The task prepares the confirmation path. It does not authorize the `evaluate`
command or parse official validation. All Python execution, including tests,
analysis and the Base WikiText measurement, goes through Miyabi PBS.

`research/artifacts/T-0014/sealed-manifest.json` records the four downloads. Only
byte sizes and SHA256 were examined on login; all four files were made read-only.
WikiText uses Salesforce/wikitext @ b08601e04326c79dfdd32d625aee71d232d685c3,
wikitext-103-raw-v1/test. The sealed contract is E-0013@3 (DOC-0128@3), SHA256
`a25529bc011fd8b639378a8aa3318b0e4cde835ba3489cdf51fa6da2ba4b60e7`.

## Rehearsal

`src/v5_rehearse.pbs` takes `SR_CODE_COMMIT`, `SR_ATTEMPT_ID`, and
`SR_SPEC_DIGEST` from the registered run. It verifies its live PBS allocation,
then runs the synthetic tests, the dev analysis, and Base WikiText once. Set
`V5_SKIP_WIKITEXT=1` for a later regression that can reuse an already measured
WikiText timing. This flag does not exist on the confirmation command.

The V1 artifacts used as software proxies are Base ATT-0024-1, F_out
ATT-0025-1, C_up ATT-0028-1, and LoRA28 r2 ATT-0032-1. Seed 0 is copied to
three files. The proxy F_out has rank 32 and the proxy LoRA has rank 2. They
are explicitly not independent V5 seeds and produce no scientific evidence.
The bootstrap runs all 10,000 replicates with RNG 20261029, drawing PIQA then
BoolQ within each replicate. Sampling indices are shared by every arm and seed;
seeds remain fixed. Rehearsal outputs are `analysis/analysis.json` and `.md`.

## Confirmation lock

Create this only after all nine final training runs, review and rehearsal have
passed. The coordinator's later unsealing task owns its exact contents and
commits the lock. `v5_confirm.file_reference(path)` supplies an absolute path,
positive byte size, and SHA256. The JSON lock contains these fields:

| Field | Required content |
| --- | --- |
| `schema_version` | `1` |
| `experiment` | `E-0013@3` |
| `code_commit` | Full immutable source commit; must match `SR_CODE_COMMIT` or the snapshot directory name |
| `code_files` | Relative filename to SHA256 for every name in `v5_confirm.REQUIRED_CODE` |
| `model` | `Qwen/Qwen3-0.6B-Base` |
| `model_revision`, `tokenizer_revision` | `da87bfb608c14b7cf20ba1ce41287e8de496c0cd` |
| `max_length`, `wikitext_block_length` | `512`, `1024` |
| `contract` | File reference to sealed E-0013@3; its hash is enforced by the code |
| `data` | Complete `files` object copied from the sealed manifest; source revisions and hashes are also enforced by the code |
| `prerequisites.code_review` | `{ "id": "RV-...", "passed": true, "answer": <file reference> }` |
| `prerequisites.dev_rehearsal` | File reference to `analysis/analysis.json` whose `split` is `dev` |
| `unseal_receipt` | Absolute shared path for an exclusive receipt, identical across all attempts |
| `learning_rate_lock` | File reference to T-0013's final LR lock with contract `E-0013@3` and all three `locked_lrs` |
| `checkpoints` | Exactly nine checkpoint objects, covering V5-C/V5-F/V5-L × seeds 1/2/3 |

Each checkpoint object has `arm`, integer `seed`, absolute adapter directory
`path`, `expected_parameters` (524288 for C/F, 630784 for L), positive
`learning_rate` (identical across the three seeds of an arm), and `files`
containing file references for **both** `config.json` and `adapter.pt`. It also
has `training_manifest`: a file reference to the final manifest with
`status=complete`, matching top-level `arm`/`seed`, and
`metrics.reload.identical=true`, `metrics.learning_rate`, and
`metrics.params.trainable` matching the lock. Each entry also has
`checkpoint_inventory`: a file reference to that final run's `checkpoint.json`,
which must be listed in its training manifest artifacts. Its two file references
must exactly match the locked paths, byte sizes and hashes. The inventory and
training manifest belong to the same run directory, and `path` is its `adapter`
subdirectory. This binds the loaded bytes to the run whose seed is reported.
The training manifest must additionally have `contract=E-0013@3`, `phase=train`,
`validator.passed=true`, actual `train.examples=58329` and `train.steps=29165`,
and `metrics.lr_lock.sha256` equal to the locked LR file hash. Planned counts
alone do not pass. Checkpoint directories must be distinct.
Configuration layers, ranks, fusions, branch type, model revision and
LoRA alpha are verified. Adapter loading uses the existing V1 `load_adapter`
methods and never imports training code. Trainable parameters must have the
registered count and finite FP32 values.

`verify_lock` checks all metadata, code, contract, prerequisite and checkpoint
hashes before even hashing the validation files. It then hashes all four data
files. Only after this entire check succeeds can `read_validation` parse data.
It checks raw counts 1838/3270 before encoding. Known V1 candidate encoding
failures are excluded once for all models; other errors propagate. Rates over
0.5% are protocol deviations, reported with raw/kept counts and excluded IDs.

The later unsealing job runs from the locked snapshot, using the locked
environment, and invokes:

```bash
python src/v5_confirm.py evaluate --lock /absolute/committed-lock.json \
  --output /work/xg24i002/x10041/bypass-lora/runs/ATT-ID --unseal
```

An exclusive `flock` on the receipt's permanent `.guard` file is held through
evaluation, recovery and all receipt transitions. A concurrent initial attempt
or retry cannot reach the semantic reader. Process exit releases ownership.
The guard file is never removed, so all contenders lock the same inode.

The receipt stores the SHA256 of the entire lock plus the originating attempt
and absolute output directory for every admitted run. An engineering retry
before all predictions complete requires the same lock and `--resume`, with a
fresh attempt output directory. Each model writes its metrics and an inventory
with prediction/metric paths, byte sizes and SHA256 before atomically publishing
`predictions.jsonl`. All JSON and prediction publication uses flushed temporary
files followed by atomic replacement, preserving the previous receipt on a
failed replacement.

Under exclusive ownership, a retry reconciles the persisted bundles of every
prior attempt before any raw-data read. If all ten exist and their hashes match,
it regenerates and durably writes `analysis-input.json` before marking the
receipt `predictions_complete`, then refuses further evaluation. Thus a crash
immediately after the tenth prediction publication still selects analysis-only
recovery. An interrupted analysis-manifest write or receipt replacement can be
retried through the same reconciliation path. After that refusal, run only
`v5_analysis.py` on the recovered `analysis-input.json` in a PBS job.
No checkpoint/method/rule changes are permitted after any validation output.

## Analysis interface

`v5_analysis.py --manifest PATH --split dev|validation --output DIR` uses an
`entries` array. Every entry has `arm`, `seed` (`null` for Base), `predictions`
(V1 JSON array or JSONL), and optionally `metrics`. Relative paths resolve from
the manifest directory. All nine trained entries are required for either split.
Base is optional for dev and required for validation. IDs and gold labels must
match across models; duplicates and differing omissions fail.

Confirmation metrics sidecars contain `{ "metrics": { "wikitext.nll": ...,
... }, "validation": { "piqa": <audit>, "boolq": <audit> }, ... }`.
Each audit has `raw_examples`, `examples`, `excluded_ids`, and
`protocol_deviation`. Validation analysis checks complete raw counts, disjoint
kept/excluded IDs and identical audits across all ten models. WikiText NLL for
Base and all nine checkpoints is mandatory in validation. Dev can omit WikiText
or provide only the measured Base reference. Direct WikiText inputs can instead
be supplied as `wikitext[arm][seed]`, and `wikitext.Base.nll`.

Accuracy uses correct/example counts; task NLL is total gold CE divided by total
gold supervised tokens, and macro NLL averages the two task NLL values. Seed SD
uses `ddof=1`. Point-estimate gates use exact count ratios to preserve inclusive
thresholds. WikiText concatenates raw rows without added delimiters, tokenizes
without special tokens, uses nonoverlapping 1024-token blocks and drops the short
tail; only the 1023 within-block predicted positions contribute per block.

Intervals quantify data uncertainty conditional on the three trained seeds.
The +1 pp and per-task -1 pp conditions are point-estimate gates; supported does
not establish a 95% guarantee of at least +1 pp or taskwise noninferiority, and
challenged does not establish equivalence. Secondary intervals are raw and are
not Holm-adjusted simultaneous intervals. S1 is a budget comparison rather than
an isolated fusion-location intervention. Training/latency cost measurements
are owned by T-0013 and will be reported with the eventual scientific delivery.
