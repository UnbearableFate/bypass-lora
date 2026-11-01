# T-0020 confirmation lock and no-data preflight

The confirmation lock is committed in a09c3181ba9e2eb9c4ca2e63004845c54269455d.
Its SHA256 is f7b52c1b5c0d856a1e704d183076337134df997f1bfcfb39d9bc8bb3615d05e2,
14646 bytes. The immutable runtime path is
/work/xg24i002/x10041/bypass-lora/snapshots/a09c3181ba9e2eb9c4ca2e63004845c54269455d/research/artifacts/T-0020/confirmation-lock.json.
The worktree copy has the same bytes and is included in the review scope.

Scientific evaluation source is deliberately separate:
2b5c809f6101b5ba132d5565ebbf73ef91e105da, the formal V5 training snapshot.
All nine REQUIRED_CODE source/dependency files are byte-identical to the
RV-0039-reviewed334fdb4 versions (git diff exited0), and the lock pins each
SHA256. No src files were changed for this task. Current master includes later
memory-route changes and is not used for evaluation.

The lock copies learning_rate_lock and all nine checkpoint entries exactly
from ATT-0174-1/checkpoint-inventory.json:10830 bytes, SHA256
2886d4981d2fe954bfc022b2d5b6641af638d1d0529d0012755538bd60619619.
Each final checkpoint is tied to its matching arm/seed training manifest and
checkpoint.json inventory. Learning-rate lock SHA256 is
70ba5356203afc128cbf3e4c14a7b4474ece54a55d28b8401b852d957069e01a;
all arms use0.001. All nine final runs record29165 steps,58329 examples,
4173300 nonpadding input tokens and exact same-job reload. C/F parameters
are524288, L630784. No alternate LR lock or slice comparison is used.

Contract pin is E-0013@3, SHA256
a25529bc011fd8b639378a8aa3318b0e4cde835ba3489cdf51fa6da2ba4b60e7.
The contract and RV-0039 answer FileRefs use immutable files in2b5c809.
Dev prerequisite is ATT-0116-1/analysis/analysis.json, explicitly split=dev;
its duplicated proxy seeds only establish software rehearsal. All four sealed
data FileRefs are copied from the T-0014 sealed manifest, with pinned source
revisions, hashes and byte sizes. No data contents, lengths or labels were
inspected during lock assembly.

The one receipt for all confirmation attempts is
/work/xg24i002/x10041/bypass-lora/research/artifacts/T-0020/unseal-receipt.json.
Neither it nor its permanent .guard file existed at preflight. The reviewed
flock/receipt/recovery machinery is unchanged. Any incomplete engineering
retry must use the identical lock/source and a fresh attempt/output with
--resume. A complete prediction grid admits analysis-only recovery.

## Observed preflight

ATT-0176-1 / RES-0174 / PBS3767773.opbs completed with Exit0 and a complete,
passed application manifest. Its allocation artifact confirms a live Miyabi-G
compute allocation before application execution. PBS measured2 seconds;
application wall was0.131773 seconds. The native measured usage export will
be imported once when available (2 MIG seconds =0.000138889 node hours).

research/notes/T-0020-preflight.pbs ran the locked verify_lock, wrapping only
its verify_file boundary. It verified40 metadata/checkpoint files plus the
nine source/dependency hashes, and checked all four sealed-data reference
structures before stopping at the first sealed-file hash. builtins.open and
Path.open were guarded against every sealed data path; validation/WikiText
readers and model/tokenizer entry points were forbidden. No sealed file was
opened, no receipt/guard was created and there were no blocked access attempts.

Original result paths:

- /work/xg24i002/x10041/bypass-lora/runs/ATT-0176-1/preflight.json
- /work/xg24i002/x10041/bypass-lora/runs/ATT-0176-1/allocation.json
- /work/xg24i002/x10041/bypass-lora/runs/ATT-0176-1/manifest.json

Exact preflight/allocation copies are committed as preflight-result.json and
preflight-allocation.json under research/artifacts/T-0020 for this review.
The preflight is engineering evidence only; no H-0008 outcome is observed.

## Measured-cost opening plan

The final nine V5 dev reload evaluations sum with the measured Base reference
to1820.880600 seconds. Preflight preserves each run's unrounded timing. Use
the larger prespecified row expansion3270/1000=3.27 and300 seconds for all
WikiText passes, loading, encoding and analysis:6254.279563 expected seconds
(104.24 minutes,0.434325 node hours). The300-second allowance builds on the
T-0014 Base WikiText measurement (15.15s forward,23.79s end to end) for ten
models and overhead. Validation sequence lengths remain unobserved.

Expected x1.5 is9381.42 seconds. Propose short-mig02:40:00 (9600s), one MIG;
requested upper cost0.666667 node hours, within T-0020's1-node-hour cap and
the shared E-0013 budget. This is an estimate, not validation timing. T-0013
has separately recorded5.81138889 node hours for all its work.

research/notes/T-0020-unseal.pbs preserves the reviewed evaluator, pinned
environment and immutable lock. It requires the allocation check, records
hardware metadata only (D-0018/D-0019), then invokes evaluate --unseal. The
evaluator already calls the locked10000-draw analysis after the ten model
bundles are complete; it is not necessary to rerun successful inference to
produce analysis. Analysis-only recovery remains available if needed.

No unseal job is submitted until this lock/preflight review passes. After
opening, inputs/methods/checkpoints/rules remain fixed. Any evidence edge is
dictated only by the sealed four-part supported rule or upper-CI<+1pp
challenged rule; the coordinator owns the hypothesis conclusion. The final
report will state fixed-seed data uncertainty, point-estimate gates, raw
secondary intervals, S1's budget-comparison scope, exclusions, WikiText and
the separately measured training/latency cost tradeoffs.
