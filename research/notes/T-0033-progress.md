# T-0033 / E-0030 execution state

Implementation source (initial preflight): `e7b1c6c25e08bc0289bc156a61106a8b1ce64e44`.
Branch: `sr/researcher-2/T-0033`. A2 remains sealed.

The new path supports seeds 3, 4, 5 without changing the E-0027 optimizer,
sample/step budget, architecture or LR lock. Keep's dev iv_off check runs
inside its training allocation after final-checkpoint reload; no dev iv_swap
or iv_reset is run. All four arms retain initialization, 25/50/75% and final
dev predictions. The legacy LoRA CUDA initialization is unchanged; every run
records torch's actual MIG name and PBS job ID. This follows MSG-0934.

The E-0030 pre-opening data loader hashes source files but replaces held PIQA
rows with ID-only placeholders before JSON/label conversion, and filters held
BoolQ rows in Arrow before Python row conversion. Its PBS smoke checks all
six order streams, including the previously registered seed 0/1/2 hashes,
and compares the exclusion audit to ATT-0306-1. This follows MSG-0936.

Before registering the new runs, the recorded E-0019 through E-0029 run specs
were inspected through `rl run show`; the only seed values present were 0,
1 and 2. Seeds 3, 4 and 5 are new within the recorded RB/R1 experiments.

PBS preflight: RUN-0374 / ATT-0374-1, job `3779857.opbs`, debug-mig, one MIG,
15-minute walltime based on ATT-0304-1's 358-second predecessor plus additional
loader/tests. It was observed running at submission. Output:
`/work/xg24i002/x10041/bypass-lora/research/artifacts/ATT-0374-1/`.
Its results, actual cost and hardware have not yet been validated. Notification
`N-20261103-024958-7df8` (`T0033-preflight`) targets the coordinator's listener;
notifyd is running. The small source snapshot is read-only. The declared
RunSpec uses the resource's default `runs/` planned location, while the actual
job explicitly writes the session-required `research/artifacts/` location.

Next: validate/import the preflight result and exact measured scheduler usage;
register seed 3/4/5 order hashes; request the full E-0030 changed-code review;
run deterministic checks at the final reviewed source; submit the twelve
formal training runs only after review and the <=5.5 node-hour projection pass.
The A2 reader/evaluator/analysis is implemented but unreviewed and unexecuted.
No A2 job may be submitted until all pre-opening checks pass and the coordinator
confirms the committed A2 lock SHA256 with a decision.

## Failed first preflight: ATT-0374-1

PBS `3779857.opbs` ended with exit 1 after 73 scheduler seconds on mg1101,
actual 2g.24gb. The deterministic suite reported 394 passed and three failed,
all in the new rules tests. It stopped before loading task data or timing.
Measured usage is U-0383: 0.020278 MIG-hour = 0.0050695 node-hour.

The complete `checks.log` and inspected fixture code establish three causes:

- The exact-quantity test asserted a zero lower secondary interval after only
  32 draws. With numpy's linear quantile interpolation that is not guaranteed
  even if one draw is zero. The test now uses the preregistered 10,000 draws
  and explicitly asserts a zero lower bound; the positive point estimate alone
  must still fail the secondary confirmation criterion.
- The fixed-sequence test declared 500/600 correct rows but supplied unrelated
  100/1-row vectors. Production correctly rejected the mismatch. The corrected
  vectors are generated from the declared counts, so it reaches the intended
  fixed-sequence gate with Keep identical to the better control.
- The malformed-length test expected the text `row count`, while the correct
  error is `one-dimensional and match count totals`. It now matches that exact
  validation error.

No production threshold, validation requirement or analysis behavior changed
for these fixes. A fresh PBS preflight is required; no result is assumed.

Replacement preflight RUN-0375 / ATT-0375-1 executes commit
`967f551ede63b8a7f1850bd395f805ada4b9b7ca` as PBS `3779991.opbs`, one
debug-mig instance, 15 minutes. Submission observed it running. Results are
pending under `research/artifacts/ATT-0375-1/`; notification
`N-20261103-030242-b40e` is sealed and notifyd is running. After it ends,
resume the validation and training-source registration/review steps above.

## Validated replacement preflight: ATT-0375-1

The complete manifest and its raw checks.log/XML, plan, initial-digest and cost
outputs were inspected. PBS `3779991.opbs` ran on mg1097_mig2, where the in-job
guard and torch both report 2g.24gb. The suite passed 397 cases, 0 skipped,
0 failures/errors. Four smoke arms reloaded identically after 132 uniform
training steps each, with full parameter gradient coverage and expected state
shapes/accounting. No A2 rows were selected, encoded or scored.

The masking loader reproduces the registered seed 0/1/2 order SHA256 values,
26,230 examples and 13,115 steps each, plus the old exact input-token totals
2,000,069 / 2,005,134 / 2,002,226. Its exclusion audit equals ATT-0306-1;
the pinned 159-byte file's SHA256 is
`b67a489da0a81f8c6c3fcfd112182c9a0e77df5fd468c6d1547166fec83780b5`.

The three new plans are registered in src/seg_data.py before formal training:

| seed | order SHA256 | examples | steps | total full-protocol tokens | segment-2 tokens |
| --- | --- | ---: | ---: | ---: | ---: |
| 3 | 108aec8c1c9684c5e4a58f91da0c5b7aa56e91aab25c7b07d45874e418774b14 | 26230 | 13115 | 2003216 | 567856 |
| 4 | 1c94716a5e449bb711b7135751f6161f062c2456a29065de5e557b144e1f5644 | 26230 | 13115 | 2001319 | 566215 |
| 5 | 6005c84bf0e8610f1dda085736c9fc8f3cdd3aa3d63a0a38211939c5aa22bb23 | 26230 | 13115 | 2003710 | 566084 |

SEG-05/06/07 initial parameter digests agree elementwise within each seed,
with distinct digests across the three seeds. SEG-04 also has distinct
initializations across seeds; its digest remains slice-dependent by design.

Measured usage U-0384 is 0.1 MIG-hour = 0.025 node-hour. Both preflights cost
0.0300695 node-hour. The measured part model projects 4.27048606125148
node-hour before actual smoke usage, or 4.30055556125148 including both
preflights, below the 5.5-node-hour cap. Projected training seconds per arm:
SEG-04 3545.31, SEG-05 7481.36, SEG-06 3523.50, SEG-07 3301.95.
At 1.5x rounded up to 10 minutes, request 3h10m for each Keep job and 1h30m
for each other training job on short-mig. Estimated A2 evaluation is 2356.22s
per seed (future; requires the confirmed lock before submission).

Pinned preflight output SHA256 values:

- manifest.json: ced4505f542913c3ad41cc70d59322af0106317717d6edd37463fa35e2a14203
- seed-plans.json: 6b62b5129e8ddf8676e7dade35aa597f28a1c12f7ef5ef2520927d90ab788f0a
- initial-digests.json: a2a54283da3e1e8ef4c56f0222e465978bc8a2fabfae6703c4cdedff842cfec7
- cost-smoke.json: 265ff9fc95732ba95a056bc1d0eb62a2405365768a45fe5110e68e4f9af1d39f

Formal runs remain unregistered pending the changed-code review and final-source
deterministic checks. A2 remains sealed.

RES-0349 imports the validated successful preflight; U-0384 imports its measured
usage. New order constants are committed at
`61cd97b1881dfe3c6045c425076023f31ca91b03`. Changed-code review RV-0097 has been
requested against E-0027's 054a54a baseline, including the new masked training
path, lock, reader, evaluator, rules, analysis and PBS boundary. Reviewed files
must remain unchanged until its answer.

Candidate-source deterministic checks RUN-0376 / ATT-0376-1 are concurrently
running as PBS `3780105.opbs` (debug-mig, 2-minute walltime from measured
57.5-second suite plus allocation/startup overhead). Outputs:
`research/artifacts/ATT-0376-1/`; notification `N-20261103-031339-bceb`.
The coordinator must resume this role on the review answer or job notification.
Before formal runs, inspect/import these outputs and measured usage, resolve
review findings and rerun affected checks if the candidate source changes.
