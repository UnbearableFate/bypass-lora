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
