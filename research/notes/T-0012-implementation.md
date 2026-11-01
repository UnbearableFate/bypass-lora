# T-0012 implementation and execution notes

E-0012@1 is the source of the measurement definitions. `src/v2_layer_diagnostics.py`
captures both feature families in one unadapted Base forward through layer 14,
separately for support and question. It stops after the attention residual norm,
avoiding the unused later layers and vocabulary logits. Base features are retained
as BF16 tensors in the attempt output, with train/calibration/dev identity order.

Offsets are checked against the released token IDs and fact rendering. L0 uses
the final name token even for bare multi-token names. A checkpoint from each tap
checks projection input equality in support and query phases (exact tolerance),
and checks explicit-slot read reconstruction against the deployed branch.

Softmax probes standardize using fitting episodes only. Full runs fit on train
indices 0--14335 and select on 14336--16383. Selection maximizes internal top-1,
then minimizes CE, then prefers stronger L2 and earlier epoch on exact ties. The
L2 grid is 0, 1e-4, 1e-2; Adam learning rates are .03 for softmax probes and .003
for bilinear probes. L0 has up to 64 epochs; other softmax and bilinear probes
have up to 32. Fitting never receives calibration or dev labels. No refit uses
the internal validation episodes. Constant baselines use frequencies in all train.
The six L0 controls must all reach 90% before interpreting a full run.

L2 compares all ten checkpoints with the seed-0 initialization generated through
the existing `v2_train.adapted` path, once per feature family. Both question-head
and last-question positions are reported. Written keys are positive normalized
features; queries additionally include Rq. L4 uses these same positive keys and
the actual backend scoring scale. Fusion amplitude uses the FP32 P increment
for F_out and the actual modified-minus-Base BF16 FFN output before residual add
for C_up/C_dual. The denominator is the layer-14 input residual norm; the
post-attention residual denominator is also reported.

L3 invokes `src/v2_probe.py` unchanged for every checkpoint, `--limit 4096`, which
reads train[:4096] and all 2048 calibration episodes. Its split checks and ridge
remain unchanged. Counts are recovered from the exact binary-fraction accuracy
and the reported held-out count. No confirmation file is opened.

All execution, including unit tests and smoke, goes through miyabi-mcp. A smoke
uses 512 train / 128 calibration / 64 dev episodes, two representative arms,
reduced probe grids, and 64/64 L3. It measures runtime and plumbing only; it does
not produce interpretable L1--L4 evidence if L0 fails. Full-run table expectations
are 120 probe rows, 10 L3 arms, and 14 L4 rows (ten trained, two random feature
families each scored with both backends so score margins are comparable).

## Recorded validation

- Initial implementation commit: `fed91c959a4ba3460f3956d98ef24a4e6a089139`.
- `RV-0027`: pass, no findings. The full L0 gate, full-size runtime/memory and
  all-checkpoint completion remain execution requirements.
- `ATT-0094-1` / `RES-0095`: PBS `3762986.opbs`, confirmed allocation `mg1098`.
  Five tests passed, no skips. Smoke completed 52 probe rows, two L3 arms and six
  L4 entries; pipeline 25.0819 seconds, measured PBS walltime 48 seconds on one
  MIG instance (0.003333 node·h). Both tap families had exact support/query
  projection-input equality and passing read reconstruction.
- The four-epoch smoke L0 qknorm tail/head controls were 1754/2048 and 1829/2048
  (85.64% / 89.31%); other controls were above 90%. The smoke is explicitly
  uninterpretable. Full mode selects among up to 64 epochs and three L2 values
  using train-internal validation and aborts before L1--L4 if any L0 fails.
- `src/v2_layer_cost.py`, added without modifying reviewed files, times full-size
  synthetic probe epochs/validation and all four bilinear ranks. It also executes
  all ten trained L4 geometry paths on the 64 cached smoke dev episodes and the
  unchanged L3 path at its full 4096/2048 shape. These are cost/plumbing checks,
  not evidence interpreted before the full L0 gate.
- `ATT-0095-1` / `RES-0096`: PBS `3763092.opbs`, confirmed `mg1098`, exit 0.
  Full-shape benchmark completed in 57.9269 pipeline seconds / 66 measured MIG
  walltime seconds (0.004583 node·h). It exercised all ten trained geometry paths,
  including soft_slot/C_up/C_dual, on 64 dev episodes and the unchanged L3 probe
  on 4096 train / 2048 calibration episodes. Observed environment matches the
  assigned torch 2.10.0+cu128, CUDA 12.8, Python 3.13.13, GH200 MIG 2g.24gb.
- Cost components from full-shape benchmark: extrapolated extraction 42.811 s,
  softmax fits 236.960 s, bilinear fits 111.335 s, ten L3 calls 555.481 s.
  A 20% allowance gives 1136 expected seconds (0.0789 node·h); 1.5x requests
  1704 seconds = 28m24s in debug-mig. Including the two checks, the expected
  E-0012 total is approximately 0.0868 node·h, below the 0.5 node·h ceiling.
