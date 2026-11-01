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
