# T-0032 / E-0029 implementation and release gates

The eight prescribed arms are G-T, G-QN, G-N, Q-N, P-T, P-N, L-56 and L-48.
All use the sealed E-0029@1 protocol. The entrypoint is `src/rb_improve_train.py`;
`rb_train.py` supplies only the shared factory, train objective and arm registry.

Q-N receives the exact input_layernorm output for Q/K and exact v_proj output
for V. Q-N and X-01 have the same parameter shapes and initial values on Qwen3.
The up-only LoRA subclass uses the existing fan-in A/zero-B initialization and
checkpoint format, with only `mlp.up_proj` targets and alpha/rank = 1.

I-G uses the fixed 256-sample pooled RB train selection from the contract seed.
Population order is PIQA then BoolQ in source train order, after confirmation
and failed-encoding exclusions; sampling is Python Random.sample without replacement.
It records IDs and token counts. Its full-value sufficient statistics are chunked
to bound temporary memory. It collects every valid input position, including
indirect prompt gradients, at the actual up output. The formal example-mean loss
is divided by batch size2, with no parameter update. G is FP32. SVD retains32
left vectors; a covariance calculation supplies the scale, which is then checked
with an actual second forward pass on all256 examples (relative RMS <=1e-3).
Only A_v changes. New optimizers are created by formal training.

Read diagnostics use the first64 of those fixed256 training examples at step0
and final, for every side arm and both random/trained reference checkpoints.
They report self/max weight and n_eff by task, layer, all-input/answer-prediction
scope, and prefix bins1,2-8,9-32,33-128,129-512. Probability-based metrics normalize
weights to sum one; actual epsilon-affected self weight and weight sum are also kept.

The smoke never reads dev. It tests rule1, times80 steps for all eight arms,
runs one full G-T I-G collection including SVD/scale verification, times a full
4096-row calibration pass (not an LR candidate), and estimates the full cost.
Initial code-review and LoRA LR locking precede all new dev access. The LoRA
calibration uses V0 seeds/order,200k first-full-step token stop, the prescribed
three LRs and at most one endpoint extension. Locks record candidate manifests,
configs, predictions and step-log SHA256s and reconstruct their selection.

Formal training requires a valid LoRA LR lock even for side arms; it uses R1
seed0,13115 steps/26230 examples, inherited side LR1e-3, and the unchanged dev
schedule. It records D-0060 step0 identity with hardware-aware references and
requires reload identity. All runs use immutable source snapshots through
`research/notes/T-0032-run.pbs` and verify their live allocation before imports.

Analysis uses exact Fraction arithmetic for rules2-7 and the shared10000-draw,
seed20261101 task-stratified bootstrap for descriptive intervals. It withholds
scientific outcomes if any implementation check fails or E-0028 X-01 rule1 is
pending/invalid. It emits rule-specific screening directions, never hypothesis
status changes. The coordinator owns conclusion review and route decisions.

Budget: cap3.5 node hours, MIG elapsed instance hours/4. Full training waits for
the measured smoke cost table, a passing code review on the merged E-0028 base,
and the LoRA LR lock. A predicted excess is referred to the coordinator before
training. Source integration was explicitly authorized in MSG-0844; merge the
later reviewed E-0028 commit when the coordinator announces it.
