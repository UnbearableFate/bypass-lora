# T-0017 implementation state

Contract E-0016@1, researcher-1 branch sr/researcher-1/T-0017, reference master
e1660c954ff1f1efb2738a1708d3eb095f060541. Runs are D1/D2a/D2b/D3 only;
shared support/schedule hooks also serve T-0018 D4/D5 (MSG-0430). No
confirmation access or hypothesis decisions. Task estimate about0.5 node hours;
the full E-0016 contract ceiling is0.9 including other researchers' arms.

New protocols live in v2r16_memory.py, leaving v2_memory.py untouched for the
parallel E-0015 branch. v2_train.train gains optional forward/curve callbacks;
defaults select its original functions. An old/new exact regression runs the
same initialization, gradients, optimizer steps and six controls from both
immutable code trees, using v2r_default_regression.py.

D1 masks the native memory's valid slots to the target's public slot_columns
mapping. It is explicitly supervised. D2 loads only its saved adapter into a
fresh wrapper; train() creates fresh AdamW state and the full schedule warmups
for300 steps. D2a freezes A_v/Ro/P and trains only A_q/A_k/Rq. New D2a/b orders
use derive_seed(20261029,concat_loop_v2,V2R16,arm,0)+epoch for epochs0..2.

D3 builds support once, repeats each question16 times in a batch and injects
each corresponding stored value only at the last question position. Address
log weights come from the normal differentiable SoftmaxSlot query/key score
at that same pre-read position. The objective mixes answer probabilities via
logsumexp; no target-fact metadata enters its training branch. Hard/soft reads
and marginal NLL are separate protocols; only hard last-position H3 is gated.

Common evaluation uses global cyclic shuffled donors, reports per-example
diagnostics and position0 versus position1plus strata. All-position address
means are weighted by valid question tokens. For D3 those weights outside the
last position are counterfactual diagnostics; no increment is injected there.

Coordinator MSG-0405 fixes D1 plateau: regress per-step batch-mean answer CE
against step over5001..6000; fitted decline=-999*slope; plateau if decline<.05
nat. Also report each raw500-step mean and final1000-step mean. The ending LR
is below about8% of peak; plateau describes this schedule, not convergence.
D1 failure still requires both tf margin<20pp AND plateau.

Coordinator MSG-0407 requires the actual allocated MIG2g.24gb profile. Every
smoke/formal job checks nvidia-smi's selected UUID against CUDA_VISIBLE_DEVICES
and checks torch's actual device name before tests, timing or training. A host
listing containing an unrelated2g slice is insufficient. Unknown or1g profiles
fail fast. No work falls back to login.

MSG-0430 adds keyword support_fn=build_support to v2r16_memory.train_forward,
evaluate_batch and evaluate. Both recipient and shuffled donor support use it.
The standard training path executes the same operations in the same order as
legacy train_forward; exact logits/loss/gradients are checked both with the
argument omitted and with an explicit wrapper of build_support. The evaluator
also checks full-record equality and donor propagation. make_arm_schedule
accepts D4 (released order) and D5 (derived-seed continuation order); this does
not add D4/D5 to the T-0017 runner's CLI. resume_provenance stays D1-specific:
R1 lacks its contract/arm/checkpoint inventory schema and uses nested metrics
with a normal-only reload check. Researcher-3 owns the R1 source validator.

The initial source7354cdf smoke ATT-0150-1 / RES-0138 passed all43 tests,
old/new exact default identity, four protocol save/reload checks and the D3
cost gate on the verified allocated MIG2g.24gb. See T-0017-smoke.md.
Next: validate the shared hooks in a registered PBS smoke, then request the
independent code review of the final implementation including those hooks.
After passing review, D1 and D3 run independently; D2a/b are released from
validated D1 final checkpoint provenance. All full jobs use6000x8, seed0,
LR3e-3 and final checkpoints. Only screening evidence will be recorded.
