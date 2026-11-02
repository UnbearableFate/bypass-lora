# T-0028 protocol, before any SEG scores

E-0025@1 / H-0019@1, D-0044, MSG-0726. This is the segmented F protocol,
separate from ordinary F and memory-only results. No confirmation examples
are evaluated. The coordinator owns hypothesis status and route decisions.

Encoding keeps V1's longest-candidate reserve and a **total 512-token budget
across both segments**. BoolQ segment 1 is exactly the V1 retained Passage
prefix. PIQA independently encodes `Goal: {goal}\n` and `Solution:` and crops
Goal tokens from the left within that same budget. Candidate tokens are
unchanged. Each row records prompt lengths and changed token spans at the
seam against the full V1 encoding. This choice was confirmed in MSG-0726
before any SEG score was read.

Two uncached backbone forwards, segment 2 positions start at zero, original
causal mask unchanged. Only four KernelMatrix M,c,counts tensors cross, with
their autograd graph. Segment 1 still executes side reads and residual exits.
Reset/T0/LoRA/Base skip segment 1; deterministic tests also execute and discard
it and require exact equality of the second-segment logits.

Rule 6 follows D-0044: if 0 < primary gain32 < 3 pp, release r128 once. A
gain128 >=3 pp with swap drop >= half the gain supports r128 only. Challenges
after widening requires **both gain32 <1 pp and gain128 <1 pp**; all other
outcomes are descriptive. Rules 3/6 use the selected primary task, including
BoolQ fallback. Rule 8 is task-specific; secondary-only success is inconclusive
and does not authorize another width, tap, or loop.

Deterministic tests and a timing smoke precede code review. The timing smoke
discards quality metrics and predictions. Formal training and inspection of
SEG dev quality require passing code review and a measured <=3 node-hour
cost projection, including the conditional r128 branch and all failed attempts.
