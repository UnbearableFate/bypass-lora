# T-0015 implementation notes (E-0014@1)

## Data: E ∈ {1, 4, 8} variants (ATT-0112-1, RES-0103)

- Generator `src/v2r_data.py`, version `concat-loop-v2-t1-e{E}-1`, phase `V2R`,
  base seed `derive_seed(20261029, "concat_loop_v2", "V2R", split, E)`.
- Episode RNG `derive_seed(base_seed, version, core_layout, index, attempt)`. Heads are
  `rng.sample` (distinct within an episode); tails come from `rng.choices` with V2's
  3:1 frequency weights. Relations: E=1 uniform over four, E=4 each once, E=8 each
  twice (shuffled). The target is `rng.randrange(E)`. Support order alternates
  natural/shuffled by index, as in V2.
- Templates, the V2 seen name pool (bytes copied, SHA-256 pinned
  `a6bde954…`), single-token answer checks, delimiter checks and the semantic content
  hash are the V2 ones.
- Deduplication. The order of generation is dev, then calibration, then train.
  - dev and calibration contents are unique within and across both splits. A colliding candidate is redrawn with the next `attempt`; the attempt is stored in the row (`generation_attempt`).
  - Train contents are disjoint from dev and calibration, again by redraw.
  - Repeated contents inside train are kept and counted. Reason: at E=1 only 92·4·92 = 33,856 contents exist. Forcing 19,432 unique released contents would push the high:low tail ratio from 3:1 to about 2:1 (estimate).
  - Generating the evaluation splits first keeps dev/calibration on the intended distribution.
  - This departs from V2's literal "raise on any duplicate", which V2 never needed. The departure is flagged for review.
- Confirmation: an identity registry only (`identity`, `episode_id`), with no content and no content hash, as the contract says. Before any later opening, the registry's content would need the same redraw rule against released contents.
- Results:

  | E | train unique contents / repeated rows / redrawn | cal redrawn | dev redrawn | prior dev EM | metadata SHA-256 |
  |---|---|---|---|---|---|
  | 1 | 11,993 / 4,391 / 1,877 | 134 | 22 | 0.019 | 46fa324c… |
  | 4 | 16,384 / 0 / 0 | 0 | 0 | 0.012 | ccc271d4… |
  | 8 | 16,384 / 0 / 0 | 0 | 0 | 0.014 | 58143a37… |

  - E=1 high-frequency answer share: dev 742/1000, train 12,227/16,384 (nominal 141/186 = 0.758).
  - Metadata hashes are pinned in `METADATA_SHA256`. `load_variant` checks metadata, split bytes, version, split and fact count.

## Training options (all default off; `src/v2_train.py` CLI)

| option | arm | behaviour |
|---|---|---|
| `--step-budget N` | all | the first N batch-8 steps of V2's three-epoch order. On V2 data, N=6000 gives exactly V2's 4M-token schedule (same batches, 4,000,470 tokens, warmup 300; tested). |
| `--train-facts E --variant-data E=DIR` | R1, R2 | train on the variant's train split (same order seed; every split has 16,384 rows) |
| `--curriculum 1:1000,4:1000,8:1000,16:3000` | R3 | one schedule over the concatenated train sets. Segment k uses the first `steps` batches of its own step schedule, so steps 1–1000 are E=1, … 3001–6000 are E=16. One warmup/cosine over 6000 steps. At each segment end, normal EM on that E's calibration (monitor only). |
| `--prior-rank 32 --prior-steps 1000 --prior-lr 3e-4` | R4 | `PriorPath`: `SiLU(u A) P` added to the layer-14 output. A ~ N(0,1/d), P = 0. It is applied only in the M-line question forward (`phase='query'`); the support pass never sees it. Stage 1: memory parameters have `requires_grad=False`; the query reads an empty (reset) memory after the normal support pass; only `prior.*` are optimized for 1000 steps at LR 3e-4 (5% warmup and cosine, the first 1000 batches of the V2 order, AdamW wd 0, clip 1). Stage 2: `prior.*` frozen, 6000 memory steps. The run asserts the memory tensors are unchanged by stage 1 and the prior tensors by stage 2. The prior path is on in every evaluation control. |
| `--address-loss-weight 1` | R5 | loss = answer CE + λ·mean(−log w_target) at the last valid question position. w is SoftmaxSlot's softmax over the actual memory slots, recomputed from the captured query projection; the target slot is the (t+1)-th valid slot. Uses `target_fact_index` (extra supervision). The address term reaches only A_q, Rq and A_k (tested). |
| `--backend centered_cos_slot` | R6 | score_i = τ·cos(q, k_i − mean_{valid j in the same row} k_j), τ = exp(θ), θ₀ = log 10 (`reader.log_temperature`); softmax over valid slots as SoftmaxSlot. q and k are the same ELU+1 features as soft_slot. |
| `--extra-dev E` | R1–R3 | after training, variant dev with normal/zero/shuffled |

- New parameters are created after the V2 ones (the prior path last; τ uses no RNG), so a new option leaves the memory path's initialization bitwise unchanged (tested).
- With the defaults, the side `state_dict` keys and values are the V2 ones. `AdapterConfig` gains `prior_rank=0`; a V2 `config.json` without that field still loads.

## Evaluation additions (measurement only; EM/NLL/controls unchanged)

- Per control, `last_position`: argmax slot counts by support position at the last question position, the rate of hitting the target, the same-relation rate, the mean target weight and n_eff there. Fusion and prior increment norms are divided by the layer-14 input residual and by the post-attention hidden state (the E-0012 L4 definitions: F_out FP32 z@P). These are given for the last position and for all valid question positions.
- The H3 target weight stays V2's `diagnostics['address.target_weight']`, the mean over valid question tokens.

## Validation

- `src/test_v2r.py`: new unit tests.
- `src/v2r_default_regression.py`: old tree 4c5a57c vs new tree on V2 data. Three V2 arms; first-batch loss, logits and gradients; four formal `train()` steps; all six evaluation controls. Exact equality is required, with an old-vs-old determinism baseline.
- Smoke runs of R1–R6 run in the same job (ATT-0115-1).

### Results (ATT-0115-1, RES-0106, code 05b0841; 0.061 MIG-h)

- pytest `test_v2r test_core test_model test_v2_memory test_v2_train`: 159 passed.
- Default-off regression: old (4c5a57c) vs new is **bitwise identical**, and so is old vs old. The comparison covers losses, logits, every gradient, four `train()` steps (history and parameters), and the six controls' logits and scalar diagnostics, for V2-03, V2-01 and V2-07-like C_up.
- Smoke runs (8 steps, 16 dev rows):
  - All six arms complete; R4 and R5 use V2's order hash 0d4b55ed….
  - Gradients: R1 updates only fusion.P, A_v and Ro. With one slot the address weight is 1, so A_q, A_k and Rq never receive gradient at E=1; R1's E=16 evaluation therefore reads with untrained addressing.
  - R4: stage 1 updates only prior.*, and the freeze assertions held.
  - R6: log_temperature gets gradient.
  - R3: the monitors fired at steps 2, 4, 6, 8 for e1, e4, e8, e16.
  - Read reconstruction max abs ≤ 3e-6.
  - Smoke step seconds 0.13 (E=1) to 0.23 (R6); these include warmup.
