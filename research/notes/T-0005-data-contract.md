# T-0005 V2 data contract and independence argument

Implementation: `src/v2_data.py`, `src/v2_prepare.py`, `src/v2_eval.py`,
`src/test_v2_data.py`. Generator `concat-loop-v2-t1-1`; Qwen3 model/tokenizer
revision `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`.

## Legacy independence (coordinator authorization MSG-0093)

Read-only audit at LoopLora commit
`a255e814740ff85ab3b4ec4836c88f0a8f802b2d`:

- `src/looplora/data/render.py:19-23`: names are loaded from
  `assets/qwen3_names.v1.json`; 16 fixed relation phrases are knows, trusts,
  visits, helps, follows, mentors, admires, supports, greets, teaches, calls,
  thanks, hires, invites, guides, remembers.
- `render.py:77-88,91-102`: the name asset has distinct names/answer tokens,
  including a reserved Unknown; leading-space encodings are checked.
- `render.py:105-110`: every other nonnegative relation ID renders as
  `has relation {relation_id} to`.
- `render.py:113-128`: questions identify the start and relation, then request
  a one-hop answer, chain endpoint, or fixed-length walk, with optional
  update/missing instructions.
- `render.py:163-176`: each entity comes from the fixed name asset; facts render
  as `{head} {relation_phrase} {tail}.\n`.
- `src/looplora/data/relations.py:319-347`: old splits use independent episode
  seeds derived from base seed, `relations`, `relations-v1`, layout digest,
  split and index; seen splits also use minimum-entity modulo-three partitions.

New facts use relation labels **emails, photographs, tutors, rescues**. None
belongs to either the old finite phrase set or its numeric fallback. Every new
episode has 16 facts, each containing one of these labels. Therefore its set of
named `(head, relation, tail)` triples cannot equal any old nonempty fact set
under this renderer; adding the question to the comparison preserves that
non-equality. This is a construction proof of zero *labeled content* overlap,
not an empirical enumeration of unavailable old facts, a proof of different
graph isomorphism classes, or a claim of semantic novelty. Reusing some common
personal names is harmless to this argument. Renaming split IDs alone is not
used as evidence of independence.

The registry inventory found all 169 rows report `relations-v1`. The prepare
job checks this and pins the exact legacy files by SHA256:

- render.py: `eecc3ba0188e5815523dc15d7bd1046bb0a8e77cd05c948f8506e5c8ab1c9aab`
- relations.py: `94f8df780515d158e64fdd415d6a684e45a7805bb6d52aaad0422d8302d1f3a5`
- experiment_registry.csv: `f2419a7a6992b9a62a536641023d868baff77c7a8362c3d7b960f26efe53a251`

Source copies remain in ignored project `data/looplora-a255e81/`. The prepare
job parses the relation constant with AST literal inspection only; it neither
imports nor executes old code. `legacy_disjointness.json` records the check.

## Generation and sampling choices

All arm/seeds share the same train examples. Train size 16,384 is chosen to
cover each arm's 4M input-token budget within three epochs, with headroom; full
generation checks measured `(support + question + answer)` token coverage.
Eight arms reuse those data, so 32M aggregate tokens do not require eight
distinct train datasets. Calibration=2,048, dev=1,000, confirmation=2,000.
Smoke data have explicitly smaller sizes and must not be used for the final
Base evaluation.

The new hand-curated name candidate list is filtered only by exact single-token
leading-space encoding and decode. Seen/unseen pools are disjoint and stratified
by bare-name token length and designed high/low answer frequency (3:1 sampling
weight). This is **synthetic occurrence frequency**, not an estimate of name
popularity or backbone pretraining frequency. Both bare and leading-space token
lengths are recorded, along with actual support entity counts and train answer
counts. All current generated data use seen names. Unseen names are reserved
for sealed future layouts; no unseen examples or evaluations are generated.

Each episode independently samples 16 unique heads, four occurrences of each
relation, and 16 independent weighted tails. Tails may equal heads and repeat;
there is no answer exclusion based on the question head. The queried fact is
uniformly selected independently of its answer. Natural order and a seeded
shuffle alternate across indices, with an independent order RNG; explicitly
switching the order preserves facts, question and answer. There is one question,
15 distractor facts and no auxiliary path supervision. The constant prior is
chosen solely from train answer counts (tie: lowest token ID), then scored on
dev.

Seeds use the existing `v1_data.derive_seed(20261029, "concat_loop_v2", "V2",
purpose)`. Identity includes generator version, task/layout, base seed, split,
and index. Canonical content SHA256 sorts named triples and includes the query
head, relation and hop; it excludes support order and identity. Generation
rejects identity or content duplicates within/across all splits and checks
two independently regenerated stream hashes. No adaptive rejection changes the
fixed seed mapping: a collision is a hard failure requiring an explicit fix.

Multi-hop identities exist for train/dev hops 2/3 and sealed hops 4/6/8;
materializing them raises an error. Confirmation writes only identity and
hashes. Structural content is computed transiently to obtain its content hash;
no text, target, answer, token IDs, per-example statistics or model predictions
are written for confirmation, and the materialization/evaluation API rejects it.

## JSONL interface for T-0006 (MSG-0095)

Released rows contain:

- `identity`, `episode_id` (SHA256 of canonical identity), `content_sha256`.
- `support_text`, `support_input_ids`: exactly the 16 lines
  `{head} {relation} {tail} ;\n`, encoded without special tokens.
- `question_text`, `question_input_ids`: `Who does {head} {verb}?\nAnswer:`,
  encoded separately without special tokens. No leading answer space is in
  the question; it is part of `query.answer_text`.
- `facts`: in rendered support order, each `{head, relation, tail,
  delimiter_char_span: [start, end], delimiter_token_index}`. Character spans
  are zero-based, half-open offsets into `support_text` covering exactly the
  semicolon. Token indices are zero-based offsets into `support_input_ids`
  with `add_special_tokens=False`. A token may also cover adjacent whitespace;
  tokenizer offsets must cover each semicolon exactly once, and all 16 token
  positions must be distinct. Positions are calculated from text only.
- `query`: `{head, relation, target_fact_index, answer_text, answer_token_id}`.
  The target index refers to the rendered `facts` list, after shuffling.
- `axes`: hop, n_facts, n_distractors, name_pool, support_order, queried
  relation, answer_frequency_stratum, answer_bare_tokens,
  answer_leading_space_tokens, support/question/full-prompt token lengths,
  support/question character lengths.

**Offline-only fields:** fact triples, target_fact_index, and answer metadata
are for diagnostics/labels; they must not select write positions or enter the
model forward. The training pipeline should pass support tokens plus the
format-derived write mask, and question tokens plus all-valid-question read
mask. Answer token IDs are labels only. These arrays contain no padding; batch
padding must use sequence lengths, including when PAD and EOS coincide.

## Base evaluation

Only seen-name dev is accepted. M prompt contains question text only; F prompt
contains full support plus question, re-tokenized jointly. Both completion
boundaries are checked against a single leading-space answer token. Backbone
is frozen BF16, eager attention, eval mode, no cache across examples. Batches
are left-padded with explicit attention mask and positions. Full-vocabulary
argmax EM and FP32 one-token cross entropy are computed at the last prompt
position, without restricting logits to the name pool. Nonfinite logits fail.
Confirmation cannot be selected via an evaluation flag.

Runtime validation and actual artifact hashes will be recorded after PBS tests.
