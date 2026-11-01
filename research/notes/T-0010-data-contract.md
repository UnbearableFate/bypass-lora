# T-0010 T2 implementation choices and validation

E-0011 is the controlling contract. Implementation adds `v3_data.py`,
`v3_prepare.py`, `test_v3_data.py`, and `v3_eval.py`, without changing V2's
generator, dependencies, or M-line code.

## Sampling and graph invariants

"Composition" means an **ordered composition** of twelve edges, uniformly
sampled from all admissible sequences. It does not mean uniform sampling over
integer partitions. This is the interpretation consistent with the contract's
approximate six-chain and 17% random-endpoint calculations. All lengths are
1–4 and at least two are >=3; the graph RNG contains neither hop nor support
order. Query selection uniformly chooses among chains long enough for the
requested hop, each contributing exactly one start at that distance from its
endpoint. Relations are uniform random draws from V2's four labels.

Chain vertices use V2's weights without replacement. Noise heads are distinct;
apart from the designated queried endpoint they belong to other r chains, so
no extra noise is placed on the queried chain. A designated endpoint's noise
tail is sampled from off-chain names to ensure traversal really terminates at
a wrong entity if relation labels are ignored. Other noise tails use the full
weighted pool and may overlap chain vertices. The untyped baseline ignores
all relation labels, follows the last-presented outgoing edge, and counts
cycles as failures. The queried trajectory has no outgoing-edge ambiguity.

The contract explicitly alternates **both** support order and terminal noise
by index. Therefore natural order always has terminal noise and shuffled order
never does. Untyped traversal consequently scores 0%/100% within those order
strata and about 50% within each hop. This prescribed coupling is reported in
the shortcut metadata; it was not changed to improve a gate. Per-order model
results must be read with this design limitation.

## Identity, sealing, and independence

Phase V3/V4 is explicit, with phase-specific `derive_seed` purposes and full
identity validation. Sealed confirmation, seen-name hops 4/6/8, and unseen-name
hops 1–3 have separate split purposes. Their files contain only identity,
episode ID, and semantic-content SHA256. Structural graphs are computed
transiently to hash them, without rendering text, answers, paths, or token IDs.
Only aggregate chain-count/composition distributions are reported for sealed
slices, to document their distribution shift.

The exact V2 name-pool file SHA256 is pinned and its bytes copied unchanged.
V2 split digests are verified against ATT-0039-1 metadata, and all original
content hashes are compared. A second comparison normalizes away task/hop
hash-schema differences for every readable V2 row. V2's sealed fact-set
nonoverlap also follows structurally: V2 has four edges per relation, whereas
every T2 graph has twelve edges of its queried relation. The old-registry
disjoint-label proof from T-0005 applies unchanged to T2 fact sets; preparation
rechecks the pinned legacy sources and registry rather than executing them.

Two full independent generation passes must match every file's SHA256.
Identity, task-specific semantic content, and task-normalized fact/query
content must each be unique across and within all slices.

## Tokens and offline fields

Train is generated in blocks of 1,024 until three epochs contain at least
5.5M measured **support+question** tokens. Metadata separately records
`prompt_input_tokens` and V2-compatible `input_tokens` including the one answer
token, so the answer is never used to inflate the 5.5M prompt coverage check.

Released rows preserve V2's text, token arrays, visible delimiter positions,
query labels, and axes, and add `path.entities` / `path.fact_indices` for
offline diagnostics. Hop is never rendered in the question. Every answer and
both M/F prompt-completion boundaries are tokenizer-checked. M Base sees only
question text; F Base sees support plus question. Evaluation accepts released
V3 seen-name train/calibration/dev data only, checks metadata and file hashes,
and rejects failed shortcut gates before loading any model. The Base command
itself evaluates dev only and reports full-vocabulary EM/NLL and subject copy,
including hop × support-order strata. No sealed model evaluation is exposed.

## Initial PBS smoke

Source `5f45c9770fd65911e8efb9c85c57d85244a38fcc`, ATT-0073-1,
PBS 3761734.opbs, compute host mg1098 (GH200 MIG): all nine tests passed in
2.54 s. The 480-row smoke (60 per slice) was independently byte-reproduced,
with 480 unique identities/contents, zero raw/normalized V2 overlap, verified
legacy disjointness, and passing shortcut gates. Generation took 1.1773 s;
entrypoint elapsed time excluding imports was 4.3610 s. Scheduler measured
15 s, or approximately 0.001042 node·h. These smoke data are not the final
V3 dataset. The Base evaluator was added after this snapshot and has not yet
run; full generation and all model evaluation await code review.
