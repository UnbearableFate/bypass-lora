# T-0005 legacy data availability

Read-only probe commit: `5e6c4664dfb447af5eeb8c851a0275eaf4e2ed63`.
Run `RUN-0035`, attempt `ATT-0035-1`, result `RES-0027`.
PBS `3760163.opbs` ran on `mg1098`, exited 0, and produced a complete manifest
with a passing probe validator. Application elapsed time was 7.115 s; measured
PBS walltime was 9 s. Imported executor usage: 0.0025 MIG-instance-h, equivalent
to 0.000625 node-h (do not add both units as separate cost).

Artifact:
`/work/xg24i002/x10041/bypass-lora/runs/ATT-0035-1/legacy_inventory.json`.
It records SHA256 and sizes for all fetched legacy data and the GitHub tree.
The read-only source assets are under the ignored shared `data/looplora-a255e81/`.

## Observed availability

- The pinned public repository's `reports/experiment_registry.csv` has 169
  run-level entries, all reporting `relations-v1`, and 129 distinct referenced
  sample paths. None of these paths exists locally.
- The pinned tree includes nine committed `samples.npz` files. All nine were
  inspected using `numpy.load(..., allow_pickle=False)` without importing or
  executing legacy code. They contain episode/query IDs, answer/prediction
  fields, and evaluation diagnostics; none contains support facts or questions.
- These assets therefore do not presently enable the contract's exact comparison
  by fact set plus question. Distinct new IDs/seeds alone cannot establish it.
- Qwen3 tokenizer at the pinned revision was loaded offline. `;` is token 26,
  ` ;` is token 2587, newline is 198, and ` |` is 760. Full-template delimiter
  positions still need validation in the generator tests.

## Pending coordinator decision

Supply the old episode facts/questions (or their canonical content hashes), or
authorize a read-only source audit and independent reconstruction of the old
episodes, with the comparison scope stated explicitly. The current task request
permits reading only old data/registry files and prohibits running old code.
No claim of zero overlap has been made. No V2 data or Base evaluation has run.

After resolving this prerequisite: implement deterministic generator and split
registry, validate on PBS, request core-path review, and evaluate Base dev M/F.
