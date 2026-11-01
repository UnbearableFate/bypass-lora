# T-0014 delivery: sealed V5 evaluation and analysis

T-0014 is complete against E-0013@3. Official validation remains unopened: only
its downloaded bytes were hashed; its contents, lengths and labels were not
parsed. No H-0008 evidence or scientific conclusion was recorded.

## Code and review

Branch: `sr/researcher-1/T-0014`.
Final evaluated/reviewed source commit:
`334fdb4646fae0ad01caba6f8101f5a1887e59e8`.
Preceding implementation commits: `bd4fbe76f654eb2e099296e29c974dde47384d49`
and `4f944a2d37fad600ab17ec6f065c73b24241f68e`.
The final handoff adds documentation only after that source snapshot.

- `src/v5_confirm.py`: complete pre-unseal lock/hash/final-run checks, V1
  candidate scoring, common exclusions, WikiText NLL, process-lifetime ownership
  and atomic recovery that refuses reevaluation of complete predictions.
- `src/v5_analysis.py`: one dev/validation interface, exact point gates,
  10,000 paired task-stratified bootstrap draws, empirical secondary p-values,
  Holm correction, seed mean/sample SD, WikiText changes and JSON/Markdown.
- `src/test_v5_confirm.py`, `src/test_v5_analysis.py`,
  `src/test_v5_recovery.py`: numerical boundaries, pairing/stratification,
  provenance failures, V1 scoring parity, exclusions, WikiText blocks, process
  exclusion and fault-injected recovery.
- `src/v5_rehearsal.py` and `src/v5_rehearse.pbs`: immutable-snapshot PBS
  regression and dev-only rehearsal with a live allocation guard.
- `research/notes/T-0014-lock-schema.md`: exact final lock schema and recovery
  instructions; `T-0014-RV-0036-fixes.md` describes the three accepted fixes.

RV-0036 initially found one high and two medium provenance/lifecycle issues.
**RV-0039 passed with zero findings**, resolving all three at the final source
commit. Its answer is in the shared project at
`research/reviews/RV-0039.md`. Numerical methods, sealed manifest and dependency
files remained unchanged by the fixes. The training producer owner confirmed
the final-run inventory and LR-lock schema in MSG-0347; the reviewer also
inspected that producer for compatibility.

## Seal

The committed source/source-revision/URL inventory is
`research/artifacts/T-0014/sealed-manifest.json`; an identical metadata copy is
under the main project's same relative path. Raw files stay outside Git in
`/work/xg24i002/x10041/bypass-lora/data/v5-sealed/`. All four have mode 440;
their byte sizes and SHA256 were rechecked at delivery without parsing data.

| File | Bytes | SHA256 |
| --- | ---: | --- |
| piqa-valid.jsonl | 495929 | 93503cc97c679e459b065c3d13e848282e44b2a25213985bed3e5d458abef72d |
| piqa-valid-labels.lst | 3676 | b4192dc3a2a0363d9d60ccf79800cbbe2f32ebb17726efdde6970e0b8131bceb |
| boolq-validation.parquet | 1257630 | 52355d11524b4b874a9b9dcc278feb10f672d52c4f4eff9872e695ede59820f8 |
| wikitext-103-test.parquet | 732610 | 5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91 |

PIQA revision: `21edab439af693b961be2f069e8690a88d3b4e37`;
BoolQ revision: `35b264d03638db9f4ce671b711558bf7ff0f80d5`;
Salesforce/wikitext revision: `b08601e04326c79dfdd32d625aee71d232d685c3`,
configuration `wikitext-103-raw-v1`, split `test`.

## Observed validation and cost

All workloads ran in confirmed Miyabi-G PBS MIG allocations using the existing
3c916b51337d-cuda environment; no dependencies were changed.

| Attempt/result | PBS job/node | Application result | Charged MIG seconds | MIG instance hours |
| --- | --- | --- | ---: | ---: |
| ATT-0099-1 / RES-0100 | 3764379.opbs / mg1098 | 43 tests; Base WikiText; complete dev rehearsal | 37 | 0.010278 |
| ATT-0111-1 / RES-0102 | 3764474.opbs / mg1100 | 44 tests; complete synthetic orchestration and dev rehearsal | 23 | 0.006389 |
| ATT-0116-1 / RES-0105 | 3764621.opbs / mg1098 | **56 passed**, zero failures/errors/skips; provenance, concurrency, recovery; complete dev rehearsal | 26 | 0.007222 |

Total measured usage: **0.023889 MIG instance hours = 0.00597225 node hours**
using the executor's rounded records and the project's factor of 0.25, within
the 0.3-node-hour task budget. Execution end, results and usage were imported.
For ATT-0116-1, ledger export initially lagged at running; a verified terminal
`job_get` observation and the normal accounting reference recorded its Exit0
and 26-second measured usage. The later executor export must reuse that same
accounting reference rather than add another charge.

Final test time was 17.48 s; full application time was 24.6411 s. Pytest emitted
three Python 3.13 fork deprecation warnings in the two-process test, with no
failure. Lock contention was exercised on one compute node; the reviewer
confirmed the shared Lustre mount uses `flock`, not `localflock`.

Latest dev rehearsal:
`/work/xg24i002/x10041/bypass-lora/runs/ATT-0116-1/analysis/analysis.json`
and `analysis.md`; test details are in that attempt's `pytest.xml`/`pytest.log`.
The report runs all 10,000 replicates over 1000 dev rows per task. V1 seed 0
was copied to three procedural files; these are not independent V5 results.

Base WikiText measurement:
`/work/xg24i002/x10041/bypass-lora/runs/ATT-0099-1/base-wikitext/metrics.json`.
NLL **2.647376056184969** over **297693** predicted positions, using **291**
full 1024-token blocks from 298938 tokens and discarding a 954-token tail.
NLL evaluation took **15.1502 s**; the full WikiText operation including
tokenization/loading took **23.7888 s**. The tokenizer's long-string warning
was expected before blocking; only 1024-token blocks reached model forwards.

## Later unsealing task

Proposed short-mig walltime is **02:35:00**, based on a conservative roughly
103-minute estimate and the required 1.5 multiplier. The central estimate is
about 81 minutes. These estimates use V1 dev timings plus measured Base
WikiText, not validation token statistics; see `T-0014-rehearsal.md` for the
formula. Update from T-0013's final V5 dev timings before submission.

The coordinator should accept/merge this task branch, then assemble and commit
the actual nine-final-checkpoint lock after T-0013 completes. The required
training manifests, checkpoint inventories and LR lock belong to that later
preflight. No real final nine checkpoints, complete opening lock or actual
unsealing job was evaluated in T-0014. The code review pass establishes code
readiness; it does not establish the scientific hypothesis. No unresolved
review finding remains in this task's scope.
