# T-0014 rehearsal and unsealing cost estimate

Official validation remains sealed. Its contents, lengths and labels have not
been inspected. This note records software validation and a Base-only WikiText
measurement; it is not evidence about H-0008.

## Observed run

Code commit: `bd4fbe76f654eb2e099296e29c974dde47384d49`.
RUN-0099 / ATT-0099-1 / RES-0100 / PBS 3764379.opbs completed on mg1098,
GH200 MIG 2g.24gb. The allocation artifact confirms a running Miyabi-G
allocation before any workload. PBS exit status was 0; application manifest
status is complete and its validator passed. The final executor observation
and measured usage were imported together; usage is U-0114.

- 43 tests passed, with zero failures, errors or skips; pytest elapsed 9.889 s.
- Application elapsed 35.2993 s; PBS charged walltime 37 s.
- 0.010278 measured MIG instance hours, approximately 0.0025695 node hours
  after the project's quarter-node conversion. No second usage record is added.
- Environment observed: Python 3.13.13, torch 2.10.0+cu128, CUDA 12.8,
  transformers 4.57.1; the existing 3c916b51337d-cuda environment was unchanged.
- Dev analysis completed 10,000 paired stratified bootstrap replicates in
  0.3801 s on 1000 PIQA and 1000 BoolQ rows. Source seed 0 was duplicated
  into seeds 1/2/3 solely for procedural rehearsal; no H-0008 evidence was added.
- WikiText Base NLL: **2.647376056184969**; total CE 788105.320293872 over
  297693 predicted positions. Tokenization produced 298938 tokens, 291 full
  1024-token blocks and a discarded tail of 954 tokens. Forward NLL evaluation
  took 15.1502 s, Base loading 0.7100 s and the entire WikiText operation
  including tokenizer preparation took 23.7888 s.

The tokenizer warned that the concatenated token string exceeds the model's
nominal maximum. This string is only tokenized: model forwards receive one
1024-token block each. No indexing error occurred. The warning did not change
tokenization or truncation.

Artifacts (absolute shared paths):

- `/work/xg24i002/x10041/bypass-lora/runs/ATT-0099-1/manifest.json`
- `/work/xg24i002/x10041/bypass-lora/runs/ATT-0099-1/pytest.xml`
- `/work/xg24i002/x10041/bypass-lora/runs/ATT-0099-1/base-wikitext/metrics.json`
- `/work/xg24i002/x10041/bypass-lora/runs/ATT-0099-1/analysis/analysis.json`
- `/work/xg24i002/x10041/bypass-lora/runs/ATT-0099-1/analysis/analysis.md`

The subsequent regression adds a synthetic end-to-end orchestration test and
reports validation exclusion audits and Base WikiText NLL visibly in Markdown.
It will reuse the Base timing above; the model evaluation code is unchanged.

## Planning estimate, not a validation measurement

V1 E-0003 complete dev evaluation measurements on the same hardware:
Base ATT-0024-1: 143.918 s; F_out ATT-0025-1: 158.244 s;
C_up ATT-0028-1: 154.523 s; LoRA28 r2 ATT-0032-1: 246.709 s.
The nine-checkpoint-plus-Base dev-equivalent time is
143.918 + 3 × (158.244 + 154.523 + 246.709) = approximately 1822.35 s.
These are proxy timings: V5 F_out has rank 64 instead of 32 and LoRA28 rank 1
instead of 2. T-0013 supplies the final V5 dev timings and latency measurements.

Raw task sizes are prescribed by the contract (not measured before opening).
The average row expansion (1838 + 3270) / (1000 + 1000) = 2.554 gives roughly
77.6 minutes for candidate evaluation. Adding about 200 s for all ten WikiText
passes and loading/tokenization gives a central estimate of about 81 minutes.

Because task-specific timing and validation token lengths remain unknown,
use the larger task expansion 3270/1000 = 3.27 for a conservative estimate:
1822.35 × 3.27 + 200 s = about 102.7 minutes. Applying the required 1.5 factor
and rounding up gives a proposed **short-mig walltime of 02:35:00** for the
single future unsealing job. Requested cost is at most about 0.646 node hours;
the measured estimate is about 0.428 node hours. These are planning estimates,
not measured usage. Revise from T-0013's final measured V5 timings before actual
submission, while keeping methods and the sealed validation unchanged.
