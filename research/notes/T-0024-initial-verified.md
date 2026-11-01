# T-0024 verified initial calibration

All twelve initial-grid manifests passed the PBS artifact validation in ATT-0222-1 / RES-0205 / PBS 3768977.opbs, source 9be4a86fe1b676c44802af667e02104c982708a8. This is a separate artifact-only report module; the scientific trainer, data, model/core and dependency files are byte-identical to the RV-0062 reviewed snapshot.

Every candidate has complete/passed status, failed=false and finite positive NLL; seed0/phaseV0/calibration and representative arm/family match; model/tokenizer revision and uv.lock match; the allocation check confirms the registered compute job; full calibration contains exactly 2048 PIQA and 2048 BoolQ rows. The report recomputed the registered NLL from the saved gold-token sums/counts, checked no dev metrics/monitor/checkpoint/export/reload, and authenticated source/config/split/order/budget/accounting.

All candidates: 1096 steps, 2192 example presentations, 200229 nonpadding input tokens, same order SHA 053284831520676982857bf9677a9fa3e338b0bbad4d2ce2af1c3f88d20457d3 and A2 split SHA 34277aa4627271dbf4e0bcb9592e3014c6130d87b1766caad17c2d6b97a8a7cc. The step before the final step is below 200000 tokens.

Reviewed extension_lr returns:
- RBc-L: raw initial minimum at 1e-3, NLL 1.583488263911061; upper extension 3e-3.
- RBc-T: raw initial minimum at 1e-3, NLL 1.5967088477796325; upper extension 3e-3.
- RBc-R: raw initial minimum at 3e-3, NLL 1.559168157001273; upper extension 1e-2.
- RBc-C: raw initial minimum at 3e-3, NLL 1.585611915833004; upper extension 1e-2.

These are the only four new candidates. The final 1% rule runs after their results. No LR is locked yet.

Results: ATT-0210-1..ATT-0221-1 -> RES-0193..RES-0204 in the same order; detailed refs are in verified-initial-jobs.json. Every result was imported once with no mismatches. Each exact PBS walltime was imported once under miyabi-pbs:<job>. Total initial training: 7230 MIG seconds = 0.5020833333333333 node·h. Artifact report: 13s = 0.0009027777777777777 node·h. Total so far: 7243s = 0.5029861111111111 node·h. This exact scheduler sum supersedes the rounded total in MSG-0619.

The endpoint jobs will use expected 660s and walltime 00:16:30 (1.5× rounded observed maximum 654s). Four endpoints project another 0.183333 node·h; the total remains comfortably below the 1.5 cap. Ordinary profile metadata remains nonfatal; no slice comparison or sensitivity analysis.
