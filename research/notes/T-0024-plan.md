# T-0024 execution plan

Contract E-0020@1 (sealed DOC-0231@1). Task claimed by researcher-2. Base commit cbfa5d80f68dc4e410c53722888acd9607444dcb includes the coordinator's merge of T-0022.

The complete training/data/core/environment path is byte-identical to RV-0062's reviewed snapshot de4443b130653bfbd6c1418ce69d582ee75b57af. No scientific code, data, dependency, tolerance or metric changes are planned.

Run the twelve committed initial-grid configurations in one PBS notification group. Each candidate executes the reviewed rb_train.py run --stage calibration, seed 0, phase V0, full 200k input-token budget and full calibration. No dev monitoring, checkpoint export, fresh reload, RB confirmation or official validation evaluation. The A2 ID manifest is reused, not regenerated.

Per the measured one-pass RB0 forecast, each candidate is approximately 0.042–0.0435 node·h (600–630 MIG seconds). Request 00:16:00 per initial candidate (expected 640s, about 1.5x), debug-mig, one MIG instance. Total predicted usage is 0.51 node·h for the initial grid and 0.68 node·h with all four extensions, within the 1.5 cap. All actual PBS seconds are converted exactly by /14400 and imported once. Ordinary job profile checks only record metadata per D-0018/D-0019.

After application verification and imports, call the reviewed extension_lr separately for each family's exact initial grid in a PBS report job. Stop on a family without finite candidates. Submit only the required one-time extensions in a second notification group. Then execute the reviewed lock CLI over all expected calibration manifests in PBS, preserving manifest SHA256 references and the split SHA256. Supplement its output with authenticated common plan order metadata if needed; do not change its LR decisions.

Register one diagnostic EV about E-0020/H-0013 with no supports/challenges direction. Deliver the committed lock, candidate table, explicit expansion/1%/boundary decisions, exact usage and ATT/RES/EV references. No RB1 training is authorized by this task.
