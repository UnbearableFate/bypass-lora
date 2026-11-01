# T-0020: V5 single-opening confirmation delivery

The authorized confirmation and locked analysis are complete. The sealed
rule classifies the confirmatory evidence as **challenged**: C_up minus
LoRA macro accuracy is-0.417652pp,95% bootstrap interval
[-1.020109,+0.197062]pp, whose upper bound is below the registered+1pp
minimum benefit. EV-0110 is the one confirmatory/valid evidence item and
adds only the required challenges edge to H-0008@2. The coordinator owns the
hypothesis-status decision and conclusion review. No route decision is made
by this delivery.

The interval includes zero. This evidence excludes the prespecified+1pp
benefit under its rule; it does not establish equivalence or LoRA superiority.
Intervals reflect evaluation-data uncertainty conditional on these three
trained seeds. Seeds are fixed, not resampled. The+1pp and per-task-1pp
rules are point-estimate gates; they do not provide a95% guarantee of at
least+1pp or taskwise noninferiority.

## Lock, source and the one opening

Lock: research/artifacts/T-0020/confirmation-lock.json,14646 bytes, SHA256
f7b52c1b5c0d856a1e704d183076337134df997f1bfcfb39d9bc8bb3615d05e2.
It was committed in a09c3181ba9e2eb9c4ca2e63004845c54269455d and used from
the read-only absolute path
/work/xg24i002/x10041/bypass-lora/snapshots/a09c3181ba9e2eb9c4ca2e63004845c54269455d/research/artifacts/T-0020/confirmation-lock.json.

Evaluation/analysis source is2b5c809f6101b5ba132d5565ebbf73ef91e105da;
its nine REQUIRED_CODE files match the RV-0039-reviewed334fdb4 versions.
The reviewed launcher is in snapshot03de68fc8b27826bf48bdfdf458571ec30ac99c9.
RV-0056 passed with0 findings and admitted the lock/preflight/launcher;
the coordinator authorized the opening in MSG-0546. No lock, source, data,
checkpoint or method changed after opening. No inference rerun occurred.

The fixed receipt is
/work/xg24i002/x10041/bypass-lora/research/artifacts/T-0020/unseal-receipt.json.
It has status complete, the exact lock digest/source, and exactly one run:
ATT-0179-1, output /work/xg24i002/x10041/bypass-lora/runs/ATT-0179-1.
PBS3767829.opbs completed Exit0 and the application manifest is complete
with its validator passed. Base and all nine final checkpoints were evaluated
in one job/process, using unmerged LoRA for task scoring.

The generic review checker retains a stale label because the separately
reviewed external lock/control files and unused slice-check modules are
absent from2b5c809. Their placement is intentional under the T-0020 brief;
the exact used external bytes were admitted by RV-0056/MSG-0546. Executed
source and its local import closure match the locked reviewed files.

## Primary comparison

Macro accuracy is the mean of PIQA and BoolQ accuracy. Each row below uses
the paired seed's independently trained final checkpoints.

| Seed | C_up macro accuracy | LoRA macro accuracy | C_up-LoRA | PIQA difference | BoolQ difference |
|---|---:|---:|---:|---:|---:|
|1|77.396935%|77.672347%|-0.275412pp|-0.489663pp|-0.061162pp|
|2|77.922153%|78.323450%|-0.401297pp|-0.924918pp|+0.122324pp|
|3|77.133435%|77.709683%|-0.576248pp|-0.816104pp|-0.336391pp|
|seed mean|77.484174%|77.901827%|-0.417652pp|-0.743562pp|-0.091743pp|

The95% paired task-stratified bootstrap interval is[-1.020109,+0.197062]pp.
The locked analysis uses10000 replicates and numpy.random.default_rng(20261029),
sharing sampled row indices across every arm and seed within each task.

| Sealed supported condition | Observed | Satisfied |
|---|---|---|
|seed-mean macro difference>=+1pp|-0.417652pp|no|
|95% interval lower bound>0|-1.020109pp|no|
|all three seed differences>0|all three are negative|no|
|each task seed-mean difference>=-1pp|PIQA-0.743562pp;BoolQ-0.091743pp|yes|
|sealed challenged condition: interval upper bound<+1pp|+0.197062pp|yes|

## Descriptive accuracy and NLL

Accuracy entries are seed mean plus/minus sample SD in percentage points.
Base is untrained and has no seed SD.

| Arm | PIQA accuracy | BoolQ accuracy | Macro accuracy |
|---|---:|---:|---:|
|Base|71.327530%|66.666667%|68.997098%|
|V5-C|72.052956% +/-0.547685pp|82.915392% +/-0.260089pp|77.484174% +/-0.401531pp|
|V5-F|71.962278% +/-0.517105pp|82.538226% +/-0.340536pp|77.250252% +/-0.248460pp|
|V5-L|72.796518% +/-0.661889pp|83.007136% +/-0.115778pp|77.901827% +/-0.365614pp|

| Arm | PIQA NLL mean +/- SD | BoolQ NLL mean +/- SD |
|---|---:|---:|
|Base|2.780734|1.823280|
|V5-C|2.552831 +/-0.003318|0.707940 +/-0.007182|
|V5-F|2.551289 +/-0.001751|0.683212 +/-0.027592|
|V5-L|2.574450 +/-0.004396|0.699242 +/-0.019120|

Task NLL is total gold CE divided by gold supervised tokens, then macro NLL
averages task NLLs. These NLL summaries and dev monitoring are descriptive,
not confirmation gates. Per-seed validation counts and scores, all final dev
metrics and27 epoch-monitor records remain in the source artifacts.

## Secondary comparisons

These95% intervals are raw; they are not Holm-adjusted simultaneous intervals.

| Comparison | Seed-mean macro difference | Raw95% interval | Raw two-sided p | Holm p | Reject at0.05 |
|---|---:|---:|---:|---:|---|
|S1 C_up-F_out|+0.233922pp|[-0.239300,+0.708736]pp|0.3356|0.3356|no|
|S2 F_out-LoRA|-0.651575pp|[-1.252136,-0.037860]pp|0.0346|0.0692|no|

S1's per-task mean differences are PIQA+0.090678pp,BoolQ+0.377166pp;
S2's are PIQA-0.834240pp,BoolQ-0.468909pp. Full per-seed and per-task
differences/intervals are in analysis.json. S1 compares two equal parameter
budgets while changing both interface and rank; it is not an isolated
fusion-location intervention. Neither secondary comparison rejects after Holm.

## WikiText capability-retention measurement

Base NLL is2.647376056. All ten models used298938 input tokens,291
nonoverlapping1024-token blocks,297693 predicted positions and a discarded
954-token tail. The source tokenization warning concerned the concatenated
string; model forwards used the locked1024-token blocks.

| Arm | Delta NLL vs Base,seed mean +/- sample SD |
|---|---:|
|V5-C|+0.200122 +/-0.031665|
|V5-F|+0.238017 +/-0.018026|
|V5-L|+0.336505 +/-0.015557|

These are relative capability-retention measurements and do not enter the
decision. Pretraining-data overlap is not excluded by this protocol.

## Parameters, mergeability and measured costs

Training/dev/latency costs are reused from authenticated T-0013 artifacts
(RES-0172 latency,RES-0173 cost tabulation), rather than measured again.
Training statistics below are means/sample SDs of three runs; peak CUDA is
the maximum allocated memory recorded by the trainer, not total device memory.

| Arm | Trainable FP32 parameters | Train seconds mean +/- SD | Peak CUDA GiB max | Static merge |
|---|---:|---:|---:|---|
|V5-C:C_up r32,4 layers|524288|5387.581 +/-103.557|3.0773|no|
|V5-F:F_out r64,4 layers|524288|5332.368 +/-263.386|3.0756|no|
|V5-L:all28/all7 rank1|630784|11397.997 +/-589.205|3.9846|yes|

| Inference mode | Mean latency ms | Median ms | p95 ms |
|---|---:|---:|---:|
|Base|36.650580|35.837197|42.620985|
|V5-C|37.803234|37.782364|38.540544|
|V5-F|38.090068|38.069726|39.004932|
|V5-L unmerged|55.760592|55.736401|56.867382|
|V5-L merged|35.929942|35.915564|36.759474|

Latency used the same fixed256 dev candidate inputs,batch1,16 warmups and
three timed passes (768 timings), with synchronized host forward walltime.
Merged LoRA is timing-only; official candidate scoring used the unmerged
checkpoint. C/F have equal parameter counts, while LoRA has about20% more;
this does not establish equal compute, equal state or parameter efficiency.

## Integrity, exclusions and reproducibility

The read-only closing audit ATT-0194-1 / RES-0189 / PBS3768665.opbs passed:

- The fixed receipt is complete and names exactly ATT-0179-1; there was no retry.
- All ten bundles' arm/seed/lock identities, absolute paths, byte sizes and
  prediction/metric SHA256 match; no attempts were mixed.
- Raw and retained counts are PIQA1838 and BoolQ3270 for all models, with
  zero excluded IDs, zero exclusions and no protocol deviation.
- Paired rows, IDs, labels and omission audits satisfy the locked analysis.
- All ten WikiText block metadata and NLL sums/denominators agree.
- The complete locked10000-draw bootstrap/Holm report reproduced exactly
  from saved predictions under the original source; raw sealed file opens
  were blocked and none was attempted. No model inference was repeated.

The T-0014 dev rehearsal used proxy duplicated seeds only for software tests.
This validation uses the actual nine independently trained V5 checkpoints,
seeds1/2/3, selected before opening. The fixed validation is now consumed and
cannot confirm a modified method. LoRA scope is rank1 and this short-calibration
LR/three-epoch recipe. V1 train/dev had already selected interfaces; this is
not zero-tuning transfer. D-0018/D-0019 are followed: profile metadata only,
no slice-related rerun, analysis or flag. The opening recorded2g.24gb on mg1110.

## Records, artifacts and actual compute

| Purpose | Attempt | Result | PBS | Measured MIG-h | Node-h |
|---|---|---|---|---:|---:|
|metadata preflight|ATT-0176-1|RES-0174|3767773.opbs|0.000556|0.0001390|
|single opening + locked analysis|ATT-0179-1|RES-0187|3767829.opbs|1.267778|0.3169445|
|read-only closing audit/reproduction|ATT-0194-1|RES-0189|3768665.opbs|0.000556|0.0001390|
|T-0020 total||||1.268890|0.3172225|

The jobs charged2+4564+2=4568 MIG seconds. The table uses the native ledger's
rounded MIG-hour quantities and the user rule node-h=MIG-h x0.25. Usage was
imported once per accounting reference. Total is below the approximate0.5
task estimate and1-node-hour cap. T-0013 separately reports5.81138889 node
hours for its complete work; earlier T-0014 costs are separate. E-0013's
12-node-hour aggregate is managed by the coordinator; no subtotal is double-counted.

EV-0110 is confirmatory/valid, challenges H-0008@2 under the sealed rule,
and cites the opening result, both opening/audit attempts and the audit/table
artifacts. No supported edge or hypothesis-status update was made. Code
reviews RV-0039 and RV-0056 have0 unresolved findings in this task's scope.

Durable artifacts under /work/xg24i002/x10041/bypass-lora:

- runs/ATT-0179-1/manifest.json and analysis-input.json
- runs/ATT-0179-1/analysis/analysis.json and analysis.md
- research/artifacts/T-0020/unseal-receipt.json (complete,fixed,retained)
- runs/ATT-0194-1/closing-audit.json
- runs/ATT-0194-1/analysis/analysis.json and analysis.md (exact reproduction)
- runs/ATT-0194-1/delivery-tables.json and delivery-tables.md
- runs/ATT-0175-1/cost-summary.json (authenticated cost source)

T-0020 execution is complete. Coordinator acceptance,conclusion review,
H-0008 status and subsequent route decisions remain. The branch has not been
merged or pushed by this worker.
