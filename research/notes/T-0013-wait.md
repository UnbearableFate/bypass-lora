# T-0013: V5 implementation and first smoke

Task remains owned by researcher-2; not delivered. Coordinator owns the research
decisions. Official validation has never been read by this session or code.

## Current contract and source

- Released contract is **E-0013@3**, hypothesis H-0008@2. Coordinator MSG-0288:
  conclusion RV-0031/RV-0032 passed, D-0011 releases V5; LR calibration can start
  once the training code review passes.
- MSG-0280 corrects the original brief: V1 **example-mean objective**, cross-epoch
  pending batches, 29,165 steps, dev monitoring after steps 9,722/19,443/29,165.
  Raw finite initial-grid NLL minimum triggers one endpoint extension; afterwards
  choose the smallest finite LR within1% of the final minimum. These are in @3.
- Training commit `f4b2c19cdd9eef5a3a01d2272129877fe852ddb2`: src/v5_train.py,
  src/test_v5_train.py, src/v5_smoke.py. V1 files/dependencies unchanged.
- Latency commit `0343896` adds only src/v5_latency.py and src/test_v5_latency.py;
  measured inputs are first64 V1 dev rows per task, both candidates =256 inputs,
  batch1,16 warmups,3 timing passes. Seed1 per arm. Merged LoRA is timing only;
  merged BF16 rounding is disclosed and unmerged checkpoints remain for scoring.
- **RV-0033** reviews the training path; **RV-0034** reviews latency. Both pending.
  Files in both scopes are frozen until their answers. No need to merge/rebase.

## Smoke and scheduler facts

- RUN-0098, spec digest
  `blake2b:b50b731bd3c9b50740a3a372ad60a0b842b68da55658c24af3079608d402f565`.
- ATT-0098-1 / PBS3764323.opbs failed **before Python**: launcher used relative
  src/v5_smoke.py without explicit cd; PBS began in /home/x10041. No application
  manifest. Recorded ended/failed from job_get; measured2s×0.25/3600 node·h in
  U-0112 =0.0001388889 node·h. Do not import duplicate usage from an executor
  export. Its stale ledger export still showed running after job_get/history;
  this is why the verified end was explicitly recorded with a separate source.
- **ATT-0098-2 / PBS3764329.opbs** runs the identical f4b2c19 commit, corrected
  launcher with explicit cd. debug-mig,1 instance, expected360s/request540s.
- At20:11:28 JST, PBS was still R. Log reports **36 tests passed in5.70s**:
  unchanged V1 data/LoRA/train suites and V5 regressions/real-arm checks. The
  end-to-end three-arm smoke is still running; do not call it passed yet.
- Outputs: `/work/xg24i002/x10041/bypass-lora/runs/ATT-0098-2/`:
  manifest.json, checks.xml, V5-{C,F,L}/{manifest.json,adapter/,predictions.jsonl,
  dev-epoch-{1,2,3}.json}. allocation.json confirms mg1098, matching PBS_JOBID
  and nodefile, allocation_confirmed and live job state R before workloads.
- Notification **N-20261101-200901-c205**, expected1, sealed/all_ended and early
  failure. Daemon was started on miyabi-g2. Coordinator continuation listener:
  `/work/xg24i002/x10041/miyabi-toolbox/miyabi-mcp/bin/miyabi-mcp notify listen --id N-20261101-200901-c205 --notifier miyabi-g2`.

## Resume

1. Read reviewer answers and job completion; inspect the application manifests,
   exactly three epoch dev records and init/reload checks. Report concrete fixes
   through the prescribed recheck; never edit files during a pending review.
2. Import verified final scheduler observations and measured usage once; import
   the smoke manifest. Add diagnostic-only evidence for E-0013, no hypothesis
   supports/challenges. Send exact evidence to reviewer-2 (RV-0033).
3. The new latency unit test is **not yet run**: submit it through PBS from the
   latency commit or include it in a subsequent necessary regression job.
4. After RV-0033 passes, run each arm's seed0 initial LR grid {1e-4,3e-4,1e-3},
  200k tokens, calibration NLL only. Register each run against E-0013@3, measured
   costs from smoke, confirmed allocation and immutable snapshot, notify groups.
   Endpoint extensions follow raw NLL minimum; no dev selection. Selection/lock
   CLI is `v5_train.py lock --manifests ... --output ...`; execute in PBS.
5. Commit the LR lock/selection table before seed1–3. Register nine full runs;
   `v5_train.py run --phase train --arm V5-C|V5-F|V5-L --seed1|2|3 --lr ...
   --lr-lock ... --data-dir /work/.../data/v1 --output /work/.../runs/ATT...`.
   Each has standard adapter config.json+adapter.pt; complete manifest includes
   arm,seed,metrics.learning_rate,params.trainable,plan.order_sha256,train.steps,
   train.input_tokens,reload.identical; final dev predictions are JSONL V1 fields.
6. T-0014 researcher-1 confirmed schema and owns confirmation/WikiText/analysis.
   This task also owns full Base repeated-dev noise and fixed256 latency. Send
   final checkpoint inventories and manifest SHA256/bytes to researcher-1.
7. Budget E-0013≤12 node·h; T-0013 planning share~7. Re-estimate with measured
   timings before launching; report to coordinator if total would exceed it.

All tests, analysis and runtime imports must run through miyabi-mcp PBS. Current
local plane was confirmed login/control on miyabi-g2. Use the unchanged lock
environment3c916b51337d-cuda, cuda/12.8, model revision da87bfb...496c0cd,
HF_HUB_OFFLINE=1. Do not run workloads locally or change dependencies.
