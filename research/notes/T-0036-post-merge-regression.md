# T-0036: post-merge write-mask regression

This is maintenance validation of T-0034/T-0035, with no research claim or evidence changes.

## Contract inspection

Retain `a1317f2df93039e75136ecdf5f9882a5aea90af3` unchanged. Its revised assertions in
`src/test_model.py::test_support_delimiter_write_scope_and_capacity` match the intentional
`640bac75416df14c27c91e9155fac67e36d2eede` contract in `BranchModel.forward`:

- Query-phase `write_mask` raises `write_mask is not accepted in the read-only query phase`
  for each of the three tested backends. The test supplies valid support memory so this
  checks the write-mask boundary rather than missing-memory validation.
- F-phase `write_mask` requires the memory branch, KernelMatrix backend, no diagnostic
  runtime, no incoming cache, and `use_cache=False`. The revised test rejects the two slot
  backends and rejects `use_cache=True` for KernelMatrix, matching the actual condition.
- The support delimiter, padding/subset and capacity checks remain in place.
- Existing `src/test_seg_w.py` tests exercise successful masked uncached KernelMatrix
  forwards: all-true-mask logits and gradients match the unmasked path; W1/W2 write only
  the selected positions, read every valid position, and carry the resulting state into
  segment 2. Its validation test also rejects invalid mask dtype, padding writes,
  diagnostic runtime, and the tokenwise branch.

No production defect was identified in this inspection; no production or test source was
changed. `git diff a1317f2 HEAD -- src` was empty at submission.

## Validation scope and execution

The original 20-module suite in
`/work/xg24i002/x10041/bypass-lora-maintenance/post-merge-tests/post-merge-tests-2e66017.sh`
is retained. A repository search of test modules found 12 referencing `bypass_model`,
all already included: `test_model.py`, `test_rb_core.py`, `test_rb_read_diagnostics.py`,
`test_rb_improve.py`, `test_rb_qvu.py`, `test_rb_model.py`, `test_rb_tap.py`,
`test_rb_train.py`, `test_seg_seeds.py`, `test_seg_model.py`, `test_seg_w.py`, and
`test_v2r.py`.

- Tested source commit: `b44c246475de01e836405277b64d9631b21d55b7` (branch head at submission).
- Script: `/work/xg24i002/x10041/bypass-lora-maintenance/post-merge-tests/T-0036-post-merge-tests.sh`.
- PBS job: `3789321.opbs`, Miyabi-G `debug-mig`, group `xg24i002`, one MIG instance.
- Requested walltime: `00:03:15`, based on the previous same-suite PBS job
  `3789025.opbs` measured walltime of `00:02:07` (approximately 1.5x).
- Interpreter: `/work/xg24i002/x10041/bypass-lora/envs/3c916b51337d-cuda/bin/python`;
  CUDA module `cuda/12.8`, offline HF cache `/work/xg24i002/x10041/hf_home`.
- Outputs: `/work/xg24i002/x10041/bypass-lora-maintenance/post-merge-tests/T-0036-b44c246475de01e836405277b64d9631b21d55b7/`.

The script checks the live allocation through Miyabi MCP before workload execution,
archives the exact commit into node-local storage and makes the source read-only,
then runs all 20 modules in one pytest invocation. It saves the source commit,
module list, allocation JSON, GPU identity, pytest log and JUnit XML outside the
worktree. The shell syntax check passed before submission. No tests or ML imports
were run on the login node.

## Verified result

PBS `3789321.opbs` completed on confirmed compute host `mg1097` with exit status 0.
The pytest log reports **604 passed, 1 warning in 128.16 seconds**, followed by
`pytest_exit=0`. The JUnit suite attributes independently confirm `tests="604"`,
`failures="0"`, `errors="0"`, and `skipped="0"`. Thus no test in this requested
20-module suite was failed or skipped.

The warning is the existing requires-grad tensor-to-scalar warning at
`src/test_v2r.py:70`; it also appears in the original failed post-merge job's log.
The earlier 3 failures were the stale error-message assertions now fixed by
`a1317f2`, not a reproduced production regression. Its fix is retained without
amendment, and there are no production or test changes in T-0036.

Measured PBS walltime was `00:02:13`: `0.036944` MIG instance-hours, approximately
`0.009236` node-hours under the project's quarter-node MIG accounting. The test
log, JUnit XML and allocation evidence named above are the durable result.

Only this maintenance note is added after the tested source commit. No research
run, result or evidence records are required or created for this task.
