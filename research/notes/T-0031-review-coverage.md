# E-0028 review coverage before formal launch

RV-0088 passed with zero findings, at the exact scientific source
`0f22cd60f45674ec20b800fdd455f80c2b6d46dd`. The scientific files and original
two launchers have not changed. The measured cost gate also passed; see
`T-0031-cost.md`. The local shell is login/control only.

First formal registration: **RUN-0321 / ATT-0321-1**, armX-01, seed0,
LR0.001, scientific source0f22cd6. It is **prepared only; no job was
submitted**. Explicit reviews named were RV-0076,0077,0080,0081,0082,0088.
Its automatic coverage check reports four inherited stale paths:

| Path | Old review versions named by the check | Inspected scope/evidence |
|---|---|---|
| src/bypass_model.py | RV-0002/0009/0060/0064 | Byte-identical between accepted7d8b346 and0f22cd6 (emptygitdiff). RV-0078 reviewed the initial_memory guard and unchanged default path; RV-0079 verified this file unchanged. RV-0088 also checked the unchanged initial_memory default path. E-0028 supplies no initial_memory. |
| src/rb_sink_read.py | RV-0077 | Later T-0027 diagnostic changes, latest582b444. It is imported only by rb_sink_batchcheck and its tests. No E-0028 training/eval/intervention/analysis source imports either sink module. |
| src/test_rb_sink_read.py | RV-0077 | Tests of that unrelated diagnostic module, not imported at runtime and not in the E-0028 selected test suite. |
| src/test_rb1_artifacts.py | RV-0064/0069 | Last changed8bf0146 to restore wrong-family LR rejection coverage. It is a test-only file; the complete file was included in the112passing tests at0f22cd6. The runtime rb1_artifacts.py is included in RV-0088. |

The reviewer answer RV-0088 explicitly leaves inherited internals to earlier
reviews. No code change is proposed merely to remove the stale flag. The
researcher asked the coordinator for disposition before costly launch, as
T-0031's work pack requires for any remaining stale coverage. Existing
RV-0078/0079 can additionally be named for bypass_model in future registrations.
The coordinator owns any scoped approval or further review request.

After authorization, reuse prepared ATT-0321-1 for X-01 (do not register or
launch a duplicate). Register the remaining12new training arms and the one
baseline evaluation from `T-0031-formal-plan.json`, keep SR_CODE_COMMIT exactly
0f22cd6, record explicit reviews and the coordinator's disposition for each
flagged attempt, then submit through MCP and establish the group notification.
If further coverage is required, get it before execution. All checkpoints,
confirmation seals, cost/eta-lock gates and source constraints remain intact.
