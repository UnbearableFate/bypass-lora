# Supplementary inherited review coverage before formal submission

All scientific source remains49f70094b78474ef0158dd631e3b25ea6ebdd174.
RV-0107 covers E-0031's new core paths. Initial training registrations
ATT-0409-1 and ATT-0410-1 named only RV-0107; the immutable checks retained
three inherited stale paths relative to the much older RV-0064/0069.
Neither prepared attempt has ever been submitted or assigned an execution ID.

The correct already-passing reviews are RV-0094 (both inherited tests) and
RV-0095 (portable up-only LoRA addition, preserving legacy LoRA). Read the
complete scope/answers and verify byte identity with git diff --exit-code:

- src/test_rb1_artifacts.py and src/test_rb_read_diagnostics.py are identical
  between reviewed f46a13dc4c65f6bf1292e3fafa4198af4fbd59f2 and49f7009.
- src/v1_lora.py is identical between reviewed
  dc82cfc6c1f642f9380596a6be3244dafd4147a4 and49f7009.

Explicitly naming RV-0107,RV-0094,RV-0095 makes all four legacy reevaluation
registrations ATT-0411-1 through ATT-0414-1 report covered for49f7009.
The same explicit review list was supplied to the original training request
IDs, but idempotent registration returns the immutable original records and
cannot refresh their check. Therefore retire the two prepared attempts with
cancelled_before_submission observations, no execution reference and no
fabricated usage, then register replacement attempts of the SAME immutable
RunSpecs with the full review list. Only the replacement attempts may be
submitted. This is bookkeeping to preserve the honest review record; it
changes no scientific code, data, seed, arm or training budget and launches
no duplicate experiment. The retired preparations consumed no PBS allocation.
