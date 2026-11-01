# T-0024 registration description correction

The twelve prepared records ATT-0198-1, ATT-0199-1, ATT-0200-1, ATT-0201-1, ATT-0202-1, ATT-0203-1, ATT-0204-1, ATT-0205-1, ATT-0206-1, ATT-0207-1, ATT-0208-1, ATT-0209-1 were withdrawn before any submission. Their metric description incorrectly included "per example". This was a registration description error. No code, contract, split, score or actual workload changed.

Correct calibration metric: for each task, sum gold supervised-token CE over its full 2048 calibration examples and divide by total gold supervised-token count; then take the arithmetic mean of the PIQA and BoolQ task NLLs. This is exactly the reviewed v1_run.evaluate used by rb_train.py and E-0020.

The immutable RunSpecs are superseded with new registrations under request suffix v2. Withdrawn records used no compute. They must not enter lock_lrs or the candidate table. No job was submitted for any of them.
