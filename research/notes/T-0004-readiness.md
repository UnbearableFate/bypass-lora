# T-0004 memory-path readiness

Implementation commit: `3ddd3a107cdc7e342025e8bd16c306522d9fd50e`.
Production change removes the kernel_matrix exemption from support write-capacity
validation. All backends accept two delimiter writes at capacity two and reject
them at capacity one. No dependencies or tolerances changed.

The new real-model state check covers layer 14, ranks 32/128, all three backends,
inclusive/strict, and S=1/4 (24 cases), with mixed left/right padding and nonzero
exits. It compares FP32 M/c and slot keys/values, and checks counts and slot-valid
masks. Each floating component has a one-element corruption rejection control.
Chunk1 is recorded without an equivalence gate. The runner accepts additional
pytest paths so the same registered job includes all existing V1 tests.

## Observed validation

ATT-0034-1 / RES-0026 / EV-0023: **130/130 passed**, pytest 35.69 seconds.
Compute plane: mg1098, PBS 3760151.opbs, NVIDIA GH200 120GB MIG 2g.24gb;
allocation guard confirmed the host and job before application execution.
Environment: 3c916b51337d-cuda, cuda/12.8, pinned Qwen3-0.6B-Base revision
da87bfb608c14b7cf20ba1ce41287e8de496c0cd, BF16 backbone and FP32 side parameters.

| Rank | chunk3 state max absolute error | chunk1 diagnostic max absolute error |
| --- | --- | --- |
| 32 | 0 | 0.17662876844406128 |
| 128 | 0.000003814697265625 | 0.2529714107513428 |

These values hold across all backend/mode/step combinations. The chunk3 checks
pass the locked FP32 atol/rtol of 2e-5. No cause investigation is needed by the
contract because there is no threshold violation; this does not establish the
source of the smaller numerical discrepancy. Chunk1 equality is not claimed.
All state corruption controls and all capacity cases passed.

No training tokens consumed. The scheduler still reported running after the
result was written; final execution observation and measured node-hours remain
to be recorded after scheduler completion. Requested usage was 0.033333 MIG
instance-hours (two minutes); requested usage is not actual usage.

Independent code review is required before delivery. These are implementation
diagnostics, not evidence for a training or mechanism claim.
