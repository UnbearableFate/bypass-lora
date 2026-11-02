# E-0029 pre-review validation

Runtime source: `df20616cadf870710705aeb5d52c373746aa8d45`.
The branch contains E-0028 reviewed commit `0f22cd6`; coordinator MSG-0859
confirmed RV-0088 passed there and no later source merge is needed.

ATT-0338-1 / PBS3775354.opbs ran in a verified allocation on mg1097.
Its complete manifest and validator passed; RES-0300 was imported.
The test log reports112 passed in11.36seconds. Every prescribed arm completed
its80-step train-only timing with registered stored/trainable/gradient counts,
and every child reports dev_read=false. Steady steps range0.2040–0.2388seconds;
peak allocated CUDA memory is2.698–2.733billion bytes.

The full G-T initialization uses256 fixed training examples,16766 nonpadding
tokens, and512 forwards including the independent RMS verification pass.
Only the four A_v matrices changed. Stored-basis orthogonality errors are
6.23e-9–1.08e-8; actual readout relative RMS errors are8.78e-9–3.41e-8.
The full G-QN and G-N checks are separately registered as ATT-0340-1 and
ATT-0341-1, using the same source and sample SHA256. Both passed and were
imported as RES-0301/RES-0302; each used512 forwards and33532 input-token
exposures, including actual RMS verification. Their largest stored-basis
orthogonality error is3.27e-8 and largest RMS error is1.42e-7. Only the four
A_v matrices changed in each. PBS3775515 on mg1097 and3775518 on mg1098
each used62seconds (0.017222MIG instance hours). The eight-arm timing smoke used random I-G
matrices for timing; it is not evidence that G-QN/G-N initialization passed.

Earlier initialization failures were investigated before changing tolerances.
ATT-0337-1 / RES-0296 compares four SVD choices on identical real G matrices:
FP32 gesvdj has true orthogonality error approximately4.7e-4; FP32 gesvd
passes at approximately2e-6; both FP64 solvers agree, and the production FP64
gesvd vectors stored to FP32 pass below1.5e-8. TF32 was disabled. The
intermediate hypothesis that the FP32 check itself caused the failure was
refuted; FP64 measurement alone did not fix the stored basis. G accumulation
remains FP32 as specified; only its decomposition and validation use FP64.
EV-0142 records this diagnostic comparison and makes no scientific accuracy claim.

Measured smoke usage:660seconds =0.183333MIG instance hours, U-0333.
The immutable source cost table projects2.4827246182node hours. The updated
table in T-0032-measured-cost.json adds0.019097node hours of prior tests,
failed smokes and solver checks, and replaces application-only smoke timing
with actual PBS walltime. Its conservative projection is2.5033858733node hours
against the3.5node-hour cap. Remaining full I-G checks consume the reserved
three-initialization allowance. Review changes or failed follow-up checks
must be included before releasing formal training.

Still pending: independent code review, up-only LoRA LR calibration/lock,
all new full training, complete dev step0 identities, final reload checks,
reference checkpoint reads, final curves/bootstrap/rules and evidence.
No new dev, A2 confirmation or official-validation data has been opened.
The reviewed-source initialization paths are G-T at
`runs/ATT-0338-1/ig-G-T`, G-QN at `runs/ATT-0340-1`, and G-N at
`runs/ATT-0341-1` under the main project root. Reuse them only if the passing
review still covers their source and initialization implementation; a changed
initialization path requires fresh artifacts and a revised cost projection.

