# E-0031 producer checks before the initial locked analysis

Every scientific producer and the counted smoke uses49f70094b78474ef0158dd631e3b25ea6ebdd174.
This check applies to the initial rule6-release analysis and must be repeated
for any final analysis. RV-0107's low findings are handled as agreed in
reviewer MSG-1048, with evidence visible to conclusion review.

| Arm | Selected reevaluation | Result | Actual/required profile | Dev flips |
| --- | --- | --- | --- | ---: |
| RB1-01 | ATT-0411-1/PBS3786999 | RES-0388 | 2g/2g | 0 |
| RB1-02 | ATT-0412-3/PBS3788306 | RES-0400 | 1g/1g | 0 |
| RB1-07 | ATT-0413-1/PBS3787002 | RES-0389 | 2g/2g | 0 |
| RB1-09 | ATT-0414-2/PBS3787113 | RES-0392 | 2g/2g | 0 |

All four are complete/passed, same_profile=true/flips0, with full1000dev and
2048calibration rows per task and historical checkpoint provenance verified.
No smoke subdirectory and no unavailable exception is used as a producer.
The actual profile evidence for each is:

- `/work/xg24i002/x10041/bypass-lora/runs/ATT-0411-1/mig-profile.json`
- `/work/xg24i002/x10041/bypass-lora/runs/ATT-0412-3/mig-profile.json`
- `/work/xg24i002/x10041/bypass-lora/runs/ATT-0413-1/mig-profile.json`
- `/work/xg24i002/x10041/bypass-lora/runs/ATT-0414-2/mig-profile.json`

Path/SHA256 references for these guards, manifests and provenance are copied
into the immutable initial analysis input and
`/work/xg24i002/x10041/bypass-lora/research/artifacts/T-0034/profile-bindings-initial.json`.
The code will check producer/row/source/compatibility again inside PBS;
same-profile true is also explicitly verified during this input preparation.

RB1-02 producer choice was fixed by coordinator MSG-1061 before outcomes:
minimum successful numeric job ID. ATT-0412-3/PBS3788306 is selected over
duplicate ATT-0412-5/PBS3788308 (RES-0401). Both passed actual1g and0dev flips.
Both full dev prediction JSON files have SHA256
`7f2bde91fcb508adabe047d704c1c7263945dc8b51b3fc5fe19bd06dfce9038f`;
both full calibration prediction JSON files have SHA256
`3c0edbf14efe909073d2ab208e4b93c6b8b511045bcee45749ecec13b3d20356`.
Thus the files, including per-row order, gold, predictions, candidate scores,
NLL and token counts, are byte-identical. All duplicate and failed-attempt
usage is charged. RB1-02 had3wrong-profile rejections in total, then succeeded;
the coordinator's10-failure exception was not invoked.

H-QVU intervention ATT-0421-1/PBS3788305 (RES-0399) is complete on2g,
its provenance names successful training ATT-0409-3, and the runtime validator
checks every normal dev prediction key against that training's dev prediction.
`normal.identical_to_training=true`; normal/swap conditions are retained for
the locked analysis. Old cancelled ATT-0420-1 is not used.

Measured ended usage before analysis is1.04451475node-hours, including every
profile rejection and duplicate. Running the locked initial analysis is within
the1.3initial cap; conditional U25 controls remain unreleased until exact rule6.
