# T-0025 training verification and locked selection

All nine formal R1/seed0 runs ATT-0239-1..ATT-0247-1 are complete and
validator-passed, with same-job reload.identical=true. Their native scheduler
observations and usage are imported once; all nine final manifests are imported
as RES-0228..RES-0236 in arm order. Training source is the covered immutable
7bf091d4af24c2a9fdc666a5f885c31817b97174.

The covered ffa4c71 analysis source passed the full real-artifact audit in
ATT-0258-1 / PBS3769518 (17 scheduler seconds). It validates each final
checkpoint's inventory and adapter bytes, the accepted LR lock and all16
calibration input hashes, source/model/tokenizer/environment pins, exact shared
sample order, first-whole-step stopping, finite steps and LR schedule, full
paired dev predictions, all27 intermediate curves and per-layer accounting.

All arms have13115steps,26230presentations and2000069nonpadding input tokens.
The common R1 order SHA256 is
33759bc20322d0c9981128dd8574438546fc3439194ab32b7c79e35e0f0fc2bf.
Curves are at steps3279,6558,9837; final dev is1000rows per task. The exact
accepted A2 split and LR-lock hashes remain those in training-jobs.json.

Locked choices: **S_F*=S2 (RB1-03)** and **C*_RB=C_up (RB1-09)**. Both concat
arms are effective and within the strictly-less-than1pp near-best band; C_up
has fewer parameters. G0 passes; GR,GL,GC are false. D-0032 and the exact
Fraction rules therefore give rule9 (T0/gradient checks before at most one
probe), with no RB5 matrix or RB1p pilot. This is a pre-registered screening
action, not a hypothesis-status decision. Final evidence waits for the complete
intervention/latency/paired10000-bootstrap report.

| Arm | PIQA % | BoolQ % | Macro % |
| --- | ---: | ---: | ---: |
| Base | 72.3 | 68.0 | 70.15 |
| RB1-01 | 73.3 | 81.3 | 77.30 |
| RB1-02 | 72.4 | 81.8 | 77.10 |
| RB1-03 | 72.7 | 82.2 | 77.45 |
| RB1-04 | 72.8 | 81.0 | 76.90 |
| RB1-05 | 72.9 | 81.3 | 77.10 |
| RB1-06 | 72.7 | 80.3 | 76.50 |
| RB1-07 | 74.4 | 83.5 | 78.95 |
| RB1-08 | 73.3 | 81.6 | 77.45 |
| RB1-09 | 73.2 | 81.8 | 77.50 |

Measured nine-arm training:35018MIG seconds =2.43180556node hours.
All tests:252MIG seconds =0.0175node hours. The17-second selection/audit
brings measured usage so far to35287MIG seconds =2.45048611node hours,
leaving1.54951389under the4node-hour cap. Frozen intervention throughput is
smoke-measured before the full5-job grid; comparative timing requires2g,
ordinary evaluation records hardware metadata only under D-0018/D-0019.
