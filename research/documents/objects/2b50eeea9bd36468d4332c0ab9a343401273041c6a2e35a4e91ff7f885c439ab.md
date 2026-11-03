# T-0033 / E-0030 final results

The frozen primary outcome is **inconclusive** (E-0030@1, H-0029@1, A2 BoolQ, locked r128 setting, new training seeds 3/4/5).
G = 61/3000 = +2.0333 percentage points; paired 95% interval [-0.3000, +3.8667] percentage points. Rule 5 emits inconclusive: G_positive=true and G_ci_lower_positive=false.

The fixed-sequence secondary confirmation was not reached: evaluated=false, confirmed=false, reason=fixed_sequence_not_reached. L = 23/750 = +3.0667 percentage points, interval [+0.7325, +5.4333] percentage points, is descriptive. No hypothesis status was changed.

Exact emitted values (CI units here are accuracy proportions):

```json
{
  "direction": "inconclusive",
  "G": {
    "denominator": 3000,
    "numerator": 61,
    "pp": 2.033333333333333
  },
  "L": {
    "denominator": 750,
    "numerator": 23,
    "pp": 3.066666666666667
  },
  "ci95": {
    "G": [
      -0.003,
      0.03866666666666667
    ],
    "L": [
      0.007325000000000008,
      0.05433333333333333
    ]
  },
  "secondary_confirmed": false,
  "eligible_edge": null
}
```

## BoolQ results

Accuracies are percent; contrasts are percentage points. All cells below are emitted by the frozen analysis.

| Seed | Keep | Reset | T0 | LoRA | Swap | Keep − max(Reset,T0) | Keep − LoRA | Keep − Swap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | 68.2 | 65.0 | 64.8 | 63.8 | 62.4 | +3.2 | +4.4 | +5.8 |
| 4 | 66.5 | 65.5 | 65.5 | 64.1 | 61.0 | +1.0 | +2.4 | +5.5 |
| 5 | 66.8 | 64.2 | 64.9 | 64.4 | 61.9 | +1.9 | +2.4 | +4.9 |

G uses equal seed weights. The bootstrap uses 10,000 BoolQ row draws with numpy.default_rng(20261101), shared indices for all seeds/arms, recomputes max(Reset,T0) on every draw and uses the linear quantile. The interval covers A2 row sampling only, not variation across training seeds.

## Descriptive results

| A2 metric | Mean Keep − max control | Mean Keep − LoRA | Mean Keep − Swap |
| --- | --- | --- | --- |
| boolq | +2.0333 | +3.0667 | +5.4000 |
| piqa | +1.6000 | +2.3000 | +4.9333 |
| macro | +1.8500 | +2.6833 | +5.1667 |

These PIQA/macro/swap contrasts are descriptive. On dev, the emitted BoolQ gain is +3.2333 pp and Keep−LoRA is +3.0667 pp. The pinned E-0027 dev comparisons emit gain difference +0.3000 pp and LoRA-contrast difference +1.0000 pp; these remain descriptive.

Per-seed A2 NLL summaries are descriptive; full-precision values, condition accuracies, dev summaries, donor metadata and rule-7 flags remain in analysis.json.

| Seed | Condition | PIQA NLL | BoolQ NLL | Macro NLL |
| --- | --- | --- | --- | --- |
| 3 | SEG-04 | 2.996162 | 0.778347 | 1.887254 |
| 3 | SEG-05 | 2.827795 | 0.611658 | 1.719727 |
| 3 | SEG-06 | 3.048170 | 0.628989 | 1.838579 |
| 3 | SEG-07 | 3.040551 | 0.622052 | 1.831301 |
| 3 | SEG-05/iv_swap | 3.501894 | 0.685852 | 2.093873 |
| 4 | SEG-04 | 2.994612 | 0.795986 | 1.895299 |
| 4 | SEG-05 | 2.839073 | 0.625319 | 1.732196 |
| 4 | SEG-06 | 3.044385 | 0.630599 | 1.837492 |
| 4 | SEG-07 | 3.037822 | 0.623183 | 1.830503 |
| 4 | SEG-05/iv_swap | 3.486478 | 0.703228 | 2.094853 |
| 5 | SEG-04 | 2.994946 | 0.810808 | 1.902877 |
| 5 | SEG-05 | 2.827865 | 0.612680 | 1.720273 |
| 5 | SEG-06 | 3.041000 | 0.631967 | 1.836483 |
| 5 | SEG-07 | 3.036567 | 0.621228 | 1.828898 |
| 5 | SEG-05/iv_swap | 3.489565 | 0.670568 | 2.080066 |

## Validation and provenance

Rules 1 and 2 passed for all three seeds. Every selected evaluation has an exact same-profile dev gate (zero mismatches), off identical to same-job Base, 1,000 encodable A2 rows per task and a valid D-0073 receipt/provenance. Cross-job Base prediction mismatches versus seed 3 are zero for each seed. The audit finds exactly one complete opened attempt per seed, equal to the selected root, with no unknown-seed openings or manifest-less attempts.

| Seed | Selected complete attempt | PBS job | Actual MIG | Result |
| --- | --- | --- | --- | --- |
| 3 | ATT-0399-2 | 3782470.opbs | 2g.24gb | RES-0374 |
| 4 | ATT-0397-1 | 3781926.opbs | 1g.24gb | RES-0371 |
| 5 | ATT-0398-2 | 3782471.opbs | 1g.24gb | RES-0375 |

The two earlier profile misses (ATT-0398-1, ATT-0399-1) were partial/unopened and produced no dev/A2/receipt. They are retained and costed. Three dev-replay imports have the disclosed dev-replay/A2 arm-label mismatch and serve as engineering diagnostics only (D-0073); the selected A2 results have matching arm A2.

Training source: `f7419525d12ff511845a11fae7ae0fae5100e8f7` (RV-0098). Evaluation/analysis source: `4e6e8dfad40feb35fab60eecf7cf2df428083b1d` (RV-0100, zero findings; 434 exact-source PBS tests). Inherited reviewed versions are covered by RV-0081/82/88/92/94/95. Training budgets are 13,115 steps / 26,230 examples per arm; matched r128 initial digests, LR/order/split pins, reloads and curve snapshots passed. All legacy LoRA training used 2g.24gb without repeats.

D-0072 defines the opening failure policy; D-0073 confirms the unchanged lock and supersedes its old first-attempt/five-miss operational fields. Runtime source, lock, checkpoints, comparisons and rules remained unchanged after opening. No A2 result was used to change them.

Analysis ATT-0400-1 ran on confirmed PBS allocation mg1104, job 3782870.opbs, actual 2g.24gb. Its application validator passed, analysis SHA256 matched, and receipt/cost audits passed. RES-0376 complete for ATT-0400-1.

## Full measured cost

**4.7108335 node-hours** (18.843334 MIG-instance hours), across all 29 executed task attempts, within the 5.5-node-hour cap. Conversion is measured MIG-instance hours × 0.25. The frozen analysis includes the 28 prior attempts at 4.70979175 node-hours and explicitly leaves its own usage pending; this delivery appends the measured 0.00104175 node-hour analysis cost without modifying the analysis artifact.

| Category | Executed attempts | Measured node-hours |
| --- | --- | --- |
| Diagnostics / smoke / regressions | 7 | 0.05513900 |
| Formal training | 12 | 4.10138875 |
| Lock validation | 1 | 0.00118050 |
| Dev-only replays | 3 | 0.07527775 |
| Profile misses | 2 | 0.00201400 |
| Complete A2 evaluations | 3 | 0.47479175 |
| Frozen analysis | 1 | 0.00104175 |

All 29 measured terminal usage records are included. The twelve superseded training preparations were never submitted; their U-0388 cost is reported zero, and they are excluded from measured execution cost. Ledger 2g labels are unreliable for some slots; actual training/evaluation profiles were checked inside jobs. Unit conversion is unchanged.

## Artifacts and limits

- Frozen report: `/work/xg24i002/x10041/bypass-lora/research/artifacts/ATT-0400-1/analysis.json`.
- Analysis SHA256: `d9af79c3b8468fac6d66fad295682213d71af563c139927396db504c632e1b55`.
- Confirmed lock: `/work/xg24i002/x10041/bypass-lora/research/artifacts/ATT-0393-1/a2-lock.json`.
- Lock SHA256: `cfd2e9b054c26da3d43ed22dbf0c410060599d6599d5fe88c4b8cfe1b78838c5`.
- Analysis input SHA256: `738931eda0438a668f21f08ebd5cec670053bb003f73d37202fe59b6b2a147c6`.
- Complete accounting and verification: `/work/xg24i002/x10041/bypass-lora/research/artifacts/T-0033/delivery-cost.json`, `ledger-observations-final.json`, `analysis-verified.json`.

The result is limited to A2 BoolQ at the locked segmented r128 configuration and these three new seeds. Row-bootstrap intervals exclude seed-to-seed uncertainty. PIQA, macro accuracy, NLL, per-seed results, swap and dev contrasts are descriptive. The secondary comparison is not confirmed because the fixed-sequence gate was not reached. A2 is consumed. Referenced checkpoints and metadata are retained for coordinator review; hypothesis status and route decisions remain with the coordinator.
