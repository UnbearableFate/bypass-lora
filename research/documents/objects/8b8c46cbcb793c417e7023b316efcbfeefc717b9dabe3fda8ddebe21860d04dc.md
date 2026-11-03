<!-- RV-0101 answer · entry codex:gpt-6-astra@xhigh · observed model unverified · session unknown · received 2026-11-03T00:49:07+00:00 -->

# RV-0101 conclusion review — H-0029@1

Verdict: **pass**, with **0 findings**.

The records support recording H-0029@1 as inconclusive and paused under E-0030@1 rule 5, at the stated confirmatory scope: A2 BoolQ, the locked segmented r128 configuration, and training seeds 3, 4 and 5. They also support the proposed stage pause as a research decision. They do not establish absence of an effect, confirm the LoRA comparison, or justify a formal-experiment recommendation; the proposal does not make those claims.

The exact point estimate and interval in EV-0174 agree with ATT-0400-1/RES-0376: G = 61/3000, or +2.033333 pp, with the paired row interval [-0.300000, +3.866667] pp. Rule 5 emits G_positive=true, G_ci_lower_positive=false and outcome=inconclusive. The three recorded contrasts are +3.2, +1.0 and +1.9 pp. The 3000 denominator is the equal-seed average of counts on the same 1000 BoolQ rows, not 3000 independent rows. The locked bootstrap shares each row draw across seeds and arms and recomputes max(Reset,T0) in each draw. Both the contract and proposal explicitly exclude training-seed uncertainty.

Rule 6 is not reached: its output has evaluated=false, confirmed=false and reason=fixed_sequence_not_reached. L = 23/750, or +3.066667 pp, and its interval [+0.732500, +5.433333] pp are therefore descriptive even though that interval is positive. PIQA, macro accuracy, NLL, swap and dev comparisons remain descriptive. The proposal correctly avoids treating six positive dev/A2 contrasts as six independent confirmatory replications.

The recorded provenance supports interpreting the outcome. I independently matched the current analysis and lock file SHA256 values to d9af79c3b8468fac6d66fad295682213d71af563c139927396db504c632e1b55 and cfd2e9b054c26da3d43ed22dbf0c410060599d6599d5fe88c4b8cfe1b78838c5. D-0073 was recorded before the first opening receipt. It selected training-profile-only attempts without changing the scientific comparison, checkpoint choices, rules or locked sources. The selected manifests report reviewed source 4e6e8df, matching training/evaluation hardware profiles and passing dev gates. The locked analysis reports all rule-1 and rule-2 checks passed, zero dev mismatches and A2 off/Base identity for every seed.

The attempt audit identifies exactly one complete opened result for each seed: ATT-0399-2, ATT-0397-1 and ATT-0398-2 for seeds 3, 4 and 5 respectively. Their receipts all name D-0073 and the same lock. ATT-0398-1 and ATT-0399-1 are explicitly partial, unopened profile misses; they are engineering attempts, not negative scientific results. The audit reports no unknown-seed openings, manifest-less attempts or other lock-matching attempts. The dev-replay spec mismatches are disclosed in D-0073 and are not used as confirmatory A2 results. Preopening rule-1 checks, per-attempt dev gates, A2 off/Base checks and the final opening audit have distinct timings; the supporting records preserve that distinction.

The pause rationale matches E-0027 evidence EV-0145/EV-0146, D-0062 and its analysis: dev gains +4.4/+2.6/+1.8 pp and mean +2.933333 pp, below the registered +3 pp threshold. Seed 0's selection role is acknowledged. H-0019 and H-0020 already have paused/inconclusive status. Keeping that status while pausing H-0029 does not reinterpret the earlier evidence or establish that the route is futile. The proposed reopening conditions are prospective conditions, not a validated sample-size design. A2 cannot be reused for confirmation under E-0030 and the recorded constraints. This review targets H-0029; it does not replace the earlier conclusion reviews of H-0019 or H-0020.

This was a records and artifact review on the login/control plane. I ran no workloads, tests, inference or bootstrap recomputation. I relied on the recorded PBS analysis and existing implementation reviews for execution correctness, while inspecting the locked statistics source and relevant manifests directly. I did not independently rehash all checkpoints, regenerate data splits, audit every raw prediction or prove absence of unrecorded activity. No changes to experiment code, contracts or hypothesis state were made.

```json
{
  "verdict": "pass",
  "findings": [],
  "checked": [
    "RV-0101 request matches its sealed document; inspected E-0030@1 rules 1-8, EV-0174, RES-0376 and ATT-0400-1 analysis/manifest/allocation records",
    "G=61/3000 (+2.033333 pp), CI [-0.300000,+3.866667] pp and per-seed gains 3.2/1.0/1.9 pp match the proposal and emit inconclusive under rule 5",
    "L=23/750 (+3.066667 pp), CI [+0.732500,+5.433333] pp is descriptive because rule 6 is not reached",
    "At locked source 4e6e8df, inspected shared-row bootstrap, per-draw max control, equal seed weights, strict lower-bound rule and fixed-sequence handling; scope excludes seed uncertainty",
    "Independently verified analysis SHA256 d9af79c3b8468fac6d66fad295682213d71af563c139927396db504c632e1b55 and lock SHA256 cfd2e9b054c26da3d43ed22dbf0c410060599d6599d5fe88c4b8cfe1b78838c5",
    "D-0072/D-0073 opening protocol, confirmation before receipts, reviewed training/evaluation sources, selected complete manifests and receipts, rule-1/rule-2 outputs, and audit of the two unopened profile misses",
    "RV-0098 and RV-0100 implementation review conclusions and their documented limits",
    "E-0027 EV-0145/EV-0146, D-0062, prior analysis and pinned reference agree on gains 4.4/2.6/1.8 pp and mean 2.933333 pp below the 3 pp gate",
    "Existing paused/inconclusive H-0019/H-0020 states, descriptive six-gain rationale, A2 non-reuse rule, and prospective reopening conditions are consistent with a stage pause"
  ],
  "unchecked": [
    "No reviewer-run tests, training, evaluation, bootstrap recomputation or other workloads; relied on recorded PBS execution and prior code reviews",
    "Did not independently regenerate data splits, verify every raw prediction or rehash all checkpoints and runtime snapshot files",
    "Recorded provenance was checked for consistency; absence of all possible unrecorded accesses or changes was not independently proven",
    "No new confirmation-set design, power calculation, seed-generalization claim, formal-experiment recommendation or revised conclusion for H-0019/H-0020 is certified by this H-0029 review"
  ]
}
```
