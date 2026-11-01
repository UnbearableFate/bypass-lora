# E-0003 full Base rerun

ATT-0024-1 / RES-0024, reviewed runtime commit 1fe7d3d, PBS 3759531.opbs: complete, validator passed, exact reload. Command used E3 full plan, 2M threshold and batch2 without smoke/eval limits; Base consumed no training tokens and had zero trainable/adapter parameters. Full dev1000/task: PIQA accuracy .723, BoolQ .680, macro .7015; PIQA NLL2.750281812879043, BoolQ NLL1.9246741668730973, macro NLL2.3374779898760703. PBS307seconds; ledger .085278MIG instance-h = .0213195node-h.

ATT-0033-1 / RES-0025, deterministic read-only comparator source f3891b0, PBS3759603.opbs on mg1098: public formal Base reader accepted the rerun. All2000 dev prediction records (including candidate scores and gold NLL), all dev metrics, and split-map content exactly equal E-0002 ATT-0011-2. Complete/validator passed. PBS1second; .000278MIG instance-h = .0000695node-h (ledger six-decimal rounding); no training. This establishes comparability of the reference, not any interface advantage.

Eight trained arms ATT-0025-1 through ATT-0032-1 remain pending. No changes to their immutable runtime snapshot or reviewed analysis scope were made.
