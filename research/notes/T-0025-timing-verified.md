# T-0025 timing verification and full frozen-evaluation release

ATT-0261-1 (PBS 3769570, RES-0241) completed successfully in 59 scheduler
seconds. The profile guard bound the selected CUDA UUID to a 2g.24gb device
on mg1098. The application manifest is complete/passed, source is reviewed
ffa4c7124939bbd82b1319f56be8e27913e0ff59, and the latency artifact retains
all five required trained arms, merged LoRA, and Base. Each has five raw
prefill samples and 128 raw decode samples; the driver checked all finite
values and recomputed summaries. KernelMatrix M+c is 16,896 bytes for four
layers, with 32 counter bytes reported separately. T0 and LoRA have zero
additional persistent history state. Latency does not separately measure the
selected S2 or C_up; the fixed protocol measures S4 and C_dual.

ATT-0260-1 (PBS 3769569, RES-0242) completed all five cost-smoke targets on
2g.24gb in 335 scheduler seconds. It used 32 dev rows/task, all six controls
for RB1-02/03/08/09, and swap for RB1-01. Every condition has 64 paired rows;
normal/intervened read strata cover all tasks, layers and actual first/last
queries (16 strata for S1, 32 for S2). iv_off exactly matched independently
scored same-allocation Base, and T0 swap exactly matched normal. Quality
scores were discarded; this run changed no selection or scientific gate.

The part model separately measured preparation, loading and each condition,
then scaled evaluation time by 1000/32 and added 10 seconds/bootstrap run.
It projects 0.697336 node hours for the full grid. Actual E-0021 usage before
the grid is 35,681 MIG seconds = 2.477847 node hours, including all five test
attempts, nine training jobs, audit/selection, cost smoke and latency. The
projected post-grid subtotal is 3.175184, plus final analysis. Even all five
requested walltimes sum to only 1.166667 node hours, leaving 0.355486 below
the four-node-hour cap for analysis/overhead if each job reaches its request.

All five full interventions were registered as covered by RV-0069 at the
immutable ffa4c71 source. S_F*=RB1-03 and C*_RB=RB1-09 remain locked. Full
evaluation records hardware metadata only under D-0018/D-0019. The launcher
calls the unchanged reviewed rb1_intervene CLI; no confirmation is accessed.
Exact job IDs, walltimes and the single notification listener are in
T-0025-intervention-jobs.json. Native terminal observations and usage for the
two completed timing jobs were imported once; no duplicate manual usage was
added. Final scientific evidence awaits the complete grid and locked analysis.
