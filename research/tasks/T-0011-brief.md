# T-0011：V2 M 线 2×2×2 析因与 C_dual 探针（E-0010）

## 目标

用已审核的正式 M 线训练程序跑 E-0010 的十组（V2-01 至 V2-10），并按契约中预先锁定的规则完成分析：H-0003 与 H-0004 的判定输入、V3 配置选择，以及描述性分析。判定本身由协调者在结论审核后记录。

## 规格

- 契约：`research/experiments/E-0010.md`（最新修订，含 Release 段：开始条件、每个读算子的学习率、步数、评估与资源）。共同配方见 `research/experiments/E-0008.md` 的 Design 段；训练程序与评估项见 `research/experiments/E-0009.md` 的 Design 段。
- 训练程序：master 上的 `src/v2_train.py`（T-0008 交付，RV-0017 与 RV-0019 审核通过；说明见 `research/notes/T-0008-training.md`）。不改训练与评估路径；确需改动时先问协调者，改动后需要复查。
- 十组配置与学习率：按 Release 段。各组 seed 0、6,000 步、4,000,470 tokens、warmup 300，最终 checkpoint 在 dev 与 calibration 上评估全部对照并重载复评，25/50/75% 只在 dev 上评估 normal、zero、shuffled。每组报告 stored、trainable 与实际有梯度的参数数。
- 分析脚本（新增，例如 `src/v2_factorial_analysis.py` 与测试）：只读十组的输出，计算
  - H-0003：每组的合格判定（EM_normal − max(EM_zero, EM_shuffled) ≥ 10 pp 且 NLL_normal < min(NLL_zero, NLL_shuffled)，dev 全部 1,000 条、最终 checkpoint、点估计）与读取非退化（normal 下平均目标权重 ≥ 0.125），以及规则给出的 supported / challenged / inconclusive；
  - H-0004：四格（特征 × 融合，只用 F_out 与 C_up）的逐格判定与总判定；
  - V3 配置选择（只在 H-0003 为 supported 时）；
  - 描述性分析：主效应与交互、C_dual 探针对同格 C_up 与 F_out 的差、normal 对 zero/shuffled/permuted 的差、以 episode 为单位的配对 bootstrap 95% 区间（10,000 次，seed 20261029，各 arm 与对照共用抽样索引）、key/value oracle 定位、分层、答案 ≠ 问题主语子集、学习曲线、归一化读误差。
  - 规则必须与契约逐字一致；脚本输出每条规则的输入数值与结论，便于核对。
- 不读 confirmation；不改数据。

## 环境

- 沿用 master 的 `pyproject.toml`/`uv.lock`（lock 3c916b51337d，cuda extra，环境 `/work/xg24i002/x10041/bypass-lora/envs/3c916b51337d-cuda`，cuda/12.8，torch 2.10.0+cu128，transformers 4.57.1，Python 3.13.13）。不要改依赖；确需改动时先问协调者（几个研究者并行工作，lock 改动会互相冲突）。
- 模型 Qwen/Qwen3-0.6B-Base，model 与 tokenizer revision da87bfb608c14b7cf20ba1ce41287e8de496c0cd，`HF_HOME=/work/xg24i002/x10041/hf_home`，作业中 `HF_HUB_OFFLINE=1`。BF16 冻结主干、FP32 新参数、eager attention。硬件 GH200 MIG 2g.24gb。

## 计算规则

协调者在登录节点上（`compute.coordinator: login`）。所有 Python 工作负载（pytest、冒烟、数据生成、评估、分析）都通过 miyabi-mcp 提交 PBS 作业，不在登录节点运行，也不回退到本地执行。

1. 提交只含源码的 commit，然后 `rl run register --experiment <E-x> --resource miyabi --code-commit <完整 sha> ...`。
2. 按 `rl compute steps --attempt <ATT>` 暂存只读快照 `/work/xg24i002/x10041/bypass-lora/snapshots/<sha>/`；作业只写 `/work/xg24i002/x10041/bypass-lora/runs/<ATT>/`，最后写 `manifest.json`。
3. `job_submit` 先 dry run，再带 `plan_digest` 提交；request key 用 `sr:<ATT>`，client_ref 用 attempt id；作业脚本先运行 allocation check。
4. 队列用 debug-mig 或 short-mig；walltime 按实测 × 1.5 申请；作业不得跨越 2026-11-25 09:00 JST 的计划停机。
5. 几分钟的作业用 `job_wait`（每次 ≤ 45 s，不要循环轮询）；更长的作业带 `notify` 提交，然后 `rl task wait`，在 WAIT 中给出 `daemon.listen_command`，由协调者在后台运行。
6. PBS 的 FINISH 或 Exit_status=0 不代表应用成功，要检查输出。作业事实用 `ledger_query` 的 smooth-research 格式导出后 `rl run record --observation @file`；结果用 `rl run import-result` 导入。用量换算：node·h = 账本 mig_instance_hours × 0.25。

## 审核轮次（协调者常设规则）

一轮审核最多两次（首审与一次 `--recheck-of`）。两次用完后仍有 medium 或更高的问题时，只要你认同这些问题，并且修复不改契约、容差、数据切分或指标定义，就直接修复，跑相应的 PBS 回归，然后用 `--recheck-of <上一份 RV> --new-round` 开新一轮，reason 写“协调者常设规则 2026-11-01：认同的 medium 问题已修复，修复不改契约、容差、数据或指标；本轮只复查该修复”。你不认同某个问题、修复需要改契约/容差/数据/指标，或希望带着未解决的问题继续时，交给协调者。审核请求由协调者分配给审核员。

## 审核

分析脚本在十组 run 结束前写好并请求代码审核（`rl review request --target E-0010 ... --author-model <你的模型 id>`），范围是规则实现、bootstrap 与输入校验；可以用 E-0009 的校准输出或合成输入做测试。十组 run 用已审核的训练程序，不需要新的训练路径审核；registration 时 `code review` 应显示 covered。

## 预算与交付

- 预算：E-0010 ≤ 7 node·h（V2 合计 ≤ 10 node·h）；预计约 1.1 node·h。
- 提交：十组用一个 notify 组（short-mig，walltime 按 Release 段）；提交后 `rl task wait`，在 WAIT 中给出 listen 命令。
- 证据：每个 run 都要登记、记录并导入结果。分析完成后添加 screening 证据：按锁定规则的结果，用 `--supports` 或 `--challenges` 关联 H-0003 与 H-0004（inconclusive 时两者都不加，在 summary 中说明）；每条证据的 summary 写明规则输入数值。
- 交付摘要：十组的 normal/zero/shuffled/permuted EM 与 NLL、目标权重与 n_eff；H-0003 与 H-0004 按规则的结果；V3 配置选择；oracle 定位结论；主要描述性结果与区间；各 run 的 ATT/RES/EV 编号；实际 node·h；审核编号与仍未解决的问题。不要写超出规则的结论。
