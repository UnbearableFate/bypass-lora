# T-0003：V1 八组 FFN 融合接口筛选（E-0003）

## 目标

按 E-0003 契约跑完 V1-01 至 V1-08 八组（seed 0，单 seed 筛选），完成契约中预先规定的分析，作为 V1 阶段报告的依据。

## 规格

- 契约：`research/experiments/E-0003.md`，特别是 Release 段：学习率、batch、训练行排除、停止规则、中间评估、Base 参考值、checkpoint 规则与分析规则。
- 在 master 现有的 V1 流水线上增加四项：
  1. 训练行排除：按契约实现；记录删去的行数、任务、id 与 token 长度；任何任务超过 1% 时停止并报告。
  2. 中间评估：在 25%、50%、75% 进度只评估 dev，只用于学习曲线。评估不能改变训练状态：train/eval 模式要正确切换，优化器状态、学习率日程、数据顺序与随机数流都不受影响。
  3. 新 arm 的配置：F_pre、C_gate、C_up、线性 Z=UA + C_dual、28 层 all-7 LoRA rank2。V0 代码已有全部融合接口，`src/v1_lora.py` 已支持两种 LoRA 布局，直接复用。每个 arm 加载后统计 stored、trainable 与实际有梯度的参数数，与契约表比对；不一致时先查实现。
  4. 分析脚本：按契约计算各 arm 的 dev 宏平均、各任务准确率与 dev NLL，以样本为单位的配对 bootstrap 区间（各 arm 共用抽样索引；重复次数与随机种子在第一个正式 run 之前写进 notes），以及 F*、C* 与额外比较。分析脚本只读 runs 输出。
- 第一个正式 arm 之前，用实际实现复现日程（T-0002 估计为 13,077 步、2,000,040 tokens，评估点为第 3,270、6,539、9,808 步）；不一致时先查原因。
- 冒烟：8 个 arm 各跑几步，检查参数数、初始化时输出与 Base 相同、重载后结果一致，并测出 V1-08 的步时（T-0002 只给了一个宽松的上界）。
- Base 参考值沿用 E-0002 的 ATT-0011-2；评估代码若有改动，则重跑 Base。
- 官方 validation 仍然封存，不读取、不评估。

## 审核

训练行排除改动了训练数据路径，中间评估与新 arm 配置改动了训练和评估路径，分析脚本决定结论中的数字。冒烟通过后，用 `rl review request --target E-0003 ...` 请求代码审核，范围包括上面四项。审核轮次按协调者的常设规则（MSG-0046）处理。审核通过前不要提交 8 个正式 run。

## 计算规则

协调者在登录节点上（`compute.coordinator: login`）。所有工作负载都通过 miyabi-mcp 提交 PBS 作业，不在登录节点运行。

1. 提交只含源码的 commit，然后 `rl run register --experiment E-0003 --resource miyabi --code-commit <完整 sha> ...`，arm 名用契约中的 ID（如 `V1-05`）。每个 arm 单独登记、单独提交作业。
2. 按 `rl compute steps --attempt <ATT>` 暂存只读快照 `/work/xg24i002/x10041/bypass-lora/snapshots/<sha>/`；作业只写 `/work/xg24i002/x10041/bypass-lora/runs/<ATT>/`，最后写 `manifest.json`。
3. `job_submit` 先 dry run，再带 `plan_digest` 提交；request key 用 `sr:<ATT>`，client_ref 用 attempt id；作业脚本先运行 allocation check。
4. 硬件与 E-0002 相同（GH200 MIG 2g.24gb）。walltime 按实测 × 1.5 申请，选能容纳该 walltime 的 MIG 队列；不再另设单作业时长与并发作业数的上限。作业不得跨越 2026-11-25 09:00 JST 的计划停机。
5. 8 个正式 run 提交时用一个 `notify` 组（expected 8），然后 `rl task wait`，在 WAIT 中给出 `daemon.listen_command`，由协调者在后台运行。
6. PBS 的 FINISH 或 Exit_status=0 不代表应用成功，要检查输出。作业事实用 `ledger_query` 的 smooth-research 格式导出后 `rl run record --observation @file`；结果用 `rl run import-result` 导入。

## 预算与交付

- 预算：E-0003 ≤ 10 node·h（node·h = 账本 mig_instance_hours × 0.25）。没有 tokens 上限，但每组 2M tokens 是实验设计，各组相同。将要超出时用 `rl task block --reason budget` 停下来。
- 证据：每个 arm 都要登记、记录并导入结果。E-0003 是单 seed 筛选，证据用 `rl evidence add --level screening`，按契约的分析规则如实写明支持、挑战还是无法区分 H-0002；假设状态由协调者在结论审核后决定。
- 交付摘要包括：每个 arm 的结果表（dev 宏平均、各任务准确率、dev NLL、三种参数数、步数与 tokens、wall time、MIG instance-h 与 node·h）；学习曲线；配对 bootstrap 区间；按契约规则得出的 F*、C* 与额外比较；训练行排除记录；各 run 的 ATT/RES/EV 编号；审核编号；仍未解决的问题。
