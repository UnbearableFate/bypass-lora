# T-0018：记忆路线训练筛选——单独编码写入（E-0016 的 D4、D5）

## 目标

按 E-0016 实现并运行 D4、D5（seed 0 筛选），交付两个 arm 在 E=16 dev 上按合同的门槛结果与诊断。只测量与报告；H-0012 的判定与后续路线由协调者负责。

## 为什么现在做

D4/D5 以 E-0015 规则 5 成立为前提，现已成立（T-0016，EV-0087）：
- R1 在 E=16 上的 iso_oracle_value：第 0 条 83.4%，第 1 条及以后 86.7%；
- 上下文编码的 oracle_value：第 0 条 83.4%，第 1 条及以后 1.6%；
- iso_wrong_oracle 1.5%，答案先验 1.9%。

也就是说，R1 学到的 value 映射只对“不带前文编码”的事实特征有效。D4/D5 检验：写入时每条事实单独编码，单跳读取能否在 E=16 上用起来。寻址仍须只从答案 CE 学。

## 规格

以封存的契约 `research/experiments/E-0016.md` 为准，本节只是摘要。

### 共同配置

- 同 E-0014 的 V2-03：direct_hidden、soft_slot、F_out，第 14 层 r128 S=1，LR 3e-3，每段 6000 步、每步 8 个 episode（`--step-budget 6000`），seed 0。
- 数据：E=16 train/calibration/dev 用 `runs/ATT-0039-1/data`。
- 所有新选项默认关闭；关闭时输出与梯度须与当前 master 一致，用旧/新回归确认（同 E-0014 的 `v2r_default_regression.py` 做法）。
- 只用最终 checkpoint，保存后重新加载，评估须逐例一致（在 guard 确认的 2g 切片上）。

### 单独编码写入

- 支持中每条事实单独编码，格式与 E=1 数据完全相同（`"Head relation Tail ;\n"`）；
- 在其分隔符处取 key 与 value，按原支持顺序写入同一槽位；
- 问题侧不变；
- 与 E-0015 的单独编码共用同一实现：`src/v2r_diagnostics.py` 的 iso 构造，已随 T-0016 合并到 master（c1e03bf）。训练路径要复用它，或把它抽到共用模块后两边都用，不要另写一份。训练时这条路径要有梯度（E-0015 只做评估）。

### Arms

- D4：单独编码写入，从零训练，只用答案 CE，读取协议不变。
- D5：同 D4，从 R1 最终 adapter（ATT-0122-1）续训：
  - 只加载 adapter 参数；AdamW 状态重置；重新 warmup 300 步再 cosine；
  - 数据顺序种子：`derive_seed(20261029, "concat_loop_v2", "V2R16", arm, 0)`。

### 与 T-0017 的分工

researcher-1 在 T-0017（分支 `sr/researcher-1/T-0017`，当前 7354cdf，代码审核前）实现 D1–D3，其中包括：
- 从 checkpoint 续训：只载 adapter、重置优化器、新调度；
- E-0016 的评估：标准对照、目标权重、n_eff、argmax 分布、范数比、按目标位置分层、学习曲线。

同一合同条目不要实现两份：
- D5 的续训与 E-0016 的评估复用 T-0017 的模块。它的代码审核通过后，协调者通知你，再把该分支合并进你的分支。在那之前先做单独编码写入的训练路径、单元测试、默认关闭回归与 D4 冒烟。
- 不要改 T-0017 新建的文件；需要改时告诉协调者。
- 两边都可能要在 `v2_train.py` 加接入点：只加最小的接入点，新逻辑放新模块，减少合并冲突。

### 评估

E=16 dev（1,000）上：
- 所有对照都在单独编码写入下做：normal、zero、shuffled、permuted、oracle_key、oracle_value；shuffled 的 donor 规则不变；
- EM 与 NLL；
- 目标权重（全问题位置平均与最后位置）；
- n_eff、最后位置 argmax 分布、融合增量与残差的范数比；
- 以上都按目标位置分层（第 0 条与第 1 条及以后）；
- 每 500 步记训练 CE；25/50/75% 学习曲线。

另外，为了和 E-0015 对照，请在 E=16 dev 上补报两个 checkpoint 的上下文编码 normal 与 oracle_value：同一 adapter，写入用普通上下文编码。这一项只作描述，不进判定。

### 判定

D4、D5 用 E-0010@2 的 H3 门槛，在单独编码写入、标准读取协议下同时满足三条：
- EM_normal − max(EM_zero, EM_shuffled) ≥ 10 pp；
- NLL_normal < min(NLL_zero, NLL_shuffled)；
- 目标平均地址权重 ≥ 0.125（全问题位置平均）；另报最后位置的权重。

单独编码写入是新登记的协议，结果与 E-0010 的统计量分开报告，不称为通过同一 H3。

## 审核

全量运行前请求代码审核：

```
rl review request --target E-0016 ... --author-model <你的模型 id>
```

审核范围：
- 默认关闭时与 master 一致；
- 单独编码的格式与槽位顺序，训练与评估用的是同一实现；
- 训练时梯度经单独编码写入路径到达 key/value 参数；
- D5 续训：只载 adapter、不带入优化器状态、调度从头开始（若复用的是已审核的 T-0017 模块，只审接入）；
- 保存/重载一致。

冒烟与单元测试可以先跑。D4 不依赖 T-0017，可以先提交；D5 等续训模块就绪后再提交。

## 预算与交付

- 预算：本任务约 0.25 node·h（D4、D5 各约 0.1，加冒烟与测试）。E-0016 合计上限 0.9 node·h，其中 T-0017 约 0.5。
- 证据：每个 arm 一条 screening 证据，关联 E-0016；只在达到 H3 时对 H-0012 加 `--supports`，其余由协调者判定。
- 交付摘要包括：
  - 两个 arm 的门槛表与分层；
  - 对照表与诊断；
  - ATT/RES/EV 编号；
  - 实际 node·h；
  - 审核编号与未解决的问题。

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
7. PBS 可能把作业放到 MIG 1g.24gb 切片上（账本仍记为 2g.24gb），1g 上的 BF16 结果与 2g 不逐位一致（EV-0086）。需要跨作业逐例一致（reload 检查、跨 arm 或跨 checkpoint 的评估）或要测时间的作业，在作业开头用 nvidia-smi 核对分配到的设备是 MIG 2g.24gb，不是就立即失败，并把 profile 写进输出。

## 审核轮次（协调者常设规则）

一轮审核最多两次（首审与一次 `--recheck-of`）。两次用完后仍有 medium 或更高的问题时，只要你认同这些问题，并且修复不改契约、容差、数据切分或指标定义，就直接修复，跑相应的 PBS 回归，然后用 `--recheck-of <上一份 RV> --new-round` 开新一轮，reason 写“协调者常设规则 2026-11-01：认同的 medium 问题已修复，修复不改契约、容差、数据或指标；本轮只复查该修复”。你不认同某个问题、修复需要改契约/容差/数据/指标，或希望带着未解决的问题继续时，交给协调者。审核请求由协调者分配给审核员。
