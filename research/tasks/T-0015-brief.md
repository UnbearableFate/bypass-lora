# T-0015：记忆路线修复筛选（E-0014）

## 目标

按 E-0014 实现并运行六个单改动 arm（seed 0 筛选），交付每个 arm 在 E=16 dev 上的 H3 门槛结果与诊断，以及 R1/R2/R3 在各自 E 上的结果。只测量与报告；H-0009 的判定与后续路线由协调者负责。

## 背景

- V2 的 M 线（第 14 层 r128 S=1，支持只写、问题只读）十组都没有可用读取（EV-0070）。E-0012 的诊断（EV-0072）发现：
  - 最后问题位置的 argmax 槽位固定为 0；
  - 融合增量远大于残差；
  - 读出接近先验；
  - 冻结特征仍可线性解码。
- 路线按预期价值停止（D-0008）后，用户已授权计划外改动，路线重开（D-0012）。

## 规格

以封存的契约 `research/experiments/E-0014.md`（E-0014@1）为准，本节只是摘要。

- 基础配置为 V2-03：direct_hidden、soft_slot、F_out，LR 3e-3（E-0009），沿用 `src/v2_train.py` 的正式训练程序（RV-0017/RV-0019 审核过）。
- 预算按步数计：每个 arm 6000 步、每步 8 个 episode（TRAIN_BATCH_SIZE），所以各 arm 的 episode 呈现数相同；tokens 只记录。需要加一个以步数为预算的选项，默认仍用 V2 的 token 预算。
- 所有新选项默认关闭。关闭时，输出与梯度须与现有代码一致，用回归测试确认。
- 新数据：E ∈ {1, 4, 8} 的 T1 变体。
  - 版本号为 concat-loop-v2-t1-e{E}-1；种子为 derive_seed(20261029, "concat_loop_v2", "V2R", split, E)。
  - 头实体在 episode 内互不相同；尾实体按频率权重抽取；关系：E=4 每种一次，E=8 每种两次，E=1 均匀抽一种；目标均匀抽取。
  - 模板、名字池、单 token 答案检查与跨 split 去重同 V2。
  - 规模为 train 16,384、calibration 2,048、dev 1,000；confirmation 只记 identity；记录 SHA256。
  - E=16 继续用 `runs/ATT-0039-1/data`。
- Arms：
  - R1：E=1。
  - R2：E=4。
  - R3：课程，第 1–1000 步 E=1，1001–2000 步 E=4，2001–3000 步 E=8，3001–6000 步 E=16。
  - R4：先验通路。第 14 层加 tokenwise F_out r32，只作用于问题位置，P 零初始化。阶段 1 中记忆输出置零，只训练这条通路 1000 步（LR 3e-4，E=16 train）；阶段 2 冻结它，再训练记忆通路 6000 步。
  - R5：在最后一个问题位置加地址辅助损失 −log w_target，λ=1。这一项属于额外监督。
  - R6：去中心化余弦寻址。k̃_i = k_i − 同一 episode 有效槽位的均值，score = τ·cos(q, k̃_i)，τ = exp(θ)，初值 τ=10。
  - 详细定义见合同。
- 评估：最终 checkpoint 保存后重新加载，评估须逐例一致。
  - E=16 dev 上报告 EM 与 NLL，覆盖 normal、zero、shuffled、permuted、oracle_key、oracle_value 六种对照（v2_memory 的定义）。
  - 另报目标平均地址权重、n_eff、最后问题位置的 argmax 槽位分布，以及融合增量与残差的范数比（E-0012 L4 的定义，可以复用 `src/v2_layer_diagnostics.py`）。
  - R1、R2、R3 另在各自 E 的 dev 上报告 normal、zero、shuffled 的结果。
  - 每 500 步记录一次训练 CE。
- 判定（点估计，附正确数/样本数；区间按 V2 的配对 bootstrap 只作描述）：
  - H3：EM 差 ≥ 10 pp，NLL_normal < min(NLL_zero, NLL_shuffled)，且目标平均权重 ≥ 0.125。
  - R1：E=1 dev 上 EM 差 ≥ 50 pp。
  - R2：E=4 dev 上 EM 差 ≥ 10 pp，NLL 条件同上，目标平均权重 ≥ 0.5。

## 审核

全量运行前请求代码审核：

```
rl review request --target E-0014 ... --author-model <你的模型 id>
```

审核范围：
- 新选项默认关闭时与 V2 一致；
- 新数据的生成规则与去重；
- 课程按步数切换；
- 先验通路只作用于问题位置，阶段 2 中被冻结；
- 辅助损失只用最后问题位置的目标索引；
- 去中心化只用同一 episode 的有效槽位；
- 对照与评估不变。

冒烟与单元测试可以先跑。审核员由协调者分配；审核轮次规则见下文。

## 预算与交付

- 预算：E-0014 ≤ 1.5 node·h（估计约 0.8）。按 V2 实测，每个 arm 约 0.41 MIG-h；六个 arm 可以同时提交，带 notify 后进入 WAIT。
- 证据：每个 arm 的结果记为 screening，关联 E-0014。只有在 H3 门槛按合同达标时，才对 H-0009 加 `--supports`；R5 的 summary 写明“有地址监督”。其他结果不加 supports/challenges，由协调者判定。
- 交付摘要包括：
  - 六个 arm 的 H3 表，R1/R2 门槛表；
  - argmax 分布与范数比；
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

## 审核轮次（协调者常设规则）

一轮审核最多两次（首审与一次 `--recheck-of`）。两次用完后仍有 medium 或更高的问题时，只要你认同这些问题，并且修复不改契约、容差、数据切分或指标定义，就直接修复，跑相应的 PBS 回归，然后用 `--recheck-of <上一份 RV> --new-round` 开新一轮，reason 写“协调者常设规则 2026-11-01：认同的 medium 问题已修复，修复不改契约、容差、数据或指标；本轮只复查该修复”。你不认同某个问题、修复需要改契约/容差/数据/指标，或希望带着未解决的问题继续时，交给协调者。审核请求由协调者分配给审核员。
