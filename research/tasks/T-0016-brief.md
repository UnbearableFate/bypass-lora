# T-0016：记忆路线诊断（E-0015，只做评估）

## 目标

按 E-0015 在 E-0014 的已有 checkpoint 上加读取替换对照、单独编码替换和线性探针，交付把 value 内容、解码、非末位混合读取、上下文迁移几种解释分开的数字。不训练。只测量与报告；E-0016 的设计与 H-0009 的判定由协调者负责。

## 背景

- E-0014（EV-0080–EV-0085，D-0013）中只有 R1 过了 value 门槛（E=1 dev +82.6 pp）。
- R2 在 E=4 dev 的最后问题位置，目标权重 0.983，argmax 全部命中目标，EM 却只有 8.2%（T-0015 笔记里的 0.371 是全问题位置的平均）。
- R5 在 E=16 上的 oracle_value 只有 10.7%。
- AD-0001 指出、协调者已核对：R1 在 E=16 dev 上的 oracle_value 只在目标为第 0 条事实时有效（45/52 = 87%），其余位置约 1%（runs/ATT-0122-1 的 dev.metrics.json，target_fact_position 分层）。由此怀疑支持侧上下文改变了写入位置的特征（上下文迁移）。
- 顾问轮次 1（AD-0001、AD-0002）还在进行；若其建议需要补充诊断，协调者会修订 E-0015 并通知你。

## 规格

以封存的契约 `research/experiments/E-0015.md`（E-0015@3）为准，本节只是摘要。修订 2/3 采纳了 AD-0001 的建议（`research/advice/AD-0001.md` 第 2 节），在任何运行之前完成：
- 诊断改在 calibration 上做；
- 增加负对照、幅度扫描和探针基线。

- checkpoint（只用最终 checkpoint，按记录的配置加载）：
  - R1：ATT-0122-1
  - R2：ATT-0123-1
  - R5：ATT-0126-1
  - V2-03：ATT-0082-1
- 加载检查：每个 checkpoint 在其已记录 split（dev）上的 normal 预测，须与记录的预测文件逐例一致；不一致就停下报告。dev 只用于这项检查。
- 诊断 split：E=1、E=4 calibration（ATT-0112-1），E=16 calibration（ATT-0039-1），各 2,048 条。
- 探针：在同一 E 的 train 前 4,096 条（文件顺序）上拟合，在 calibration 上测试。
- 读取替换对照：
  - 已有：normal、zero、shuffled、oracle_value。
  - 新增：last_only、oracle_last、oracle_last_only、uniform；
  - 负对照：last_only_shuffled（与 last_only 协议匹配）、wrong_oracle（第 (target+1) mod E 条事实的 value）。
  - 算术与现有 oracle 相同，“最后位置”即 answer_logits 用的位置。
- 单独编码替换（E=4、E=16）：
  - 每条事实单独编码，格式与 E=1 数据完全相同；在其分隔符处取 key 与 value，按原支持顺序放回同一槽位。
  - 四个对照：iso_kv、iso_v（上下文 key + 单独编码 value）、iso_oracle_value、iso_wrong_oracle。
  - 检查单独编码到分隔符为止的 token 与 E=1 格式逐 token 一致。
- 幅度扫描（只在 R1@E=1、R2@E=4、R5@E=16）：α ∈ {0.1, 0.3}，三种方式：
  - normal 下全位置缩放；
  - oracle_value 下全位置缩放；
  - normal 下只缩放非末位位置。
- 所有 EM 与 NLL 按目标位置分层：第 0 条、第 1 条及以后，附逐位置计数。
- 探针：闭式 ridge（λ=1，同 independent_probes），calibration 上拟合，dev 上测试。
  - Pv：训练出的 value，每个写入槽位一个样本。
  - Pf：投影前的冻结特征。
  - Pz：最后位置的融合增量，分 normal 与 oracle_value。
  - Pv 与 Pf 在上下文编码和单独编码上各做一次，并按事实位置分层（0、1–3、≥4）。
  - 测试集中未出现在拟合集的类别计为错误，并报告数量。
  - 基线：答案先验、标签打乱、随机投影到 128 维的 Pf。
  - R1 的 Pv 另做四种跨编码测试（拟合编码→测试编码）：单独→单独、单独→上下文、上下文→上下文、上下文→单独。
- 判定规则见合同第 1–8 条：≥ 10 pp 的比较都要求 NLL 同向改善。报告时给出 Pv、Pz 与各 oracle 对照的数字，不只给判定。

## 审核

新代码只在诊断路径：新对照、单独编码记忆、探针收集、评估入口。训练路径与已有对照的输出不得改变，用测试确认（已有对照新旧代码逐例一致）。

请求代码审核：

```
rl review request --target E-0015 ... --author-model <你的模型 id>
```

审核范围：
- 各对照只替换声明的位置，增量算术同 oracle；
- 单独编码的格式与 E=1 数据一致，槽位顺序与原支持一致；
- 探针的拟合与测试不重叠，标签对齐槽位顺序；
- 已有对照不变。

评估很便宜，可以在审核答复前先跑。只有审核通过后才记录结果；修复若改变代码，就重跑。审核员由协调者分配。

## 预算与交付

- 预算：E-0015 ≤ 0.25 node·h（估计约 0.13）。
- 证据：每个 checkpoint 一条 diagnostic 证据，关联 E-0015，不加 supports/challenges。
- 交付摘要包括：
  - 各 checkpoint × split × 对照的 EM（分层）；
  - 探针表；
  - 第 1–8 条规则各自是否成立；
  - ATT/RES/EV 编号、实际 node·h、审核编号与未解决的问题。
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
