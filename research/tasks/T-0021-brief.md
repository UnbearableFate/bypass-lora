# T-0021：V3 前置——T2 链终点任务上的读取迁移、无训练深度扫描与循环实现检查（E-0018）

## 目标

按 E-0018 完成三件事：在单独编码写入的 M 线代码中接入 T2 链终点数据与 S>1 累加循环并做实现检查；不训练，把三个 D5 adapter 用到 T2 dev 上，扫描 S_eval=1/2/4/8；测 T2 上 S=1/2/4 的训练步耗时。只测量与报告；V3 训练矩阵由协调者根据结果设计。

## 背景

- E-0017 复现后 H-0012 记为 supported（D-0023），V3/V4 路线重开（D-0024），H-0005 的范围改为单独编码写入加 D5 配方（H-0005@2）。
- 三个 D5 adapter：ATT-0159-1（seed 0）、ATT-0172-1（seed 1，起点 ATT-0166-1）、ATT-0173-1（seed 2，起点 ATT-0167-1）。加载前核对已记录的 digest。
- T2 数据：E-0011 的生成器 `src/v3_data.py`（phase V3），已发布 train/calibration/dev；封存切片（confirmation、ood_h4/6/8、ood_unseen）不做任何评估。问题模板与 T1 不同，hop1 的迁移本身就是要测的量。
- 你在 T-0018、T-0019 写的单独编码写入代码（`src/v2_isolated.py`、`src/v2r16_isolated.py` 等）是起点；master 已合并到 cefc70d 之后。

## 规格

以封存的契约 `research/experiments/E-0018.md` 为准，本节只是摘要。
- 新选项默认关闭，关闭时与 master 一致（回归测试）；S=1 时 T1 E=16 dev 上与 P1 的记录逐例一致。
- S>1：同一问题位置各步共享固定记忆 M，z 按 plan §3.5 累加（η=0.25），记录每步 query、地址权重（对路径上第 j 跳目标事实）、读出与状态范数。
- 评估：分 hop 的 EM/NLL，对照 normal、zero、shuffled、permuted，hop1 另报 oracle_key、oracle_value；hop1 在 S_eval=1 下按 E-0010@2 的 H3 口径判断，其余描述。
- 计时：从 seed 0 的 D5 adapter 出发，T2 train 上 S=1/2/4 各约 100 步冒烟，记录稳定步耗时与峰值显存；冒烟不作证据。
- 代码审核：T2 的单独编码写入路径与 S>1 循环是新的评估路径，正式评估作业前请求审核（`rl review request --target E-0018 ...`）。只换 S_eval 不需要新审核。

## 预算与交付

- 预算约 0.2 node·h，上限 0.4 node·h。
- 证据：每个 adapter 一条 diagnostic 证据，关联 E-0018，不加 supports/challenges。
- 交付摘要包括：实现检查结果与审核编号；三个 adapter 的分 hop × S_eval 表（各对照）与每步地址权重表；计时表与 V3 每组 node·h 估计（按 1M、2M、5M 样本预算各给一个数）；ATT/RES/EV 编号；实际 node·h；未解决的问题。

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
7. 切片（D-0018、D-0019）：PBS 可能把作业放到 MIG 1g.24gb 切片上，账本却记为 2g.24gb。用户决定不考虑这点 BF16 舍入差异：切片型号只作元数据记录，不因切片重跑、分析或标记；跨作业逐位比对的不一致如果能用切片不同解释，记下后继续。guard 只记录、不让作业失败；只有比较计时（延迟、吞吐）的作业要求 2g，不是 2g 就失败。

## 审核轮次（协调者常设规则）

一轮审核最多两次（首审与一次 `--recheck-of`）。两次用完后仍有 medium 或更高的问题时，只要你认同这些问题，并且修复不改契约、容差、数据切分或指标定义，就直接修复，跑相应的 PBS 回归，然后用 `--recheck-of <上一份 RV> --new-round` 开新一轮，reason 写“协调者常设规则 2026-11-01：认同的 medium 问题已修复，修复不改契约、容差、数据或指标；本轮只复查该修复”。你不认同某个问题、修复需要改契约/容差/数据/指标，或希望带着未解决的问题继续时，交给协调者。审核请求由协调者分配给审核员。
