# T-0010：V3 T2 链终点任务生成器、登记与难度检查（E-0011）

## 目标

为 V3（多次读取的独立价值，H-0005）准备数据：实现 phase 参数化的 T2 链终点生成器，生成 phase V3 的 train、calibration 与 dev，登记封存切片，计算无模型的结构捷径基线，并做 Base 与 F 线 LoRA 小 pilot 的难度检查。本任务不训练 M 线；V3 的训练要等 V2 的结论。

## 规格

- 契约：`research/experiments/E-0011.md`（任务定义、图结构约束、链长构成、hop 与支持顺序轮换、切片与身份、独立性、捷径基线与门槛、难度检查、代码审核）。
- 背景：V2 的 T1 数据与生成器见 `src/v2_data.py`、`src/v2_prepare.py`、`research/notes/T-0005-data-contract.md`，最终数据在 `runs/ATT-0039-1/data/`。T2 沿用同样的渲染、可见分隔符写入位置、名字池与身份/seed 规则（phase 换成 V3）。
- 新增 `src/v3_data.py`（以及准备脚本和 `src/test_v3_data.py`）；可以从 `src/v2_data.py` 导入工具函数，但不要改 `src/v2_data.py`、`src/v2_prepare.py`、`src/bypass_*.py`、`src/v1_*.py`、`src/v2_memory.py`、`src/v2_train.py`。名字池必须与 V2 的 `name_pools.json` 完全相同（核对 sha256）。
- 测试至少覆盖：图约束（每个实体对每个关系至多一条出边，r 链无环且入度 ≤ 1）；答案等于沿 r 走到底的实体；起点到终点的距离恰为 hop；问题文本不含跳数；hops 1–3 的结构分布与 hop 无关；轮换与均衡；封存切片只有身份与哈希；两次生成哈希一致；单 token 答案与答案边界不变。
- 捷径基线与门槛按契约计算；门槛不满足时只在本任务内、任何模型评估之前调整并重新生成，记下前后数值。
- 难度检查：M 线 Base 与 F 线 Base 在 dev 上分 hop、分支持顺序评估（可以复用 `src/v2_eval.py` 的评分）；F 线 LoRA 小 pilot 复用 T-0009 合并后的 `src/v2_fline.py`（T-0009 正在由 researcher-3 完成，合并到 master 前不要改这个文件；合并后再基于 master 做 pilot，只加数据选择参数，V2 默认行为与测试不变），学习率 1e-3（E-0008 的选择结果），0.2M tokens。

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

全量生成与任何模型评估之前，用 `rl review request --target E-0011 ... --author-model <你的模型 id>` 请求代码审核，范围包括图结构约束、终止规则、hop 与起点选择、身份与 seed、封存切片不可读、捷径基线的计算。冒烟与确定性检查可以先跑。pilot 若改了 `src/v2_fline.py`，对改动部分请求复查。

## 预算与交付

- 预算：E-0011 ≤ 0.3 node·h（含测试、生成、Base 与 pilot）。
- 证据：每个 run 都要登记、记录并导入结果；数据与难度检查是 diagnostic 证据（`rl evidence add --level diagnostic`），不支持也不挑战 H-0005。
- 交付摘要：生成器与测试结果；各切片的大小、sha256、tokens 与各轴统计；与 V2 的重叠检查；捷径基线表（分 hop、分支持顺序）与门槛判定；M 线 Base、F 线 Base 与 LoRA pilot 的分 hop EM/NLL；各 run 的 ATT/RES/EV 编号；实际 node·h；审核编号与仍未解决的问题。
