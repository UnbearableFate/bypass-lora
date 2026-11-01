# T-0007：F 线 Base 难度诊断（E-0007）

## 目标

查清 E-0005 中 Base 在 F 线（支持事实在上下文中）几乎答不对的原因：Base 能否在上下文中找到正确实体，只是完整词表 argmax 被“复制问题主语”压过；还是在 E=16 时根本找不到；或者问题出在提示格式或评估实现。本任务只评估 Base，不训练。

## 规格

- 契约：`research/experiments/E-0007.md`（Design 段列出全部输出、候选集合定义、事实数梯度、测试要求、分层与判定）。
- 背景：协调者只读统计了 `runs/ATT-0043-1/predictions.jsonl`，dev 上 F 线 top-1 为问题主语 944/1000，M 线 767/1000（另有 ` In` 222）。这只是快速查看，不算登记的结果；正式数字以本任务在 calibration 上的 run 为准。
- 新增诊断入口与测试，例如 `src/v2_base_diag.py` 与 `src/test_v2_base_diag.py`。只 import `src/v2_eval.py`、`src/v2_data.py` 的函数（提示构造、评分、渲染、`derive_seed`），不修改 `src/v2_*.py`、`src/v1_*.py` 与 `src/bypass_*.py` 的现有行为。
- 数据只用 calibration 切片：`/work/xg24i002/x10041/bypass-lora/runs/ATT-0039-1/data/calibration.jsonl`，作业中先核对 sha256 `c08753365165216b5a9f4c3d629271b2183e1323973886256867e826c1712f18`。dev 不再评估，confirmation 不读取。
- 第一个作业先在当前 master（已合并 T-0005 与 T-0006）上跑完整测试套件，作为合并后的回归，单独登记为一个 run（`--no-review "post-merge regression"`）。
- 逐条输出写入 `runs/<ATT>/`（jsonl），汇总写入 `metrics.json`。notes 中列出 5 个 calibration 例子的完整提示文本与 top-5 解码结果，覆盖 E=1 与 E=16。

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

诊断脚本决定阶段报告中的数字。测试通过、诊断 run 完成之后、交付之前，用 `rl review request --target E-0007 ... --author-model <你的模型 id>` 请求审核，范围包括诊断代码、测试与输出数字（候选集合、类别划分、事实数梯度的渲染）。审核员会对照代码检查输出。诊断 run 只需几十秒，审核前可以先跑，登记时写 `--no-review "diagnostic Base evaluation, seconds; reviewed with outputs before delivery"`；审核要求修改时重跑即可。

## 预算与交付

- 预算：E-0007 ≤ 0.05 node·h，不训练。
- 证据：每个 run 都要登记、记录并导入结果。用 `rl evidence add --level diagnostic`，不支持也不挑战 H-0003。
- 交付摘要：合并后回归的结果；calibration 上 M/F 的一致性数字；M/F 的 top-1 类别分布、正确答案秩与概率的分布、问题主语复制率；限定候选 (a)(b)(c) 及随机基线；E ∈ {1,2,4,8,16} 的完整表；E=16 的分层表；按 outcome_actions 的判定；各 run 的 ATT/RES/EV 编号；实际 node·h；审核编号与仍未解决的问题。
