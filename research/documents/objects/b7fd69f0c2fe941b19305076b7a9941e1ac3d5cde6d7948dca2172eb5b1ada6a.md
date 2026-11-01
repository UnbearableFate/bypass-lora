# T-0004：记忆路径就绪检查（E-0004）

## 目标

按 E-0004 实现并验证 RV-0002 的两条跟进（FP32 chunk3 记忆状态比较；所有 backend 一致的支持写入容量保护），使记忆路径可用于后续的记忆家族校准与 V2。本任务不训练。

## 规格

- 契约：`research/experiments/E-0004.md`（Design 段的检查表与配置）。
- 背景：`research/reviews/RV-0002.md` 的两条 low 发现，以及 `research/notes/T-0001-implementation.md` 中的记录。相关代码在 `src/bypass_core.py`、`src/bypass_model.py`、`src/test_model.py`（V0 已审核）。
- 改动只限这两项。V0 与 V1 的全部现有测试（`src/test_core.py`、`src/test_model.py`、`src/test_v1_*.py`）要继续通过。容差文件 `src/tolerances.json` 不改。
- 并行注意：researcher-1 正在 T-0003 中修改 `src/v1_*.py`，researcher-3 正在 T-0005 中新增数据生成器文件。不要修改 `src/v1_*.py`，也不要改依赖。

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

容量保护与状态检查属于记忆路径的核心代码。检查全部通过后，用 `rl review request --target E-0004 ... --author-model <你的模型 id>` 请求代码审核，按上面的审核轮次规则处理。

## 预算与交付

- 预算：E-0004 ≤ 0.5 node·h，不消耗训练 tokens。
- 证据：每个检查 run 都要登记、记录并导入结果；`rl evidence add --level diagnostic` 支持或挑战 H-0001。
- 交付摘要：各检查结果（含 r128 与 r32 的 chunk3 状态误差、chunk1 诊断值）及其 ATT/RES/EV 编号；若 chunk3 超出容差，给出原因定位；实际 node·h；审核编号与仍未解决的问题。
