# T-0006：V2 M 线训练与诊断流水线（E-0006）

## 目标

按 E-0006 实现 V2 的 M 线训练与评估入口、§8.3 诊断、因果对照与 §8.4 oracle 开关，并完成小规模冒烟，为记忆家族学习率校准和 V2 做好准备。本任务不做正式训练，V2 训练仍待用户释放。

## 规格

- 契约：`research/experiments/E-0006.md`（Design 段的协议来源、数据接口、检查与冒烟）。
- 复用 `src/bypass_model.py` 的记忆阶段与 T-0004 合并后的容量保护。若 `BranchModel` 需要改动，保持最小并纳入代码审核。不要修改 `src/v1_*.py`（researcher-1）与 `src/v2_data.py`（researcher-3）。
- 数据接口与 researcher-3 的 T-0005 对接：它会在 notes 中写明 episode 字段格式（支持文本、每条事实的分隔符位置、问题、答案 token id，以及只用于离线诊断的 target fact index 与事实三元组）。T-0005 合并到 master 之前，用字段相同的小 fixture；合并之后用少量 E-0005 数据做一次端到端冒烟（协调者会通知你）。

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

M 线训练与评估路径、诊断与因果对照都是新的核心路径。检查与 fixture 冒烟通过后，用 `rl review request --target E-0006 ... --author-model <你的模型 id>` 请求代码审核，按上面的审核轮次规则处理。

## 预算与交付

- 预算：E-0006 ≤ 1 node·h；只做冒烟训练。
- 证据：每个 run 都要登记、记录并导入结果；`rl evidence add --level diagnostic`，内容是流水线的正确性，不支持也不挑战 H-0003 的记忆读取主张。
- 交付摘要：检查结果及 ATT/RES/EV 编号；每种组合的步时、显存与吞吐，以及按此估算的记忆家族校准（KernelMatrix、SoftSlot 各 {3e-4, 3e-3} × 0.2M tokens）与 V2 八组（每组 4M tokens）的 node·h；实际 node·h；审核编号与仍未解决的问题。
