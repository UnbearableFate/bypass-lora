# T-0005：M 线合成任务生成器与 Base 上下限（E-0005）

## 目标

按 E-0005 实现 concat_loop_v2 合成任务的生成器、episode registry 与 V2 切分，核对它与旧 LoopLora 数据不重叠，并在 dev 上评估 Base 在 M 线（只有问题）与 F 线（支持事实在上下文中）的上下限。本任务不训练任何模型，也不运行 V2 的训练。

## 规格

- 契约：`research/experiments/E-0005.md`（Design 段的任务格式、身份与切分、旧 registry 核对、Base 上下限与记录项）。协议来源：`docs/plan.md` 的 §8.1、§13.1–13.2、§16。
- 新增文件，例如 `src/v2_data.py`（生成器、registry、切分）、`src/test_v2_data.py` 与一个 Base 评估入口。可以 import `src/v1_data.py` 中的 `derive_seed` 和 `src/v1_run.py` 中的评估工具函数，但不要修改 `src/v1_*.py`（researcher-1 正在 T-0003 中修改），也不要修改 `src/bypass_*.py`（researcher-2 正在 T-0004 中修改）。
- 生成也是 Python 工作负载，要在 PBS 作业中运行。生成的数据只写在该作业的 `runs/<ATT>/` 下，用 sha256 记录；后续任务按路径与 sha256 引用。confirmation 切片只生成身份与哈希清单。
- 旧 registry：只读获取 UnbearableFate/LoopLora @ a255e814740ff85ab3b4ec4836c88f0a8f802b2d（公开仓库，可在登录节点下载），放在被 git 忽略的 `data/` 下；只读取它的数据或 registry 文件，不 import、不运行它的代码，也不复制进 `src/`。
- 单 token 答案、名字池的 tokenization 长度分层都要用 Qwen3 tokenizer（锁定的 revision）检查。

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

数据切分与 registry 是 V2 所有结论依赖的新核心路径。生成器与 registry 的测试通过后、Base 上下限评估之前，用 `rl review request --target E-0005 ... --author-model <你的模型 id>` 请求代码审核，按上面的审核轮次规则处理。

## 预算与交付

- 预算：E-0005 ≤ 1 node·h，不训练。
- 证据：每个 run 都要登记、记录并导入结果。Base 上下限是 diagnostic 证据（`rl evidence add --level diagnostic`），说明任务是否适合 V2；它不支持也不挑战 H-0003 的记忆读取主张。
- 交付摘要：生成器版本与配置；各 split 的样本数、文件路径与 sha256；各轴统计与 tokenization 长度分布；写入位置检查；与旧 registry 的核对结果；Base 在 M 线与 F 线上的 EM、NLL 与先验基线，以及按 outcome_actions 的判定；各 run 的 ATT/RES/EV 编号；实际 node·h；审核编号与仍未解决的问题。
