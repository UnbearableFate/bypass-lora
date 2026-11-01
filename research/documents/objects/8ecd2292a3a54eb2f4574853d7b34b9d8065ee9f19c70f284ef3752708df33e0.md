# T-0009：V2 F 线训练后正对照（E-0008）

## 目标

在与 V2 M 线相同的数据、样本顺序与 4M tokens 预算下，训练事实在上下文中的 LoRA 正对照，判断 E=16 的单跳任务是否可学，并给出 E-0010 是否开始的放行判定。

## 规格

- 契约：`research/experiments/E-0008.md`（Design 段：V2 共同训练配方与 E-0008 设计，包括放行门槛）。
- 新增入口，例如 `src/v2_fline.py` 与 `src/test_v2_fline.py`。LoRA 用 `src/v1_lora.py`，布局与 V1-07 相同（层 [6,13,20,27]，all-7，rank 10，alpha = rank）；F 线提示用 `src/v2_eval.py` 的 `prompt_tokens`（训练行传 `split='train'`，confirmation 总被拒绝）；只在最后一个提示位置对答案 token 计算 CE。不改 `src/bypass_*.py`、`src/v1_*.py`、`src/v2_data.py`、`src/v2_memory.py`。
- 样本顺序、计数、batch、停止规则与日程按共同配方实现。researcher-2（T-0008）在写正式 M 线训练程序，请用 `rl agent tell researcher-2` 与它核对样本顺序 sha256 和 4M、0.2M 预算下的步数，两边必须一致。
- 测试：初始化时 logits 与 Base 相同（LoRA 的 B 为零）；loss 只在答案位置；样本顺序与步数符合配方；重载复评逐条一致。
- 学习率校准：{1e-4, 3e-4, 1e-3} 各 0.2M tokens，按共同配方的规则选择，端点时扩展一次。
- 正式 run：选中的学习率，4M tokens；最终 checkpoint 在 calibration 与 dev 上评估并重载复评；报告完整词表 EM 与 NLL（全部 episode 与答案 ≠ 问题主语的子集）、问题主语复制率、正确答案的秩；放行判定：calibration EM 是否 ≥ calibration 先验基线 + 20 pp。

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

正式 4M run 之前，对训练路径（数据顺序、计数、日程、loss 位置）与评估路径用 `rl review request --target E-0008 ... --author-model <你的模型 id>` 请求代码审核。校准 run 可以在测试与冒烟通过后先跑；审核要求改动训练路径时重做校准。

## 预算与交付

- 预算：E-0008 ≤ 1 node·h（含冒烟、校准与正式 run）。
- 证据：每个 run 都要登记、记录并导入结果。正对照是 diagnostic 证据（`rl evidence add --level diagnostic`），说明任务在上下文中是否可学，不支持也不挑战 H-0003。
- 交付摘要：样本顺序 sha256 与步数（以及与 researcher-2 的核对结果）；校准表与选中的学习率；正式 run 在 calibration 与 dev 上的 EM、NLL、答案 ≠ 主语子集上的 EM、问题主语复制率与学习曲线；放行判定；各 run 的 ATT/RES/EV 编号；实际 node·h；审核编号与仍未解决的问题。
