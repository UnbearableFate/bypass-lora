# T-0008：正式 M 线训练程序与记忆家族学习率校准（E-0009）

## 目标

实现 V2 的正式 M 线训练程序（供 E-0009 与 E-0010 使用），并完成 KernelMatrix 与 SoftmaxSlot 两个读算子的学习率校准。本任务不跑 E-0010 的十组正式 run。

## 规格

- 契约：`research/experiments/E-0009.md`（训练程序、评估项、保存与重载、测试、校准），以及 `research/experiments/E-0008.md` Design 段中的 V2 共同训练配方（数据与哈希、计数、样本顺序、batch 8、停止规则、AdamW 与日程、评估 batch 16、学习率选择规则）。E-0010 的判定规则已封存（`research/experiments/E-0010.md`），训练程序要能输出它需要的全部量。
- 新增正式入口，例如 `src/v2_train.py` 与 `src/test_v2_train.py`。复用 `src/v2_memory.py` 的 `collate`、`train_forward`、`evaluate_batch` 与对照；加入 C_up。不改 `src/bypass_*.py`、`src/v1_*.py`、`src/v2_data.py`；确需改 `src/v2_memory.py` 时，E-0006 的测试必须继续通过。
- 第一个校准 run 之前，算出并在 notes 中报告：样本顺序 sha256、4M 与 0.2M 预算下的步数与 tokens、warmup 步数。researcher-3（T-0009，F 线正对照）按同一配方实现，请用 `rl agent tell researcher-3` 与它核对样本顺序 sha256 和步数，两边必须一致。
- 冒烟：12 种配置（2 特征 × 2 读算子 × 3 融合）各跑几十步，检查出口零初始化时与 Base 相同，测出步时、吞吐与显存，用于估算 E-0010 每组的 walltime。
- 校准：代表配置 qknorm_pre_rope + F_out，两个读算子 × {3e-4, 1e-3, 3e-3}，各 0.2M tokens；按共同配方的规则选择，端点时扩展一次。

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

正式训练程序是 E-0010 全部结论依赖的新训练与评估路径。测试与冒烟通过后，用 `rl review request --target E-0009 ... --author-model <你的模型 id>` 请求代码审核，范围包括训练循环与日程、对照与诊断、checkpoint 保存与重载、C_up。校准 run 可以与审核同时进行；审核要求改动训练或评估路径时，重做受影响的校准。E-0010 的正式 run 必须在审核通过之后。

## 预算与交付

- 预算：E-0009 ≤ 1 node·h（含测试、冒烟与校准）。
- 证据：每个 run 都要登记、记录并导入结果。校准用 `rl evidence add --level diagnostic`，只用于选学习率，不支持也不挑战 H-0003。
- 交付摘要：训练程序与测试结果；样本顺序 sha256 与步数（以及与 researcher-3 的核对结果）；12 种配置的步时、吞吐与显存，以及 E-0010 每组 walltime 的估算；校准表（读算子 × 学习率：calibration 的 normal/zero/shuffled NLL 与 EM、梯度统计）与选中的学习率；各 run 的 ATT/RES/EV 编号；实际 node·h；审核编号与仍未解决的问题。
