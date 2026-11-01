# T-0020：V5 官方 validation 的单次开封与确认分析（E-0013）

## 目标

按 E-0013@3 与 T-0014 的锁格式（`research/notes/T-0014-lock-schema.md`），组装并提交确认锁，审核通过后把 PIQA、BoolQ 官方 validation 与 WikiText-103 test 开封一次，评估九个最终 checkpoint 和 Base，运行锁定的分析，按封存规则记录确认性证据并交付。H-0008 的结论由协调者在结论审核后作出。

## 背景

- T-0013 已验收（master f0d9283）：九个正式 run（C_up r32、F_out r64、LoRA28 all-7 rank1 × seed 1、2、3），每个 29165 步、58329 次呈现、4173300 个非 padding tokens，最终 checkpoint 作业内重载一致，三个 arm 的样本顺序逐 seed 配对；三个 arm 的 LR 都锁在 1e-3（LR 锁 sha 70ba5356…e01a，D-0018 后为最终版本）。
- 九个 checkpoint 的清单：`/work/xg24i002/x10041/bypass-lora/runs/ATT-0174-1/checkpoint-inventory.json`（10830 bytes，sha256 2886d4981d2fe954bfc022b2d5b6641af638d1d0529d0012755538bd60619619），PBS 中已认证（EV-0101）；延迟 EV-0106；成本见 `research/notes/T-0013-delivery.md`。
- 评估与分析代码是你在 T-0014 写的 `src/v5_confirm.py`、`src/v5_analysis.py`，RV-0039 通过（0 个问题），dev 排练 ATT-0116-1。官方 validation 至今没有被解析过。

## 代码版本（重要）

锁里的 `code_commit` 用 **2b5c809f6101b5ba132d5565ebbf73ef91e105da**（T-0013 的正式训练源码），作业从它的只读快照运行。协调者核对过：`v5_confirm.REQUIRED_CODE` 列出的九个文件在 2b5c809 与 RV-0039 审过的 334fdb4 上逐字节相同。master 上的 `src/bypass_model.py`、`src/bypass_core.py` 后来被记忆路线改过，不要用 master 或你的任务分支作 code_commit。锁文件本身提交在你的任务分支上（例如 `research/artifacts/T-0020/confirmation-lock.json`），用绝对路径传给 `--lock`。

## 步骤

1. 组装锁：字段与校验见锁格式说明。`learning_rate_lock` 与 `checkpoints` 从 T-0013 的清单复制；`prerequisites.code_review` 用 RV-0039；`prerequisites.dev_rehearsal` 用 ATT-0116-1 的 `analysis/analysis.json`；`unseal_receipt` 用一个固定的共享绝对路径。提交锁。
2. 预检：在 PBS 中对这份锁运行不读 validation 的完整检查（元数据、代码哈希、合同、前置条件、九个 checkpoint 的文件与训练清单），只到 validation 文件之前为止。若 `v5_confirm.py` 没有单独的预检入口，写一个只调用 `verify_lock` 前半部分的作业脚本，不改 `src/`。
3. 审核：用 `rl review request --target E-0013` 请求审核，范围是锁文件与预检结果（锁的每个字段是否与 T-0013 交付、封存清单、RV-0039 一致；code_commit 的选择是否正确）。审核通过前不开封。
4. 开封：审核通过后提交一次 `python src/v5_confirm.py evaluate --lock <锁> --output runs/<ATT> --unseal`。walltime 按 T-0013 的最终 dev 计时更新估计，× 1.5，short-mig。工程失败时按锁格式说明用 `--resume` 和新的 attempt 输出目录重试；不得改锁、代码或数据后重开。
5. 分析：在 PBS 中运行锁定的 `v5_analysis.py`（10000 次配对分层 bootstrap，RNG 20261029，次要比较 Holm 校正）。
6. 证据：一条 confirmatory 证据，关联 E-0013，按封存规则加边：四条判据（seed 平均宏平均差 Δ ≥ +1 pp；95% 区间下界 > 0；三个 seed 的 Δ_s 都 > 0；每个任务 ≥ −1 pp）全满足时 `--supports H-0008`；主差 95% 区间上界 < +1 pp 时 `--challenges H-0008`；其余情况不加边，在 summary 中写明 inconclusive。次要比较（F_out vs C_up 等）、WikiText NLL 变化、参数、可合并性、训练时间、峰值显存、延迟都写进证据或交付摘要。

## 预算与交付

- 预算约 0.5 node·h，上限 1 node·h（E-0013 总上限 12 node·h）。
- 交付摘要包括：锁的路径与 sha、审核编号、开封作业、主比较表（各 seed 的 Δ_s、seed 平均、区间、每个任务）、次要比较与 Holm 结果、WikiText 与成本表、排除的样本数、ATT/RES/EV 编号、实际 node·h、未解决的问题。
- 确认集只开封一次：开封后的任何结果都不能用于改方法后再确认。
## 环境

- 沿用 master 的 `pyproject.toml`/`uv.lock`（lock 3c916b51337d，cuda extra，环境 `/work/xg24i002/x10041/bypass-lora/envs/3c916b51337d-cuda`，cuda/12.8，torch 2.10.0+cu128，transformers 4.57.1，Python 3.13.13）。不要改依赖；确需改动时先问协调者（几个研究者并行工作，lock 改动会互相冲突）。
- 模型 Qwen/Qwen3-0.6B-Base，model 与 tokenizer revision da87bfb608c14b7cf20ba1ce41287e8de496c0cd，`HF_HOME=/work/xg24i002/x10041/hf_home`，作业中 `HF_HUB_OFFLINE=1`。BF16 冻结主干、FP32 新参数、eager attention。硬件 GH200 MIG 2g.24gb。

## 计算规则

协调者在登录节点上（`compute.coordinator: login`）。所有 Python 工作负载（pytest、冒烟、数据生成、评估、分析）都通过 miyabi-mcp 提交 PBS 作业，不在登录节点运行，也不回退到本地执行。

1. 本任务的 code commit 固定为 2b5c809（见上文）；`rl run register --experiment E-0013 --resource miyabi --code-commit 2b5c809f6101b5ba132d5565ebbf73ef91e105da ...`。
2. 按 `rl compute steps --attempt <ATT>` 暂存只读快照 `/work/xg24i002/x10041/bypass-lora/snapshots/<sha>/`；作业只写 `/work/xg24i002/x10041/bypass-lora/runs/<ATT>/`，最后写 `manifest.json`。
3. `job_submit` 先 dry run，再带 `plan_digest` 提交；request key 用 `sr:<ATT>`，client_ref 用 attempt id；作业脚本先运行 allocation check。
4. 队列用 debug-mig 或 short-mig；walltime 按实测 × 1.5 申请；作业不得跨越 2026-11-25 09:00 JST 的计划停机。
5. 几分钟的作业用 `job_wait`（每次 ≤ 45 s，不要循环轮询）；更长的作业带 `notify` 提交，然后 `rl task wait`，在 WAIT 中给出 `daemon.listen_command`，由协调者在后台运行。
6. PBS 的 FINISH 或 Exit_status=0 不代表应用成功，要检查输出。作业事实用 `ledger_query` 的 smooth-research 格式导出后 `rl run record --observation @file`；结果用 `rl run import-result` 导入。用量换算：node·h = 账本 mig_instance_hours × 0.25。
7. 切片（D-0018、D-0019）：PBS 可能把作业放到 MIG 1g.24gb 切片上，账本却记为 2g.24gb。用户决定不考虑这点 BF16 舍入差异：切片型号只作元数据记录，不因切片重跑、分析或标记；跨作业逐位比对的不一致如果能用切片不同解释，记下后继续。guard 只记录、不让作业失败；只有比较计时（延迟、吞吐）的作业要求 2g，不是 2g 就失败。

## 审核轮次（协调者常设规则）

一轮审核最多两次（首审与一次 `--recheck-of`）。两次用完后仍有 medium 或更高的问题时，只要你认同这些问题，并且修复不改契约、容差、数据切分或指标定义，就直接修复，跑相应的 PBS 回归，然后用 `--recheck-of <上一份 RV> --new-round` 开新一轮，reason 写“协调者常设规则 2026-11-01：认同的 medium 问题已修复，修复不改契约、容差、数据或指标；本轮只复查该修复”。你不认同某个问题、修复需要改契约/容差/数据/指标，或希望带着未解决的问题继续时，交给协调者。审核请求由协调者分配给审核员。
