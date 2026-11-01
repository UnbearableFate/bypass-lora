# T-0022：RB 线 RB0——残差旁路的新实现与计时（E-0019）

## 目标

按 E-0019 实现并检查 RB 线（plan2 v2.1，`docs/plan2.md`）需要的新 F 线路径，测出训练与推理成本，提交代码审核。只测量与报告；RBc（E-0020）与 RB1（E-0021）的放行由协调者在 RB0 审核通过后决定。

## 背景

- 用户 2026-11-02 把 plan2 作为另一条线加入（`docs/constraints.md`）。提取与命名见 `research/notes/plan2-extraction.md`；协调者的决定见 D-0025。
- RB 线的假设：H-0013（主线 R_res+KernelMatrix S1 的实用价值）、H-0014（侧支历史）、H-0015（同一旁路内的循环）、H-0016（gate/up 条件融合）、H-0017（新实现正确，本任务检验）、H-0018（candidate 探针）。
- 已有代码：`src/bypass_core.py`（KernelMatrix、KernelSlotReference、LatentReader、process_sequence 逐 token 参考扫描）、`src/bypass_model.py`（AdapterConfig、SideBranch、Fusion：F_out 即 R_res，C_gate/C_up/C_dual）、`src/v1_run.py`、`src/v1_lora.py`；V0 测试 E-0001、E-0004（H-0001 supported）。缺口清单见 E-0019 Design 的“现状与缺口”。
- 注意：`src/bypass_core.py` 与 `src/bypass_model.py` 是 M 线、V5 共用的代码。新选项一律默认关闭（impl 默认 reference，fixed_query 默认 False），关闭时与 master 逐位一致；同时跑 M 线与 V5 F 线路径的回归。

## 规格

以封存的契约 `research/experiments/E-0019.md` 为准，本节只是摘要。
1. KernelMatrix 向量化因果前缀实现（impl=prefix）、T0（branch=kernel_tokenwise）、AdapterConfig 的 fixed_query 与 impl、F 线侧支历史干预（iv_off、iv_window1/8、iv_strict、iv_swap、iv_permute、当前 token 读权重）、RB 训练入口（arm 表与参数断言、residual_la_v21 种子、2M tokens/3 epochs 日程、LR 校准模式）、latency 脚本。
2. RB 确认集切分（D-0025，方案 A2）：从 V1 train 中每任务切出 1,000 条，按 `derive_seed(20261101, "residual_la_v21", "RB5", "confirmation")` 在任务内均匀抽取，生成 id 清单、计数与 SHA256，提交清单；这些行不进入 RB 的训练、校准、dev 分析或任何模型评估。RB train 为 V1 train 去掉这些行。
3. Design 中“检查”表的全部检查在 PBS 中运行；FP32 容差在运行前写进测试，GPU BF16 容差沿用 E-0001/E-0004 的锁定值，不看结果后放宽。
4. 计时：在 2g.24gb 上 fail-fast（D-0019），按 Design“计时”一节测 ≥100 个稳定步，给出每个 RB1 arm（2M tokens）与 RB5 run（3 epochs）的 node·h 估计。
5. 代码审核：全部完成后提交一个审核包（`rl review request --target E-0019 ...`），范围见 Design“代码审核”。审核通过前不开始任何 RB 训练。

## 预算与交付

- 预算约 0.5 node·h，上限 1.0 node·h。
- 证据：一条 diagnostic 证据，关联 E-0019；全部检查通过时 `--supports H-0017`，某项失败且本任务内无法修复时 `--challenges H-0017` 并停下报告。
- 交付摘要包括：检查表（通过/失败与误差）、计时表与 node·h 估计（RB1 各 arm、RBc 每个 run、RB5 每个 run）、确认集清单路径与 SHA256、审核编号、ATT/RES/EV 编号、实际 node·h、未解决的问题。

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
