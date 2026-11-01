# T-0019：记忆路线 D5 配方的 seed 复现（E-0017）

## 目标

按 E-0017 跑 P1、P2（主判定）和 C1、C2（描述），交付各 run 在单独编码写入协议下的 H3 门槛结果与诊断。只测量与报告；H-0012 的判定与是否重开 H-0003 路线由协调者负责。

## 背景

- E-0016 的 D5（你在 T-0018 做的）在 seed 0 上达到 H3（EV-0095）；同协议从零训练的 D4 失败（EV-0096）。
- 按 E-0016 的 outcome_actions，不用标签的配方达标后要用 seed 1、2 复现。D5 的配方包括阶段 A（E=1 上的 R1）和阶段 B（E=16 上单独编码写入续训），所以复现两段都换 seed。
- C1、C2 从现成的 R1 seed 0 续训，只换续训 seed，用来把阶段 A 与续训的随机性分开。

## 规格

以封存的契约 `research/experiments/E-0017.md` 为准，本节只是摘要。

| arm | 阶段 A | 阶段 B |
| --- | --- | --- |
| P1 | R1，seed 1（E=1，6000 步） | D5，续训 seed 1 |
| P2 | R1，seed 2（E=1，6000 步） | D5，续训 seed 2 |
| C1 | 现成的 R1 seed 0（ATT-0122-1） | D5，续训 seed 1 |
| C2 | 现成的 R1 seed 0（ATT-0122-1） | D5，续训 seed 2 |

- 阶段 A：E-0014 的 R1 配方，E=1 数据（ATT-0112-1），初始化与数据顺序按 seed 0 的同一派生规则换成 seed s。
- 阶段 B：E-0016 的 D5 配方，数据顺序 `derive_seed(20261029, "concat_loop_v2", "V2R16", "D5", s)`；只载 adapter、AdamW 重置、warmup 300 再 cosine。
- 来源校验：阶段 B 的起点 adapter 要校验来源（P_s 是本任务阶段 A 的输出，C_s 是 ATT-0122-1），沿用你在 T-0018 写的 R1 来源校验，推广到任意起点 attempt。
- 代码用 master（85cbcdc 起）上已审核的实现。只换 seed 不需要新审核。如果要改 `src/` 的核心路径（训练、评估、写入、续训），先请求审核；只改作业编排脚本不需要。

## 评估与判定

见合同 Design 的“评估”与“判定”：
- 每个阶段 B 用同一个 h3_gate，在 E=16 dev 上按单独编码写入协议判定，并报分层、最后位置权重与 n_eff；
- P1、P2 的阶段 A 另在 E=1 dev 上报 normal、zero、shuffled；
- 描述项：E=16 calibration 上单独编码写入的 normal、zero、shuffled，E=16 dev 上上下文编码的 normal 与 oracle_value。

复现判定只看 P1、P2；C1、C2 只作描述。

## 预算与交付

- 预算约 0.55 node·h，上限 0.8 node·h。
- C1、C2 和两个阶段 A 可以立即一起提交；P1、P2 的阶段 B 等对应的阶段 A 结束后提交。
- 证据：每个阶段 B 一条 screening 证据，关联 E-0017。达到 H3 的对 H-0012 加 `--supports`；P1、P2 都不达标时不加 challenges，由协调者判定。阶段 A 的结果写在对应阶段 B 证据的 summary 里，或单独一条 diagnostic 证据。
- 交付摘要包括：
  - 四个阶段 B 的门槛表与分层；
  - 阶段 A 在 E=1 的表；
  - 描述项；
  - ATT/RES/EV 编号；
  - 实际 node·h；
  - 未解决的问题。

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
