# T-0025：RB 线 RB1：干预运行器、读诊断与分析脚本，随后九组筛选（E-0021）

## 目标

按 E-0021 分两个阶段完成 RB1。

- 阶段 1：先实现四样东西，并通过代码审核：
  - 冻结侧支历史干预运行器；
  - 读权重诊断：当前 token 权重、n_eff、熵；
  - 分析脚本：规则 1–11、描述性 bootstrap、成本表；
  - 延迟测量驱动。
- 阶段 2：RBc 的 LR lock 合并到 master 后，训练九个 arm，做干预与延迟测量，按预登记规则分析并交付。

## 背景

- 代码起点：master 6e2a2ed，已合并 T-0022 的 RB0 实现。

  | 模块 | 内容 |
  | --- | --- |
  | `bypass_core` | prefix KernelMatrix 与 LatentReader |
  | `bypass_model` | 模型接线 |
  | `rb_history.py` | iv_off、window、strict、swap、permute。按 RV-0060 第 2 条，swap 与 permute 只接受从位置 0 开始的连续有效 token |
  | `rb_train.py` | RB1 训练入口。RV-0060、RV-0062 已审核；ATT-0197-1 冒烟过 RB1-01、RB1-02、RB1-07 |
  | `rb_latency.py`、`rb_report.py` | 延迟与报告 |
  | `rb_data.py` | A2 切分 |

- researcher-2 正在用同一套代码做 RBc（T-0024）。RBc 的 lock 合并到 master 之前，RB1 不训练。
- 不要改 `rb_train.py` 的校准路径与 lock 格式。需要改 `rb_train.py`、`bypass_core.py`、`bypass_model.py` 或 `rb_history.py` 时，先用 `rl agent tell researcher-2` 说明改哪个函数、为什么（MSG-0548/0549 的约定）。新功能尽量放进新模块。
- 跨线（E-0021 的联动节）：
  - 读诊断与 M 线 `src/v2_memory.py` 的 read_metrics 用同一定义；
  - 循环用 bypass_core 的同一个 LatentReader。
- 成本：T-0022 的成本表（`research/notes/T-0022-cost-tables.md`）给出九组核心训练约 2.44 node·h；干预、分析与延迟另计。

## 规格

以封存的契约 `research/experiments/E-0021.md` 为准，本节只是摘要。

### 阶段 1：现在开始

1. 干预运行器：
   - 对象：RB1-02、S_F* 对应的 arm、RB1-08、RB1-09。
   - 在 dev 上跑 iv_off（须等于 Base）、iv_window1、iv_window8、iv_strict、iv_swap、iv_permute。
   - RB1-01 只跑 iv_swap，作无操作检查。
   - 四层同时作用，原 attention 与 KV 不变。
   - 输出：每种干预的 dev 宏平均与每任务准确率，以及与正常路径的配对差和描述性区间。
2. 读诊断（只作描述）：
   - 当前 token 读权重；
   - 读权重的 n_eff = 1/Σp² 与熵 −Σp·log p，其中 p 为当前位置对因果前缀各位置的归一化读权重，定义同 `src/v2_memory.py` 的 read_metrics；
   - 另报 n_eff 除以可读位置数；
   - S>1 时报第 1 步与最后一步。
3. 分析脚本：
   - 规则 1–11 逐条判定，用精确分数，阈值含等号；
   - 描述性的配对分层 bootstrap：10,000 次，seed 20261101，各 arm 共用重采样索引；
   - C*_RB 的选择规则；
   - 成本表与 plan2 §16.5 的记录项。
   - Base 沿用 ATT-0024-1（EV-0022）；评估代码有改动则重跑 Base。
4. 延迟：
   - 对象：RB1-01、RB1-02、RB1-04、RB1-08、RB1-07，另报 LoRA 合并后的参考。
   - 只在 2g.24gb 上运行，不是 2g 就失败（fail-fast）。
   - 测量：batch 1、512 上下文的 prefill 吞吐；128 token decode 的 p50/p95；峰值显存；状态字节。
5. 测试与审核：
   - 新路径的单元测试与回归在 PBS 中运行。
   - 然后请求一次代码审核，一个包覆盖干预运行器、读诊断、分析脚本与延迟驱动。
   - 审核之前可以跑冒烟与确定性检查。
   - 分析脚本在读取任何 RB1 dev 结果之前通过审核。

阶段 1 完成后以 WAIT 结束，等协调者通知 lock 已合并。

### 阶段 2：协调者通知 RBc lock 已合并后

6. 训练：
   - 九个 arm 放进同一个通知组。
   - 共同设置与 E-0021 一致：2M tokens 或 3 epochs（先到为准），LR 取自 lock，最终 checkpoint，重载一致，25/50/75% dev 曲线。
   - 如果按实测成本需要缩短，所有 arm 统一缩短，并在训练前登记。
7. 分析与证据：
   - 干预、延迟与分析都在 PBS 中运行。
   - 按规则 11，每个问题记一条 screening 证据，关联 E-0021。
   - 假设状态由协调者在结论审核后再改。

## 预算与交付

- 预算上限 4 node·h（E-0021）。阶段 1 的测试与冒烟约 0.1–0.2 node·h。将超出时先停下报告。
- 阶段 1 交付：测试结果与审核编号。
- 阶段 2 交付摘要包括：
  - 主表：九组与 Base 的 PIQA、BoolQ、宏平均与 NLL；
  - 规则 1–11 的逐条判定；
  - 干预表与读诊断；
  - 延迟与状态；
  - 成本表；
  - ATT、RES、EV 编号；
  - 未解决的问题。
- 等长作业时以 WAIT 结束本轮，并给出 listen 命令；唤醒由协调者负责。

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
