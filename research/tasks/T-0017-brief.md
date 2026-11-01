# T-0017：记忆路线训练筛选——解码器先行与槽位边际似然（E-0016 的 D1、D2a、D2b、D3）

## 目标

按 E-0016 实现并运行 D1、D2a、D2b、D3（seed 0 筛选），交付各 arm 在 E=16 dev 上按合同的门槛结果与诊断。D4/D5（单独编码写入）不在本任务内：它们取决于 E-0015 的结果，由另一个任务做。只测量与报告；H-0010、H-0011 的判定与后续路线由协调者负责。

## 背景

- E-0014（EV-0080–EV-0085）的六个单改动 arm 都没有在 E=16 上达到 H3；只有 R1（只用 E=1 训练）证明 value 通路在 E=1 可用。
- 顾问轮次 1：`research/advice/AD-0001.md`、`research/advice/AD-0002.md`，先读 AD-0002 §1、§4。
  - 同一 value 映射对第 0 条事实和有前文的事实互不迁移；
  - 没有一个 E=16 arm 的解码器在干净读取上训练过；
  - S=1、increment、F_out 下，value 到残差是线性路径 W = A_v·R_o·P。
- E-0016 把解码器学习与寻址学习在训练中分开：
  - D1：只给目标槽位的 teacher-forced 读取，先学解码器；
  - D2a/D2b：在 D1 的基础上，再看寻址能否只从答案 CE 学到；
  - D3：把训练目标换成槽位隐变量的边际似然。
- E-0015（researcher-3，T-0016）与本任务并行，只做评估。两边都会改 `src/v2_memory.py`。请把新的训练目标与选项尽量放在新模块里，`v2_train.py` 与 `v2_memory.py` 只加最小的接入点，减少合并冲突。

## 规格

以封存的契约 `research/experiments/E-0016.md` 为准，本节只是摘要。

### 共同配置

- 基础配置同 E-0014 的 V2-03：direct_hidden、soft_slot、F_out，第 14 层 r128 S=1，LR 3e-3，每段 6000 步、每步 8 个 episode（`--step-budget 6000`），seed 0。
- 数据：E=16 train/calibration/dev 用 `runs/ATT-0039-1/data`；E=1 dev 用 ATT-0112-1。

### Arms

D1：teacher-forced 单槽训练。
- 训练时记忆的有效槽位只留目标槽位（用 `slot_columns` 定位），所有问题位置读到的就是目标 value。
- 用到目标索引，属额外监督。

D2a/D2b：从 D1 的最终 adapter 续训。
- 只加载 adapter 参数；AdamW 状态重置；重新 warmup 300 步再 cosine。
- 数据顺序种子：`derive_seed(20261029, "concat_loop_v2", "V2R16", arm, 0)`。
- 恢复全部槽位、正常读取、只用答案 CE。
- D2a 只训练寻址参数：query 与 key 的投影、reader.Rq。value 投影、reader.Ro、融合 P 冻结。
- D2b 所有 adapter 参数都训练。
- 用测试断言 D2a 只有寻址参数拿到非零梯度。

D3：槽位隐变量边际似然。
- L = −log Σ_i w_i·p(answer | 最后位置只读 v_i)，在 log 空间计算（logsumexp）。
  - w_i：最后问题位置的正常 SoftmaxSlot 权重；
  - p(answer | 只读 v_i)：最后位置的读取替换为 v_i、其他问题位置不加增量时的答案概率；
  - 支持侧只前向一次，问题侧每个槽位一次（可把 16 份问题拼成 batch）。
- 先做冒烟量步时；超过 V2-03 步时的 3 倍就停下报告。

### 评估

E=16 dev 上：
- EM 与 NLL，覆盖六种标准对照；
- 目标权重（全问题位置平均与最后位置）；
- n_eff、最后位置 argmax 分布、融合增量与残差的范数比；
- 以上都按目标位置分层（第 0 条与第 1 条及以后）；
- 每 500 步记训练 CE；25/50/75% 学习曲线。

另外：
- D1：
  - teacher-forced 对照：tf_normal、tf_shuffled（donor 同一索引的槽位）、tf_wrong（本 episode 第 (target+1) mod 16 条）、zero；
  - D1、D2a、D2b 另在 E=1 dev 上报 normal、zero、shuffled。
- D3：硬读取（最后位置读 argmax 槽位的 value）、软读取（最后位置读 Σ w_i v_i）、边际 NLL；zero、shuffled、permuted 都在只读最后位置的协议下做。
- 保存后重新加载，评估须逐例一致。

### 判定

见合同：
- D1：tf 差 ≥ 50 pp 为达标；< 20 pp 且训练 CE 已平台为未达标；其间为部分。
- D2a、D2b：E-0010@2 的 H3。
- D3：硬读取协议下的 H3。这是新协议的统计量，软读取结果并列报告。

## 审核

新选项默认关闭时，输出与梯度须与当前 master 一致：用旧/新回归确认，同 E-0014 的 `v2r_default_regression.py` 做法。

全量运行前请求代码审核：

```
rl review request --target E-0016 ... --author-model <你的模型 id>
```

审核范围：
- 默认关闭时与 master 一致；
- teacher-forced 掩码与评估对照的掩码正确；
- 续训不带入优化器状态，调度从头开始，D2a 的梯度只到寻址参数；
- 边际似然的数值稳定性，每个槽位的前向只替换最后位置；
- 硬/软读取的定义。

冒烟与单元测试可以先跑。D2a/D2b 依赖 D1 的 checkpoint，所以 D1 与 D3 先并行提交，D1 结束后再提交 D2a/D2b。

## 预算与交付

- 预算：本任务约 0.5 node·h（D1、D2a、D2b 各约 0.1，D3 约 0.15–0.2）；E-0016 合计上限 0.9 node·h。
- 证据：每个 arm 一条 screening 证据，关联 E-0016。
  - D1 支持或挑战 H-0010 的阶段 1 部分，写在 summary 里，不加 supports/challenges；
  - D2a/D2b 只在达到 H3 时对 H-0010 加 `--supports`；
  - D3 只在达到硬读取 H3 时对 H-0011 加 `--supports`；
  - 其余由协调者判定。
- 交付摘要包括：
  - 各 arm 的门槛表与分层；
  - D1 的 teacher-forced 表、D3 的硬/软读取表；
  - ATT/RES/EV 编号；
  - 实际 node·h；
  - 审核编号与未解决的问题。
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
