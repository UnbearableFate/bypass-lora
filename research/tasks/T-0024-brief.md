# T-0024：RBc：残差旁路线四个训练家族的学习率校准（E-0020）

## 目标

按 E-0020：四个训练家族各用 seed 0 做短训练，在完整的 V1 calibration（每任务 2,048 条）上计算 NLL，按预登记规则锁定学习率，写出 lr-lock 文件，供 RB1（E-0021）使用。四个家族：

- RBc-L：同四层 all-7 LoRA rank7；
- RBc-T：R_res+T0 S1；
- RBc-R：R_res+KernelMatrix S1；
- RBc-C：C_dual+KernelMatrix S1。

## 背景

- 代码起点：master（已合并 T-0022）的 `src/rb_train.py run --stage calibration`。
  - 审核：RV-0060 及其 re-check 覆盖这段代码。
  - 已执行过的入口：ATT-0197-1 已经跑过三个 CLI 冒烟，覆盖 RB1-02、RB1-01、RB1-07 三类 arm。
  - 按 RV-0060 第 3 条，校准阶段不做 export 与 fresh reload，只有 calibration NLL 进入 `lock_lrs`。
- 成本：以 T-0022 的 one-pass RBc 计时行为准（T-0022 的 cost tables）。
- 审核：只跑已审核的校准路径，不需要新的代码审核。若要改 `rb_train.py`、`bypass_core.py`、`bypass_model.py` 或数据代码，先停下，按审核规则请求审核。
- 跨线：E-0013 的校准（三个 arm 都锁在 1e-3，扩到 3e-3 更差）、E-0002 与 E-0009 的结果只作先验参照，不沿用（E-0020 comparator）。

## 规格

以封存的契约 `research/experiments/E-0020.md` 为准，本节只是摘要。

1. 初始网格共 12 个候选：
   - L 与 T：{1e-4, 3e-4, 1e-3}；
   - R 与 C：{3e-4, 1e-3, 3e-3}。

   每个候选在首次累计达到 200,000 个非 padding 输入 tokens 的完整步停止。初始化与训练顺序为 `derive_seed(20261101, "residual_la_v21", "V0", "init"|"train", 0)`，同家族各候选的初始参数相同。日程形状、loss 与优化器同 RB1。训练数据是 RB train，即 V1 train 去掉 A2 确认集行，split hash 必须与 T-0022 的 `rb-confirmation.json` 一致。12 个作业放进同一个通知组提交，也可以按 RB0 计时打包成少数作业。
2. 扩展只做一次，看各家族初始网格中有限值的原始最小 NLL：
   - 最小在上端：追加上扩展点，L/T 为 3e-3，R/C 为 1e-2；
   - 最小在下端：追加下扩展点，L/T 为 3e-5，R/C 为 1e-4。

   扩展候选放进第二个通知组。扩展后最小值仍在新端点时，照规则选取，标注“网格边界”，不再扩展。
3. 选择：非有限的候选淘汰。在最终的有限候选中，取 (NLL − min)/min ≤ 0.01 的最小 LR。用已审核的 `lock_lrs` 写 lr-lock 文件，内容包括：
   - 各 manifest 的 SHA256；
   - split manifest 的 SHA256；
   - plan order 的 SHA256。

   提交到任务分支。RB1 在 lock 合并之前不训练。
4. 某个家族的候选全部非有限：停下报告，回到 RB0 查实现，不扩网格。
5. 不看 dev 与任何确认集（A2、官方 validation）。不做 RB1 训练。

## 预算与交付

- 预算上限 1.5 node·h（E-0020）。预计将超出时，先停下报告。
- 证据：一条 diagnostic 证据，说明各家族锁定的 LR、是否扩展、是否标注网格边界。关联 E-0020 与 H-0013，不写 supports/challenges 方向。
- 交付摘要包括：
  - 候选表：家族、LR、步数、tokens、calibration NLL、是否有限、步时、node·h；
  - 扩展与选择的逐条判定；
  - lr-lock 文件的路径与 SHA256；
  - ATT、RES、EV 编号；
  - 实测总用量；
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
