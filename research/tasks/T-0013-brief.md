# T-0013：V5 训练（E-0013）

## 目标

按 E-0013 实现 V5 训练路径，完成每个 arm 的 LR 校准和九个正式训练 run（3 arm × seed 1–3），交付锁定的 LR、最终 checkpoint、dev 监控结果、reload 检查与成本测量。官方 validation 已封存：本任务不下载、不读取、不引用它。封存与确认评估由 T-0014 负责，开封在本任务交付后另立任务。

## 规格

以封存的契约 `research/experiments/E-0013.md`（E-0013@2，按 RV-0031 修订）为准，本节只是摘要。

- Arms（4 层为 [6,13,20,27]）：
  - V5-C：C_up，tokenwise SiLU，r32，4 层，524,288 参数。
  - V5-F：F_out，tokenwise SiLU，r64，4 层，524,288 参数。
  - V5-L：LoRA，全 28 层 all-7，rank1，alpha=rank，630,784 参数。
  - 代码断言参数数，且可训练参数均为 FP32。
- 代码：新建 `src/v5_train.py`（及测试），复用 `v1_data`、`v1_lora`、`bypass_model` 与 `v1_run` 中的函数；V1 的行为与测试保持不变。checkpoint 用现有 `save_adapter` 格式（config.json + adapter.pt），保证 `BranchModel.load_adapter` / `LoRAModel.load_adapter` 不经训练代码就能加载。T-0014 的确认评估只依赖这个格式。
- 数据：沿用 V1 的切分、模板、候选打分与 max_length 512，以及已登记的 PIQA train 行 5528 排除规则。先验证过滤后每个 epoch 为 19,443 行。
- Seed s ∈ {0（只用于 LR 校准）, 1, 2, 3}：
  - 顺序：Random(derive_seed(20261029, "concat_loop_v2", "V5", "train", s) + epoch)，交替规则同 V1。
  - 初始化：新参数初始化前调用 torch.manual_seed(derive_seed(20261029, "concat_loop_v2", "V5", "init", s))。
  - 同一 seed 下三个 arm 的顺序与初始化种子相同。记录每个 run 的 order SHA256。
- Schedule：
  - 正好 3 epochs，沿用 V1 的 plan_batches（未满的 batch 跨 epoch 边界延续，只在上限处补出最后一个），29,165 步、约 4.46M 非 padding 输入 tokens；记录实际步数与 tokens。
  - 有效 batch 2、microbatch 1；训练目标同 V1（MSG-0025 的更正）：每个样本先对监督 token 的 CE 取平均，再在一步的样本间平均，复用 `example_mean_objective`；不用 token 加权。
  - AdamW（0.9/0.999、eps 1e-8、weight decay 0.01），梯度裁剪 1.0，前 5% 步 warmup，之后 cosine 降到 0。
  - 只用最终 checkpoint。dev 监控在消耗每个 epoch 最后一个样本的那一步之后进行（预计第 9,722、19,443、29,165 步），只作监控。
  - 保存后重新加载到新的 Base 上，重复 dev 评估，逐例分数与预测须完全一致。Base 重复评估一次，测同设备噪声。
- LR 校准：
  - 每个 arm 用 seed 0 跑 0.2M tokens（schedule 形状同上，按本 run 的步数计算），网格为 {1e-4, 3e-4, 1e-3}。
  - 按初始网格中有限值的原始最小 NLL 决定扩展：最小值在 1e-3 时追加 3e-3，在 1e-4 时追加 3e-5；每端只扩一次。
  - 指标为 calibration NLL（V1 定义，两任务 NLL 的平均）。非有限值的候选淘汰。在最终的有限候选集中，取满足 (NLL − 最小 NLL)/最小 NLL ≤ 0.01 的最小 LR。
  - 把选择表与锁定配置提交为文件（如 `research/artifacts/T-0013/lr-lock.json`），然后再开始 seed 1–3 的训练。
- dev 预测：每个 run 保存最终 dev 的逐例预测 JSONL，字段同 V1 的 `evaluate`（task、split、id、gold、prediction、scores、gold_nll_sum、gold_loss_tokens），T-0014 会用它做分析演练。
- 成本：
  - 每个 run 记录墙钟时间、稳定步时（去掉前 10 步）、峰值显存，以及加载、保存、评估时间。
  - 训练完成后跑一个作业测推理延迟：batch 1，对固定的 256 个 dev 候选输入做前向，比较 Base、V5-C、V5-F、V5-L（合并前与合并后），记录 GPU 类型。
- 不允许：读取官方 validation；用 dev 选 checkpoint；各 arm 用不同样本预算。若实测成本推算超出预算，先停下报告；需要缩短时所有 arm 统一缩短。

## 审核

在 LR 校准与正式训练之前，请求代码审核：

```
rl review request --target E-0013 ... --author-model <你的模型 id>
```

审核范围：
- 新 arm（F_out r64、LoRA28 r1）的注入位置与参数数；
- V5 的 seed、顺序与初始化派生；
- 3-epoch schedule 与排除规则；
- checkpoint 的保存与 reload；
- LR 选择规则；
- 不访问 validation。

冒烟与单元测试可以先跑。审核员由协调者分配；审核轮次规则见下文。

## 预算与交付

- 预算：E-0013 合计 ≤ 12 node·h，本任务约 7 node·h（LR 校准约 0.4，九个 run 约 6.5）。按 V1 实测，稳定步时 C/F 约 0.21 s，L 约 0.44 s。九个 run 可以同时提交。
- 长作业带 notify 提交，然后 `rl task wait`，在 WAIT 中给出 listen_command。
- 证据：LR 校准与 dev 监控只记为 diagnostic，关联 E-0013，不加 `--supports`/`--challenges`。确认性证据来自开封。
- 交付摘要包括：
  - LR 表与锁定文件；
  - 九个 run 的 ATT 编号、checkpoint 路径与 SHA256、步数与 tokens、各 epoch 的 dev 指标、reload 检查结果；
  - 延迟与成本表；
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
