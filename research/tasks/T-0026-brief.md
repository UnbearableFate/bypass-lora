# T-0026：V3 续：T2 准备度检查与分阶段循环矩阵（E-0023）

## 目标

按 E-0023 做三件事：

1. 阶段 0：在未用过的 T2 calibration 上，检查共同起点 ATT-0196-1 是否满足准备度规则 R。
2. 阶段 A：R 成立后，训练 V3-01、V3-02、V3-03。
3. 按规则 A1-A4，决定释放阶段 B，或者先做诊断与修复。

完成后交付。

## 背景

- 决定 D-0031：顾问轮次 2（AD-0003、AD-0004）。EV-0111 保留为原门槛的失败结果；E-0022 不再追加运行。
- 代码：master 8609f93，已合并你的 T-0023（`src/v3_train.py`，RV-0058 -> RV-0059 pass）。
- 共同起点：ATT-0196-1。adapter 的 digest 已记录，在加载前后核对。
- 新增的描述性诊断需要新代码：
  - 终点边 value oracle；
  - 每步对第 s 条路径事实的读权重；
  - 按答案名频率分层的 EM。

  这些新路径在阶段 A 的评估之前过一次代码审核。阶段 0 与阶段 A 的训练用已审核的代码，不必等这次审核。
- 规则 4 的冻结转移诊断只在 A3 或 A4 时实现，实现后同样先审核。
- 准备度 R 第 3、4 条的统计量：
  - 已审核的评估器已经输出的（例如 metrics.json 里 evaluation.by_hop 的末位权重），直接用；
  - 没有输出的（例如末位 argmax 槽位分布），在 PBS 中用确定性脚本从逐条记录计算，不改评估器。
- 跨线：E-0023 的联动节同 E-0022。循环用 bypass_core 的同一个 LatentReader。researcher-1 正在做 RB1（T-0025），也用 LatentReader；改共享类之前先用 `rl agent tell researcher-1` 说明。

## 规格

以封存的契约 `research/experiments/E-0023.md` 为准，本节只是摘要。

1. 阶段 0：
   - 一个评估作业：calibration，S=1，六个对照。
   - 按 R 的四条逐条判定，写进 notes，并记一条 diagnostic 证据。
   - R 不成立：不训练任何 arm，按 E-0023 做一次第二起点周期，再在 calibration 上按 R 判定。仍不成立时停下报告。
2. 阶段 A：
   - R 成立后，V3-01、V3-02、V3-03 三个作业放进同一个通知组。
   - 设置与评估同 E-0022：S_eval 扫描、轨迹、各个对照、重载一致。
3. 新增诊断：
   - 实现与测试，请求一次代码审核，一个包覆盖三项诊断。
   - 审核通过后，对阶段 A 保存的最终 checkpoint 运行。
4. 判定：按规则 A1-A4。
   - A1：阶段 B 的各组（S_M*=2 时含 V3-05b）放进一个通知组。
   - A2：先训练 V3-09、V3-10。
   - A3、A4：实现冻结转移诊断（新代码，先审核）并运行，然后按对应关系至多做一个修复 arm，再把结论交给协调者。
5. 证据：
   - 阶段 0 记一条 diagnostic。
   - 阶段 A、B 按 E-0023 规则 6，每个问题记一条 screening。
   - 每条证据都注明准备度规则的来由：在看到起点 dev 结果后制定，在未用过的 calibration 上检查。

## 预算与交付

- 预算上限 2.6 node·h（E-0023）。阶段 A 之后不释放阶段 B 时约 0.4 node·h。将超出时，先停下报告。
- 交付摘要包括：
  - 准备度表（四条）；
  - 阶段 A 主表：各 hop 的 EM 与 NLL、各个对照、S_eval 扫描；
  - 每步路径权重、终点边 oracle、频率分层；
  - 规则的逐条判定；
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
