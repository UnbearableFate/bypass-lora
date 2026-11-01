# T-0023：V3 训练矩阵：共同起点阶段与 plan §9.2 十组（E-0022）

## 目标

按 E-0022：先做共同起点阶段（D5 seed 0 在 T2 hop1 上续训 6,000 步），按 H3 门槛判定；过门槛后，从同一起点训练十组 V3-01..V3-10，评估、扫描、诊断，按预登记规则逐条报告。

## 背景

- E-0018（你的 T-0021，EV-0107..0109）：三个 D5 adapter 在 T2 hop1 上都没过 H3；oracle_value 为 76–80%，oracle_key 只到 0.14–0.15。因此起点阶段训练全部侧支参数，key 也要改变。
- 代码起点：master（已合并 T-0021，3908c16）中的 `src/v3_loop.py`、`src/v3_data.py` 与单独编码写入代码；起点 adapter 为 ATT-0159-1，加载前按已记录的 digest 核对。
- 共享代码：各组的循环都走 bypass_core 的 LatentReader（steps、fixed_query、update、readout）。researcher-2 的 RB0（T-0022）也改 bypass_model 与调度；改共享类之前先用 `rl agent tell researcher-2` 说明改哪个函数、为什么（MSG-0548/0549）。
- 跨线：E-0022 的联动节与 E-0021 的相同，结果以后与 RB1 按方向并列。

## 规格

以封存的契约 `research/experiments/E-0022.md` 为准，本节只是摘要。

1. 实现：T2 上的单独编码写入训练路径（起点阶段只取 hop1）；V3-06 同源 tokenwise 核；V3-07 更宽 S1（宽度按 plan §15 的 MAC 公式或 2g 计时 profile 锁定，写进 notes）；V3-08 不共享的 Rq/Ro；V3-07、V3-08 的初始化检查。新选项默认关闭，关闭时与 master 一致（回归测试）。
2. 代码审核：T2 训练路径、V3-06/07/08 与分析脚本都是新路径，起点阶段的正式作业之前请求一次审核（一个包覆盖起点阶段、十组与分析）。只改已审核的选项（S、η、fixed_query、update、readout）不需要新审核。
3. 起点阶段：一个作业，6,000 步，最终 checkpoint，重载一致，判 H3。不过：停下报告，等协调者（会做一次咨询）。
4. 十组：起点通过后，十个作业放进同一个通知组提交；每组 6,000 步，数据顺序相同，最终 checkpoint，重载一致。
5. 评估：S_eval=S_train 为主，另做 S_eval 扫描；各对照；轨迹；S_M* 组的中间状态干预；bootstrap 描述区间（seed 20261029）。分析脚本在读取十组 dev 结果之前写好并通过审核。
6. 计时作业只在 2g.24gb 上运行（fail-fast）；训练与评估可用任意 MIG 切片，切片只记元数据（D-0018、D-0019）。

## 预算与交付

- 预算约 1.9 node·h，上限 3 node·h（E-0022）；将超出时先停下报告。
- 证据：起点阶段一条 diagnostic；十组按规则 10，每个问题一条 screening，关联 E-0022，按规则写 supports 或 challenges；假设状态由协调者在结论审核后改。
- 交付摘要：起点门槛表与曲线；十组主表（hop1、hop2、hop3、hops2–3、NLL、各对照）；S_eval 扫描表；轨迹与干预摘要；规则 1–9 的逐条判定；成本表；ATT/RES/EV 编号；未解决的问题。
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
