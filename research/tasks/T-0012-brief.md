# T-0012：V2 记忆失败层次定位（E-0012）

## 目标

按 E-0012 的 Design 完成 L0–L4 诊断测量并交付。只测量：不训练 M 线，不选择后续探针，不写因果结论；E-0013 的选择与 H-0003 的后续决定由协调者负责。

## 规格

契约 `research/experiments/E-0012.md`（已封存为 E-0012@1）为准，本节只是摘要。

- 输入：数据 `runs/ATT-0039-1/data`（train 16,384、calibration 2,048、dev 1,000）；E-0010 十个最终 checkpoint `runs/ATT-0080-1` … `runs/ATT-0089-1`（V2-01…V2-10，对应关系见 `research/artifacts/T-0011/training-run-map.json`）。不读 confirmation。
- 特征：第 14 层，direct_hidden 与 qknorm_pre_rope（q 侧 q_norm、k 侧 k_norm，均为 RoPE 前；v 侧 v_proj 输出）。support 与问题分别前向，从不带 adapter 的 Base 取。每个特征集用一个 E-0010 checkpoint 验证 hook 到的投影输入与提取的特征一致，记录最大绝对误差。
- 位置：分隔符 `delimiter_token_index`；问题位置 (a) 头实体名字最后一个 token（tokenizer offset 定位），(b) 最后一个问题 token。
- 拟合与报告：train 按 index 最后 2,048 个 episode 作内部验证（只选 L2 与轮数），其余拟合；报告 calibration 全部的 top-1，附正确数、样本数、随机水平与 train 频率常数基线。
- L0 正对照：尾实体名字最后一个 token 的 v 侧 -> 尾实体；头实体名字最后一个 token 的 k 侧 -> 头实体；问题位置 (a) 的 q 侧 -> 头实体；每个特征集都做。任一 < 90%：先查位置、tokenization 与拟合，修复后重跑，交付中说明。
- L1：分隔符 v 侧 -> 尾实体，k 侧 -> 头实体、关系；问题位置 (a)(b) q 侧 -> 头实体、关系；双线性寻址探针 s_i = (x_q U)·(x_i V)，秩 128/256/512/1024，episode 内 16 条事实 softmax、目标事实交叉熵，报告 top-1。
- L2：十个 checkpoint 与每个特征集一个随机初始化投影（v2_train 在 seed 0 下 step 0 的初始化）：value -> 尾实体，key -> 头实体、关系，query -> 头实体、关系。
- L3：`src/v2_probe.py` 跑十个 checkpoint，fit 用 train 前 4,096 个 episode，held-out 用 calibration 全部；不改它的 ridge 与 split 检查语义（可以加批大小等运行参数）。报告 clean_to_clean、read_to_read、clean_to_read。
- L4（dev 1,000）：normal memory 下各问题位置读取 argmax 槽位按支持顺序位置、同关系与否、是否目标的分布（最后一个问题位置单列）；episode 内 key 两两余弦（均值、分位数）与目标 margin，训练后对随机初始化；问题位置融合输出范数与第 14 层残差范数之比。
- 输出：`runs/<ATT>/` 下的报告 JSON 与 Markdown 表（按层次汇总，每个数字附样本数），源文件与输入 hash。

## 审核

新的特征提取、探针与几何脚本在全量运行前请求代码审核（`rl review request --target E-0012 ... --author-model <你的模型 id>`），范围：特征位置与协议一致、L0 对照、train/calibration 不混用、超参数只在 train 内部验证上选择、checkpoint 与随机初始化读取正确、寻址探针按 episode 归组。冒烟与合成测试可以先跑。

## 预算与交付

- 预算：E-0012 ≤ 0.5 node·h（V2 合计 ≤ 10 node·h，已用约 1.33）。
- 证据：diagnostic 级，关联 E-0012；不加 `--supports`/`--challenges`。summary 写各层次的关键数值。
- 交付摘要：L0 是否通过；L1–L4 的主要表（两个特征集、十个 checkpoint）；可解码性在哪些层次上下降的事实描述（不写原因或建议的探针）；ATT/RES/EV 编号；实际 node·h；审核编号与未解决问题。
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
