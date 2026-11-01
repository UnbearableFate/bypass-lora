# T-0014：V5 确认集封存、确认评估与分析脚本（E-0013）

## 目标

为 E-0013 的唯一一次开封做好准备，全程不读取官方 validation 的任何内容：
- 下载并按哈希封存官方 validation 与 WikiText-103 test；
- 写好由锁定文件约束的确认评估脚本和分析脚本，并配上测试；
- 通过代码审核；
- 用 dev 预测演练一遍。

本任务不开封 validation。开封在 T-0013（训练）交付后另立任务。

## 规格

以封存的契约 `research/experiments/E-0013.md`（E-0013@1）为准，本节只是摘要。契约若因结论审核 RV-0031 修订，协调者会通知你。

- 封存：
  - 在登录节点下载到 `data/v5-sealed/`（只允许下载与计算哈希，其他工作负载走 PBS）。
  - PIQA：valid.jsonl 与 valid-labels.lst，来自 V1 同一来源 ybisk/ybisk.github.io @ 21edab439af693b961be2f069e8690a88d3b4e37（参照 `src/v1_run.py` 的 SOURCES 与 train 文件路径）。
  - BoolQ：validation parquet，来自 google/boolq @ 35b264d03638db9f4ce671b711558bf7ff0f80d5。
  - 只记录来源 URL、revision、字节数与 SHA256，写入 `research/artifacts/T-0014/sealed-manifest.json` 并提交（不提交数据），然后把文件设为只读。
  - 不打开文件，不统计行数、长度或标签。行数 1,838/3,270 在开封时由脚本检查。
- WikiText-103：
  - Salesforce/wikitext，wikitext-103-raw-v1，test split，固定 revision，记录文件 SHA256。
  - 它不是确认集，可以在 PBS 作业中读取，但不用它做任何选择。
- 确认评估：`src/v5_confirm.py`。
  - 输入：锁定文件，列出每个 checkpoint 的 arm、seed、路径、SHA256 与预期参数数，以及代码 commit、数据与 WikiText 哈希、合同版本。
  - 先校验所有哈希；锁定文件缺失、字段缺失或哈希不符时，拒绝读取 validation。
  - 用 `BranchModel.load_adapter` / `LoRAModel.load_adapter` 加载 checkpoint（T-0013 使用 V1 的 config.json + adapter.pt 格式），不依赖训练代码。
  - 评估 Base 与各 checkpoint：PIQA/BoolQ validation 的候选打分与 V1 `evaluate` 完全一致。PIQA 标签为 0/1；BoolQ 的 True 对应候选 " yes"（index 0）。max_length 为 512。
  - 无法编码的行在所有模型中同样剔除并计数；任一任务剔除超过 0.5% 时标为协议偏离。行数不为 1,838/3,270 时停止。
  - 输出逐例预测 JSONL（字段同 V1）与指标。
  - 同一脚本计算 WikiText NLL：各行按原顺序直接拼接（行内已含换行），不加特殊 token 分词，切成不重叠的 1024-token 块，丢弃最后不满的块，取所有预测位置的平均 NLL。
- 分析：`src/v5_analysis.py`。输入逐例预测，按 split 名运行，同一份代码既能跑 dev 演练，也能跑 validation。
  - 各 arm/seed/task 的准确率与 NLL，均报 seed 均值 ± SD。
  - 主判据：Δ_s、Δ、δ_task。
  - 配对分层 bootstrap：10,000 次，numpy.random.default_rng(20261029)；在 PIQA、BoolQ 内各自重抽，所有 arm 与 seed 共用索引；百分位 95% 区间。
  - 判定规则：supported 为 Δ ≥ +1 pp、下界 > 0、三个 Δ_s > 0，且每任务 ≥ −1 pp；challenged 为上界 < +1 pp；其余为 inconclusive。
  - 次要比较 S1（V5-C − V5-F）与 S2（V5-F − V5-L）：区间、双侧 p = min(1, 2·min(P(Δ* ≤ 0), P(Δ* ≥ 0)))，在 α = 0.05 下做 Holm。
  - WikiText 各 arm 相对 Base 的 ΔNLL。
  - 输出 JSON 与 Markdown 报告。
- 测试：`src/test_v5_*.py`，用合成预测覆盖：
  - 三种判定的边界情形；
  - bootstrap 在 arm/seed 间共用索引，并按任务分层；
  - Holm 的排序与拒绝；
  - 剔除行的处理；
  - 无锁定文件或哈希不符时拒绝；
  - WikiText 分块（用玩具 tokenizer）。
  测试与演练都走 PBS。
- 演练：分析脚本在 dev 预测上完整跑一遍。
  - T-0013 的 dev 预测届时还没有的话，用 V1 E-0003 的 dev 预测：F_out 为 ATT-0024-1，C_up 为 ATT-0027-1，LoRA28 r2 见 `rl run show E-0003`。
  - 把 seed 0 的预测复制成三份充当三个 seed，只用于检验流程，不是证据。
  - 同时在 Base 上测一次 WikiText NLL 与耗时，用来估计开封作业的 walltime。

## 审核

开封前请求代码审核：

```
rl review request --target E-0013 ... --author-model <你的模型 id>
```

审核范围：
- validation 读取器与标签映射；
- 打分与 V1 `evaluate` 一致；
- 锁定文件与哈希的强制检查；
- 其他代码路径不读取 validation；
- bootstrap、Holm 与判定规则与 E-0013 一致；
- WikiText 协议。

附上演练报告。审核员由协调者分配；审核轮次规则见下文。

## 预算与交付

- 预算：测试、演练与 Base WikiText 测量 ≤ 0.3 node·h（MIG）。
- 不需要证据记录（不加 `--supports`/`--challenges`）；测试与演练的作业照常 register/record。
- 交付摘要包括：
  - 封存清单（哈希与字节数）；
  - 脚本与测试的 commit；
  - 审核编号；
  - 演练报告路径；
  - 开封作业的预计 walltime；
  - 实际 node·h 与未解决的问题。

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
