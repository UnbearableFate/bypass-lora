# T-0002：V1 任务切片、训练评估流水线与家族学习率校准（E-0002）

## 目标

实现 V1 的数据切分、模板、训练与评估流水线和 LoRA 基线，经代码审核后完成 Base 评估与 6 个校准 run，按 E-0002 的规则选出三个家族的学习率，并用实测成本给出 E-0003（八组接口筛选）的 release 草案。E-0003 的 run 不属于本任务。

## 规格

- 契约：`research/experiments/E-0002.md`，包括 Design 段的切分、模板、训练配方、calibration NLL 定义、选择规则、切片检查与成本记录。
- 配方来源：封存计划 `docs/plan.md` 的 §6.4（校准预算与学习率候选）、§7.1–7.3（F 线 tokenwise proxy、接口定义、八组）、§13.1（`derive_seed`）、§13.3（训练默认值，PAD 与 EOS 同 ID 时的屏蔽）、§15（参数数）。
- 复用 V0 代码：`src/bypass_model.py` 已实现 tokenwise proxy（`branch='tokenwise_silu'` 与 `'tokenwise_linear'`）和全部融合接口，直接使用，不另写一份。V0 检查（`src/test_core.py`、`src/test_model.py`）要继续通过。
- 本任务新增：
  1. 数据：下载、记录来源与 revision、确定性切分、模板与截断、两任务混合与顺序。
  2. LoRA 基线：4 层 `[6,13,20,27]` all-7 rank10，以及 E-0003 要用的 28 层 all-7 rank2。可以自己实现，也可以用 peft。需要检查：初始化时输出与 Base 相同、主干冻结、实际可训练参数数（4 层 rank10 为 901,120，28 层 rank2 为 1,261,568）。侧分支各融合的参数数要与 E-0003 表一致（F_out/F_pre 262,144，C_gate/C_up 524,288，C_dual 917,504）。
  3. 训练循环：AdamW、warmup 占实际优化步数 5% 后 cosine、global-norm clip 1.0 并记录 pre-clip 范数、非 padding 输入 tokens 计数与停止规则、新参数 FP32。
  4. 评估：PIQA 长度归一化选择、BoolQ 标签概率比较、calibration NLL 与准确率。所有 arm 在相同样本上评估。
  5. checkpoint 保存与重载：侧分支沿用 `export_adapter`/`load_adapter`，LoRA 也要能重载，重载后复现同样的评估结果。
- 训练与评估入口由配置选择 arm，E-0003 的八组直接复用，不要为校准写一次性脚本。
- RV-0002 的两条 low 跟进只约束记忆路径与 V2，本任务不需要处理。
- 旧仓库 UnbearableFate/LoopLora @ a255e81 只能作只读参考，不能 import，也不要复制进 `src/`。

## 数据

- PIQA、BoolQ 在登录节点下载（公开数据，允许），放在 `HF_HOME=/work/xg24i002/x10041/hf_home` 或项目工作区内被 git 忽略的目录；作业中离线读取（`HF_HUB_OFFLINE=1`，用 datasets 时再设 `HF_DATASETS_OFFLINE=1`）。不要运行数据集仓库里的加载脚本（不用 `trust_remote_code`），直接读取 parquet 或 jsonl 文件。
- 记录每个数据集的来源（repo id 或官方 URL）、revision（commit sha）和所用文件的 sha256。
- 官方 validation 封存：本任务不读取、不评估，也不统计它的任何内容（包括长度分布）。切分只在官方 train 内进行。
- 切分结果（样本 id 到 split 的映射）保存成文件并记录 sha256，E-0003 直接复用。

## 环境

- 沿用 `uv.lock` c073f466bd0e（cuda extra、cuda/12.8、torch 2.10.0+cu128、transformers 4.57.1、Python 3.13.13），不改 torch、transformers 和 CUDA。确需新依赖（例如读 parquet）时，加进同一份 `pyproject.toml`/`uv.lock`，在登录节点用 `uv sync --frozen` 建 `envs/<新 lock id>-cuda`。换环境后，先在作业里重跑 V0 的 function-preservation 与冻结检查，再做校准。
- 硬件：V1 的全部 run（包括之后的 E-0003）用同一种硬件，默认 GH200 MIG 2g.24gb（V0 容差在这上面测得）。显存不够需要换整卡时，先说明理由，并在 `--environment-ref` 中写明。
- attention 保持 eager（V0 只校准了 eager）。

## 计算规则

协调者在登录节点上（`compute.coordinator: login`）。pytest、冒烟检查和所有 Python 工作负载都通过 miyabi-mcp 提交 PBS 作业，不在登录节点运行，也不回退到本地执行。

1. 提交只含源码的 commit，然后 `rl run register --experiment E-0002 --resource miyabi --code-commit <完整 sha> ...`，arm 名用契约表中的名字（如 `cal-lora-1e-4`、`base-eval`）。
2. 按 `rl compute steps --attempt <ATT>` 的输出，把 commit 暂存为只读快照 `/work/xg24i002/x10041/bypass-lora/snapshots/<sha>/`。作业只向 `/work/xg24i002/x10041/bypass-lora/runs/<ATT>/` 写输出，最后写 `manifest.json`。
3. 用 `job_submit` 先 dry run，再带 `plan_digest` 提交。request key 用 `sr:<ATT>`，client_ref 用 attempt id。作业脚本内先运行 allocation check，再跑工作负载。
4. 队列：debug-mig、short-mig 通常启动快。walltime 按实测 × 1.5 申请；普通作业 ≤ 1 h，debug 队列 ≤ 30 min；同时运行的作业 ≤ 4 个。6 个校准 run 可以放在一个作业里顺序运行，也可以分开提交，但每个 run 都要单独登记、记录和导入结果。
5. 几分钟的作业用 `job_wait`（每次 ≤ 45 s，不要循环轮询）。更长的作业提交时带 `notify`，然后 `rl task wait`，在 WAIT 与最终消息里给出返回的 `daemon.listen_command`，由协调者在后台运行。
6. PBS 显示 FINISH 或 Exit_status=0 不代表应用成功，要检查输出。作业事实用 `ledger_query` 的 smooth-research 格式导出，再用 `rl run record --observation @file` 记录；结果用 `rl run import-result` 导入。

## 审核

数据切分、loss 掩码、训练与评估路径、LoRA 注入都是新的核心路径。冒烟检查（几十步、少量样本）可以先跑；6 个校准 run 之前用 `rl review request --target E-0002 ... --author-model <你的模型 id>` 请求代码审核，审核者由协调者安排。修复真实问题后，用 `--recheck-of RV-x` 复核一次。

## 预算与交付

- 预算：E-0002 ≤ 1.2M 训练输入 tokens（6 × 0.2M），另加评估；冒烟检查的训练 tokens 也要记录，合计不超过 0.1M；≤ 5 node·h。超出前用 `rl task block --reason budget` 停下来。
- 证据：每个 run 都要登记、记录并导入结果。校准结果是 diagnostic 证据（`rl evidence add --level diagnostic`），内容是切片检查与学习率选择；它不比较接口，不用来支持或挑战 H-0002 的接口主张。
- release：模板、截断规则、切分文件 sha256、混合方式与顺序、有效 batch、LoRA 缩放与初始化、`derive_seed` 定义、选出的三个学习率。在第一个校准 run 之前写进 `research/notes/`，结果出来后只追加选出的学习率，不改前面的内容。
- E-0003 release 草案（写在 notes 中，由协调者写进 E-0003 契约）：每组的优化步数（2M 非 padding 输入 tokens 与 3 epochs 中先到者）、有效 batch、25%/50%/75% 的 dev 评估点、按实测 × 1.5 估计的每组 walltime 与 node·h。
- 交付摘要包括：Base 在 dev 与 calibration 上的结果；6 个 run 的 calibration NLL、准确率与选出的学习率；切片检查结论；各 run 的 ATT/RES/EV 编号；environment ref（lock id、extra、modules、torch/CUDA/transformers 版本、model 与 tokenizer revision、数据集 revision、硬件）；实际用掉的 node·h 与训练 tokens；审核编号与仍未解决的问题。
