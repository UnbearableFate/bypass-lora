# T-0001：V0 侧分支实现与不变量检查（E-0001）

## 目标

在 `src/` 实现新的低维侧分支，使 E-0001 的全部检查在 Miyabi-G 作业中通过，并在交付前取得代码审核。交付的代码应能直接用于 V1 校准（E-0002）。

## 规格

- 契约：`research/experiments/E-0001.md`，包括 Design 段的检查表、容差规则与负对照。
- 公式与数据流来自封存计划 `docs/plan.md`：§3.1–3.6（Qwen3 结构、QKNorm 后 RoPE 前的 tap、正核状态、累加循环、concat 融合）、§4.1 与 §4.5（恒等式与退化关系）、§6.3（检查表）、§16.1–16.3（模块边界、因果读取伪代码、原块数据流）。伪代码只说明算法，训练实现要用可求导的函数式写法。
- 范围：实现 V1 需要的部分（F 线四层 tokenwise proxy Z=SiLU(UA) 与线性 Z=UA，融合接口 F_out、F_pre、F_attnres、C_gate、C_up、C_dual），以及 V2 需要的部分（M 线单层 index14，KernelMatrix、KernelSlot 参考实现、SoftmaxSlot，support write-only 与 query read-only 协议）。LoRA 基线、任务数据与训练循环属于后续任务，本任务不做。
- 旧仓库 UnbearableFate/LoopLora @ a255e81 只能作只读参考，不能 import，也不要复制进 `src/`。

## 环境

- 项目使用 Python 3.13（`.python-version`）。所有资源共用一份 `pyproject.toml` 与 `uv.lock`，并保留 rl 的依赖（latticedb、pyyaml）。做法见 smooth-research skill 的 `references/workflow.md` 中的 "Python environments" 一节：GPU 节点用 cuda extra，所有变体钉同一个 torch 版本，并写 `required-environments` 覆盖 aarch64。
- 先在一个小作业里查清 GPU 型号、驱动与 CUDA（`nvidia-smi`、`module avail cuda`），再选 torch 的 CUDA index。transformers 要选支持 Qwen3 的版本（≥ 4.51），并记录精确版本。
- 作业用的环境放在 `/work/xg24i002/x10041/bypass-lora/envs/<lock id>-<extra>`。先确认计算节点能否联网；不能联网时，在登录节点用 `uv sync --frozen` 下载预编译 wheel 建环境。登录节点上不做源码编译，也不 import 应用代码。
- 模型：`HF_HOME=/work/xg24i002/x10041/hf_home` 中已有 Qwen/Qwen3-0.6B-Base。从缓存的 refs 锁定 model 与 tokenizer revision，作业中设 `HF_HUB_OFFLINE=1`。

## 计算规则

协调者在登录节点上（`compute.coordinator: login`）。pytest、冒烟检查和所有 Python 工作负载都通过 miyabi-mcp 提交 PBS 作业，不在登录节点运行，也不回退到本地执行。

1. 提交只含源码的 commit，然后 `rl run register --experiment E-0001 --resource miyabi --code-commit <完整 sha> ...`。
2. 按 `rl compute steps --attempt <ATT>` 的输出，把这个 commit 暂存为只读快照 `/work/xg24i002/x10041/bypass-lora/snapshots/<sha>/`。作业只向 `/work/xg24i002/x10041/bypass-lora/runs/<ATT>/` 写输出，最后写 `manifest.json`。
3. 用 `job_submit` 先 dry run，再带 `plan_digest` 提交。request key 用 `sr:<ATT>`，client_ref 用 attempt id。作业脚本内先运行 allocation check，再跑工作负载。
4. 队列：debug-g、short-g 经常满载；debug-mig、short-mig（GH200 MIG 2g.24gb，24 GB）通常启动快，足够跑 0.6B 的检查。在 `--environment-ref` 中写明硬件（MIG profile 或整卡），BF16 容差必须在同一种硬件上测得。
5. 几分钟的作业用 `job_wait`（每次 ≤ 45 s，不要循环轮询）。更长的作业提交时带 `notify`，然后 `rl task wait`，在 WAIT 与最终消息里给出返回的 `daemon.listen_command`，由协调者在后台运行。
6. PBS 显示 FINISH 或 Exit_status=0 不代表应用成功，要检查输出。作业事实用 `ledger_query` 的 smooth-research 格式导出，再用 `rl run record --observation @file` 记录；结果用 `rl run import-result` 导入。

## 审核

memory backend、融合与 adapter 注入属于新的核心路径。全部检查通过后、交付前，用 `rl review request --target E-0001 ... --author-model <你的模型 id>` 请求代码审核，审核者是 Claude。修复真实问题后，用 `--recheck-of RV-x` 复核一次。

## 预算与交付

- 预算：E-0001 ≤ 3 node·h，不消耗训练 tokens。超出前用 `rl task block --reason budget` 停下来。
- 证据：每个检查 run 都要登记、记录并导入结果；`rl evidence add --level diagnostic` 支持或挑战 H-0001。
- 交付摘要包括：各项检查结果及其 ATT/RES/EV 编号；environment ref（lock id、extra、modules、torch/CUDA/transformers 版本、model 与 tokenizer revision、硬件）；实际用掉的 node·h；审核编号与仍未解决的问题。
