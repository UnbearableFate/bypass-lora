# Review package RV-0005

Independent code review for a research experiment. Question: Verify RV-0004 reload-only nonfinite now remains engineering failure: partial, validator false, failure.stage reload_evaluation, measured usage retained, no run.failed_lr/train.diverged. Initial adapted evaluation failures still failed-LR; Base and finite reload mismatches still errors. ATT-0010-1 RES-0010 EV-0009:106tests passed including real candidate NaN/+Inf during reload after finite initial evaluation;3actual smoke roundtrips passed.

Scope: Only RV-0004 residual medium fix: one-line exception-classification change in src/v1_run.py and added reload-only NaN/+Inf regression in src/test_v1_run.py at0117986. No contract, tolerance, split or metric change. Prior accepted RV-0003/RV-0004 paths unchanged.

This re-checks only what the earlier review raised, after fixes: RV-0004 [review/answered/issues, medium open] Recheck RV-0003 two findings: all candidate losses/sums finite-checked; adapted evaluation nonfinites complete failed-LR with usage retained, Base remains error, checkpoint mismatch unchanged. Strict smoke requires completed steps/finite metrics/exact reload. ATT-0009-1 RES-0009 EV-0008:104tests passed including requested failure regressions, all3actual5step paths passed. Confirm fixes before Base and6calibration runs.

## Method definition

E-0002@1 [E2] V1 任务切片与家族学习率校准 (contract DOC-0019@1: research/experiments/E-0002.md)
  question: PIQA+BoolQ（从官方 train 切出 train/dev/calibration，纯文本固定模板）在 4 层 r32 设置下是否处于可学习且未饱和的区间？LoRA、tokenwise 加法、tokenwise concat 三个家族在 calibration 上各取哪个学习率？
  intervention: Base 只评估不训练；三个家族各两个学习率 {1e-4, 3e-4}，每个 run 0.2M 非 padding 输入 tokens，seed 0，相同数据顺序：LoRA（4 层 [6,13,20,27] all-7 rank10）、tokenwise 加法（proxy Z=SiLU(UA) + F_out）、tokenwise concat（proxy Z=SiLU(UA) + C_dual）
  comparator: Base；同一家族的两个学习率之间
  metric: calibration NLL（选择依据，运行前固定）与准确率；Base 与 LoRA 在 dev 上的 PIQA、BoolQ 准确率（地板/天花板检查）；实测吞吐（tokens/s）、每步时间、加载/评估/保存时间与 node·h
  scope: F 线 context-retained；max length 512；AdamW betas (0.9, 0.999)、eps 1e-8、矩阵 weight decay 0.01；warmup 占实际优化步数 5% 后 cosine；global-norm clip 1.0 并记录 pre-clip 范数；dropout 0；PAD 与 EOS 同 ID 时按实际 padding 位置屏蔽；模板、标签 token、候选打分与多 token 标签归一化规则在本实验内固定并写入 release；官方 validation 不开封
  outcome_actions: Base 与 LoRA 显示可学习且未饱和 -> 锁定每个家族的学习率，按实测吞吐生成 V1 八组 release（步数、batch、walltime）；全部近随机 -> 先查实现与任务可学性；全部近天花板 -> 按预先规则对所有 arm 换同一个非天花板设置，不据此排名
  budget: ≤ 1.2M 训练输入 tokens（3 家族 × 2 学习率 × 0.2M）加评估；≤ 5 node·h
  stop: 6 个校准 run 与 Base 评估完成即止；不追加候选学习率（额外调参属于储备，须记录理由）

## Files under review

Read them where they are, in /work/xg24i002/x10041/bypass-lora-worktrees/researcher-1 (the requester's worktree); the package does not copy them. Cite findings by path and line.

- /work/xg24i002/x10041/bypass-lora-worktrees/researcher-1/src/test_v1_run.py
- /work/xg24i002/x10041/bypass-lora-worktrees/researcher-1/src/v1_run.py

## What to return

Review the code against the method definition. For each problem give: location (file:line), the evidence
that reproduces or demonstrates it, its impact (result_error | invalid_experiment | resource_waste |
nonblocking), its severity, the minimal fix (a suggested patch is welcome; you do not edit files yourself),
and how to verify the fix. Severity: high = the result would be wrong or the experiment invalid; medium =
likely to distort the result or waste much of the budget, to be fixed or explicitly accepted; low = worth
fixing, does not change the conclusion. State what you checked and what you did not. Do not restate the
package. End with exactly one ```json block:

```json
{"verdict": "pass | issues | blocking",
 "findings": [{"location": "", "evidence": "", "impact": "", "severity": "high | medium | low", "fix": "",
               "verify": ""}],
 "checked": [""], "unchecked": [""]}
```
