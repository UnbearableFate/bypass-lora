# 实验产物清理（artifact cleanup）

- 工具：`src/artifact_cleanup.py`，测试 `src/test_artifact_cleanup.py`（47 项，PBS 中通过）。默认只做 dry run；apply 只执行审阅过的计划，并且必须带确认参数。
- 用户规则（2026-11-02）：每个假说的相关任务完成后，清理以后用不到的大文件（checkpoint）和日志等。由协调者审阅计划后执行 apply。
- 本文数字来自 2026-11-02 的 dry run。records 的 data_version 是 2743，由 `rl --json show` 逐条读出后重建，不是 `rl export`。计划在 `/work/xg24i002/x10041/bypass-lora-maintenance/artifact-cleanup/20261102-56aea1d/plans/`，不在 git 中。正式清理前要用 `rl export` 重新出计划。
- 大小有两种。表观大小是文件长度。释放量是删除后真正腾出的空间：envs 里的文件大多是 uv cache 的硬链接，删掉也不腾空间。

## 1. 盘点与配额

配额（`lfs quota`，2026-11-02）如下：
- 用户 x10041 在 /work 上用了 1.006 TB，没有用户上限。
- 组 xg24i002 用了 14 TB，上限 27 TB。

本项目各区合计：表观大小 25.4 GB，自有数据约 12.3 GB。/work 上其余用量在项目之外（hf_home、uv cache、其他项目），不在本工具范围内。

| 区域 | 表观 | 候选（现在可删） | hold（等事件） | keep |
|---|---|---|---|---|
| runs/ | 11.804 GB | 10.260 GB | 0.340 GB | 1.204 GB |
| snapshots/ | 0.354 GB | 0.120 GB（53 个） | 0.180 GB（39 个） | 0.054 GB（24 h 内暂存的 8 个） |
| envs/ | 13.046 GB（自有 4 MB） | c073f466bd0e：只释放 2 MB | 3c916b51337d（所有 checkout 的 HEAD） | – |
| research/、data/、.miyabi-mcp/、.venv、worktrees | 0.05 / 0.02 / 0.002 / 0.07 / 0.08 GB | – | – | 全部保留 |

runs/ 按类别分：
- 候选：feature-cache 4.970 GB（6 个），regression-dump 4.423 GB（17 个），test-temp 0.428 GB（275 个），checkpoint 0.420 GB（142 个），trace 0.018 GB（8 个）。
- hold：checkpoint 0.189 GB，test-temp 0.110 GB，trace 0.041 GB。
- keep：predictions 0.673 GB，data-copy 0.272 GB，metadata 0.258 GB。

## 2. 规则

每个路径得到三种决定之一：keep（本工具永不删除）、hold（可删类型，但仍被需要，并写明解除事件）、候选（可删类型，而且没有任何开放工作需要它）。

**可删类型**（只在 runs/、snapshots/、envs/ 下）：
- C1 checkpoint、C2 optimizer、C3 intermediate-checkpoint（step-/epoch- 目录下的张量）。attempt 已结束；manifest、metrics、predictions 保留，需要时可按记录的 spec 重训。
- C4 regression-dump（regression/ 下的 old/new 张量）：回归检查的结论文件保留。
- C5 feature-cache：缓存的特征，可以重新抽取；分析报告保留。
- C6 test-temp：复制进 run 目录的 pytest 临时目录 `tmp/test_*`。
- C7 trace：不小于 1 MiB 的训练轨迹（首行带 `step` 的 jsonl）和 .log、.out、.err。摘要在 manifest 和 metrics 里。
- C8 cache：run 目录里的 `__pycache__`、`.pytest_cache`。
- C9 snapshot：已结束 attempt 的只读代码快照，commit 仍在 git 里。commit 缺失或 git 不可读时按 K13 保留。
- C10 environment：没有任何 checkout 的 `HEAD:uv.lock` 用到，也没有 hold 住的 attempt 用到。`uv sync --frozen` 可以重建。

**keep**（K1–K15）：
- 记录与账目：research/、.miyabi-mcp 账目、data/、各 worktree、.venv。
- run 目录里的小文件：metadata（manifest、metrics、config、report、小日志）、predictions（结果和证据由它算出）、data-copy（后续 run 以产生它的 attempt 为数据源）。
- 需要人工看的：unclassified（其他张量）、unregistered（记录里没有的 run 目录）。
- 链接（link）。
- 快照的两种保留：recent-snapshot（24 h 内暂存）和 snapshot-no-commit。
- K15 evidence：evidence 记录（artifact_refs 或摘要）点名的路径。

**hold**，以及它的解除事件：
- 未结束的 attempt，以及它的 run spec 读到的东西 → `ATT-x ended`。
- 开放实验的东西 → `E-x closed`。开放实验是指：有任务未 accepted、有 attempt 未结束，或者还没有任务。会被 hold 的有：
  - 它的 attempt、契约、run spec 和结果所点名的东西；
  - 它 depends_on、code_from 的实验的全部 attempt。
- 已关闭的实验如果仍在检验 active 或 candidate 的假说，它的契约所点名的 attempt 和快照 → `H-x paused or stopped`。
- 开放任务的东西 → `T-x accepted`。包括任务记录、brief、任务点名的研究文档，以及主 checkout 和每个研究者 worktree 里 `research/artifacts/T-x/` 下的 lock。
- lock 的内容会递归读取。
- decision 的 reopen condition 点名的东西 → 该条件不再适用时解除。
- 被 hold 的 attempt 的起点 checkpoint（resume_from，逐级传递）→ 同一事件。

**范围门槛**：范围内只要有任务未 accepted，或有 attempt 未结束，整份计划就是 REFUSED：没有候选，apply 也拒绝执行。stopped 任务同样会挡住（协调者的规则）。

结果的 manifest 列出的 checkpoint 仍然是候选，在 summary 里单独列出（见第 7 节第 1 条）。`rl doctor` 会跳过已删除的文件，结果记录里的 digest 仍然保留。

## 3. 可回收空间（data_version 2743）

- **现在可删：10.382 GB**（502 项）。其中最大的三类：
  - E-0012 的特征缓存 4.970 GB；
  - E-0014..E-0017 的回归张量 4.423 GB；
  - checkpoint 和 pytest 临时目录各约 0.4 GB。
- **等事件后可删：0.522 GB**。下表只算某个事件发生后不再被别的来源 hold 的部分：

  | 事件 | 可释放 |
  |---|---|
  | E-0022 关闭（T-0023 accepted） | 0.140 GB |
  | E-0013 关闭（T-0020 accepted，ATT-0179-1 结束） | 0.051 GB；连同其他事件共 0.143 GB |
  | H-0005 paused 或 stopped | 0.021 GB |
  | T-0020 accepted、E-0019 关闭 | 各 0.008 GB |
  | D-0021 的 reopen condition 不再适用 | 0.007 GB |
  | H-0003 或 H-0009 paused 或 stopped | 各 0.006 GB |

- **始终保留：1.489 GB**（predictions、data-copy、metadata、research 等）。

## 4. 每个假说完成后的清理流程

1. **触发条件**：某个假说 H-x 的全部任务都已 accepted，全部 attempt 都已结束。也可以按单个实验 E-x 做。
2. **导出记录**：在登录节点运行 `rl export --out <dir>`，得到一份一致的记录快照。
3. **出计划**：在 PBS 中运行 `plan --scope H-x`，代码从只读快照 `snapshots/<commit>/src` 读取。作业脚本可以照 `/work/xg24i002/x10041/bypass-lora-maintenance/artifact-cleanup/20261102-56aea1d/job.sh` 写：先做 allocation check，并设 `TMPDIR=/local/$PBS_JOBID`。用 debug-mig 一个 MIG 实例，大约 2 分钟。
4. **审阅** summary.md：
   - gate 必须是 closed；
   - warnings 会列出仍然 active 的假说；
   - 看 Holds by release event、被结果引用的候选，以及 Large files kept。
   - 要保留某个候选，就把它的条目从 plan.json 中删掉，另存为新文件（例如 `plan.reviewed.json`），再用新文件的 sha256 作确认值。apply 只删计划里列出的候选。
5. **执行**：再运行一次 `rl export`，然后在 PBS 中运行 apply，由协调者执行。
6. **检查**：看计划目录里的 `deletions.jsonl`（apply-start、deleted、apply-end），再运行 `rl doctor`。

已关闭实验的 checkpoint，如果后续工作还要用，就由第 2 节的 hold 规则保护。例如后续契约点名的起点、数据源，或者 reopen condition 点名的 checkpoint。

## 5. 旧范围（2026-11-02 的计划）

| 范围（假说） | 删除 | 释放 | 保留 | 仍要保留的文件 → 解除事件 |
|---|---|---|---|---|
| V1：E-0002+E-0003（H-0002 paused），T-0002、T-0003 accepted | adapter 34 个 0.095 GB；pytest 临时目录 111 个 0.204 GB；steps.jsonl 8 个（ATT-0025-1..0032-1，每个 2.3 MB）0.018 GB；快照 12 个 0.006 GB | 0.324 GB | metadata 0.029 GB，predictions 0.012 GB | 快照 1fe7d3d：它是 ATT-0024-1 的代码，E-0021（RB1）契约沿用 ATT-0024-1 的 Base 评估 → E-0021 关闭。D-0004 的 reopen 是用多个 seed 复核，需要新的训练，不需要旧 adapter |
| V2：E-0009+E-0010（H-0003 active，H-0004 paused），T-0008、T-0011 accepted | adapter 29 个 0.099 GB（都被结果 manifest 引用）；临时目录 20 个 0.088 GB；快照 2 个 | 0.189 GB | predictions 0.161 GB，metadata 0.048 GB | ATT-0082-1 的 F_out adapter 和快照 9145309：E-0014、E-0015、E-0016 的契约点名 → H-0009 和 H-0012 都 paused 或 stopped。快照 631f87e、d8cc3d4：E-0010 契约点名 → H-0003 paused 或 stopped |
| E-0012（H-0003），T-0012 accepted | 特征缓存 6 个 4.970 GB（ATT-0096-1 的 train 一项就有 3.970 GB）；快照 4 个 | 4.983 GB | metadata 0.001 GB | 无。D-0008 的 reopen 要新的探针运行，没有点名这些缓存 |
| E-0014（H-0009），T-0015 accepted | 回归张量 3 个 0.643 GB；adapter 10 个 0.023 GB；临时目录 16 个；快照 1 个 | 0.707 GB | predictions 0.066 GB；data-copy 0.105 GB（ATT-0112-1 的 e1/e4/e8 数据）；metadata 0.019 GB | ATT-0112-1：数据源，E-0015..E-0017 契约点名。ATT-0123-1、ATT-0126-1：E-0015 契约点名。快照 c6037d3、f9eba25。这些 → H-0009（和 H-0012）paused 或 stopped。ATT-0122-1（stage-A F_out）另外是 D5 ATT-0159-1 的起点 → 还要等 T-0023 accepted、H-0005 paused 或 stopped |
| E-0015（H-0009），T-0016 accepted | 回归张量 4 个 0.858 GB；临时目录 11 个；快照 4 个 0.020 GB | 0.888 GB | predictions 0.207 GB | 无 |
| E-0016（H-0010、H-0011 paused，H-0012 active），T-0017、T-0018 accepted | 回归张量 2 个 0.584 GB；adapter 16 个 0.036 GB（其中 14 个被结果引用）；临时目录 21 个；快照 3 个 | 0.647 GB | predictions 0.093 GB，metadata 0.013 GB | ATT-0154-1 的 D1 adapter 和快照 6da33b3：D-0021 reopen 的 (a) 点名 → 该条件不再适用。ATT-0159-1 的 D5 adapter 和快照 415cac3：E-0018、E-0022 契约和 T-0023 点名 → T-0023 accepted、E-0022 关闭、H-0005 paused 或 stopped |
| **以上合计** | 317 项，互不重叠 | **7.738 GB** | | |
| 另：E-0017（H-0009、H-0012），T-0019 accepted | 回归张量 8 个 2.338 GB；adapter 21 个；临时目录 44 个；快照 1 个 | 2.423 GB | predictions 0.108 GB | ATT-0166-1、ATT-0167-1：ATT-0172-1、ATT-0173-1 的起点。ATT-0172-1、ATT-0173-1：E-0018 契约点名。这些 → H-0005 paused 或 stopped |
| 另：E-0020（RBc，H-0013），T-0024 accepted | 尚未出计划 | – | | lr-lock（research/artifacts/T-0024/lr-lock.json）登记的 16 个 calibration manifest：RB1 的分析、干预与延迟脚本会重新校验它们的 sha256（RV-0069）→ T-0025 accepted、E-0021 分析完成。manifest 本来属于 keep，这里只是写明依赖 |
| 另：H-0005（E-0011、E-0018、E-0022、E-0023），T-0010、T-0021、T-0023、T-0026 accepted，H-0005@2 paused（D-0036） | pytest 临时目录 435 个 0.424 GB；checkpoint 27 个 0.063 GB；快照 22 个 0.182 GB | 0.669 GB | data-copy 0.093 GB，metadata 0.128 GB，predictions 0.028 GB | 被结果引用的 8 个 checkpoint 从计划中删去：共同起点 ATT-0196-1（重开时可复用）、阶段 A 的 ATT-0233/0234/0235-1、pilot 的 ATT-0256/0257-1、E-0011 的 ATT-0091/0092-1。计划目录 20261102-h0005，确认值见 confirm.txt，等用户同意后执行 |
| 另：E-0013（V5，H-0008） | 无：REFUSED（T-0020 waiting，ATT-0179-1 queued） | 0 | | 0.132 GB 一直 hold 到 T-0020 accepted、ATT-0179-1 结束 |

按假说出计划的例子：
- H-0002 与 V1 相同。
- H-0003（E-0005..E-0010、E-0012）可释放 5.307 GB。
- H-0009（E-0014、E-0015、E-0017）可释放 4.018 GB。
- H-0012（E-0016、E-0017）可释放 3.071 GB。
- H-0005 是 REFUSED，因为 T-0023 active。

## 6. 运行方法

```
rl export --out <dir>                                   # 登录节点，只读
python src/artifact_cleanup.py plan --root /work/xg24i002/x10041/bypass-lora \
    --records <dir>/snapshot.json --scope H-0002 --out <新目录>          # PBS
python src/artifact_cleanup.py apply --plan <新目录>/plan.json \
    --records <新 export>/snapshot.json --confirm <summary.md 里的 sha256>   # PBS，审阅后
```

- dry run 不改动磁盘，测试比较了运行前后的整棵树。plan.json 永远不会被覆盖。
- apply 会先用新的记录和当前磁盘重新计划同一个范围。出现下列任一情况就全部拒绝，什么都不删：
  - 确认值不符；
  - 记录比计划旧；
  - 范围重新打开；
  - 某个候选不在 runs/、snapshots/、envs/ 下（解析链接后判断）；
  - 某个候选已消失；
  - 某个候选变成了 hold 或 keep；
  - 某个候选的类别、大小、stat 或内容 sha256 变了。
- 删除过程中某个候选消失，或某个文件的 stat 变了，apply 立即停止。每一步都写入 `deletions.jsonl`。

## 7. 待决定和未解决的问题

1. **结果引用的 adapter**：57 个候选（0.173 GB）列在 Result manifest 的 artifacts 里。V2 有 29 个，E-0016 有 14 个，E-0008 有 6 个，E-0014 有 4 个，E-0011 和 E-0017 各 2 个。按“清理 checkpoint”的规则它们可以删；要保留，就按第 4 节第 4 步把它们从计划中去掉。请决定。
2. **E-0012 的特征缓存**（4.970 GB）是最大的一项。H-0003 仍是 active/challenged。删除后，可以用冻结模型和 git 中的代码重新抽取，但要花 GPU 时间。
3. **stopped 任务会永久挡住所在范围**。现在还没有 stopped 任务；出现时需要另定规则。
4. **轨迹阈值是 1 MiB**。V1 的 8 个 steps.jsonl 是候选；V5（E-0013）的 steps.jsonl 被结果引用，目前处于 hold。
5. **predictions（0.67 GB）和 data-copy（0.27 GB）一律保留**。
6. **不在工具区域内的东西**：
   - `bypass-lora-worktrees/` 下的 plan2-sandbox（36 MB）和 plan2-snapshot（9 MB）是记录沙盒；
   - hf_home 和 uv cache。
   这些需要人工决定。
7. **本次计划的 records 是重建的**，因为本会话不能用 `rl export`。收集期间 data_version 从 2743 变成了 2744。apply 前必须用 `rl export` 重新出计划；apply 本身也会按新的 export 重新核对每个候选。
8. **维护快照**：b09639b、1d0070f、56aea1d、8e9c646 是本次维护作业的快照，没有 attempt 引用，24 h 后会变成候选。如果这些 commit 以后从仓库中消失（例如 squash 合并后被 gc），它们就按 K13 保留，需要人工删除。
9. **compared_with 不算 hold**。E-0021 compared_with E-0003 用的是 V1 的切分与训练约定、保留下来的结果和证据；它要沿用的 Base 评估 ATT-0024-1 已由 E-0021 契约点名。code_from 和 depends_on 会 hold 被依赖实验的全部 attempt，例如 E-0019 code_from E-0004 会 hold ATT-0034-1。
10. **不要直接 apply 全项目的 "all" 计划**。"all" 计划等于所有已关闭实验的范围，加上未用的快照和环境。它会把 E-0011、E-0018 的文件也列为候选，而它们属于 H-0005，H-0005 还有 T-0023 未完成。按假说规则，应当逐个范围执行。

## 8. 执行记录（2026-11-02）

用户在 2026-11-02 下午回复“同意清理”，执行第 5 节里等待同意的三份计划。执行前核对了每个范围 confirm.txt 的值等于 plan.reviewed.json 的 sha256，并为三个计划目录各做了一次新的 `rl export`（data_version 4058，放在各自的 records-apply/）。c5b961b 里 03:24 的旧导出改名为 records-apply.stale-0324，没有删除。

PBS 作业 3774200（debug-mig，脚本 `/work/xg24i002/x10041/bypass-lora-maintenance/artifact-cleanup/apply-20261102-all.sh`，用时 1 分 11 秒，0.0197 MIG-h）依次运行三个目录的 apply-job.sh，全部 exit=0：

| 计划目录 | 范围 | 删除项 | 释放 |
|---|---|---:|---:|
| 20261102-c5b961b | E-0002+E-0003 | 165 | 0.324 GB |
| | E-0009+E-0010 | 22 | 0.089 GB |
| | E-0012 | 10 | 4.983 GB |
| | E-0014 | 26 | 0.697 GB |
| | E-0015 | 19 | 0.888 GB |
| | E-0016 | 28 | 0.616 GB |
| | E-0017 | 72 | 2.419 GB |
| 20261102-e0013 | E-0013 | 53 | 0.068 GB |
| 20261102-h0005 | H-0005 | 484 | 0.669 GB |
| 合计 | | 879 | 10.753 GB |

每个范围的 deletions.jsonl 都有一条 apply-start、逐项 deleted 与一条 apply-end。之后 `rl doctor` 检查了 322 个封存对象，另报两个未登记对象（6515b817…、8569bc2c…，提示为中断的封存，无害），与这次删除无关。

## 9. 执行记录（2026-11-02 晚）：H-0018、H-0019、H-0020

触发：E-0027 交付后 T-0030 accepted；H-0020 记为 inconclusive 并暂停（D-0062，结论审核 RV-0089 pass），H-0019 暂停（D-0063）；H-0018 此前已记为 challenged 并暂停（D-0056）。计划目录 `/work/xg24i002/x10041/bypass-lora-maintenance/artifact-cleanup/20261102-h0018-h0020`，工具取自快照 c5b961b（artifact_cleanup.py 此后没有改动）。

- 计划：`rl export`（data_version 4362）后，PBS 作业 3775595（debug-mig，10 秒，0.0028 MIG-h）出三个范围的计划。
  - H-0018（E-0026）：没有候选。9 个 checkpoint、3 个轨迹与 5 个快照（共 0.176 GB）由 E-0028 hold，因为 E-0028 code_from E-0026。E-0028 关闭后再出计划。
  - H-0019（E-0025）：gate closed，候选 25 项 0.155 GB（checkpoint 14 个 0.087 GB，steps.jsonl 7 个 0.016 GB，快照 4 个 0.052 GB），没有 hold。
  - H-0020（E-0027）：gate closed，候选 22 项 0.154 GB（checkpoint 12 个 0.107 GB，steps.jsonl 8 个 0.018 GB，快照 2 个 0.029 GB），没有 hold。
- 审阅：两份计划都没有 warning。被结果引用的候选只有 steps.jsonl（训练轨迹，每个约 2 MB），摘要已在 manifest 与 metrics 里。D-0056、D-0062、D-0063 的重开条件都要求新的训练，没有点名这些 checkpoint，所以按原计划执行，没有删减条目。
- 执行：再做一次 `rl export`（records-apply/，data_version 4363），PBS 作业 3775615（debug-mig）按 confirm.txt 的 plan.json sha256 执行 apply，两个范围都 exit=0：H-0019 删除 25 项 0.155 GB，H-0020 删除 22 项 0.154 GB。每个范围的 deletions.jsonl 都有 apply-start、逐项 deleted 与 apply-end。之后 `rl doctor` 检查了 335 个封存对象，只报此前已知的两个未登记对象。

## 10. 执行记录（2026-11-02 夜）：H-0018

触发：E-0028 交付后 T-0031 accepted，第 9 节里 E-0028 对 E-0026 的 hold 随之解除。计划目录 `/work/xg24i002/x10041/bypass-lora-maintenance/artifact-cleanup/20261102-h0018-post-e0028`，工具仍取自快照 c5b961b。

- 计划：`rl export`（data_version 4543）后，PBS 作业 3777631（debug-mig，6 秒，0.0017 MIG-h）出计划。H-0018（E-0026，T-0029 accepted）gate closed，候选 17 项 0.176 GB（checkpoint 9 个 0.100 GB，steps.jsonl 3 个 0.007 GB，快照 5 个 0.070 GB），没有 hold，保留 160 项 0.012 GB（manifest、metrics、predictions 等）。
- 审阅：计划没有 warning。仍开着的 E-0029 code_from E-0028，它的契约、源码快照 f46a13d 与任务笔记都没有引用 E-0026 的 attempt 或 checkpoint；`src/test_seg_seeds.py` 里出现的 ATT-0300-1 只是 tmp_path 下的合成 fixture 名。被结果引用的候选只有 3 个 steps.jsonl（RES-0270 至 RES-0272），摘要已在 manifest 与 metrics 里。D-0056 的重开条件要求新的训练，没有点名这些 checkpoint；H-0029 候选的 A2 确认也按新 seed 重新训练。因此按原计划执行，没有删减条目。
- 执行：再做一次 `rl export`（records-apply/，data_version 4549），PBS 作业 3777667（debug-mig）按 confirm.txt 的 plan.json sha256（8266c755...）执行 apply，exit=0，删除 17 项 0.176 GB；deletions.jsonl 有 apply-start、逐项 deleted 与 apply-end。之后 `rl doctor` 检查了 347 个封存对象，只报此前已知的两个未登记对象。
- 待办：E-0028 的 checkpoint 等 E-0029 关闭后，再按 H-0021、H-0022 的范围出计划，因为 E-0029 code_from E-0028，且它的 X-01 比较绑定 E-0028 的参考 checkpoint。

## 11. 执行记录（2026-11-03 凌晨）：H-0021 至 H-0026

触发：T-0031（E-0028）与 T-0032（E-0029）都已 accepted，H-0021、H-0022 已暂停（D-0066、D-0067），H-0023 至 H-0026 也已暂停（D-0068 至 D-0071）。第 10 节的待办（E-0029 关闭后再清 E-0028）随之完成。计划目录 `/work/xg24i002/x10041/bypass-lora-maintenance/artifact-cleanup/20261103-e0028-e0029`，工具仍取自快照 c5b961b。

- 计划：`rl export`（data_version 4810）后，PBS 作业 3779649（debug-mig，10 秒）按两个范围出计划，gate 都是 closed，没有 hold，也没有 warning。
  - H-0021、H-0022（E-0028，21 个 attempt）：候选 37 项 0.106 GB（checkpoint 21 个 0.072 GB，steps.jsonl 13 个 0.029 GB，快照 3 个 0.004 GB），保留 407 项 0.057 GB。
  - H-0023 至 H-0026（E-0029，36 个 attempt）：候选 55 项 0.247 GB（checkpoint 41 个 0.138 GB，steps.jsonl 8 个 0.018 GB，源码快照 6 个 0.091 GB），保留 649 项 0.036 GB。
- 审阅：被结果引用的候选只有 steps.jsonl（RES-0303 至 RES-0316、RES-0337 至 RES-0344），摘要已在 manifest 与 metrics 里，与第 10 节的处理一致。manifest、metrics、predictions 与两份锁定分析都保留。D-0066 至 D-0071 的重开条件都要求新的封存契约与新的训练，没有点名这些 checkpoint；H-0029 的 A2 确认用新 seed 重新训练，也不用它们。因此按原计划执行，没有删减条目。
- 执行：再做一次 `rl export`（records-apply/，data_version 4811），PBS 作业 3779701（debug-mig，11 秒）按 confirm.txt 的两个 plan.json sha256（406084c3...、a9ce9bfe...）执行 apply，都 exit=0，共删除 92 项 0.353 GB；两份 deletions.jsonl 都有 apply-start、逐项 deleted 与 apply-end。之后 `rl doctor` 检查了 357 个封存对象，只报此前已知的两个未登记对象。

## 12. 执行记录（2026-11-03 上午）：H-0029

触发：E-0030 交付后 T-0033 accepted；H-0029 记为 inconclusive 并暂停（D-0074，结论审核 RV-0101 pass），分段线的阶段交付记为 pause（D-0075）。计划目录 `/work/xg24i002/x10041/bypass-lora-maintenance/artifact-cleanup/20261103-h0029`，工具仍取自快照 c5b961b（artifact_cleanup.py 此后没有改动）。

- 计划：`rl export`（data_version 5257）后，PBS 作业 3783356（debug-mig，8 秒）出计划。H-0029（E-0030，41 个 attempt，T-0033 accepted）gate closed，候选 7 项 0.121 GB，全部是已结束 attempt 的源码快照（e7b1c6c、967f551、61cd97b、909b5f0、f741952、3ae1974、4e6e8df），commit 都在 master 中；没有 hold，也没有 warning。
- 审阅：E-0030 的 12 个 checkpoint（adapter，共 0.107 GB）与全部 manifest、predictions、锁定文件和锁定分析都在 research/artifacts/ 下，按 K1 不在本工具范围内。锁定文件记录了这些 checkpoint 的哈希，它们是确认性结果 EV-0174 的来源，体积也小，所以保留，不另行删除。D-0074 的重开条件要求新的封存确认集与新 seed 重训，不需要它们。按原计划执行，没有删减条目。
- 执行：再做一次 `rl export`（records-apply/，data_version 5258），PBS 作业 3783386（debug-mig）按 confirm.txt 的 plan.json sha256（4c229fee...）执行 apply，exit=0，删除 7 项 0.121 GB；deletions.jsonl 有 apply-start、逐项 deleted 与 apply-end。之后 `rl doctor` 检查了 377 个封存对象，只报此前已知的两个未登记对象。

## 13. 执行记录（2026-11-03 晚）：H-0030 至 H-0033

触发：T-0034（E-0031）与 T-0035（E-0032）都已 accepted。H-0030、H-0031 记为 inconclusive 并 stop（D-0079，结论审核 RV-0111、RV-0112），普通设置下 RB1-09 改进线的阶段交付记为 stop（D-0080）；H-0032、H-0033 记为 challenged 并 stop（D-0078，结论审核 RV-0110 pass）。计划目录 `/work/xg24i002/x10041/bypass-lora-maintenance/artifact-cleanup/20261103-h0030-h0033`，工具仍取自快照 c5b961b（artifact_cleanup.py 此后没有改动）。

- 计划：`rl export`（data_version 5764）后，PBS 作业 3789060（debug-mig，13 秒）按两个范围出计划，gate 都是 closed，没有 hold，也没有 warning。
  - H-0030、H-0031（E-0031，21 个 attempt）：候选 15 项 0.090 GB（smoke 与正式训练的 adapter 共 10 个 0.025 GB，steps.jsonl 2 个 0.005 GB，源码快照 3 个 0.060 GB），保留 283 项 0.022 GB（metadata 与 predictions）。
  - H-0032、H-0033（E-0032，14 个 attempt）：候选 6 项 0.121 GB，全部是源码快照。
- 审阅：
  - 9 个源码快照（6f85da7、aedd043、49f7009；640bac7、06d191f、322bee3、b0e7a8f、9aa4d79、5b0fad0）的 commit 都在 master 中，逐个用 `git merge-base --is-ancestor` 核对过。
  - 被结果引用的候选只有 2 个 steps.jsonl（RES-0390、RES-0391），摘要已在 manifest 与 metrics 里，与第 10、11 节的处理一致。
  - 正式训练的 H-QVU（ATT-0409-3）与 T0-QVU（ATT-0410-2）adapter 各约 3 MB。D-0079、D-0080 的重开条件要求新的决定与新的训练，没有点名它们；比较用的逐行预测、干预结果与锁定分析 ATT-0424-1 都保留。E-0031 复评的 E-0021 RB1 checkpoint 不属于这两个范围，没有变动。
  - E-0032 的输出（含 smoke 与 5 条训练的 adapter，共 0.185 GB）都在 research/artifacts/ 下，按 K1 不在本工具范围内。锁定分析 ATT-0423-1 的 provenance 绑定了每条训练的 manifest、predictions 与 checkpoint.json，体积也小，所以与第 12 节的 E-0030 一样保留，不另行删除。
  - 按原计划执行，没有删减条目。
- 执行：再做一次 `rl export`（records-apply/，data_version 5765），PBS 作业 3789126（debug-mig，9 秒）按 confirm.txt 的两个 plan.json sha256（c498c8ef...、9bdca33b...）执行 apply，都 exit=0：H-0030、H-0031 删除 15 项 0.090 GB，H-0032、H-0033 删除 6 项 0.121 GB，共 21 项 0.211 GB。两份 deletions.jsonl 都有 apply-start、逐项 deleted 与 apply-end。之后 `rl doctor` 检查了 415 个封存对象，只报此前已知的两个未登记对象。
