# Constraints

Who decided each item and when. The coordinator and executors follow this; the plugin does not enforce it.
Model choices and compute resources are not kept here: they live in smooth-research.yaml (models, compute),
where the CLI reads them.

来源标记：**[用户 2026-11-01]** 为用户在本项目初始化会话中的决定；**[plan]** 为用户提供并已封存的 `docs/plan.md`（DOC-0001，plan v2，2026-10-29）；**[协调者默认]** 为协调者按计划补全的低风险细节，用户可随时修改。

- Research goal and claims in scope:
  - 研究对象 [plan §0.1, §3]：冻结 Qwen/Qwen3-0.6B-Base 的全部原参数，在原层旁加低维侧分支（原 Q/K/V 特征 → 固定记忆 → 低维循环读取），经原 FFN 的 gate/up 条件融合（concat）或加法接口写回。
  - 范围内主张（都需要证据，不预设成立）[plan §0.1, §17.3]：(1) 低维分支实际读到了什么（寻址、读取混合、解码分开）；(2) 同一低维信息应以哪种接口进入原 FFN；(3) 固定记忆下改变查询的重复读取是否有独立收益（排除幅度、参数、训练量与额外监督解释）；(4) 保留原上下文时的实用价值与成本。
  - M 线（memory-only）与 F 线（context-retained）的结果分开登记、不互相替代 [plan §2.3]。
  - 不在本范围 [plan §12, §17.3]：新颖性或领先性能声明；更大 backbone（1.7B 及以上）、14/28 层布局、长上下文、每轮更新 K/V、旧 SLOT-LOOP-v1 续训（均为 V6 或独立预算）。
  - 当前授权阶段 [用户 2026-11-01]：第一里程碑 V0+V1。V2 及以后的阶段先由协调者给出阶段报告，再由用户决定是否释放预算。
- Allowed data, and which compute resources may be used for what (the resources themselves, how they are
  reached and how code gets there, are in smooth-research.yaml, compute):
  - 模型 [plan §3.1]：Qwen/Qwen3-0.6B-Base，运行时锁定模型与 tokenizer revision；不得用非 Base 配置替代。本地 HF 缓存 `/work/xg24i002/x10041/hf_home` 已有该模型。
  - 数据 [plan §8, §9, §11.2, §13]：
    - 新的合成关系/图任务，`study_id=concat_loop_v2`，按 phase/purpose 派生新 seed，不复用旧验证集。
    - PIQA、BoolQ：从官方 train 切出 train/dev/calibration；官方 validation 封存为确认集，只在配置、训练规则与分析脚本锁定后开封一次。
    - WikiText-103 固定版本的保留 split，只作能力保持指标。
    - 公开数据集与模型可以在登录节点下载到 HF 缓存 [协调者默认]。
  - 旧证据 [plan §18；用户 2026-11-01]：UnbearableFate/LoopLora @ `a255e814740ff85ab3b4ec4836c88f0a8f802b2d`（GitHub 公开仓库），只作只读审计与参考。旧 S-RE checkpoint 不在本文件系统上，按计划 fallback 跳过冻结 S-RE 读算子诊断（plan §6.1–6.2，H-0007）。不得声称旧指标已在本地复现。
  - 计算 [用户 2026-11-01]：只通过 miyabi-mcp 向 Miyabi-G PBS（group xg24i002）提交作业。登录节点只做控制工作：编辑、记录、暂存小型源码快照、依赖解析与下载。测试、冒烟检查、训练、评估与数据分析都放进作业，即使输入很小也不在登录节点运行。
- Resource ceiling (per experiment / per batch / total; units such as CPU-h, GPU-h, node-h; queues):
  - V0+V1 总算力 [用户 2026-11-01]：≤ **200 node·h**（Miyabi-G；MIG 作业按 miyabi-mcp 账本的实际计量计入）。达到 80% 时协调者先报告，超出前必须由用户同意。
  - 训练 tokens [用户 2026-11-01]：取消训练 tokens 上限（原为 V0+V1 ≤ 18M 非 padding 输入 tokens，来自 plan §6.4、§7.3、§14）。tokens 仍按样本呈现次数如实记录，但不再作为资源上限；算力只受上面的 node·h 上限约束。[协调者默认] 实验契约中每个 arm 的 token 数属于实验设计（各 arm 相同的样本预算），不是资源上限，改动按契约修订处理。
  - 每个作业 [plan §14.4；协调者默认，用户 2026-11-01 未改]：单节点；walltime 按实测 × 1.5 申请，普通作业 ≤ 1 h，有实测结果和 scale-up 说明时 ≤ 2 h；debug 队列 ≤ 30 min；同时运行的作业 ≤ 4 个；作业不得跨越 2026-11-25 09:00 JST 的计划停机。
  - 预算核算 [plan §14.4]：每个 arm 实测加载、稳定训练步、评估与保存时间；node·h 只按实测换算。时间不足时统一缩短所有 arm 的步数，不允许各 arm 停在不同的样本预算上。
  - 角色与订阅 [用户 2026-11-01]：研究者优先用 codex，审核者用 Claude；顾问轮次 ≤ `advice.max_rounds`（3）。
- Actions that need the user's decision first:
  - 改变研究目标、范围内主张、backbone、数据集或确认集封存规则。
  - 超出上述 node·h 上限；释放 V2 及以后阶段的预算；推荐正式（确认性）实验；开启 V6 分支。
  - 把代码、数据或结果推送、发布到项目外（GitHub、共享页面等）。
  - 任何不可撤销的项目外操作，例如删除其他目录的数据、修改 miyabi-mcp 的 policy、取消非本项目提交的作业。
