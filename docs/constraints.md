# Constraints

Who decided each item and when. The coordinator and executors follow this; the plugin does not enforce it.
Model choices and compute resources are not kept here: they live in smooth-research.yaml (models, compute),
where the CLI reads them.

来源标记：**[用户 2026-11-01]** 为用户在本项目初始化会话中的决定；**[plan]** 为用户提供并已封存的 `docs/plan.md`（DOC-0001，plan v2，2026-10-29）；**[协调者默认]** 为协调者按计划补全的低风险细节，用户可随时修改。

- Research goal and claims in scope:
  - 研究对象 [plan §0.1, §3]：冻结 Qwen/Qwen3-0.6B-Base 的全部原参数，在原层旁加低维侧分支（原 Q/K/V 特征 → 固定记忆 → 低维循环读取），经原 FFN 的 gate/up 条件融合（concat）或加法接口写回。
  - 范围内主张（都需要证据，不预设成立）[plan §0.1, §17.3]：(1) 低维分支实际读到了什么（寻址、读取混合、解码分开）；(2) 同一低维信息应以哪种接口进入原 FFN；(3) 固定记忆下改变查询的重复读取是否有独立收益（排除幅度、参数、训练量与额外监督解释）；(4) 保留原上下文时的实用价值与成本。
  - M 线（memory-only）与 F 线（context-retained）的结果分开登记、不互相替代 [plan §2.3]。
  - 新增一条研究线 [用户 2026-11-02]：读取 `docs/plan2.md`（《残差旁路低维线性注意力适配：研究计划 v2.1》，2026-11-01），把它的研究计划作为本项目的另一条线加入，与现有内容互补，一起推进。plan2.md 是这条线的起点文档，加入后不再编辑。它与现有记录重叠的部分以已有证据为准，按需补充，不改写已有结论；这条线的具体假设与实验以研究记录为准。训练 tokens 同样不设上限：plan2 中的 token 数只作各 arm 的样本预算，不作资源上限。算力计入全项目 200 node·h 总上限。
    - RB 线的固定设置 [plan2 §0.3、§3、§6A；协调者默认，D-0025]：F 线（保留原 attention），冻结 Qwen3-0.6B-Base 的四层 [6,13,20,27]，r32，qknorm_pre_rope，块输出残差旁路 R_res 加因果 KernelMatrix（inclusive 先写一次再读，η=0.25，g=1）；命名空间 study_id=residual_la_v21，seed 基数 20261101。plan2 的 R1 与 V5 在记录中称 RB1 与 RB5，以免与 E-0014 的 arm 名 R1..R6 混淆。
  - 不在本范围 [plan §12, §17.3]：新颖性或领先性能声明；更大 backbone（1.7B 及以上）、14/28 层布局、长上下文、每轮更新 K/V、旧 SLOT-LOOP-v1 续训（均为 V6 或独立预算）。
  - 当前授权阶段 [用户 2026-11-01]：第一里程碑 V0+V1（已完成）。原规定 V2 及以后先报告、再由用户释放；已由下面两条用户决定取代。
  - V2 已释放 [用户 2026-11-01，看过 V0+V1 阶段报告后]：分两步进行。第一步：实现正式训练程序，训练 F 线正对照（事实在上下文中），并做记忆家族学习率校准。第二步：正对照达到 V2 契约中预先锁定的水平后，自动进入八组析因（特征 × 读算子 × 融合 F*/C*）与 C_dual 交互探针，不再等用户确认。正对照不达标就停下报告；改动每个 episode 的事实数等已封存的数据协议，需要用户再决定。V3 及以后仍需阶段报告与用户决定。
  - 自主推进 [用户 2026-11-01，V2 释放之后]：“将计划向前推进直到探索完所有目标，不需要再征求我的同意”。协调者按计划推进 V2 至 V5（范围内主张 (1)–(4)），各阶段的释放、阶段内预算分配、正式（确认性）实验（如 V4 的新 seed 确认、V5 官方 validation 按封存规则开封一次）都由协调者按证据决定，不再等用户同意；V2 正对照不达标时，也由协调者决定下一步（例如可学性诊断或新的诊断性数据变体），记录理由。每个阶段结束仍写阶段报告并告知用户，但不等待回复。仍需用户决定：总算力超出 200 node·h；改变研究目标、范围内主张、backbone、数据集或确认集封存规则；开启 V6；对项目外发布或推送。
  - 扩大自主 [用户 2026-11-01，看过 V2 阶段报告后]：用户许可重开记忆路线，并采用计划外的改动，例如减少每个 episode 的事实数或加课程、加一条吸收答案先验的独立通路、加寻址监督；这包括改动已封存的数据协议（以新登记的数据版本进行，旧数据、旧结论与旧契约不改写）。用户同时表示“今后所有有意义的尝试都可以，不用请我许可”：新的实验、数据变体、方法改动、辅助监督与探索分支（包括 V6 类分支，如更多层、更长上下文或第二规模 backbone）都由协调者按证据决定并记录理由，不再等用户同意；每个阶段仍写报告并告知用户。额外监督与额外信息（地址标签、课程、辅助损失等）按 plan §13.5 单独计数，结论只在其条件下陈述。
- Allowed data, and which compute resources may be used for what (the resources themselves, how they are
  reached and how code gets there, are in smooth-research.yaml, compute):
  - 模型 [plan §3.1]：Qwen/Qwen3-0.6B-Base，运行时锁定模型与 tokenizer revision；不得用非 Base 配置替代。本地 HF 缓存 `/work/xg24i002/x10041/hf_home` 已有该模型。
  - 数据 [plan §8, §9, §11.2, §13]：
    - 新的合成关系/图任务，`study_id=concat_loop_v2`，按 phase/purpose 派生新 seed，不复用旧验证集。
    - PIQA、BoolQ：从官方 train 切出 train/dev/calibration；官方 validation 封存为确认集，只在配置、训练规则与分析脚本锁定后开封一次。
    - RB 线确认集 [协调者默认，D-0025]：官方 validation 由 E-0013 开封后不再作确认。RB 线从 V1 train 中每任务切出 1,000 条作新的确认集，按 derive_seed(20261101, "residual_la_v21", "RB5", "confirmation") 抽取，登记 id 清单与 SHA256 后封存；RB 线训练用其余 train 行；按上面的确认集规则只开封一次。
    - WikiText-103 固定版本的保留 split，只作能力保持指标。
    - 公开数据集与模型可以在登录节点下载到 HF 缓存 [协调者默认]。
  - 旧证据 [plan §18；用户 2026-11-01]：UnbearableFate/LoopLora @ `a255e814740ff85ab3b4ec4836c88f0a8f802b2d`（GitHub 公开仓库），只作只读审计与参考。旧 S-RE checkpoint 不在本文件系统上，按计划 fallback 跳过冻结 S-RE 读算子诊断（plan §6.1–6.2，H-0007）。不得声称旧指标已在本地复现。
  - 计算 [用户 2026-11-01]：只通过 miyabi-mcp 向 Miyabi-G PBS（group xg24i002）提交作业。登录节点只做控制工作：编辑、记录、暂存小型源码快照、依赖解析与下载。测试、冒烟检查、训练、评估与数据分析都放进作业，即使输入很小也不在登录节点运行。
- Resource ceiling (per experiment / per batch / total; units such as CPU-h, GPU-h, node-h; queues):
  - V2 算力 [用户 2026-11-01]：≤ **10 node·h**（含两步与全部冒烟、审核回归）。V3 至 V5 的阶段预算由协调者按实测成本在契约中设定 [用户 2026-11-01 授权]。所有阶段合计不超过 200 node·h；换算规则相同。
  - 总算力 [用户 2026-11-01]：≤ **200 node·h**（Miyabi-G；最初为 V0+V1 设定，现作为全部阶段的总上限）。整节点作业按实际 node·h 计；MIG 队列作业按 1/4 计，即 node·h = miyabi-mcp 账本的 mig_instance_hours × 0.25 [用户 2026-11-01]。达到 80% 时协调者先报告，超出前必须由用户同意。
  - 训练 tokens [用户 2026-11-01]：取消训练 tokens 上限（原为 V0+V1 ≤ 18M 非 padding 输入 tokens，来自 plan §6.4、§7.3、§14）。tokens 仍按样本呈现次数如实记录，但不再作为资源上限；算力只受上面的 node·h 上限约束。[协调者默认] 实验契约中每个 arm 的 token 数属于实验设计（各 arm 相同的样本预算），不是资源上限，改动按契约修订处理。
  - 每个作业 [协调者默认；用户 2026-11-01：资源只看 node·h 上限]：单节点；walltime 按实测 × 1.5 申请；遵守队列自身的时限（如 debug 队列 ≤ 30 min）；作业不得跨越 2026-11-25 09:00 JST 的计划停机。不再另设单作业时长或并发作业数的上限。
  - 预算核算 [plan §14.4]：每个 arm 实测加载、稳定训练步、评估与保存时间；node·h 只按实测换算。时间不足时统一缩短所有 arm 的步数，不允许各 arm 停在不同的样本预算上。
  - 角色与订阅 [用户 2026-11-01]：研究者优先用 codex，审核者用 Claude；顾问轮次 ≤ `advice.max_rounds`（3）。
- Actions that need the user's decision first:
  - 改变研究目标本身（新增尝试、数据变体、backbone 或探索分支不算改变目标，已由用户授权，见“扩大自主”）。
  - 超出上述总 node·h 上限。
  - 确认集规则照旧，不由新的尝试改变：每份确认集在配置、训练规则与分析脚本锁定后只开封一次，开封后不再用于确认；改方法后的确认用新数据。
  - 把代码、数据或结果推送、发布到项目外（GitHub、共享页面等）。
  - 任何不可撤销的项目外操作，例如删除其他目录的数据、修改 miyabi-mcp 的 policy、取消非本项目提交的作业。
