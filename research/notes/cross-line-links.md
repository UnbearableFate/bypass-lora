# 跨线联动：plan.md 各线与 plan2 RB 线

协调者，2026-11-02。用户问 plan.md 生成的研究与 plan2.md 新增的 RB 线能否联动。本页记录联动点，以及后续契约要带上的条款。运行前固定的解读规则写在 E-0021 的“跨线联动”节。

| 问题 | plan.md 一侧 | plan2 RB 一侧 | 连接方式 |
|---|---|---|---|
| 多次读取 | H-0005@2（M 线 V3→V4，T2 链终点） | H-0015（RB1-03..06） | 两线的循环都走 bypass_core 的同一个 LatentReader（steps、fixed_query）；fixed_query 与 η/4 对照同定义；S 各自按本线选；结果按方向并列 |
| gate/up 融合 | H-0002（V1 无记忆 proxy，challenged） | H-0016（RB1-08/09，真实 KernelMatrix 状态） | RB1 检验 H-0002 的 scope 留下的有状态 concat；EV-0048、D-0004 不改写 |
| 实用价值 | H-0008@2（V5 tokenwise C_up，E-0013 开封中） | H-0013、H-0014（RB1→RB5） | 对照同为 LoRA28 r1；V5 的 checkpoint 不能复用（训练数据含 RB 确认集的行） |
| 读取选择性 | H-0004（paused/inconclusive）与 H-0012 的读诊断 | RB1 的 n_eff、熵与当前 token 读权重 | 同定义（src/v2_memory.py 的 read_metrics），按相对值并列 |
| 实现 | H-0001（V0） | H-0017（RB0） | 共用 bypass_core 与 bypass_model；改共享类之前先互相通知（MSG-0548/0549） |

后续契约要带上的条款：

- M 线 V3 训练矩阵契约（T-0021 之后）：镜像 E-0021 的循环联合解读条款；建立后 `rl link <V3 契约> compared_with E-0021`。
- RB1 任务建立、E-0021 封存时：`rl link E-0021 compared_with E-0003`（concat 的跨线解读）。
- RB5 契约：LoRA28 r1 对照的 LR 校准规则与分析约定（配对分层 bootstrap、Holm）与 E-0013 对齐，在 RB 训练行上重新训练；不复用 V5 checkpoint；用 D-0025 的 A2 确认集。
- V4（H-0004，或 KernelMatrix 版 D5）：把 RB1 的读诊断列为参照。
- 阶段报告与 manuscript：按上表列跨线主张；联合描述属 screening，随结论审核一起审。

三条线的确认数据互不污染：V5 用 official validation，只开一次（E-0013）；RB 线用从 V1 train 切出的 A2 集（D-0025），RB 训练不含这些行；M 线外推用封存的 T2 hops4/6/8。计算预算共用 200 node·h。
