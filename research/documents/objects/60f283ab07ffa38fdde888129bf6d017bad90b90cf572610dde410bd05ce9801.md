# T-0033：分段协议的 A2 确认（E-0030）

## 目标

按 E-0030 完成 H-0029 的一次预先登记确认：

1. 实现新 seed 3、4、5，A2 读取器，A2 评估阶段与锁定分析，并通过代码审核；
2. 冒烟后训练 12 个正式 run（3 个 seed × SEG-05、SEG-06、SEG-07、SEG-04），做开封前检查；
3. 写 A2 锁定文件，报给协调者，等协调者的确认；
4. 确认之后开封 A2，运行锁定分析，记录证据并交付。

## 背景

- H-0029 已启用。用户 2026-11-02 同意为分段协议动用 A2（docs/constraints.md）。H-0020 停在 inconclusive（D-0062）；E-0028、E-0029 没有更强的侧支算子（D-0066 至 D-0071），所以沿用 E-0027 的 r128 配置。
- 代码从 E-0027 的 054a54a 出发（`src/seg_*.py`）。`seg_seed_analysis.max_control_intervals` 是已审核的区间口径，规则 4 在它的基础上扩展到三个 seed 共用下标。
- 训练行已经按 `research/artifacts/T-0022/rb-confirmation.json`（SHA256 34277aa4…）排除 A2（`rb_train.load_data` 调用 `apply_confirmation`）。
- 成本参照（E-0027 实测，每个 seed）：SEG-05 约 2.08 MIG-h（short-mig，约 2 小时），SEG-06、SEG-07、SEG-04 各约 1.0 MIG-h，dev 干预作业约 0.47 MIG-h。
- 现在没有其他研究者在改共享代码。

## 规格

- 契约：`research/experiments/E-0030.md`。arm、seed、检查与规则都以它为准。
- 新 seed：在 `ORDER_SHA256_BY_SEED`、`SEED_ARMS` 等处加入 3、4、5，顺序 sha256 在训练前算出并登记。核对记录，确认这三个 seed 没有在 R1 阶段用过。
- A2 读取器（新代码）：
  - 只在 A2 作业里调用。锁定文件的路径与 SHA256 由作业传入；启动时核对锁定文件、A2 id 清单文件（SHA256 与 ids_sha256）与各 checkpoint 的 SHA256，任一不符就停止。
  - 每次调用在 run 目录写一条读取记录：时间、PBS 作业号、锁定文件 SHA256、评估的 arm、分配到的硬件。
  - 按 id 从 V1 train 行取出 A2 行，按 E-0025 的分段协议编码。
  - 测试只用合成切分。真实数据上只核对 id 与哈希，不评估模型，也不读取或汇总 A2 的文本与标签。
- A2 评估：每个 seed 一个作业，内容与顺序见契约“A2 锁定与开封”第 3 条与规则 2。iv_swap 的 derangement 种子为 derive_seed(20261101, "residual_la_v21", "A2", "iv_swap")。
- 锁定分析：实现规则 1 至 8，输出 analysis.json 与它的 SHA256，在 PBS 中运行。
- 代码审核：新 seed、A2 读取、A2 评估与锁定分析一起审，在正式训练之前通过（`rl review request --target E-0030 ...`）。冒烟与确定性检查可以先跑。审核之后再改这些路径，在写锁定文件之前重新审核。
- 正式训练：12 个 run 用一个 notification 组并行提交；每个 run 先 `rl run register`。
- 锁定：规则 1 全部通过后，写 `research/artifacts/T-0033/a2-lock.json`（内容见契约）并提交，用 NOTE 把锁定文件的 SHA256、各检查结果与实测成本报给协调者，然后等待。
- 开封：收到协调者的确认之后，才提交读取 A2 的作业。作业因工程原因失败时，只能按同一锁定文件原样重跑，每次尝试都登记。
- 证据：规则 5、6 各记一条 confirmatory 证据，规则 1、2 记 diagnostic；不改假说状态。
- 交付：`research/notes/T-0033-results.md`，包括每个 arm 在 A2 与 dev 上的 PIQA、BoolQ、宏平均与 NLL，每个 seed 的 gain、lora、swap，G、L 与区间，规则结果，A2 读取记录，以及成本。

## 不要做

- 协调者确认锁定之前，不提交任何读取 A2 的作业；测试、冒烟与开封前检查都不读取 A2 的文本或标签。
- 不按 dev 或 A2 选 checkpoint；不改 LR、步数或训练混合；不加 seed 或 arm。
- 不在普通设置下做新的 tap 或来源变体（E-0028 规则 10，D-0065）。

## 预算

≤ 5.5 node·h（node·h = MIG-h × 0.25）。冒烟实测后更新成本表；预测超过上限时先告诉协调者，等契约修订后再继续。
