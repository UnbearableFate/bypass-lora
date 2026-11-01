# T-0012 / E-0012@1：L0–L4 诊断测量交付

测量完成，六项 L0 均通过。交付的是单 seed、固定 V2 T1 数据与指定探针下的可解码性、读取分布和幅度；不改变 H-0003 或 E-0010 的结论，不选择 E-0013，不作信息缺失或唯一原因的判断。

完整 120 行探针表及十 checkpoint 的 L3/L4 主表：[report.md](/work/xg24i002/x10041/bypass-lora/runs/ATT-0096-1/report.md)。全部计数、训练内部验证分数、超参数、逐问题位置槽位分布、key/margin/幅度分位数及输入/源码 hash：[report.json](/work/xg24i002/x10041/bypass-lora/runs/ATT-0096-1/report.json)。独立校验：[checks.json](/work/xg24i002/x10041/bypass-lora/runs/ATT-0097-1/checks.json)。

## 运行与验证

| 用途 | ATT / RES | PBS（均在 mg1098） | 实测 MIG 秒 | node·h |
|---|---|---|---:|---:|
| 五项回归与缩小冒烟 | ATT-0094-1 / RES-0095 | 3762986.opbs | 48 | 0.003333 |
| 全形状成本与十 checkpoint 几何路径检查 | ATT-0095-1 / RES-0096 | 3763092.opbs | 66 | 0.004583 |
| 正式 L0–L4 测量 | ATT-0096-1 / RES-0097 | 3763105.opbs | 955 | 0.066319 |
| 独立产物校验 | ATT-0097-1 / RES-0098 | 3763545.opbs | 2 | 0.000139 |
| 合计 | 4 个作业 | 均 exit 0，应用产物完整 | 1071 | **0.074375** |

按一个 MIG instance 为 1/4 node 换算，合计低于 E-0012 的 0.5 node·h。正式测量 pipeline 用时 945.891 s，PBS 实测 walltime 955 s。作业启动均验证了匹配的 Miyabi-G allocation 与 PBS RUNNING 状态。前三次用量来自 executor ledger 兼容导出；最后一次 ledger end 尚未导出时，以 miyabi-mcp job_get 的已验证 F/Exit0/2s 生成结束观察并用同一 accounting_ref 记录，原始返回保存在该 attempt 的 job-get-final.json。

`RV-0027` 代码审核 pass，零 findings；全量执行 commit 为 `8cfcb1a8c696cec3af2b611bdb33a25a5b51e801`，审核覆盖的核心文件未改。五项回归全部通过。独立 PBS 校验不调用提取/拟合实现，检查了精确的 120 行协议组合、计数与准确率、从 released train/calibration 重算的常数基线、内部验证的 L2/epoch 网格和选中最优记录、十份原始 L3 manifest、十四组 L4 直方图/分位数，以及 55 个源码和 26 个输入 hash。正式 report SHA256 为 `bf1168eb81df6f3af1874bbb578361d2eda5d710cb7ead441a59c9e45e9db08a`。没有未解决的工程或审核问题。

冻结特征从无 adapter Base 的第 14 层、support 与 question 独立前向获取。两个 tap 的 support/query 投影输入逐元素最大绝对误差均为 0，读取重建通过预先记录的 atol/rtol 2e-5。只使用 train16384、calibration2048、dev1000；没有打开 confirmation。线性/双线性探针在 train 前 14336 episode 拟合，最后 2048 episode 仅选 L2 和 epoch，输入标准化也只用拟合部分。L3 保持原 `v2_probe.py` 不变，fit=train 前4096，held-out=calibration 全2048，ridge=1。

最初四 epoch 冒烟的 qknorm 尾/头控制为 1754/2048、1829/2048，未达门槛，已标记为不可解释；正式预定拟合（L0 最多64 epoch、三种 L2、仅内部验证选取）通过六项控制，未修改位置、切分、阈值或指标。

## L0 正对照（calibration）

| 特征 | 尾名字末 token → 尾实体 | 头名字末 token → 头实体 | 问题头名字末 token → 头实体 |
|---|---:|---:|---:|
| direct_hidden | 32764/32768 (99.9878%) | 31717/32768 (96.7926%) | 2048/2048 (100%) |
| qknorm_pre_rope | 32756/32768 (99.9634%) | 31663/32768 (96.6278%) | 2048/2048 (100%) |

## L1–L4 主要测量

完整 L1/L2 key、query、关系以及每个 rank 的表在上述 report.md；以下是尾实体、寻址、独立读出与几何的并列表。L2 的 value 探针用拟合部分所有事实，L3 只用前4096个训练 episode 的目标 value，而且分类器为 ridge；两者是分别报告的诊断，不是控制了拟合样本和分类器的对照。

All six L0 controls passed. The 120 probe rows, ten L3 arms and fourteen L4 entries passed independent artifact checks.

| Feature | L1 delimiter→tail | L2 random value→tail | L2 trained value→tail range |
|---|---:|---:|---:|
| direct_hidden | 32648/32768 (99.6338%) | 12609/32768 (38.4796%) | 8151/32768 (24.8749%) [V2-02] – 12148/32768 (37.0728%) [V2-03] |
| qknorm_pre_rope | 30433/32768 (92.8741%) | 10539/32768 (32.1625%) | 5556/32768 (16.9556%) [V2-06] – 8615/32768 (26.2909%) [V2-10] |

| Feature | L1 head-position address, ranks128–1024 | L1 last-position address, ranks128–1024 |
|---|---:|---:|
| direct_hidden | 1756/2048 (85.7422%) [Base] – 1819/2048 (88.8184%) [Base] | 1804/2048 (88.0859%) [Base] – 1847/2048 (90.1855%) [Base] |
| qknorm_pre_rope | 1614/2048 (78.8086%) [Base] – 1676/2048 (81.8359%) [Base] | 1684/2048 (82.2266%) [Base] – 1756/2048 (85.7422%) [Base] |

| L3 arm | Clean→clean | Read→read | Clean→read |
|---|---:|---:|---:|
| V2-01 | 472/2048 (23.0469%) | 40/2048 (1.9531%) | 49/2048 (2.3926%) |
| V2-02 | 455/2048 (22.2168%) | 37/2048 (1.8066%) | 62/2048 (3.0273%) |
| V2-03 | 467/2048 (22.8027%) | 43/2048 (2.0996%) | 47/2048 (2.2949%) |
| V2-04 | 447/2048 (21.8262%) | 38/2048 (1.8555%) | 41/2048 (2.0020%) |
| V2-05 | 345/2048 (16.8457%) | 42/2048 (2.0508%) | 47/2048 (2.2949%) |
| V2-06 | 356/2048 (17.3828%) | 45/2048 (2.1973%) | 44/2048 (2.1484%) |
| V2-07 | 370/2048 (18.0664%) | 43/2048 (2.0996%) | 38/2048 (1.8555%) |
| V2-08 | 341/2048 (16.6504%) | 42/2048 (2.0508%) | 42/2048 (2.0508%) |
| V2-09 | 367/2048 (17.9199%) | 47/2048 (2.2949%) | 51/2048 (2.4902%) |
| V2-10 | 382/2048 (18.6523%) | 39/2048 (1.9043%) | 47/2048 (2.2949%) |

| L4 arm | Key cosine mean / n | Last target / n | Last same relation / n | Last slot mode: count / n | Last fusion/residual mean / n |
|---|---:|---:|---:|---:|---:|
| random.direct_hidden.kernel_matrix | 0.934750 / 120000 | 48/1000 | 224/1000 | 2: 155/1000 | 0 / 1000 |
| random.direct_hidden.soft_slot | 0.934750 / 120000 | 48/1000 | 224/1000 | 2: 155/1000 | 0 / 1000 |
| random.qknorm_pre_rope.kernel_matrix | 0.932191 / 120000 | 57/1000 | 248/1000 | 7: 110/1000 | 0 / 1000 |
| random.qknorm_pre_rope.soft_slot | 0.932191 / 120000 | 57/1000 | 248/1000 | 7: 110/1000 | 0 / 1000 |
| V2-01 | 0.911460 / 120000 | 52/1000 | 249/1000 | 0: 1000/1000 | 607.753 / 1000 |
| V2-02 | 0.888802 / 120000 | 52/1000 | 249/1000 | 0: 1000/1000 | 383.455 / 1000 |
| V2-03 | 0.968558 / 120000 | 52/1000 | 249/1000 | 0: 1000/1000 | 47.5953 / 1000 |
| V2-04 | 0.961644 / 120000 | 52/1000 | 249/1000 | 0: 1000/1000 | 25.9158 / 1000 |
| V2-05 | 0.944545 / 120000 | 52/1000 | 249/1000 | 0: 1000/1000 | 318.553 / 1000 |
| V2-06 | 0.918273 / 120000 | 52/1000 | 249/1000 | 0: 1000/1000 | 185.815 / 1000 |
| V2-07 | 0.997272 / 120000 | 52/1000 | 249/1000 | 0: 1000/1000 | 33.1045 / 1000 |
| V2-08 | 0.999020 / 120000 | 52/1000 | 249/1000 | 0: 1000/1000 | 19.2867 / 1000 |
| V2-09 | 0.931175 / 120000 | 52/1000 | 249/1000 | 0: 1000/1000 | 1.12365e+06 / 1000 |
| V2-10 | 0.993508 / 120000 | 52/1000 | 249/1000 | 0: 1000/1000 | 18.6638 / 1000 |

These are observations under the specified probes and splits. L1/L2 are softmax probes with train-internal model selection; L3 uses independent ridge heads and a smaller fitting set, so its numbers are separate diagnostics.
The tables do not identify a unique cause, establish information absence, change H-0003, or select a subsequent experiment. All raw counts, per-position histograms, quantiles and selected hyperparameters remain in the full report.
