# 目前最好的方案

截至 2026-11-02。

项目冻结 Qwen3-0.6B-Base（下称 backbone），只在旁边加低维侧分支。“最好”要分用途说：

- 只看微调后的准确率，也就是 backbone 能看到全部输入的普通设置，最好的仍是普通 LoRA，没有哪个侧分支方案超过它。侧分支里最接近的是 tokenwise C_up：在唯一一次确认级比较（V5）里，它比 LoRA 低 0.42 pp（百分点），分不出高下，参数也更少。但它不读侧分支自己的历史，结构上接近加在 FFN（attention 之后的前馈网络）up 投影上的 LoRA。
- 要的如果是侧分支独有的能力，即把 backbone 看不到的信息存下来、之后再读出来，唯一达标的是记忆线（M 线）的 D5 配方，前提是每条事实单独编码。在 backbone 看不到前一段的分段设置下，保留跨段状态的 R_keep 还在验证。
- 侧链线性注意力（RB 线）里，普通设置下点估计最高的是 RB1-09：四层低维线性注意力读一次，读出接到 FFN 的 up 投影。它比参数更少的四层 LoRA 低 1.45 pp，比不读历史的对照 T0 只高 0.20 pp，结构见后面的专节。

## 基准是普通 LoRA

LoRA（低秩适配）冻结原权重 $W$，只训练两个小矩阵 $B$、$A$，它们的乘积秩不超过 $r$：

$$W' = W + \frac{\alpha}{r} BA$$

项目里取 $\alpha = r$，增量就是 $BA$。all-7 指每层的 q、k、v、o、gate、up、down 七个线性层都加。V5 的对手是全部 28 层 all-7、rank 1 的 LoRA，共 630,784 个参数。LoRA 只改权重，推理时没有随输入累积的状态。

## 准确率上没有方案超过 LoRA，C_up 最接近

C_up 在第 6、13、20、27 层的 FFN 里加一个逐 token 的分支：

$$Z = \mathrm{SiLU}(UA), \qquad Y = H + \big[\mathrm{SiLU}(UW_g) \odot (UW_u + ZC_u)\big] W_d$$

$H$ 是 attention 之后的残差，$U$ 是 $H$ 归一化后的 FFN 输入，$W_g$、$W_u$、$W_d$ 是原 FFN 冻结的 gate、up、down 权重，SiLU 是原 FFN 的 gate 也在用的激活函数。可训练的只有 $A \in \mathbb{R}^{1024\times 32}$ 和 $C_u$，四层共 524,288 个参数。$C_u$ 零初始化，训练起点就是 Base（不训练的原模型）。

$U$ 已经含有 backbone 的上下文，tokenwise 只是说侧分支没有自己的跨 token 记忆。把 SiLU 去掉，

$$UW_u + UAC_u = U(W_u + AC_u)$$

就是在 $W_u$ 上加一个秩不超过 32 的 LoRA。所以 C_up 本质上是 up 投影上的一个低秩项，中间夹了一个 SiLU。

V5（E-0013）是目前唯一的确认级比较：判定规则在训练前锁定，三个方案各用 seed（随机种子）1、2、3 训练，之后在 PIQA（1,838 条）与 BoolQ（3,270 条）的官方 validation 上只开封一次。PIQA 给一个目标、选出合适的解法，BoolQ 读一段文章回答是或否，宏平均是两者准确率的平均。次要对照 F_out 把同类分支乘一个矩阵后直接加到层输出，$Y = H + F(U) + ZP$，$F$ 是原 FFN。

| 方案 | 参数 | 宏平均 | WikiText NLL 增量 |
|---|---:|---:|---:|
| 28 层 all-7 rank 1 LoRA | 630,784 | 77.90% | +0.337 |
| C_up（四层，r32） | 524,288 | 77.48% | +0.200 |
| F_out（四层，r64） | 524,288 | 77.25% | +0.238 |

C_up 减 LoRA 为 −0.42 pp，95% 区间 [−1.02, +0.20]，三个 seed 分别是 −0.28、−0.40、−0.58，分任务 PIQA −0.74、BoolQ −0.09。上界低于封存规则的 +1 pp，H-0008 因此记为 challenged，这条路线停止（D-0028）。区间含 0，既不能说两者等价，也不能说 LoRA 更好，能说的是 C_up 没有更好。

最后一列是 WikiText-103 上 NLL（负对数似然，越低越好）相对 Base（2.647）的增量，用来看原有能力保留得怎样。这一列只作描述，C_up 的增量最小。

读侧分支自己历史的 RB 线也没有超过 LoRA。dev 是从官方 train 切出的开发集，每任务 1,000 条，在 dev 上最好的 RB1-09 比四层 LoRA r7 低 1.45 pp。残差写回加宽到 r128 后，仍比参数相当的 LoRA r31 低 1.80 pp（E-0026）。

## 独有能力上只有 D5 记忆配方达标

backbone 丢掉的内容，LoRA 拿不到。M 线专门检验侧分支能不能补上这一点：一个 episode 有 16 条事实和一个问题，事实写进第 14 层的一组记忆槽位后，attention 为这些事实存下的 key 与 value（KV）被丢掉，回答问题时只能从槽位里读。读取用 softmax：

$$a = \sum_{i=1}^{16} \mathrm{softmax}_i\!\left(\frac{q^\top k_i}{\sqrt{r}}\right) v_i$$

$k_i$、$v_i$ 由第 $i$ 条事实分隔符处第 14 层归一化后的隐藏状态投影得到，$q$ 由问题位置的隐藏状态投影得到，$r = 128$。$k_i$ 与 $q$ 投影后还要经过 RMSNorm 和 $\phi(x) = \mathrm{ELU}(x) + 1$，$q$ 另乘一个 $r\times r$ 矩阵，写法与后文 RB1-09 的 $\kappa_t$、$\rho_t$ 相同。只读一次（S=1），$a$ 经可训练矩阵映射后加到第 14 层的输出上（F_out 写回）。

D5 配方有两个要点。一是每条事实不带前文、单独过一遍 backbone，在它的分隔符处写入 key 与 value，问题侧不变。二是先在每个 episode 只有 1 条事实的任务上训练，再续训到 16 条。全程只用答案的交叉熵，不用该读哪个槽位的标签。

判定看正常记忆比清空记忆（zero）和换成另一个 episode 的记忆（shuffled）好多少，EM 是完全答对的比例：

$$\Delta_{\mathrm{EM}} = \mathrm{EM}_{\mathrm{normal}} - \max(\mathrm{EM}_{\mathrm{zero}}, \mathrm{EM}_{\mathrm{shuffled}})$$

seed 0 上正常记忆的 EM 是 42.8%，清空记忆 1.1%，换成别的记忆 1.7%，优势 41.1 pp。seed 1、2 的优势是 47.8、46.2 pp，门槛是 10 pp。另外两条也都通过：NLL 更低，全部问题位置上正确槽位的平均读权重至少 0.125。H-0012 因此记为 supported，证据是筛选级（D-0023）。它有三个限制：

- 依赖每条事实单独编码这种结构化输入。同一批训练好的参数改回上下文编码写入，也就是事实连在一起过 backbone，EM 只有约 5%。
- T2 链终点任务的问题给出起点，答案是沿关系链走到的终点。学到的寻址没有迁移到这种句式，一跳问题的优势只有 0.9 到 3.0 pp：解码迁移过去了，寻址没有。
- 在它之上读两次（V3）没有成功，S2 在 2 到 3 跳上只比读一次的 S1 高 0.60 pp，门槛是 3 pp。H-0005 记为 inconclusive，M 线暂停（D-0036）。

## 分段设置下的 R_keep 还在验证

分段设置（E-0025）把每个样本拆成两次 backbone 前向，段间丢掉 KV，位置从 0 重新开始。PIQA 的目标、BoolQ 的文章只在第 1 段，第 2 段的 backbone 看不到它们，能跨段的只有第 6、13、20、27 层的 KernelMatrix（低维线性注意力的状态）。它每读到一个 token 写入一次，之后任何位置都可以读：

$$M_t = M_{t-1} + \phi(k_t)\, v_t^\top, \qquad c_t = c_{t-1} + \phi(k_t), \qquad a_t = \frac{\phi(q_t)^\top M_t}{\phi(q_t)^\top c_t}$$

$\phi(x) = \mathrm{ELU}(x) + 1$，$k, v, q$ 是从 attention 特征压缩出的低维向量，完整写法见下一节。R_keep 把第 1 段结束时四层的 $(M, c)$ 作为第 2 段的初始状态，读出加到层输出上（残差写回）。R_reset 结构相同，但在段界清空状态；T0 参数相同，但不读历史。主比较是

$$\mathrm{gain} = \mathrm{Acc}_{\mathrm{keep}} - \max(\mathrm{Acc}_{\mathrm{T0}},\ \mathrm{Acc}_{\mathrm{reset}})$$

门槛是 +3 pp。dev 上 seed 0 的准确率（%）：

| 组 | PIQA | BoolQ |
|---|---:|---:|
| Base_seg（分段，不训练） | 66.1 | 60.9 |
| R_keep，r32 / r128 | 69.4 / 69.3 | 65.8 / 65.9 |
| R_reset，r32 / r128 | 67.8 / 66.9 | 61.2 / 61.5 |
| T0，r32 / r128 | 66.9 / 67.9 | 63.8 / 61.2 |
| 分段 LoRA（只看第 2 段） | 67.4 | 64.2 |
| 能看到全文的 LoRA | 74.4 | 83.5 |
| gain，r32 / r128 | +1.6 / +1.4 | +2.0 / +4.4 |

主任务 PIQA 的 gain 为正，但低于 +3 pp。BoolQ 在 r128 上是 +4.4 pp，把第 1 段的状态换成同任务另一个样本的状态后下降 6.5 pp。但 r128 的两个无历史对照只比 Base_seg 高 0.3 和 0.6 pp，R_keep 也只比分段 LoRA 高 1.7 pp。H-0019 记为 inconclusive（D-0057）。BoolQ 的两个新 seed 复现（E-0027）已经开始，这次还要求 R_keep 平均比分段 LoRA 高至少 1 pp。即使复现成功，分段各组也都远低于能看到全文的 LoRA，只能说明状态能把一部分信息带过段界，说明不了实用价值。

## 侧链线性注意力里点估计最高的是 RB1-09

RB 线（残差旁路线）让侧分支在每个 token 处用低维线性注意力读自己的因果历史。普通设置下，RB1 筛选（E-0021）的八个侧分支组里点估计最高的是 RB1-09：四层 KernelMatrix，读一次（S1），$r = 32$，读出接到 FFN 的 up 投影上（C_up）。分段设置下最好的是上一节的 R_keep。

只在第 6、13、20、27 层加侧分支，backbone 冻结。特征直接取原 attention 的中间结果：$\tilde Q_t$ 是 `q_norm` 之后、RoPE 之前的 query（16 头 × 128 = 2048 维），$\tilde K_t$ 是 `k_norm` 之后的 key（8 × 128 = 1024 维），$V_t$ 是 `v_proj` 的输出（1024 维）。先压到 32 维：

$$\bar q_t = \tilde Q_t A_q, \qquad \bar k_t = \tilde K_t A_k, \qquad \bar v_t = V_t A_v$$

$A_q \in \mathbb{R}^{2048\times 32}$，$A_k, A_v \in \mathbb{R}^{1024\times 32}$，随机初始化。每个样本从零状态开始，每个 token 先写后读（inclusive），所以当前 token 也在状态里：

$$\kappa_t = \phi\big(\mathrm{RMSNorm}(\bar k_t)\big), \qquad M_t = M_{t-1} + \kappa_t \bar v_t^\top, \qquad c_t = c_{t-1} + \kappa_t$$

$$\rho_t = \phi\big(\mathrm{RMSNorm}(\bar q_t)\, R_q\big), \qquad a_t = \frac{\rho_t^\top M_t}{\rho_t^\top c_t + \varepsilon}$$

RMSNorm 不带可学的缩放，$\phi$ 让每个分量为正，$R_q \in \mathbb{R}^{32\times 32}$ 初始化为单位阵。把 $M_t$、$c_t$ 展开（忽略 $\varepsilon$），读出就是对历史 value 的加权平均，权重为正、和为 1：

$$a_t = \sum_{i \le t} w_{t, i}\, \bar v_i, \qquad w_{t, i} = \frac{\rho_t^\top \kappa_i}{\sum_{j \le t} \rho_t^\top \kappa_j}$$

侧分支的输出是 $Z_t = \eta\, a_t R_o$，$\eta = 0.25$ 固定，$R_o \in \mathbb{R}^{32\times 32}$ 初始化为单位阵。最后接到原 FFN 的 up 投影上：

$$Y_t = H_t + \big[\mathrm{SiLU}(u_t W_g) \odot (u_t W_u + Z_t C_u)\big] W_d$$

$u_t$ 是 $H_t$ 经过 `post_attention_layernorm` 后的 FFN 输入。$C_u \in \mathbb{R}^{32\times 3072}$ 零初始化，训练开始时模型与 Base 完全相同。

残差写回 R_res（RB1-02）的前几步完全相同，只是把 $Z_t P_R$（$P_R \in \mathbb{R}^{32\times 1024}$）直接加到层输出上。T0（RB1-01）的参数与 R_res 相同，只用当前 token 的 $\bar q_t, \bar k_t, \bar v_t$ 做一次非线性门控，不读历史。V5 的 tokenwise C_up 也写回 up 投影，但它的 $Z$ 由 FFN 输入直接算出，RB1-09 的 $Z_t$ 是对历史的线性注意力读出。

可训练参数每层是 $A_q$ 65,536、$A_k$ 32,768、$A_v$ 32,768、$R_q$ 1,024、$R_o$ 1,024、$C_u$ 98,304，共 231,424，四层 925,696。状态每层是 $32\times 32$ 的 $M$ 加 32 维的 $c$，FP32 共 4,224 字节，四层 16,896 字节，不随长度增长。

训练设置与 RB1 其余各组相同：PIQA 与 BoolQ 混合，seed 0，13,115 步，有效批大小 2，共 26,230 个样本、2,000,069 个输入 token。训练目标是正确候选 token 上的交叉熵，按样本平均。优化器 AdamW（0.9, 0.999），权重衰减 0.01，梯度裁剪 1，前 656 步预热，之后按余弦衰减。学习率 1e-3，是 gate/up 融合家族在 calibration 上锁定的值。backbone 用 BF16 冻结，侧分支参数和状态用 FP32。

### 结果（dev，seed 0，%）

| 组 | 参数 | PIQA | BoolQ | 宏平均 |
|---|---:|---:|---:|---:|
| Base | 0 | 72.30 | 68.00 | 70.15 |
| T0（RB1-01） | 663,552 | 73.30 | 81.30 | 77.30 |
| R_res，读一次（RB1-02） | 663,552 | 72.40 | 81.80 | 77.10 |
| R_res，读两次（RB1-03） | 663,552 | 72.70 | 82.20 | 77.45 |
| C_dual，接 gate 与 up（RB1-08） | 1,318,912 | 73.30 | 81.60 | 77.45 |
| C_up（RB1-09） | 925,696 | 73.20 | 81.80 | 77.50 |
| 同四层 all-7 LoRA r7（RB1-07） | 630,784 | 74.40 | 83.50 | 78.95 |

RB1-09 和 C_dual、读两次的 R_res 只差 0.05 pp，dev 分不出来。它比不读历史的 T0 高 0.20 pp，但 T0 的参数与 R_res 相同、比 RB1-09 少，C_up 自己的同参数 T0 没有测。对读一次的 R_res，它高 0.40 pp（描述性配对区间 [−0.70, +1.50]），没到 1 pp 门槛，H-0016 记为 inconclusive（D-0041）。它比 Base 高 7.35 pp，但比参数更少的 LoRA r7 还低 1.45 pp。

### 训练后的干预

权重不动，评估时改侧分支的输出或它读到的历史，原 attention 不受影响。下表是 dev 宏平均相对正常读取的变化（pp），残差写回的 RB1-02 作对照：

| 干预 | 做法 | RB1-09 | RB1-02 |
|---|---|---:|---:|
| iv_off | 侧分支输出置零 | −7.35 | −6.95 |
| iv_window1 | 只读当前 token | −2.90 | −30.05 |
| iv_window8 | 只读最近 8 个 token | −1.10 | −14.00 |
| iv_strict | 读时排除当前 token | +0.45 | +0.05 |
| iv_swap | 过去的历史换成同任务另一条样本的 | −1.95 | −1.85 |
| iv_permute | 打乱过去 token 的 key 与 value 配对 | −0.65 | −17.15 |

RB1-09 用到了侧分支，输出置零后正好掉回 Base。它也用到一部分历史内容，swap 下降 1.95 pp，区间 [−3.15, −0.75]。只读近处和打乱配对对它的影响远小于 RB1-02，permute 的区间 [−1.50, +0.20] 含 0。E-0024 的读诊断显示，它在 BoolQ 第 6 到 20 层、PIQA 第 6 到 13 层的读权重接近均匀，相当于对历史的 value 取平均，第 27 层则有 0.55 到 0.70 的读权重落在第一个 token 上。能说的只有：它读了历史，但只比不读历史的 T0 高 0.20 pp。干预造成的下降也可能来自分布偏移，不能当作历史带来收益的证据。

## 汇总

| 方案 | 设置 | 结果 | 证据 |
|---|---|---|---|
| 普通 LoRA | backbone 看得到全部输入 | 准确率最高，没有侧分支超过它 | V5 确认级 |
| tokenwise C_up | 同上 | 比 28 层 LoRA 低 0.42 pp，区间 [−1.02, +0.20]，参数更少 | V5 确认级，3 个 seed |
| RB1-09 | 同上，读侧分支自己的历史 | 比四层 LoRA r7 低 1.45 pp，比 T0 高 0.20 pp | dev 筛选，1 个 seed |
| D5 记忆配方 | 事实的 KV 被丢掉，只能读记忆 | 优势 41.1、47.8、46.2 pp，门槛 10 pp | 筛选级，3 个 seed |
| R_keep | 分段，第 2 段看不到第 1 段 | PIQA gain +1.6 / +1.4，门槛 +3；BoolQ r128 +4.4 | dev 筛选，1 个 seed，复现中 |

V5 用的是官方 validation，其余是 dev，两边的数字不能直接比。

## 只有 V5 是确认级证据

V5 的判定规则在训练前锁定，官方 validation 只开封一次，现在已经用掉。它的 95% 区间只反映给定这三个训练 seed 时评估数据的不确定性，不含训练本身的随机性；LoRA 一侧的结论也只针对 rank 1 和这次的短校准学习率规程。其余结果都是筛选级，也就是 dev 上的初筛：RB1、E-0026、E-0025 各只有 1 个 seed，D5 有 3 个。PIQA、BoolQ 的 dev 每任务 1,000 条，只能分辨约 2 pp 以上的差别，区间也不含 seed 和选择带来的不确定性。
