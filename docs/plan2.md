# 残差旁路低维线性注意力适配：研究计划 v2.1

**主骨干：Qwen/Qwen3-0.6B-Base**  
**版本：2.1｜日期：2026-11-01（Asia/Tokyo）**  
**证据快照：UnbearableFate/LoopLora @ `a255e814740ff85ab3b4ec4836c88f0a8f802b2d`**  
**状态：已完成代码、报告与既有诊断的审查；本文新增实验尚未执行。**

> 本版修正 v2 的主线偏移：**块级残差旁路 + 低维因果 linear attention 是必须独立验证的第一条主线；循环与 FFN concat 是分别检验的增量。** 新增首轮 R1 真实旁路实验，固定后续主线的块输出接口与 KernelMatrix。旧 LoopLora 的 memory-only 诊断作为并行机制分支，不能替代或阻断保留原 attention 上下文的实用验证。本文取代 v2 的实验选择和预算安排；已有历史结果及其边界保持不变。

## 阅读导航与证据说明

- 第 0–2 节：本次改动、历史结果及其适用边界。
- 第 3–5 节：统一数学定义、关键恒等式、探索分支图。
- **第 6A 节 R1：原始残差旁路低维 linear attention 的首轮八组验证。**
- **第 11 节 V5：残差旁路、循环、concat 的九组独立确认。**
- 第 6–10 节：实现检查、可选接口筛选与并行 memory-only 机制研究。
- 第 12–16 节：条件扩展、数据与统计、预算、参数、实现与交付要求。
- 第 17 节：第一批工作与结果到研究主张的对应关系。
- 第 18 节：固定版本证据和原始参考资料。

本文使用 **“已观察”** 表示仓库报告中的结果，**“推导”** 表示可由给定公式得到的关系，**“提议”** 表示尚未执行的新实验。报告中的旧研究名称 P1L、SQ2、LSQT 等保持原义；本版保留 V0–V6 阶段号，新增 R1 表示原始残差旁路的直接筛选；它们均不等于旧研究或 SLOT-LOOP-v1 的 S0–S3。

用户提供的共享页 `https://chatgpt.com/s/t_6ae3091abbf481919d134f042045d961` 在v2审计时未能直接取得正文。此前已读取《LoopLoRA：实验诊断与下一阶段执行计划》全文 [D00]，并用仓库原报告核对其主要判断。v2.1沿用该固定证据快照，不声称取得了共享页正文或仓库后续结果。该诊断提出的显式槽位方案仍是**待验证建议**。本文未访问HPC训练权重、重新执行模型评估或向仓库提交修改。

---

## 0. 本次修正：把原始残差旁路恢复为主线

### 0.1 v2 的问题具体在哪里

用户最早提出的核心结构是：在完整 Transformer 块旁边放置一个低维线性注意力分支，必要时在该低维分支内循环，然后回写块输出。

$$Y=\mathcal B(X)+Z(X_{\le t})P_R.$$

这里 \(\mathcal B\) 已包含原 attention、原 FFN 及其残差。v2 的 `F_out=H+F(U)+ZP` 在形式上保留了它，但没有保证真正执行这个结构的独立实验：

| v2 位置 | 实际安排 | 为什么不够 |
| --- | --- | --- |
| V1-01 | `F_out` 搭配 \(Z=\operatorname{SiLU}(UA)\) | 仅有逐 token 瓶颈，没有 linear attention 的历史读写 |
| V1 的接口选择 | 从 `F_out/F_pre` 选一个 F* | 块输出旁路可能在无记忆 proxy 阶段被淘汰 |
| V2–V4 | 主要研究 memory-only | 丢弃支持文本的主干 KV，不能代替保留原上下文的适配验证 |
| V5 的配置继承 | 继承选中的接口、backend | 可能变成 FFN 输入加法或 SoftmaxSlot，原始残差 linear attention 不再出现 |

因此，本次不是只改名称，而是补上研究对象和必做实验之间缺失的对应关系。旧 LoopLora 的 o_proj 路径也不能直接代替块输出旁路：它会改变当前块 FFN 的输入，而本主线在当前块 FFN 输出之后相加。[E13][E14]

### 0.2 三个问题分别验证

1. **残差旁路是否有效？** 固定完整原块，添加真正的低维因果 KernelMatrix，先做 S=1。
2. **循环是否额外有效？** 在同一个残差旁路里比较 S=1/2/4，排除幅度、逐 token 深度、参数及运算量解释。
3. **concat 是否改善写回？** 低维状态与读算子保持同源，单独比较块输出相加与原 FFN gate/up 条件融合。

“F 线”在本文始终表示 **context-retained 协议**；“R_res”表示**块输出残差接口**，两者不要混为同一个标签。“M 线”表示 memory-only 机制协议。SoftmaxSlot 是读算子对照；即使表现更好，也不能替换 KernelMatrix 后仍声称验证了原始 linear attention 设计。

### 0.3 去哪里看原始设计的验证

| 核心问题 | 第一轮直接实验 | 后续独立确认 |
| --- | --- | --- |
| 残差旁路低维 linear attention 是否有用 | **R1-02：R_res + KernelMatrix + S1**；对比 R1-01、R1-07、Base | **V5-01** 对残差预算LoRA、Base确认实用价值；S1历史贡献仅有R1-T0初筛与冻结干预，多读历史另由V5-08确认 |
| 循环是否有额外作用 | **R1-03/04：同旁路 S2/S4**；R1-05/06 为尺度控制 | **V5-02 对 V5-01、V5-08**；固定 query、总步长与成本对照共同解释 |
| 是否使用侧支自己的跨 token 历史 | R1-02 对 R1-01，加仅改变侧支历史的冻结干预 | **V5-02 对 V5-08**，保持原 attention KV 不变 |
| concat 是否优于原残差写回 | **R1-08 对 R1-02**，同一个 KernelMatrix、同一个 S1 | **V5-01/02/03/04 四格**；V5-06/07 控制参数预算 |
| 低维读写为何成功或失败 | V0 原资产诊断；V2 读算子/特征析因 | V3/V4 的 M 线机制研究；不替代上述 F 线实验 |
| 原始 raw QKV 与 activation QKV 的区别 | 第 3.3 节定义，按第 12.1 节触发等预算消融 | 同一 R_res、KernelMatrix、S、rank、层位置下确认 |

固定主线：`context_retained + block_output_residual + kernel_matrix`。F_pre、attention-residual、SoftmaxSlot、concat 都是明确的比较分支，不得通过 proxy 排名将这三个主线字段替换掉。

**先执行 V0 + R1，训练上限仍为 18M 非 padding 输入 tokens；不需要先通过 V1 或 M 线 gate。** 完整条件路线核心 455M、储备 25M，总规划上限仍为 480M。新实验尚未执行。

---

## 1. 已有实验究竟说明了什么

### 1.1 证据账本

下表的 pp 表示百分点。不同任务、信息访问协议、数据区间或训练预算的数值不能串成同一条性能曲线。

| 研究与条件 | 已观察结果 | 对本计划的提示 | 不能推出的结论 |
| --- | --- | --- | --- |
| P1-L：0.6B、单层 14 的 o_proj、oracle memory、r128、3 seeds | hops 2–3：共享四读 M1 为 26.91%，单读 B4 为 5.36%；差 +21.55 pp，97.5% CI [20.38,22.76] | 保留“查询随循环变化”的主假设 | 新 QKV + concat、learned writer 或真实任务已经有效 |
| P5L：1.7B、同类 oracle 条件、3 seeds | 31.96% 对 6.17%；+25.78 pp，97.5% CI [24.53,27.03] | 有第二模型规模的同协议支持 | 放大 backbone 能解决写入与读取问题 |
| P1-L：r32 | hop1：B4/B8/M1 约 0.07/0.07/0.15%，无记忆 B3 为 0.03%；记忆使用 gate 失败 | 先校准接口可学性，不能在地板上解释循环差异 | r32 理论上不能工作，或多层 r32 也必然失败 |
| B8W：复制成功的 shared 解后解除共享，继续 2,000 步 | untied 32.40%，shared 31.50%；差 +0.90 pp，95% CI [0.09,1.71]；97.5% CI [-0.04,1.81]；通过非劣检查 | 比较共享时控制初始函数与续训预算 | 非共享已被证明更优，或共享是表达能力必要条件 |
| RFX：旧 R-r128，冻结 checkpoint 的析因诊断 | 原读取33.8%；校准scalar gain后：ridge/raw/fit 38.8%、delta/true/fit 39.9%、ridge/true/fit 84.2%；clean value84.4% | 旧条件存在存储与 query 的联合瓶颈；替换条件还需控制输出尺度 | 只改 query 或只改融合就足够；oracle 干预可部署 |
| DPFP：单 seed、T1、E16 | D128-3 为 60.8%；E8 另行重训 73.0%；均未过 80% 门槛 | 特征几何与负载值得诊断 | E8 是同一个 E16 模型的直接泛化结果 |
| DPFP-X：单 seed、同类 learned-writer 任务 | r256 为 66.2%；1.7B/r128 为 59.3%；均未通过旧 gate | 无诊断扩容优先级下降 | 更长训练无效；两条曲线末段仍在上升 |
| SQ2：同父 checkpoint，6,000 步继续训练，单 seed | delta 65.4%；delta+coef 63.0%；ridge 72.3%；ridge+coef 69.5% | 首先做读算子与训练收益的分解；辅助 loss 需要直接证据 | 组合方案通过了原注册规则，或所有辅助监督都无用 |
| LSQT：线性特征下从头训练 LSQ | 最佳 ridge 46.1%，读误差 0.512；处于原注册中间区间 | 不能把“从头 ridge”当未经试验的通用解决方案 | 整条 LSQ 路线已被理论否定；后续 U 对照已有结果 |
| P1L-C：读取位置与中间状态干预 | M1 只保留最后问题位置读取，hops1–3 约 33.64%→3.74%；z1 bridge patch shift +0.39 pp，95% CI [-0.13,0.78] | 控制所有读位置；可解码状态不等于因果桥 | 早位置读取是泄漏；每次循环已对应一跳 |
| CTC：学习解析器 + 固定 oracle code，单 seed | hop1 50.7%；hops2–3 多读差 +24.05 pp；候选被标记 | 可作为输入表示到 reader 的正对照 | 连续 learned key/value writer 已成功；开放词表泛化已验证 |
| LSR：oracle reader 的停止规则 | R2 改善长跳，但 0.6B hop2 下降 10.83 pp；平均读 5.60 次，需 6.60 次上层前向 | 停止方案必须同时报告短跳损失与完整计算路径 | 动态停止免费或已带来端到端加速 |

证据：[E01]–[E12]。P1-L 与 P5L 的区间条件于已训练的三个 seeds、指定 oracle 编码与布局，不覆盖所有 writer 初始化和新数据机制。SQ2、DPFP、DPFP-X、LSQT 的主要结果是一个训练 seed 上的验证筛选结果。

### 1.2 两个必须保留的细节

**SQ2 不能与历史父模型作不受控归因。** 同一组 D128-3 权重，零步换成 ridge 时已从 60.8% 到 68.7%；继续训练后到 72.3%。与此同时，delta 对照继续训练后也到 65.4%。因此，新算子的比较必须同时记录 step-0 与训练后差异。SQ2 原主对比是 ridge+coef 对 delta：+4.1 pp、读误差比 0.554，未达到其 +10 pp 与误差减半要求。ridge-only 是该表中的最佳单元，不会改写原 verdict。[E06]

**B8W 的主要结论是保留解。** 不同置信水平的区间不矛盾；本计划不把 95% 下的名义正差改写成注册的优越性结论。shared 继续训练本身也增加约 5.09 pp，说明“多训练”是需要匹配的因素。[E03]

### 1.3 不再作为默认失败解释的事项

仓库当前实现已关闭训练期 dropout 随机性，适配参数和 memory 使用 FP32，并有零出口初始化后的两步梯度启动、冻结主干输入 Jacobian 与数值一致性检查。此前确实修复过跨进程 GPU kernel history 导致的 BF16 support-state 不一致，但不能将其继续当作所有负结果的已确认原因。[E13]–[E17][E21][E22]

SQ2 中 clip=1 在所有**记录到的步**上生效，是事实；它是否抑制了辅助损失，需要逐项梯度证据。不能据此声称 AdamW 的有效学习率严格缩小为原来的十分之一。[E06]

---

## 2. 旧方法与当前方法的迁移边界

### 2.1 旧 LoopLora 的实际结构

用列向量表达旧方法，省略若干 feature-map 分支：

$$
u=Ax,\qquad z^{(0)}=u,
$$

$$
z^{(s+1)}=u+\rho\,\sigma\left(Rz^{(s)}+UM\psi(Qz^{(s)})\right),
\qquad y=Wx+\alpha Bz^{(S)}.
$$

默认旧写入使用 decay-first delta rule：

$$
\bar M=\lambda M,\qquad
M^+=\bar M+\beta(v-\bar M k)k^\top.
$$

一次问题读取的内循环中 M 固定。原 `base(x)` 只执行一次，主实验插在单个 `o_proj`。后续 DPFP、ridge 与 sweep 是明确不同的分支。[E13][E14][E18]

### 2.2 六个不能同时改变后再作单一归因的轴

| 轴 | 旧主要条件 | 当前候选 |
| --- | --- | --- |
| 输入 | o_proj 前的 attention 输出，另有 residual 变体 | 原 Q/K/V 的指定阶段，或 Norm 后隐藏态 |
| 写读 | delta / DPFP / ridge | 正核外积 + 分母归一化，或 softmax 槽位 |
| 循环 | 每轮重新锚定初始 u | 累加式 z 更新；另保留同核锚定对照 |
| readout | 完整 zS，含初始状态通路 | 循环增量 zS−z0；完整状态单独消融 |
| 融合 | o_proj 输出加法，经过原 residual/Norm/FFN | **主线为块输出残差R_res**；FFN gate/up条件项为扩展比较 |
| 布局与协议 | 单层；memory-only 为主要机制条件 | 少层/多层；context-retained 为实用条件 |

**同一个 r 不代表相近预算。** 旧单层 r32 的 stored adapter 约 103,426 参数；原 v1 的 14 层 r32 C_dual 为 4,616,192，约 44.6 倍。旧 r128 的 B4/M1 stored adapter 均 475,138，而该 reader 阶段可训练参数均为 442,368。必须区分存储参数与实际训练参数。[E01][E13]

### 2.3 信息访问协议分开登记

**M 线：memory-only 机制线。** 支持文本先产生可写特征；问答阶段丢弃支持文本的原主干 KV cache，仅保留声明的低维 memory。问题与答案不写入该 memory。每个问题从同一支持状态独立开始。

**F 线：context-retained 适配线。** 原 attention 按正常因果方式看到完整前缀；侧分支额外保留自己的状态。这里的任务增益可能来自融合、额外非线性、当前 token 通路或历史记忆，需要逐项诊断。

M 线的高分与 F 线的高分不是同一个指标。F 线没有显著侧支历史贡献时，仍可得到有用的残差或 FFN 适配器，但不能称为“侧支记忆带来的收益”。旧单层 support 截断计算优化也不能直接用于新多层前向。[E15]

---

## 3. 当前核心方法的统一数学定义

以下全部采用**行向量**；t 是 token 位置，s 是同一个 token 内的循环步，ℓ 是原 Transformer 层。循环深度记 S，训练优化步数记 N_step，避免与旧报告的训练步数 S 混淆。

### 3.1 Qwen3-0.6B-Base 的固定结构

| 项目 | 官方值 |
| --- | ---: |
| 隐藏宽度 D / FFN 宽度 m / 层数 | 1024 / 3072 / 28 |
| Q heads / KV heads / head_dim | 16 / 8 / 128 |
| Q 投影总宽度 / K、V 总宽度 | 2048 / 1024、1024 |
| FFN / Q、K 归一化 | SwiGLU / 每头 QKNorm |
| 最大位置数 / RoPE theta | 32768 / 1000000 |
| attention bias / dropout | false / 0 |

来源：[R01][R02]。运行时锁定实际模型与 tokenizer revision；不得用非 Base 配置替代。由结构计算的唯一参数约 596,049,920，仍需实际加载后核对 tied embeddings，不能把计算值称为本轮测量。

### 3.2 原 Transformer 块

$$
N=\operatorname{Norm}_1(X),\quad
H=X+\operatorname{Attn}(N),\quad U=\operatorname{Norm}_2(H),
$$

$$
F(U)=\left[\operatorname{SiLU}(UW_g)\odot UW_u\right]W_d,
\qquad Y_{\rm base}=H+F(U).
$$

其中 Wg、Wu 为 D×m，Wd 为 m×D。原 attention 每层每次前向执行一次；循环只在新增低维分支中执行。

### 3.3 低维投影与用户提出的 activation 分支

主候选读取原 Q/K 在 **QKNorm 后、RoPE 前**的特征 Qb、Kb，Vb 为原 v_proj 输出；K/V 在 GQA repeat_kv 之前压缩：

$$
\bar Q=Q_bA_q,\quad \bar K=K_bA_k,\quad \bar V=V_bA_v.
$$

默认三路输出宽度均为 r：

$$
A_q\in\mathbb R^{2048\times r},\qquad
A_k,A_v\in\mathbb R^{1024\times r}.
$$

保留以下明确命名的特征变体，不能在实现中混用：

- `direct_hidden`：从 N 分别线性投影 Q/K/V，参数为 3Dr。
- `raw_qkv`：使用原投影、QKNorm 前的 NWq、NWk、NWv，再乘 A。
- `qknorm_pre_rope`：主候选。
- `activation_raw`：例如 SiLU(NWk)Ak，Q/V 同理；激活函数及位置必须写入配置。

线性重排 NWkAk=N(WkAk) 只对纯线性路径成立。引入 QKNorm 或 activation 后不能再这样合并。即使 activation(NWk) 非负，乘一个有正负系数的 Ak 后也可能为负；正核读取仍需显式定义最终的正特征映射。

### 3.4 归一化正核状态

低维归一化采用不带可训练仿射项的 RMSNorm：

$$\mathcal N_r(z)=\frac{z}{\sqrt{\frac1r\sum_j z_j^2+\epsilon_N}},\qquad\epsilon_N=10^{-6}.$$

正特征映射为：

$$
\phi(a)=\operatorname{ELU}(a)+1,
\quad k_t=\phi(\mathcal N_r(\bar k_t)),\quad v_t=\bar v_t.
$$

对允许写入的位置集合 \(\mathcal I_t\)，定义：

$$
M_t=\sum_{i\in\mathcal I_t}k_i^\top v_i\in\mathbb R^{r\times r},
\qquad c_t=\sum_{i\in\mathcal I_t}k_i^\top\in\mathbb R^{r\times1}.
$$

F 线的 inclusive 模式中，当前有效 token 先写一次，再读 \(i\le t\)；strict 模式改为 \(i<t\)，作为自作用诊断。M 线中 \(\mathcal I\) 是支持文本已完成的写入集合，问题阶段不再写。M 线首轮的写位置与读位置按第 8.1 节锁定，实际状态容量按写入项数计算，不能一律用事实数 E 替代。

同一 t 的各 s 步共享同一 M、c；不能每轮重复写当前 token。训练不允许让所有 t 共用全序列的 KᵀV，否则会泄漏未来信息。[R03]

### 3.5 最小累加循环

$$
z_t^{(0)}=\bar q_t,
\qquad \hat q_t^{(s)}=\phi\left(\mathcal N_r(z_t^{(s)})R_q\right),
$$

$$
a_{t,\rm raw}^{(s)}=\frac{\hat q_t^{(s)}M_t}{\hat q_t^{(s)}c_t+\epsilon},
\quad a_t^{(s)}=g\,a_{t,\rm raw}^{(s)},
\qquad z_t^{(s+1)}=z_t^{(s)}+\eta a_t^{(s)}R_o.
$$

Rq、Ro 都为 r×r，循环内共享，各原模型层暂不共享。首轮固定 eta=0.25、全局读增益g=1，二者不可训练；读取分母 epsilon=1e-6。两类backend都先产生a_raw，再统一乘g。后续改变步长或增益须登记为独立变体。主 readout：

$$Z_t=z_t^{(S)}-z_t^{(0)}.$$

这里的 Z 是**循环增量**，不是 delta-rule 写入。完整 state readout Z=zS 另设实验，因为它包含初始 query 的直接通路。

空记忆显式输出零；M、c 用 FP32 累积。若分母太小、状态溢出或存在无效写入，先修复实现，不以改变任务 gate 掩盖问题。

### 3.6 核心方法：完整 Transformer 块的残差旁路

记原块 \(\mathcal B(X)=H+F(U)\)，定义主线 **R_res**：

$$
\boxed{Y_R=\mathcal B(X)+ZP_R=H+F(U)+ZP_R},
\qquad P_R\in\mathbb R^{r\times D}.
$$

Z 使用第 3.3–3.5 节的真实因果线性注意力和低维循环，由原块输入 N 的 Q/K/V taps 产生。它不是 \(\operatorname{SiLU}(UA)\) 代理。原 attention 和原 FFN 各执行一次，低维状态只在侧支内循环；当前块的 FFN 不接收 Z。下一块正常接收修改后的 Y_R。

S=1 已经是完整的低维 linear attention 适配方法；S>1 才引入重复读取的额外假设。P_R 零初始化、分支内部非零初始化，保证初始函数与 Base 一致。冻结主干参数仍须允许梯度通过主干传到更早层的适配器。

若把用户原式中的 `trans(X)` 定义为不含外层恒等路径的块增量，则 \(\mathcal B(X)=X+\operatorname{trans}(X)\)，得到 \(Y_R=X+\operatorname{trans}(X)+ZP_R\)。若 `trans` 已是完整块，不再额外加一次 X。

```mermaid
flowchart TD
    X["块输入 X"] --> B["原 attention 与原 FFN"]
    X --> T["复用原 QKV 特征"]
    T --> K["低维因果 linear attention"]
    K --> L["低维共享循环 S 次"]
    L --> P["投影 P_R 回 D 维"]
    B --> A["块输出相加"]
    P --> A
    A --> Y["下一块输入 Y"]
```

图中的 QKV taps 复用原 attention 的计算，不额外执行一次原投影。linear attention 的写入每 token 只发生一次；循环是改变 query 后读取同一份状态。

“LoRA”在本研究中指低维参数高效适配的设计动机。一般情形下，这个带动态状态与非线性读取的分支不能静态合并成某个固定 \(\Delta W\)。局部加法 \(ZP_R\) 的输出矩阵秩最多为 r，不意味着整个因果 attention 权重矩阵秩最多为 r。

### 3.7 扩展方法：Concat 到原 FFN

保留原残差与 Norm，拼接的是 \([U,Z]\)，其作用可以不实际分配 concat 张量而等价实现为：

$$
Y=H+
\left[\operatorname{SiLU}(UW_g+ZC_g)
\odot(UW_u+ZC_u)\right]W_d,
$$

$$C_g,C_u\in\mathbb R^{r\times m}.$$

输入宽度 D→D+r，FFN 中间宽度 m 与输出 D 保持原值。原 Wg/Wu/Wd 冻结，Cg/Cu 零初始化，分支内部非零初始化，因此开始时与 Base 相同。首步只有出口获得梯度是预期；出口更新后，分支内部应获得梯度。

直接把 FFN 输入改成未加原残差的 `origin-attn(X)` 会改变预训练块语义。本计划以原 H/U 为锚点，落实用户的拼接想法。

---

## 4. 本次必须增加的数学检查与机制解释

### 4.1 最关键恒等式：同核槽位与矩阵状态完全等价

固定相同 key、value、query、mask 与 epsilon：

$$
\underbrace{\frac{q(\sum_i k_i^\top v_i)}{q(\sum_i k_i^\top)+\epsilon}}_{\text{KernelMatrix}}
=\underbrace{\sum_i\frac{qk_i^\top}{\sum_jqk_j^\top+\epsilon}v_i}_{\text{KernelSlot}}.
$$

**推导结论：** 对这个读取族，保存 M、c 没有引入额外近似；它是精确的聚合表示。整体槽位信息当然比单一矩阵多，但同核 read 无法使用那些额外信息。两种实现的输出及对应输入/参数梯度应在数值容差内相等。

因此，KernelSlot 只用于正确性与数值检查，不值得当成独立科学 arm 大规模训练。

### 4.2 SoftmaxSlot 真正改变了什么

另一种读算子是：

$$
w_i^{\rm soft}(q)=\frac{\exp(qk_i^\top/(s_0\tau))}{\sum_j\exp(qk_j^\top/(s_0\tau))},
\quad a_{\rm soft,raw}(q)=\sum_iw_i^{\rm soft}(q)v_i,
\quad a_{\rm soft}(q)=g\,a_{\rm soft,raw}(q).
$$

新分支固定s0=sqrt(dk)、tau=1、g=1；旧S-RE冻结诊断使用其原DPFP分数尺度，s0=1，单独校准，不复用温度值。SoftmaxSlot改变了权重分布、选择性和温度响应，并需要显式槽位。性能差应称为**读算子差异**；不能仅归因为“保存了 KV，消除了矩阵压缩”。第一轮固定相同的正核 q、k 特征来计算两类权重，不同时再换原始 signed 特征。

所有可见 key 都同乘一个正数，在 epsilon 可忽略时不会改变归一化线性读的相对权重。因此，简单放大 q 或 k 不是可靠的锐化方法。需要看目标 key 权重、权重熵、有效读取项数与实际输出，而不只看 key top-1。

若 softmax 槽位有效而正核矩阵无效，支持的是“可选择的低维 memory 侧支”，暂不支持“常数状态 linear-attention 侧支”。后续要压缩回矩阵是另一项研究。

### 4.3 旧 delta 的干扰例子不能直接替换新算子的证明

旧 delta 存储中，从M0=0开始，两个单位 key 的内积为 c，正交 value 为 v1、v2，单位更新率、无遗忘时：

$$M_2k_1=(1-c^2)v_1+cv_2.$$

即使真实 key 排第一，读出也可被错误 value 主导。这个例子帮助解释旧 RFX 类诊断，但当前 KernelMatrix 与同核槽位等价；其问题应分析为**特征区分度、权重混合与有限读函数族**，不能直接称为同一种 delta 覆写损伤。

### 4.4 Concat 能改善什么，不能保证什么

记 \(a=UW_g,b=UW_u,\delta a=ZC_g,\delta b=ZC_u\)。小扰动的一阶展开为：

$$
\Delta F\approx
\left[\operatorname{SiLU}'(a)\odot b\odot\delta a
+\operatorname{SiLU}(a)\odot\delta b\right]W_d.
$$

因此同一个 Z 在不同 U 下可产生不同的写回方向；这是测试 FFN 条件融合的合理动机。它不保证恢复已从可访问信息中消失的事实。在 M 线中，如果两段支持文本产生完全相同的 M/c、问题与主干可访问状态相同，任何确定性的下游 concat 都无法区分它们。

F 线仍有原 attention 的上下文，能够借助 U 补充信息；这正是两条协议必须分开的原因。不能用 F 线上的恢复来证明侧支 memory 保留了全部信息。

### 4.5 必须保留的退化与等价关系

| 关系 | 精确定义与含义 |
| --- | --- |
| 线性 concat | [O,Z]W=OW1+ZW2；仅拼接后线性投影不是独立机制 |
| C_dual 与 F_pre | 若 Cg=PWg、Cu=PWu，则 C_dual=H+F(U+ZP)；训练时必须由同一个 P 构造约束 |
| 线性低维状态 | Z=UA 时，Cg/Cu 扩展退化为 gate/up 共享输入因子的 LoRA |
| raw QKV | NWkAk=N(WkAk)；不适用于 QKNorm/非线性之后 |
| 固定 query 的最小循环 | 固定 a、无额外 FFN/output norm 时，Z=S eta a Ro；这是精确幅度对照 |
| 当前 token 单槽位 | epsilon 可忽略时 a≈v；Q/K 近乎抵消，不能作为有效参数完全相同的训练基线 |
| 低秩的范围 | 加法 ZP 的表示秩受 r 限制；非线性 FFN 差分不必在全部样本上保持该全局秩；因果 attention 矩阵也不必 rank≤r |

旧 B5 固定 query 后仍有 Rz 非线性递归，不满足上述精确 S 倍关系。新对照必须使用公式定义，不能只沿用 B5 名字。[E13]

### 4.6 锚定更新与累加更新

在**同一读函数** \(f(z;M)=a(z;M)R_o\) 下比较：

$$
\text{累加： }z_{s+1}=z_s+\eta f(z_s;M),
\qquad
\text{锚定： }z_{s+1}=z_0+\eta f(z_s;M).
$$

S=1 时两者相同；S>1 才产生差别，参数量也相同。新锚定对照是受旧方法启发的严格消融，不等于完整复刻旧 M1 的 Rz 与 ReLU。

局部 Jacobian 分别为 \(I+\eta J_f\) 与 \(\eta J_f\)。旧 contraction 条件不能直接用来保证累加更新收敛；数值稳定也不等于语义正确。应同时观察状态范数、扰动放大与各跳准确率。

---

## 5. 更新后的探索分支图

### 5.1 全局路线：残差主线、融合扩展、读取诊断分开

```mermaid
flowchart TD
    V0["V0 实现与函数保持检查"] --> R["R1 真实残差 linear attention"]
    R --> S["同一旁路 S1、S2、S4 与控制"]
    S --> F["V5 残差主线独立确认"]
    R --> C["V1 可选接口筛选与 concat pilot"]
    C --> F
    V0 --> M["V2 读取诊断：memory-only"]
    M --> D{"读取信号成立？"}
    D -->|"是"| L["V3、V4 机制与循环确认"]
    D -->|"否"| STOP["限制 M 线主张；一次限额探针"]
    L -. "解释与诊断" .-> F
    F --> G{"收益与成本可接受？"}
    G -->|"是"| E["V6 多层与长上下文"]
    G -->|"否"| END["保留有效单读或结束对应分支"]
```

R1 不依赖旧 checkpoint、V1 proxy 成功或 M 线通过 gate。M 线失败只关闭对应的 M 线扩展。R1 的残差循环无收益，也不自动否定 concat 与循环的交互；可在储备内做 C-S1/C-S4 配对 pilot，或在 V5 保留预登记固定 S4 的交互检验。已失败的分支按登记规则停止，不为保留方法名称无限追加搜索。

### 5.2 按错误位置选择动作

```mermaid
flowchart TD
    A["任务没有提升"] --> B{"正对照能工作？"}
    B -->|"不能"| C["检查实现、输出接口与数据难度"]
    B -->|"能"| D{"真实读出接近目标值？"}
    D -->|"不能"| E["测 key 权重、混合干扰和特征对齐"]
    E --> F["同内容读算子替换；固定内容不扩容"]
    D -->|"能"| G{"任务仍读不出答案？"}
    G -->|"是"| H["同 Z 比较加法与 FFN 条件融合"]
    G -->|"否"| I["检查多次读取的独立收益"]
    I --> J{"失败来自终止前还是终止后？"}
    J -->|"终止前"| K["查询更新和角色对齐"]
    J -->|"终止后"| L["终态保持与低成本停止"]
```

### 5.3 各路线的默认配置

| 用途 | 初始布局 | 宽度 | 协议 | 固定结构与地位 |
| --- | --- | --- | --- | --- |
| 旧资产诊断 | 旧 index14 o_proj | S-RE key768/value128 | memory-only | 兼容性与错误定位；不是新旁路验证 |
| **R1 原始主线** | **[6,13,20,27] 四层** | **r32** | **context-retained** | **R_res + KernelMatrix 必做；S1/2/4** |
| V1 可选融合筛选 | 同四层 | r32 tokenwise proxy | context-retained | 仅选择 C* 或辅助接口；不得淘汰 R_res |
| V2 新记忆诊断 | 单层 index14 | r128 | memory-only | 特征 × 读算子 × R_res/C* |
| V3–V4 机制 | 单层 index14 | r128 | memory-only | 固定 R_res；V3 固定 KernelMatrix，V4 比较两个算子 |
| **V5 实用确认** | **同四层** | **r32 主候选** | **context-retained** | **固定 KernelMatrix；R_res 与 C* 并列，残差有专属 LoRA/T0 对照** |
| V6 扩展 | 14层 [1,3,...,27] | 已选宽度 | 单独登记 | 不能用更多层的结果替代四层主线比较 |

F 线深度记为 \(S_F^*\)，M 线深度记为 \(S_M^*\)，分别由各自 train/dev 决定。M 线的优胜 backend、深度与门槛不能直接移植成 F 线结论。本文所有 layer14 均指 0-based index14。

单层 r128 与四层 r32 是不同研究条件，不能相减当作 rank 消融。旧 r32 的 memory-only 地板也不是四层 context-retained 残差旁路必然失败的证据。

---

## 6. V0：先建立证据与实现基础

### 6.1 资产与历史复现

首先建立 `evidence_manifest.json`，区分仓库源码 commit、每个旧科学运行 commit、checkpoint 文件摘要和新实现 commit。源码快照相同不表示模型文件已经验证。

若原 HPC 资产可用，优先取 S-RE：

- 旧目录：`/work/xg24i002/x10041/LoopLora/runs/p1l/sq2/arms/S-RE/`。
- 实际 loader：`looplora.llm.sq2.load_inference`；不要送进不兼容的普通 writer loader。
- 旧验证复现参考：723/1,000；NLL 1.642127275466919。
- 必须保留实际 key 768 / value 128、DPFP feature map、读位置、归一化 buffers 与 memory-only 协议。
- 推理 `writer.pt` 与含优化器/训练期 value head 的 `checkpoint.pt` 是不同资产，不能互相替代。[D00][E06][E19]

复现旧 1,000 个验证 episode 只证明兼容性，不产生新确认结果。若 checkpoint 不可用，只完成源码与文档审计，新方案仍可从 Base 实现；不得伪称旧指标已经本地复现。

### 6.2 冻结 checkpoint 的低成本读算子诊断

在 S-RE 原有特征上，比较原生 ridge read、SoftSlot、HardSlot，保持 writer 与低维 reader 冻结。HardSlot 用模型分数 argmax，不使用真实 fact index。另加 true-key、clean-value 等明确标记的 oracle 诊断。

所有正常读取位置同时替换，不只改最终问题位置。报告未校准结果与校准结果；温度和增益只能在独立 calibration 上按 NLL 选择。建议固定网格 tau∈{0.03,0.1,0.3,1.0}、g∈{0.5,1,2}，不作逐样本拟合。

这一步回答“既有 learned K/V 在另一种 read 下能否更有效地使用”，不能证明训练后极限。旧 ridge read 与新正核 KernelMatrix 不是同一个公式；不要把第 4.1 节的恒等式套到它们之间。

此冻结诊断沿用 [D00] 的第一步，但其后续 SLOT-LOOP-v1 续训**不自动纳入本计划**。若要执行，使用独立 release 和预算，不与下文的新 QKV 分支数据混合。

### 6.3 新实现必须通过的针对性检查

| 检查 | 通过条件与目的 |
| --- | --- |
| 功能保持 | R_res 的 P_R 或 concat 的新出口全零时，与未修改 Base 一致到预先固定数值容差；逐接口检查 |
| 残差拓扑 | R_res 的 ZP_R 只在当前块原 FFN 完成后相加；给定相同块输入X，本块FFN输入H/U与原块路径相同；原块不重复执行 |
| 两步梯度启动 | 用至少两个不同有效写入项；首步出口获得梯度，更新后Q/K/V压缩与循环参数获得梯度；不拿单槽Q/K抵消作失败判据 |
| 冻结输入 Jacobian | 下层适配器能通过上层冻结主干回传梯度；主干参数不更新 |
| 同核恒等 | KernelMatrix 与 KernelSlot 的输出、Q/K/V 梯度一致 |
| 约束融合恒等 | Cg=PWg、Cu=PWu 的实现与 F_pre 输出及对同一 P 的梯度一致 |
| 因果 | 改动后缀不改变前缀输出；inclusive/strict 严格按配置执行 |
| cache | 一次 prefill、分块 prefill 与逐 token decode 一致 |
| reset/padding | M、c、slots 同时清空或重排；padding 不写、不读、不衰减 |
| 所有权 | batch 重排、beam reorder、多个 query 不串状态 |
| read/write 计数 | 同 token 内只写一次；S 变化不增加固定记忆的写次数 |
| M 线信息访问 | support KV 不进入 query forward；答案与 true-hop 不进入普通 read/stop |
| 初始化恒等对照 | shared复制成untied后，比较每步state、read与Z；或使用共同非零出口fixture比较logits。零出口logits天然等于Base，不能单独验证复制正确 |
| 导出重载 | 恢复算子、scope、step、normalization 后，预测一致 |

FP32 小张量参考可使用严格误差阈值；GPU BF16 容差根据同一 Base 的重复测量预先锁定，不能在看见失败结果后任意放宽。检查范围围绕这些具体风险，不要求重跑与改动无关的所有实验。

### 6.4 训练校准预算

总上限 **2M 输入 tokens**：5 个家族 × 2 个学习率 × 0.2M。家族为 LoRA、tokenwise 残差、concat、残差 KernelMatrix、SoftSlot。残差 KernelMatrix 的 F 线校准在 R1 前释放；SoftSlot 随 V2 释放。concat 首轮采用真实 KernelMatrix 的 S1 接口；若后来切到 proxy 或 M 协议，不能假定同一 LR 已完成公平校准，必要补校准计入储备。不要求 V0 一次跑完。

建议 LoRA/逐 token 接口先试 {1e-4,3e-4}，记忆家族先试 {3e-4,3e-3}。后一个范围参考旧分支的学习难度，但不是最优值的保证。各家族获得相同数量的候选与 token 预算；任何额外调参进入第 14 节储备。只依据 calibration/dev 锁定配方，不看确认集。

---

## 6A. R1：原始残差旁路低维 linear attention 的直接验证

**这是本计划第一批科学训练，必须先于或独立于 V1–V4 执行。** 它直接回答原始设计的问题，不用逐 token proxy 代替 linear attention。

### R1.1 固定主候选

- Backbone：Qwen3-0.6B-Base；冻结原参数，保留主干梯度传播。
- 插入层：0-based `[6,13,20,27]`；每层 r=32，各层参数独立。
- 主结构：\(Y=\mathcal B(X)+ZP_R\)，`fusion=block_output_residual`。
- QKV：默认 `qknorm_pre_rope`，V 为原 v_proj，K/V 在 GQA repeat 之前；raw/activation 变体另行登记。
- 读取：`kernel_matrix`，第 3.4 节正核归一化状态；主候选禁止自动换成 SoftmaxSlot。
- 协议：`context_retained`；原 attention 保留完整合法前缀，侧支对所有有效 token 因果写入；inclusive，先写一次，再读取 S 次。
- 低维循环：第 3.5 节；\(\eta=0.25,g=1\)，跨步共享，Z 为循环增量；仅 R1-06 改步长。
- 数据：固定 PIQA/BoolQ train/dev 混合或预先锁定的非天花板 full-context 切片；所有 arm 同一输入、训练顺序、损失与可访问信息。默认序列上限512，监督答案，不写入任何未来 token。
- 每组最多 2M 输入 token exposures；真实任务最多3 epochs，达到任一上限即停。每组同一 seed0，确认集此时封存。

### R1.2 八组矩阵：合计最多16M

| ID | 完整结构 | 对应问题 |
| --- | --- | --- |
| R1-01 | **R_res + 同源逐 token 核 T0，S1** | 无侧支历史、同有效参数的适配对照 |
| **R1-02** | **R_res + KernelMatrix，S1** | **原始残差旁路低维 linear attention 本身是否有效** |
| **R1-03** | **R_res + KernelMatrix，S2** | **增加一次低维读取是否有信号** |
| **R1-04** | **R_res + KernelMatrix，S4** | **更多低维循环是否有效** |
| R1-05 | R_res + KernelMatrix，S4，query 固定为第1步 | 排除仅重复同一读取、放大输出的解释 |
| R1-06 | R_res + KernelMatrix，S4，\(\eta=0.25/4\) | 固定总步长，检查累积幅度 |
| R1-07 | 同四层 all-7 LoRA，rank7 | 接近残差旁路参数预算的标准适配基线 |
| R1-08 | C_dual + 同一个 KernelMatrix，S1 | 在真实注意力状态上，初查 concat 对写回的作用 |

Base 只评估。R1-02/03/04 的可训练参数均为663,552；S只改变读取计算。R1-07为630,784，少约4.94%，层位置相同；R1-08为1,318,912，并非等参，不能直接由它的分数高低宣称参数效率。以上为第15节条件下的结构计数，实际加载后仍核对参数清单。

T0 保留 Q/K/V 压缩及 Rq/Ro 的计算作用，但取消侧支历史：

$$
a_{{\rm tok},t}^{(s)}=\operatorname{SiLU}\left(\mathcal N_r(z_t^{(s)})R_q+\mathcal N_r(\bar k_t)\right)\odot\bar v_t,
\qquad z_t^{(s+1)}=z_t^{(s)}+\eta a_{{\rm tok},t}^{(s)}R_o.
$$

原 QKV 特征仍含上游上下文，因此 T0 不是“整个模型无历史”。它也改变了读函数，须结合第11.3节仅改变侧支历史的冻结干预解释。单个当前 token 的归一化 attention 会令 Q/K 近乎抵消，不拿该退化式冒充同有效参数 T0。

### R1.3 必须输出的结果与进入下一步的规则

每组记录任务宏平均/逐任务准确率、答案NLL、每层新增参数、读取次数、峰值训练显存、prefill/decode延迟及新增状态字节。分别报告“模型无适配”Base、“标准低秩参数适配”LoRA、“无侧支历史”T0和真实 KernelMatrix。

先比较 R1-02 对 R1-01/R1-07/Base，回答**单读残差旁路**；再比较 R1-03/04 对 R1-02/05/06，回答**循环的初步信号**。R1的 T0 只有S1，不能排除 S4 的额外逐 token 深度解释；正式归因还需 V5-08 同深度 T0，以及按诊断登记的同成本/宽度控制。

筛选只依赖 train/dev：记录全部候选和不确定性，选择 F 线多读候选 \(S_F^*\in\{2,4\}\)，不能用 M 线的 \(S_M^*\) 替代。若残差多读无信号，仍保留有效 S1 方法；concat 多读可选择一项预登记策略：C-S1/C-S4 两臂各1M的配对pilot（从储备支出），或在V5以固定S4作交互检验。后一情形把深度称为**待检验的S4**，不称已选最优深度。

若所有 R1 组都失败，先看 LoRA/T0 正对照、零出口后的梯度启动、数据天花板/地板。只有在实现和可学性成立后，才允许从第14.3节的8M容量/特征储备中释放一次同布局 r64/128 探针或 raw/activation 特征比较；不得无上限扩容。四层r32结果只界定该配置，不能推广成所有低维残差旁路都无效。

**M线失败不阻断R1或有信号的V5；concat失败不阻断残差线；残差循环失败也不否定残差S1。**

---

## 7. V1：可选的 concat 接口筛选，不代替残差主线

### 7.1 配置与数据

本阶段在 R1 之后按需执行，预算上限保留16M；它只帮助缩小接口搜索，不能决定 R_res 是否获得真实 linear attention 验证。F 线，4 层 `[6,13,20,27]`，r32，不使用额外 memory，也不循环。低维 proxy：

$$Z=\operatorname{SiLU}(UA),\qquad A\in\mathbb R^{D\times r}.$$

线性退化组用 Z=UA。这里 U 已含原主干上下文，“tokenwise”只指侧支没有自己的跨 token memory。

主筛选使用第 13 节的真实任务训练/dev 或预先锁定的非天花板 context-retained 切片。旧简单 T2 的保留上下文对照已接近 99.9%，只适合作为流水线检查，不作为接口排名主指标。[E01]

### 7.2 六种接口的定义

设 P 为 r×D，g=UWg，u=UWu：

| 接口 | 输出 |
| --- | --- |
| R_res（旧名F_out） | H+F(U)+ZP；主线的完整版本使用真实KernelMatrix状态 |
| F_pre | H+F(U+ZP)，不在 ZP 之后额外做一次 Norm |
| F_attnres | H'=H+ZP；Y=H'+F(Norm2(H')) |
| C_gate | H+[SiLU(g+ZCg)⊙u]Wd |
| C_up | H+[SiLU(g)⊙(u+ZCu)]Wd |
| C_dual | H+[SiLU(g+ZCg)⊙(u+ZCu)]Wd |

F_attnres 保留为单层桥接/条件消融，不进入首轮全部八组，以控制搜索规模。它更接近旧 o_proj 加法路径，但仍需对齐 Z 的来源和缩放。

### 7.3 可选八组：每组最多 2M，合计 16M

| ID | 条件 | 目的 |
| --- | --- | --- |
| V1-01 | proxy + R_res | 仅筛选逐token接口；不能替代R1-02 |
| V1-02 | proxy + F_pre | 信息提前进入当前 FFN |
| V1-03 | proxy + C_gate | 只调制门 |
| V1-04 | proxy + C_up | 只调制值通路 |
| V1-05 | proxy + C_dual | 独立双通路条件融合 |
| V1-06 | 线性 Z=UA + C_dual | 共享 A 的 gate/up LoRA 退化基线 |
| V1-07 | 相同4层 all-7 LoRA，rank10 | 与 proxy C_dual 约0.92M参数接近 |
| V1-08 | 全28层 all-7 LoRA，rank2 | 约1.26M的更强全层参考，明确额外参数 |

Base 只做评估，不计训练 run。all-7 指 q/k/v/o/gate/up/down。V1 的不同接口不是严格等参；本阶段筛位置与信号，不从原始高低宣称参数效率。

### 7.4 决策

- R_res 是固定的主线接口，不参与淘汰式筛选。F_pre 仅作辅助位置对照；它即使在proxy上获胜，也不能替换 R1、V3、V4、V5 中的 R_res。本文不再使用含混的 F*。
- concat 可从 C_gate/C_up/C_dual 中选择最简单的有效者 C*；分数近似时优先少参数。选择属于proxy假设，仍须以真实KernelMatrix的接口交互验证；跳过V1时默认C*=C_dual。
- 即使 proxy concat 没赢，V2 仍保留 C_dual 作为预登记的 memory 交互探针。不能用无记忆 proxy 证明有记忆 concat 无效。
- 若所有训练组近随机，先做实现与任务可学性检查；若都近天花板，不据此排行。改变任务难度只能依据预先定义的 pilot 规则，并对所有 arm 使用同一新设置。
- 若连线性退化组都不能学习，不立即增加 S、r 或层数。

---

## 8. V2：并行 M 线诊断，拆分表示、读算子和融合

### 8.1 新 M 线协议

这是一项**从 Base 构建的新研究**，不是旧 S-RE checkpoint 的直接续训。它解释受限信息访问下的读写瓶颈；不先于 R1，也不是 F 线残差旁路的准入条件。

- 单层 0-based index14；r128；S=1。
- T1 单跳，起始 E=16 个支持事实，4 种关系，固定公开渲染模板。
- 首轮 `write_scope=visible_fact_delimiter`：每条事实在可见终止分隔符的唯一位置写一次。该位置由输入文本格式确定，不使用真实答案或选中的 fact index。所有 arm 获得相同边界；这是结构化支持输入条件，应单独声明。
- `read_scope=all_valid_question`：训练与评估均在全部有效问题位置读取。单 token 答案任务不在答案处写入，query memory 只读。
- 支持阶段 `write_only`，不把侧支读出注入支持前向；K/V 从该层原始特征产生。支持主干 KV 在问题阶段被丢弃。
- 每个问题从同一只读 M/c 或 slots 快照开始，batch 内按 episode owner 索引。
- 只用答案 CE 训练主线；address/value/路径辅助真值不进入这八组。oracle 干预仅作离线诊断。

首轮 E=16 对应 16 次有效写入，是因为显式规定了写位置。后续 `all_support_tokens` 可能有数百次写入，必须重新按实际 n_write 计算负载与状态；不能仍按 E=16 标注容量。

### 8.2 八格析因矩阵：2×2×2

| 因子 | 值1 | 值2 |
| --- | --- | --- |
| 特征 | direct_hidden | qknorm_pre_rope |
| 读算子 | KernelMatrix | SoftmaxSlot |
| 融合 | **固定R_res块输出残差** | C*；若跳过V1或其无信号则用C_dual交互探针 |

共 **8 组 × 4M = 32M 输入 tokens**，先单 seed 筛选。所有组的 facts、query、训练顺序、写读位置与任务标签相同。两类算子的 q/k/v 来源保持一致；direct_hidden 与 QKV 的参数数不同，要实际报告。

默认八格固定g=1，SoftmaxSlot固定s0=sqrt(dk)、tau=1，不在零初始化时用答案NLL选择tau/g：此时出口为零，各候选都返回同一Base输出，选择没有信息。旧S-RE已训练，其校准属于V0，不能直接迁移到新模型。

若读权重诊断触发温度/增益校准，另行登记：先用已训练、出口非零的pilot checkpoint，在独立calibration上选择；新模型的网格依据其训练侧分数尺度定义，不机械照搬DPFP网格。两backend获得对等g机会，报告默认/校准两版；正式训练要么按锁定值从相同初始化重新开始，要么将其明确定义为配对继续训练，不与原八格混称同一实验。训练消耗进入储备，tau/g仍不默认为可训练参数。

KernelSlot 的数学恒等已在 V0 检查，不另占一组训练。旧 ridge/native 模型的冻结替换只作参考，不进入本八格的因果差分。

### 8.3 不只记录答案 EM

| 层次 | 必报指标 | 可支持的诊断 |
| --- | --- | --- |
| 寻址 | 正确 key top-1、margin、目标权重、权重熵、有效项数 | 排名正确是否仍读得过于分散 |
| 读取 | 对目标 learned value 的归一化误差、方向误差、输出范数 | 实际读取是否保持任务相关值 |
| 表示 | clean-value probe 与独立 read-value probe | 分开信息可解码性和 decoder 分布偏移 |
| 下游 | 完整词表 EM、NLL、每种关系与写入年龄切片 | 融合是否把低维信息变成可用答案 |
| 因果依赖 | zero / shuffled / permuted association | 是否确实使用该 episode 的记忆 |
| 数值 | 分母分位数、状态范数、梯度范数、clip记录比例 | 识别失真与优化异常 |

归一化读误差可定义为：

$$E_{\rm read,raw}=\mathbb E\frac{\|a_{\rm raw}-v_*\|_2^2}{\|v_*\|_2^2+\epsilon}.$$

若g不为1，另列a_used=g a_raw的范数，以及相对同量纲目标g v*的误差。若还报告a_used对未缩放v*的误差，标为幅度敏感诊断；不把增益校准带来的变化全部记为存储改善。

独立 read-value probe 只在训练侧读取向量上拟合，在保留数据评估，计入诊断训练预算；不进入普通预测。clean-head→read-head 的准确率差不能直接解释为任务信息损失百分比或最终答案 EM 的可加分解。

对于正权重，令 \(\tilde w_i=w_i/\sum_jw_j\)，有效项数：

$$n_{\rm eff}=\frac1{\sum_i\tilde w_i^2}.$$

### 8.4 Oracle 诊断与进入循环的条件

冻结训练好的模型，分别观察真实 key 替换 query、直接提供目标 learned value、zero memory、另一 episode 的 memory。真实 key/目标 value 不进入可部署 forward，只定位上限与错误位置。不同 oracle 干预可能改变读出尺度，须同时报告范数与原结果。

**提议的筛选条件，需在 V2 release 前锁定：** 至少一组在新 dev 上，正常 memory EM 同时比 zero 与 shuffled memory 高 10 pp，且 NLL 更低；同时存在可用的正对照与非退化读出。它是“值得继续有限循环探索”的工程/研究门槛，不是统计确认，也不替代旧 writer 的80% gate。

若所有八组都没有信号：先定位失败层次；只允许一个预登记探针，从温度/增益校准、S=2或同布局宽度中按诊断选择，最多动用4M储备。仍无信号则停止扩大本M线记忆研究；R1与有信号的F线V5按各自证据继续，不能用此门槛关闭原始残差旁路。不能无限追加loss、rank、heads，直到某一组偶然改善。

---

## 9. V3：循环的独立价值，不等同于更多计算必然更好

### 9.1 固定前置选择

V3为M线机制分支：固定 **R_res + KernelMatrix**、单层r128及M线write/read scope；仅在V2对应的KernelMatrix残差条件已建立读取信号时推进。特征可按预登记dev规则从V2选择，不能因为SoftmaxSlot或concat更好而把主候选替换掉。循环比较改用可组合T2，所有arm从配对初始化训练。若只有其他接口/backend能工作，可另立有预算的诊断研究，但不得占用本主线名义。F线R1/V5不受该停止条件约束。

原 T2 chain-end 与多关系路径任务是不同任务。核心先使用训练 hops1–3、测试 hops4/6/8 的明确图生成器；关系含义、终止规则与答案都由可见图决定，不能把 true-hop 输入停止模块。后续多关系路径单独登记。

### 9.2 十组矩阵：每组最多5M，合计50M

| ID | 条件 | 要排除或验证的解释 |
| --- | --- | --- |
| V3-01 | 累加 S1 | 独立训练的单读参考 |
| V3-02 | 累加 S2 | 最小增加深度 |
| V3-03 | 累加 S4 | 主要多读候选 |
| V3-04 | 累加 S4，eta=0.25/4 | 固定总步长，避免纯累积幅度解释 |
| V3-05 | S4，但 query 固定为第一步query | 最小核的精确幅度对照 |
| V3-06 | 同源 tokenwise 核，S4 | 额外低维非线性深度是否已足够 |
| V3-07 | 更宽 S1 | 匹配新增运算量或实测延迟的宽度对照 |
| V3-08 | S4，解除 Rq/Ro 的跨步共享 | 用共享随机初始化复制各步，训练前先验函数相同；参数更多，单独报告 |
| V3-09 | 同核锚定 S4 | 只改变 z_s vs z_0 的外层残差锚点 |
| V3-10 | S4，Z=zS 的完整状态读出 | 检查初始query直通路径是否解释收益 |

同源 tokenwise 核为：

$$
a_{\rm tok}^{(s)}=
\operatorname{SiLU}\left(\mathcal N_r(z^{(s)})R_q+\mathcal N_r(\bar k_q)\right)\odot\bar v_q,
\qquad z^{(s+1)}=z^{(s)}+\eta a_{\rm tok}^{(s)}R_o.
$$

其中 \(\bar k_q,\bar v_q\) 来自当前问题 token；Q/K/V 三路都有实际计算作用。它改变了读算子和历史访问，因此不能作为唯一的记忆因果证据。M 线没有 support 信息时，tokenwise 地板也是协议预期，需与更强的单读、宽度与固定query组共同解释。

V3-07 的宽度由第 15 节公式或校准 profile 选定，正式前锁定；若硬件对齐使 MAC 匹配与延迟匹配无法同时成立，分开报告，不称两者都公平。SoftmaxSlot 必须按其实际 n_write 重新计算，不能套矩阵读取成本。

### 9.3 吸收 B8W：同函数续训对照

若 shared S4 有明确收益，允许从**同一个新 dev 选择的 checkpoint**复制为 untied，然后将 shared/untied 两臂都使用相同新数据、相同优化器策略继续训练。先验证每步state/read、最终Z和非零出口下的函数恒等，再比较。

这是可选的2组×1M储备实验，不能用 untied 继续训练后对 shared 的旧历史数值比较。新 optimizer 或继承 moments 必须两组一致；若同时研究 optimizer reset，需要独立2×2设计。

### 9.4 必做的无训练深度扫描与轨迹检查

- 共享核对每个正式训练深度S_train扫描S_eval∈{1,2,4,8}，保存每步状态与任务输出。S_eval8是深度外推诊断，不用于挑选主checkpoint。V3-08的untied模型只有四套权重，只扫描已定义的前缀1/2/4；禁止暗中重复最后一套权重构造八步。
- 独立训练 S1 与截断 S4 checkpoint 到一读是不同对照，两者均报告。
- 记录状态范数、读出范数、各步 query cosine、目标 key 权重、准确率随 s 的变化。
- 以随机、无关实体与匹配 donor 为控制，干预中间状态；同时干预实际所有读取位置。模型内部可解码出某实体，不足以宣称因果逐跳链。
- 若某个 s 后答案正确率下降，分开看终止前寻址错误与终止后漂移，不立即上动态 halt。

### 9.5 进入确认的原则

S2/S4 至少需要优于独立训练 S1，并有幅度、tokenwise、预算对照支持。若多读只与无记忆地板拉开差距，而不优于单读，不支持循环主张。

M线筛选可保留S2；第10节使用 \(S_M^\star\) 表示锁定的M线多读深度，默认最多4。F线的 \(S_F^\star\) 独立由R1/其配对pilot选择。若M线没有多读收益，可结束M线循环确认或保留其S1结论，不能据此删去F线循环比较。不要为了保留“循环”名字而修改已锁定阈值。

---

## 10. V4：机制确认，保留读算子与深度的交互

### 10.1 四格而非挑一个最高分

在新的 M 线 confirmation 任务登记上，**固定R_res**，使用V2/V3锁定的特征、布局、rank、训练配方与\(S_M^\star\)；读算子交互不会改变融合接口：

| 读算子，均为R_res | S1 | \(S_M^\star\) |
| --- | --- | --- |
| KernelMatrix | M1 | MS |
| SoftmaxSlot | K1 | KS |

**4组 × 3个新训练 seeds × 8M = 96M tokens 上限。** 默认确认 seeds 为17/29/43，筛选使用0；如这些 seed 与生成器已用于同一新研究命名空间，另行登记不同 seeds。确认数据必须独立于旧报告和 V0–V3 的数据。

该矩阵估计：

$$\Delta_{\rm loop}^{M}=J(M,S_M^*)-J(M,1),$$

$$\Delta_{\rm operator}^{1}=J(K,1)-J(M,1),$$

$$\Delta_{\rm interaction}=[J(K,S_M^*)-J(K,1)]-[J(M,S_M^*)-J(M,1)].$$

J 为“越大越好”的预注册任务指标；若使用 NLL，取负 NLL 或明确反转差值方向。

### 10.2 提议的确认判据

主要 contrast 是新 learned-memory 条件下 \(\Delta_{\rm loop}^{M}\)。建议预注册：hops2–3 的平均 EM 差至少 +3 pp、配对区间下界高于0、三个seed方向均正，且 hop1 差不低于 −1 pp。数值是本版新设计阈值，必须在运行前锁定，不自动沿用或改写任何旧研究 gate。

SoftmaxSlot 的循环 contrast 与交互是预注册次指标；若需要与矩阵同等确认地位，对两项主比较预先做多重比较控制。某个条件低于记忆使用门槛时，先报“接口/读取未建立”，不将其纯粹解释为循环失败。

hops4/6/8、未见名字、不同写入顺序与更高负载是独立泛化切片。主方法按已登记 in-distribution dev 选定；不得用这些封存切片反复调整 S 或停止阈值。

---

## 11. V5：保留原 attention 的实用确认

### 11.1 九组覆盖残差主线、循环与 concat

固定四层 r32、F线 context-retained、`qknorm_pre_rope + KernelMatrix`；若R1触发的等预算特征消融正式选择其他tap，先锁定并在全部相关arm一致采用。**不继承M线胜出的SoftmaxSlot，也不把R_res换成F_pre。** 可独立从Base开始，不需要M线通过gate。必要的协议smoke计入储备。

| ID | 条件 | 主要作用 |
| --- | --- | --- |
| **V5-01** | **R_res + KernelMatrix，r32，S1** | **原始残差旁路低维linear attention的确认** |
| **V5-02** | **R_res + KernelMatrix，r32，\(S_F^\star\)** | **同一个残差旁路的循环增量** |
| V5-03 | C* + 同一个KernelMatrix，r32，S1 | 单读状态下的concat收益 |
| V5-04 | C* + 同一个KernelMatrix，r32，\(S_F^\star\) | concat与循环的交互 |
| **V5-05** | **全28层all-7 LoRA，rank1** | **匹配残差r32的独立标准适配基线** |
| V5-06 | 全28层all-7 LoRA，匹配C*预算 | concat预算的标准适配基线；C_dual默认rank2 |
| V5-07 | R_res + KernelMatrix，更宽r，\(S_F^\star\)，匹配C*参数 | 排除concat仅因接口参数更多而改善 |
| **V5-08** | **R_res + 同源T0，r32，\(S_F^\star\)** | **同参数、同深度，检验残差侧支历史价值** |
| V5-09 | C* + 同源T0，r32，\(S_F^\star\) | concat自己的同参数、同深度历史对照 |

每组3个新seeds、每次最多9M输入token exposures，并受真实任务最多3epochs限制；上限 **9×3×9M=243M**。Base只评估。V5-05有630,784参数，比R_res的663,552少4.94%；其层布局与R1-07的同四层LoRA不同，两种比较分别报告。C_dual时，V5-06 rank2、V5-07 r64提供近预算比较，具体数值见第15节。

\(S_F^\star\)是从F线dev锁定的2或4。若R1多读没有信号但仍要检验concat×loop，可将九组中的多读深度统一预登记为4，明确这是固定深度假设检验；不称最优深度。若R/C两类多读均按停止规则取消，收缩为S1接口与基线比较，取消重复arm并降低预算，不把S*=1的两行当不同实验训练两次。同深度T0随保留的S同步定义。

前四组保留同rank、同backend、同低维核的 **接口×深度** 四格。不同接口的参数仍不同；第5–7组另提供参数预算视角，不能同时声称等参数、等状态、等计算。**R_res的主要比较独立登记；concat失败不影响残差主线的结论。**

### 11.2 真实任务与能力保持

初始采用 PIQA 与 BoolQ 的固定训练混合，纯文本 completion 或固定候选打分模板。每项任务从官方训练集划独立 train/dev/calibration；官方 validation 封存为确认集。在 release 中固定模板、标签token、候选概率计算方式与多token标签的归一化规则。[R05][R06]

核心报告两个任务各自准确率与宏平均，同时用固定版本WikiText-103的保留split测量语言建模NLL/困惑度变化，作为能力保持指标；它只用于微调前后相对评估，不保证排除预训练数据重叠。[R07] 真实任务数据量有限，不通过大量重复刷满9M预算。此混合是低成本实用确认，不等于开放领域长程记忆或复杂推理已解决。

如果 R1、V1或其他pilot已用这些任务的训练/dev选择过结构，不重新将它们叫作零调参迁移；封存 validation 仍只开一次。若后续根据其结果修改方法，下一轮用新任务或新登记数据，不能重复借用旧测试的确认地位。

### 11.3 侧支历史贡献必须额外测量

冻结模型，保留原 attention KV 不变，仅对侧支：

1. 去掉历史，只保留当前 token 或短窗口；
2. 置零历史状态；
3. 在 episode 间交换匹配形状的历史状态；
4. 对同一历史内部打乱 key/value 关联，同时尽量保持边际分布；
5. 比较 inclusive 与 strict 的推理诊断，并记录当前 token 权重。

这些干预可能产生分布偏移，所以对R_res使用V5-08、对concat使用V5-09的同深度训练匹配T0，同时保留V5-01/03的单读对照。不能靠一次zero/shuffle下降宣布机制证明；V1的proxy、R1的S1-T0与V3的M线T0均不能代替F线多读T0。原Q/K/V特征本身已经含上下文；即使清空侧支历史，整个模型也没有失去历史信息。

### 11.4 何时可以提出哪些结论

- R_res单读优于其预算LoRA且在独立seed稳定：支持原始残差旁路的实用适配价值，不要求concat或循环同时成立。
- R_res多读优于其单读、同深度T0及相关尺度/成本控制：支持低维重复读取的额外贡献；是否为因果逐跳推理仍需状态干预证据。
- R_res有效而concat无效：保留残差主线，停止复杂接口；不是整个研究失败。
- concat有效而R_res无效：贡献收缩为该条件融合结构；如实报告原始旁路未获支持。
- 多读只在M线有效：支持受控memory-only机制，尚未建立实用PEFT价值；两条协议不混报。
- full-context有效但历史干预和T0无支持：可称动态/非线性适配，不能把收益全归因于侧支历史。
- 参数效率提高而延迟变差：报告质量与成本取舍，不称加速。

---

## 12. V6：只根据具体诊断开启的探索分支

### 12.1 分支优先级

| 分支 | 触发证据 | 最小追加实验 | 暂不做的原因 |
| --- | --- | --- | --- |
| 同布局 r32→r64/128 | 正对照可学，但投影/接口容量有明显限制 | 固定层位置、算子、S，只改一个宽度；加匹配LoRA | 旧r32失败不是普遍容量定理 |
| 4层→14层 | 少层实用增益已稳定且参数预算允许 | 同任务、同总预算对照；先2组pilot | 不能把层数增长归给concat |
| 写入范围 | delimiter条件有效，原始token条件失效 | delimiter vs all-support-token；报告真实n_write | 事实数与写入项数不是一回事 |
| 独立 key/value/state 宽度 | 寻址差、value可解码，且读算子诊断明确 | 固定value/state，只改key宽度 | 避免一次放大全部r导致归因不清 |
| activation_raw | raw与QKNorm来源差异稳定 | raw / SiLU(raw)；其余保持一致 | 激活同时改变尺度与几何，不先全网格 |
| 低维FFN | 读出可用，但query转移有独立不足 | 增加r→2r→r的FFN，与同MAC宽度对照 | 增加非线性深度会破坏固定query的纯缩放解释 |
| 角色/转移监督 | 桥实体或下一地址在终止前就错误 | 2×2：结构改动×额外监督，所有组匹配标签 | 不能将辅助真值带来的收益称为无额外监督涌现 |
| 终态/NULL | 终止前转移可靠，结束后漂移 | 无出边语义与吸收态训练，保留短跳指标 | 旧TSM/LSR已经说明简单停止并非普遍补救 |
| 动态halt | 小型状态信号能预测“已完成”，固定S有明显浪费 | 只用低维状态的门，与固定S实测对比 | 不为每一步重跑上层主干再宣称低成本 |
| 更大backbone | 0.6B的learned-memory或实用收益已确认 | 1.7B固定结构迁移；对等LR校准 | DPFP-X未显示单纯放大解决读出 |

本次不把 ridge、coefficient loss、增大r、第二规模四项同时加入。SQ2 的组合失败与 LSQT 的中间结果要求逐项定义可证伪预测。[E06][E07]

### 12.2 显式转移语义的高成本分支

若任务要求跨步组成关系，可拆分问题条件 c、寻址状态 p 与答案状态 a：

$$q_s=F_{\rm addr}(p_s,c),\quad m_s=\operatorname{Read}(q_s),$$

$$p_{s+1}=F_{\rm step}(p_s,m_s,c),\qquad a_{s+1}=F_{\rm ans}(a_s,m_s,c).$$

逐转移监督应作用于终止前的桥实体或下一地址，区别于仅让最终状态保持不变的 stay loss。必须报告 teacher-forced 与自由 rollout 的差别；下一步使用的状态来自模型自身，不能由金标桥实体替换后称为普通测试。

一个合理的可证伪预测是：在同等辅助标签条件下，角色分离提高自由 rollout 的下一地址准确率，同时中间状态干预使最终答案按预期改变。若只改善辅助头而不改善普通答案，不能升级成多跳推理证据。

### 12.3 每轮更新K/V与长程记忆

如果改为每一循环步更新 K/V，必须为各阶段维护独立历史状态 \(M_t^{(s)},c_t^{(s)}\)。不能把各循环步的写入混进同一个状态而声称仍是固定记忆读取。状态与成本一般随 S 增长，此分支不在核心455M中。

遗忘或 delta 写入也要与当前正核公式分开命名。非负衰减外积可以仍有清晰的正权重归一化；delta 的误差修正项可能带符号，不能机械套用同一个 c 分母作为概率权重。

### 12.4 独立扩展预算

以下是可选预算模板，不包含在480M内，也不要求全部执行：

| 扩展 | pilot上限 | 确认上限 | 进入条件 |
| --- | ---: | ---: | --- |
| 14层布局 | 2组×4M=8M | 9组×3seed×9M=243M | V5残差或concat有稳定信号；重新按预算匹配LoRA |
| 一项角色/终态改动 | 4格×4M=16M | 4格×3seed×8M=96M | 对应的机制诊断成立 |
| 1.7B迁移 | 3组×4M=12M | 3组×3seed×16M=144M | 0.6B已完成确认 |
| 长上下文 | 4组×8M=32M | 重新登记 | 先验证精确cache和状态成本 |

1.7B先核对该模型自己的D、m、heads与head_dim，不能沿用0.6B的数值。其后才讨论4B、量化、跨层共享和中间FFN扩宽。

---

## 13. 数据、统计与训练协议

### 13.1 不复用旧验证集作新确认

新研究使用独立 `study_id=residual_la_v21`。episode身份包含生成器版本、task/layout、base seed、split、index。建议：

```text
train_seed = derive_seed(20261101, "residual_la_v21", phase, "train", run_seed)
eval_seed  = derive_seed(20261101, "residual_la_v21", phase, purpose)
```

eval seed 不含 arm 或 run_seed，所有arm/seeds在相同episode IDs上评估。calibration、dev、confirmation、OOD使用不同purpose；V2、V3、V4也使用独立phase身份。只改变文件名或数字窗口不足以证明数据独立，需与旧registry核对。[D00]

建议每阶段 calibration 2,048 episodes、dev 1,000、confirmation 2,000；confirmation仅在配置、训练规则和分析脚本锁定后开封。旧seed4242的多次使用验证区间只用于历史复现。

### 13.2 难度、负载与名字泛化分离

- **路径长度：**训练/dev只用hops1–3；4/6/8作为封存外推切片，除非另设新研究改变训练范围。
- **负载：**固定hop改变事实数、干扰项数量或写入token数，避免把更难的问题与更大记忆负载混成一个轴。
- **文本长度：**固定图结构，改变表述和无关上下文；F线中原attention也能使用它们，需单独解释。
- **名字：**seen/unseen名字池、tokenization长度和实体词频分层；旧闭集名词泛化不等于开放词表知识泛化。
- **顺序：**自然顺序与预先固定的随机支持顺序；不能仅把更短的文本版本称为更简单的存储。

筛选切片的天花板检查只使用Base和预登记的普通LoRA小pilot，目标是存在可学习而非饱和的区间。不得根据新方法在哪个切片赢来挑选难度。

### 13.3 训练默认值与可调范围

| 项目 | 默认原则 |
| --- | --- |
| 冻结范围 | embedding、attention、原Norm、原FFN、lm_head全部冻结 |
| 新参数/state dtype | 首轮FP32；原主干BF16；有测量依据后才优化side计算精度 |
| optimizer | AdamW；beta=(0.9,0.999)，eps=1e-8；矩阵weight decay0.01 |
| 初始步长 | eta=0.25，固定不可训练；特定对照例外 |
| LR与warmup | 使用V0家族校准；warmup占各run实际优化步数5%，之后统一cosine |
| clip | global norm1.0起步；记录pre-clip与分项梯度，不直接据此推测有效LR |
| dropout | 首轮0；若增加，所有对照规则匹配 |
| loss | 核心answer CE或固定真实任务标签loss；额外辅助loss单独研究 |
| 最大长度 | R1/V1先512；V2机制按模板锁定；V5可1024，所有arm保持相同 |
| 批量 | 以相同有效标签/queries和token预算校准；用gradient accumulation适配显存 |
| checkpoint选择 | 相同频率，dev主指标或NLL的预登记规则；最终/最佳两者不能随arm挑选 |

Base使用纯文本任务格式是设计选择，不表示其tokenizer不存在chat template。PAD与EOS相同ID时按实际padding位置屏蔽label，不能把所有EOS token都mask掉。[R10]

### 13.4 统计规则

筛选的一个seed仅用于发现信号；确认至少3个独立训练seed。主报告列出每seed结果、均值与标准差，配对bootstrap以**episode**为重采样单位，在arm、seed、hop之间共用抽样索引，保留同图多个query的相关性。

此区间主要刻画给定已训练seeds下的数据不确定性。3个seeds不足以精确估计整个训练过程分布；不得把大量episode数替代训练seed数。若需增加到5个seed，单独增加预算，例如V5再增加2×9×9M=162M，不能声称25M储备可以覆盖。

V4的M线主比较与V5的F线主比较分别预注册。V5将原始残差旁路单独列为主要研究族：

1. **残差单读实用主比较：V5-01 对 V5-05**，同一任务宏平均，使用接近参数预算的全层LoRA。
2. **残差循环增量主比较：V5-02 对 V5-01**；侧支历史归因使用V5-02对V5-08，并结合冻结干预、幅度控制及成本报告。
3. **concat扩展比较：V5-03/04对对应深度的V5-01/02**；同时对V5-06、V5-07及V5-09分别检查标准适配、参数预算和侧支历史。
4. 预定义四格中的接口效应、循环效应与交互；不能只选整个矩阵中最高分的一格称成功。

单读残差有效不以concat成功为前提。多读只改善某接口时，报告交互；不把另一接口的失败隐藏。

建议使用95%配对区间，并对并列主比较做Holm校正；也可预先选择其他控制方式，但不能看结果后切换置信水平。实用最小收益可登记为宏平均+1 pp、每任务不下降超过1 pp，同时报告延迟/状态增加；这些是待release锁定的研究偏好，不是自然定律。

### 13.5 额外监督、继续训练与超参数公平

- 任何address标签、正确fact index、桥实体、clean-value decoder、oracle code均属于额外信息，要单独计数。
- 使用旧checkpoint继续训练时，所有新旧算子从相同起点、相同optimizer策略和相同数据流开始；保存step0结果。
- 只给新结构多做HPO、却使用历史LoRA分数，不构成公平比较。
- 参数预算比较与wall-clock预算比较分别报告；一种实验不能自动支持两种效率结论。
- 普通任务没有足够样本时，训练到登记epoch上限就停止，不为了填满tokens反复暴露测试/验证标签。

---

## 14. 从低成本到高成本的预算与释放顺序

### 14.1 核心训练预算的精确分解

tokens按训练样本呈现累计的非padding输入tokens计数；缓存特征复用仍计入样本暴露，另列实际主干token-layer计算。下表是所有条件阶段均展开的规划上限，不代表必须先做完M线才能进入F线确认。

| 阶段 | 最多训练run | 单run上限 | 阶段tokens | 展开累计tokens |
| --- | ---: | ---: | ---: | ---: |
| V0 家族LR校准 | 5×2=10 | 0.2M | 2M | 2M |
| **R1 真实残差旁路筛选** | **8** | **2M** | **16M** | **18M** |
| V1 可选融合proxy | 8 | 2M | 16M | 34M |
| V2 并行M线单跳诊断 | 8 | 4M | 32M | 66M |
| V3 M线残差循环与控制 | 10 | 5M | 50M | 116M |
| V4 M线机制确认 | 4×3=12 | 8M | 96M | 212M |
| **V5 F线残差/concat独立确认** | **9×3=27** | **9M** | **243M** | **455M** |
| 有登记目的的储备 | 不预设 | 合计封顶 | 25M | 480M |

正式训练73次上限，另10次短校准；停止条件、跳过V1、真实数据3epochs上限会减少实际消耗。旧v2是448M核心+32M储备。本次增加R1的16M，将V5从252M改为243M，所以核心为 \(448+16-9=455\)M、储备25M，总上限保持480M。

### 14.2 里程碑与实际释放顺序

| 里程碑 | 训练上限 | 可回答的问题 |
| --- | ---: | --- |
| **第一批：V0+R1** | **18M** | **真实残差linear attention是否值得继续，循环与concat有无初步信号** |
| 残差/concat实用路线：V0+R1+V5 | 261M，另按需释放储备 | 直接确认原始方法及扩展，不强制先支付M线预算 |
| 实用路线加可选V1 | 277M，另按需释放储备 | 进一步选择接口形态 |
| 全部机制探索到V3 | 116M | 含R1/V1/V2/V3的筛选与替代解释，不是确认 |
| 完整条件版 | 核心455M；含储备≤480M | 两种信息访问协议、残差主线、融合与循环的分开结论 |

R1通过后可以直接释放V5，V2–V4按机制需要并行或后置。若只保留S1、取消某接口、或只确认残差主线，先登记实际arm矩阵与减少后的预算；不把取消额度自动改成无上限搜索。

### 14.3 25M储备的可核算用途

- 最多4M：V2失败后的一个预登记温度/增益、S2或宽度探针；只用于M线。
- 最多2M：shared→untied的两臂同起点继续训练。
- 最多2M：F线C-S1/C-S4的配对pilot，各1M；从同源初始化、同数据开始，不与R1较长训练直接作公平性能比较。
- 最多3M：协议smoke、单层接口桥接或必要的F线深度/同成本对照；每项预先登记。
- 最多8M：一次有证据触发的同布局宽度或特征比较。raw/SiLU(raw)/QKNorm的等预算三臂例为3×2M=6M；不能拿训练时长不同的旧结果直接作特征差分。
- 其余6M：有原因记录的实现修复、公平LR补校准或重要复跑。

总计4+2+2+3+8+6=25M。以上不是必做清单，每笔消耗入账；若F线循环机制仍缺同成本或同深度证据，只能缩小主张或另立预算，不把M线对照冒充已在F线完成。V6与旧SLOT-LOOP-v1续训另立预算。

### 14.4 node·h只用实际测量换算

对每个arm实测加载开销、100个稳定训练步、评估与导出：

$$T_{\rm job}=T_{\rm load}+N_{\rm step}\bar t_{\rm step}+T_{\rm eval}+T_{\rm save},$$

$$C_{\rm node\cdot h}=\sum_{\rm jobs}\frac{n_{\rm nodes}T_{\rm wall}}{3600}.$$

只作量纲示例：如果有效吞吐分别是1,000或3,000 token/s，18M输入的纯训练时间为5.0或1.67小时；480M为133.3或44.4小时。**这些不是本模型或新实现的实测预测**，未包括加载、测试、评估，也不能跨不同cache路径使用同一个吞吐。

旧SQ2的0.199–0.214秒/步、约2.18 node·h（含共享测试）来自特定单层、support-cache与短query设置，不能外推到新四层全上下文训练。[E06]

执行建议先给冻结诊断分配不超过1 node·h的独立上限，R1按实测成本生成release；若时间不足，统一缩小所有arm训练步数后再运行，不能各自在walltime结束时停在不同样本预算却称相同训练预算。

仓库当前submit入口实际只实现单节点。普通作业≤1小时，只有已有结果和对应scale-up文档支持时才使用≤2小时；debug队列另有30分钟限制。科学作业引用包含实现与配置的不可变完整commit；不将脚本的最大允许资源当成默认申请量。[E20]

---

## 15. 参数、运算量、状态与部署成本

### 15.1 完整新分支的参数

以下公式适用于QKV特征、宽度都为r、无bias/可训练Norm、固定eta/g/tau、循环步共享、各层独立；适配层数为La：

| 部件 | 每层参数 |
| --- | ---: |
| Q/K/V压缩 | 4096r |
| Rq、Ro | 2r² |
| 加法接口P | 1024r |
| 单Cg或Cu | 3072r |
| 双Cg/Cu | 6144r |

$$P_{R_{\rm res}}=L_a(5120r+2r^2),$$

$$P_{C\text{-single}}=L_a(7168r+2r^2),\qquad P_{C\text{-dual}}=L_a(10240r+2r^2).$$

direct_hidden少1024r/层；状态/投影宽度解耦后需按实际矩阵重新统计。额外低维FFN r→2r→r增加4r²/层。解除Rq/Ro的S步共享额外增加2(S−1)r²/层。

| 完整分支配置 | 块输出残差R_res | C单接口 | C_dual |
| --- | ---: | ---: | ---: |
| 单层r128 | 688,128 | 950,272 | 1,343,488 |
| 四层r32 | 663,552 | 925,696 | 1,318,912 |
| 十四层r32 | 2,322,432 | 3,239,936 | 4,616,192 |

V1 proxy不同：四层r32的F为262,144，C单接口524,288，C_dual917,504。不能拿proxy参数当完整memory模型的参数。

### 15.2 LoRA和宽度匹配

Qwen0.6B每层LoRA参数为rank乘以输入/输出维度之和：[R04]

| target | 每单位LoRA rank的参数 |
| --- | ---: |
| q | 3072 |
| k或v | 2048 |
| o | 3072 |
| gate、up或down之一 | 4096 |
| 全7项 | 22528 |

四层完整R_res/r32为663,552；同四层all-7 LoRA/rank7与全28层all-7 LoRA/rank1均为630,784，少约4.94%。前者控制层位置，后者提供全层标准适配比较；二者不能视作同一布局。

四层完整C_dual/r32为1,318,912；全28层all-7 LoRA/rank2为1,261,568，低约4.35%。同四层R_res/r64为1,343,488，高约1.86%。它们对应V5-06/07的近预算参考。

十四层C_dual/r32为4,616,192；全28层all-7 LoRA/rank7为4,415,488，低约4.35%。如改用C_gate/C_up，应重新搜索匹配rank，不能继续使用C_dual的匹配表。

这些比较只控制总参数附近，不同时控制每层位置、memory容量或FLOPs。正常LoRA可合并；非线性动态侧支通常不能整体静态合并。线性Z=UA的退化组是可合并例外。

### 15.3 近似新增MAC

等宽、最小矩阵循环、每token一次写入时，忽略Norm/激活/分母的标量操作：

$$\operatorname{MAC}_{R_{\rm res}}\approx L_a[5120r+(1+3S)r^2],$$

$$\operatorname{MAC}_{C\text{-dual}}\approx L_a[10240r+(1+3S)r^2].$$

该表达式不包含原Transformer计算，也不适用于SoftmaxSlot；M线的问题token不写，需删除该token的写入项。必须用实际统计区分support写入与query读取。

原FFN每token约3Dm=9,437,184 MAC；C_dual接口本身增加2rm。其占原FFN算术的比例为2r/(3D)，例如r64约4.17%。这是接口算术占比，**不是端到端延迟增幅**，也不包含QKV压缩和循环。

同MAC宽度对照应求解对应方程并按硬件可执行宽度取附近值，报告真实偏差。例如C_dual矩阵核r128/S4与r141/S1的上述MAC很接近，但硬件对齐可能更偏好r144；不能假定数学接近等于延迟接近。

### 15.4 持久状态和槽位交叉点

令key维度dk、value维度dv、写入项数n、每元素b字节：

$$B_{\rm matrix}=b(d_kd_v+d_k),\qquad B_{\rm slot}\approx bn(d_k+d_v)+B_{\rm mask/index}.$$

这表示新的归一化KernelMatrix；旧不带c的ridge状态只有第一项。两种后端各只保留自己的部署状态，不能同时缓存两种状态却只统计较小那一个。

- r32、FP32：矩阵M+c为4,224字节/层/样本；16个等宽slot为4,096字节，再加mask/index。
- 旧S-RE的768×128 FP32矩阵为393,216字节；16个768+128维slot为57,344字节，再加元数据。这是小E下的算术比较，不表示任意长度下slot更省。
- F线all-token写入时n随上下文增长；若使用SoftmaxSlot，首轮容量覆盖整个锁定最大输入长度，超出时报错，不静默淘汰。窗口/淘汰规则是后续新条件。
- 额外矩阵状态与S无关，仅因为KV固定；每步重写KV则一般需要S组状态。

当前方法保留原softmax attention和其KV cache，不能声称整个模型变成线性复杂度或消除了原缓存。若新低维矩阵加在原KV之上，是增加状态，实用价值需由任务与成本共同判断。

### 15.5 训练内存与性能测量

朴素保存每个前缀M_t会占O(BTr²)激活。r64、T2048、B1、FP32仅这些矩阵约32MiB/层；28层约896MiB，不含梯度及原主干激活。新M线只在完整support结束后读取，可只保留最终状态；新F线必须实现真实因果前缀，不能为省内存改成全序列矩阵。

先用正确参考实现，之后才开发chunk/scan/checkpoint策略。每次优化验证前向与梯度，而不是只比较loss曲线。

部署报告至少包含：同硬件、同batch、同上下文长度的prefill吞吐、decode p50/p95延迟、peak allocated memory、实际持久状态字节；同时给出合并后的LoRA参考。涉及停止规则时，逐次上层前向、同步和控制开销全部纳入。[E12]

---

## 16. 实现契约、配置与可复现交付

### 16.1 最小模块边界

建议新增模块时维持以下接口，而不改动已有报告/配置的默认语义：

| 模块 | 输入/输出 | 要求 |
| --- | --- | --- |
| FeatureTap | 原层指定位置→Qb/Kb/Vb | 明确QKNorm/RoPE/GQA位置；不多跑原投影 |
| LowDimProjection | 特征→qbar/kbar/vbar | 记录真实维度与参数 |
| MemoryBackend | write/read/reset/reorder | native、kernel_matrix、kernel_slot_reference、soft_slot各自明确 |
| LatentReader | query、只读memory、S→Z | 不暗中重写KV或重跑原Transformer |
| Fusion | 原H/U、Z→Y | R_res固定为主线；F_pre/attention-residual/concat为独立比较接口，不自动替换 |
| Protocol | support/query、scope、owner | M与F分开；真值只进loss/诊断 |
| Accounting | 每次write/read/base call计数 | 记录实际路径，不只从配置推算 |

旧 `rfx.FinalRead` 不应去掉限制后直接包装S-RE：它对DPFP/ridge等路径有不兼容条件。旧通用writer loader和SQ2专用loader也不能互换。新增适配入口保留原保护条件，并用旧路径一致性检查验证兼容。[D00][E19]

### 16.2 因果矩阵读取伪代码

以下是算法说明，尚不是仓库已有可运行API；省略batch、head和autocast细节。

```python
def kernel_read(q, M, c, eps=1e-6):
    # q: [r], M: [r, r], c: [r]
    # empty memory must return zero explicitly
    if is_empty_state(M, c):
        return zeros_like_value(q)
    return (q @ M) / (q @ c + eps)

def token_step(qbar, kbar, vbar, cache, valid, write, reset, S):
    if reset:
        cache.clear_M_c_and_slots()
    if not valid:
        return zero_Z(), cache

    z0 = qbar
    k = elu_plus_one(rmsnorm_no_affine(kbar))

    # Inclusive F mode; M-only query calls this with write=False.
    # Strict F mode moves this write to AFTER all reads.
    if write:
        cache.M += outer(k, vbar)       # exactly once
        cache.c += k

    z = z0
    for _ in range(S):
        q = elu_plus_one(rmsnorm_no_affine(z) @ Rq)
        a_raw = kernel_read(q, cache.M, cache.c)
        a = read_gain * a_raw          # default fixed gain = 1
        z = z + eta * (a @ Ro)         # shared weights, fixed memory

    return z - z0, cache
```

训练实现不能照抄带Python分支/原地更新的伪代码而破坏autograd；使用有验证的函数式scan或安全状态缓冲。`is_empty_state`应由有效写入计数/标记判断，不靠把含梯度张量转为Python数值。读写mask、分母c和owner映射必须和M一致。

**训练cache契约：**V2可缓存完全冻结的support前缀特征/QbKbVb，但Ak/Av、正特征以及M/c/slots必须随每次参数更新重新计算并保留所需梯度。不能缓存旧memory后继续声称writer在训练。F线多层中，上游adapter更新会改变下游taps，不能跨优化步复用旧下游特征。旧原生memory复现是冻结诊断，另行使用只读快照。

### 16.3 原块的数据流：首先实现残差旁路

```python
N = original_norm1(X)
O, qkv_features = original_attention_once_with_taps(N)
H = X + O
U = original_norm2(H)
Z, new_state = side_branch(qkv_features, old_state, protocol)

if fusion == "block_output_residual":
    base_Y = H + original_mlp(U)     # Original FFN executes once.
    Y = base_Y + Z @ P_R            # P_R is initially zero.
elif fusion == "c_dual":
    g = original_gate_proj(U) + Z @ Cg
    u = original_up_proj(U) + Z @ Cu
    Y = H + original_down_proj(silu(g) * u)
else:
    Y = explicitly_selected_comparison_interface(H, U, Z)
```

R_res不改变本块FFN输入；concat显式改变gate/up。两者是同一低维信息的两种写回路径，不能在实现中用一个含混的“additive”选项互相替代。原FFN仍只执行一次。V1 proxy才从U计算Z；M线support的write-only路径不注入Z。以上是待实现伪代码，未声称仓库已具备该API。

### 16.4 目标配置示例

下面首先给出 **R1-02原始残差旁路** 的待实现schema。除明确标记为不适用的字段外，`null`字段必须在release前解析；runner遇未锁定字段应拒绝正式训练/确认，不能猜测。

```yaml
study_id: residual_la_v21
phase: R1
arm_id: r1_02_qkv_kernel_block_residual
status: proposed

provenance:
  evidence_commit: a255e814740ff85ab3b4ec4836c88f0a8f802b2d
  implementation_commit: from_submission
  backbone: Qwen/Qwen3-0.6B-Base
  backbone_revision: null
  tokenizer_revision: null
  data_manifest: null
  checkpoint_digest_algorithm: blake2b-128

protocol:
  information_access: context_retained
  task: piqa_boolq_locked_mix
  write_scope: all_valid_tokens_causal
  read_scope: all_valid_tokens_causal
  causal_mode: inclusive_write_once_then_read
  backbone_prefix_kv_visible: true
  side_history_is_separate: true
  future_tokens_visible: false
  max_sequence_length: 512
  packing: false

adapter:
  layers: [6, 13, 20, 27]
  feature_tap: qknorm_pre_rope
  q_dim: 32
  key_dim: 32
  value_dim: 32
  state_dim: 32
  norm: rmsnorm_no_affine
  norm_eps: 1.0e-6
  feature_map: elu_plus_one
  reader: residual_fixed_memory
  loop_steps: 1
  step_size: 0.25
  step_size_trainable: false
  readout: increment
  fusion: block_output_residual
  zero_init_fusion: true
  cross_step_weight_sharing: true
  cross_layer_weight_sharing: false

memory:
  backend: kernel_matrix
  state_dtype: float32
  denominator_eps: 1.0e-6
  read_gain: 1.0
  read_gain_trainable: false
  score_scale: null  # Not used by normalized kernel_matrix read.
  softmax_temperature: null  # Only applies to the SoftmaxSlot comparison.
  calibration_mode: none_in_primary_matrix
  slot_capacity: null  # Not used by the fixed kernel_matrix backend.
  state_immutable_within_loop: true
  owner_index: original_sequence

training:
  seed: 0
  optimizer: adamw
  lr: null
  parameter_dtype: float32
  backbone_dtype: bfloat16
  loss: answer_ce
  auxiliary_losses: []
  max_input_token_exposures: 2000000
  max_epochs: 3
  max_steps: null
  effective_queries_per_step: null
  checkpoint_selection: locked_dev_rule

evaluation:
  score: task_accuracy_macro
  scoring_rule: null  # Lock candidate scoring or free-generation labels before release.
  report_answer_nll: true
  calibration_manifest: null
  dev_manifest: null
  confirmation_access: closed
  record_counterfactual_memory: true
```

R1-03/04仅改变loop_steps；R1-05显式固定query，R1-06显式改步长。R1-08改变fusion为c_dual并移除P_R；计数按实际有效参数更新。M线另建单层r128、delimiter写入、support write-only、query禁止写入及丢弃support主干KV的完整配置，不能只改一个information_access字段。SoftmaxSlot另行配置真实容量和overflow策略；主线KernelMatrix不分配slot缓存。

### 16.5 每次run的必须记录项

| 类别 | 字段 |
| --- | --- |
| 身份 | study/phase/arm/run_seed、完整源码commit、环境依赖锁、模型/tokenizer revision |
| 数据 | 生成器/模板/layout版本、split、episode IDs、实际input/loss/padded tokens、queries与epoch |
| 信息 | support/query边界、write/read scope、cache访问、所有辅助真值来源 |
| 参数 | stored/trainable/实际有梯度参数数、辅助head参数、原主干冻结校验 |
| 状态 | backend、实际shape、dtype、n_write、mask/index字节、所有仍保留的cache |
| 训练 | optimizer与moments来源、LR、warmup、实际steps、step0、clip记录、退出原因 |
| 计算 | 写入数、读取数、原块调用数、实际主干token-layer次数、cache构建时间、wall/node·h |
| 结果 | per-example预测、每hop与每任务指标、原始/干预状态、深度轨迹、区间算法 |
| 选择 | 所有calibration候选、dev查看记录、选择规则、锁定时间与confirmation开封记录 |

推荐目标产物：

```text
docs/prereg/RESIDUAL-LA-v2.1.md
configs/residual_la_v21/
reports/residual_la_v21/evidence_manifest.json
reports/residual_la_v21/budget_ledger.jsonl
reports/residual_la_v21/V0/identity_and_invariants.json
reports/residual_la_v21/R1/residual_linear_attention_screen.md
reports/residual_la_v21/V1/optional_interface_screen.md
reports/residual_la_v21/V2/read_factorial.md
reports/residual_la_v21/V3/depth_controls.md
reports/residual_la_v21/V4/mechanism_confirmation.md
reports/residual_la_v21/V5/residual_and_concat_confirmation.md
reports/residual_la_v21/claims_evidence.md
```

这些路径是交付约定，本轮没有在仓库创建它们。科学作业的持久化结果目录要显式设置为原HPC持久路径；源码快照的临时工作目录退出后可能删除，不能把相对`runs/`当作持久保存。[D00][E20]

---

## 17. 第一批工作与研究结论的收敛方式

### 17.1 现在应优先完成的工作

1. 锁定v2.1登记、证据快照及新的数据命名空间；旧P1L/SQ2/LSQT verdict不变。
2. **首先实现R_res：原块输出 + 真正因果KernelMatrix的低维写回**，核对本块FFN未接收Z、原块只执行一次、零出口函数保持及梯度启动。
3. 用KernelSlot验证矩阵读取恒等、因果cache与reset；分开旧ridge与新正核。
4. **先运行R1八组：单读、S2、S4、固定query、总步长、T0、近预算LoRA和单读concat；V0+R1最多18M。**
5. 单独登记残差S1是否有效、循环是否有初步增益、concat是否改善写回；有信号可直接准备V5九组确认。V1接口proxy按需执行。
6. 旧S-RE冻结诊断与V2–V4作为并行机制研究；保持各自协议，不以其失败关闭F线残差旁路。

以上为研究执行安排。本次更新的是完整计划，尚未修改GitHub源码、提交GPU作业或取得新的实验指标。

### 17.2 每种结果都应有明确去向

| 结果 | 合理研究结论 | 后续动作 |
| --- | --- | --- |
| R_res有效、concat无效 | 原始残差旁路获得支持，复杂接口未获支持 | 保留R_res，不把整项研究判失败 |
| R_res单读有效、循环无效 | 单读低维linear attention适配有价值 | 保留S1，停止相应循环主张 |
| M线无信号、F线有信号 | 两种信息访问任务的结果不同 | 分协议解释，继续有效的实用旁路 |
| C_dual不优于C_gate | 单接口足以利用该低维状态 | 保留C_gate，减少参数 |
| 同核matrix与slot不一致 | 实现/数值有错误或定义不同 | 停止性能解释并修复 |
| SoftmaxSlot胜，但KernelMatrix不胜 | 选择性读算子重要 | 明确状态和读取成本；不要继续称常数状态结果 |
| 读取较准但任务仍差，concat改善 | 融合接口是有效瓶颈之一 | 进入预算匹配确认 |
| key top1高但read差 | 排名不足以保证可用值 | 改权重几何/混合，暂缓循环 |
| S4只等于固定query幅度对照 | 暂无迭代查询证据 | 保留单轮并调尺度 |
| S4胜S1，却输给同成本更宽S1 | 循环有等参数收益，成本效率不足 | 按目标资源选择，缩小效率主张 |
| oracle有效、learned memory仍差 | 条件性reader机制成立，写读接口未解决 | 不扩展为端到端推理成功 |
| full-context有效、memory-only不成立 | 实用适配有价值，压缩记忆结论有限 | 按实际协议报告 |
| 所有有效模块都很简单 | 简化方案就是本研究结果 | 不强行保留循环或双接口 |

### 17.3 可以争取的贡献

已有LoRA、线性注意力、条件FFN、多跳memory与低维循环均有先例；低维循环甚至已在本仓库中实现。[R03][R04][R09] 本次不能把重新组合这些名称当作首创证据。

更有依据的贡献候选是：

- 在冻结Qwen的完整块旁路上，独立验证低维因果linear attention是否优于标准适配和无侧支历史控制，并明确它与o_proj/FFN内部适配的差别。
- 在已经明确的残差主线上，分解寻址、读出、状态转移与FFN融合的瓶颈，并以同内容和同预算实验验证。
- 确认何种FFN条件接口能把低维动态状态转化为稳定的任务收益。
- 明确固定记忆下的演化query何时优于单读、幅度放大或更宽状态，以及其失败边界。
- 提供可复核的语义、信息访问、状态字节与端到端计算记录，使负结果也能缩小研究空间。

最终论文或技术报告应将创新性审查与性能实验分开。v1已列相关工作的比较框架；投稿前仍需重新核对最接近方法的版本、公式和实际实现。本版不依据当前审计宣布新颖性或领先性能。

---

## 18. 证据与参考资料

所有E编号指向同一固定仓库快照。仓库报告是原作者的实验记录；本文审查其数值、判据和实现对应，未独立重跑权重。R编号为原论文或官方实现。D00为此前的内部诊断文档。

| 编号 | 内容 |
| --- | --- |
| [E01] | P1-L decision：0.6B oracle reader、r32/r128、记忆使用与等参数多读 |
| [E02] | P5L decision：1.7B同协议确认 |
| [E03] | B8W decision：同函数复制后解除共享的非劣验证 |
| [E04] | RFX：存储×query的冻结诊断 |
| [E05] | DPFP：非线性特征writer与E8重训 |
| [E06] | SQ2：delta/ridge×coefficient loss，含step0、梯度和成本 |
| [E07] | LSQT：从头LSQ训练及中间区间判定 |
| [E08] | DPFP-X：r256与1.7B writer诊断 |
| [E09] | P1L-C：读取位置与bridge patching |
| [E10] | CTC：解析器+固定oracle code正对照 |
| [E11] | TSM：symbolic终态监督与门控结果 |
| [E12] | LSR：动态停止、各跳变化与上层前向成本 |
| [E13] | core/adapters.py：旧循环与fixed-query定义 |
| [E14] | core/wrapper.py：原线性层单次执行与梯度路径 |
| [E15] | llm/protocols.py：support/query、cache与截断条件 |
| [E16] | test_gradient_flow.py：零出口启动与输入Jacobian |
| [E17] | test_llm_determinism.py：跨进程数值一致性 |
| [E18] | core/memory.py：delta与其他写读实现 |
| [E19] | llm/sq2.py：SQ2的配置、载入与继续训练 |
| [E20] | scripts/submit.py：作业入口、资源上限与commit解析 |
| [E21] | llm/inject.py：冻结原参数与FP32适配模块 |
| [E22] | test_precision.py：BF16主干与FP32分支/state契约 |
| [D00] | 《LoopLoRA：实验诊断与下一阶段执行计划》，2026-10-29 |
| [R01] | Qwen3-0.6B-Base官方config |
| [R02] | Transformers v4.51.3的Qwen3参考实现 |
| [R03] | Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention |
| [R04] | LoRA: Low-Rank Adaptation of Large Language Models |
| [R05] | PIQA官方数据/项目 |
| [R06] | BoolQ官方数据/项目 |
| [R07] | WikiText论文：Pointer Sentinel Mixture Models |
| [R08] | Language Model Evaluation Harness官方项目 |
| [R09] | GLU Variants Improve Transformer |
| [R10] | Qwen3-0.6B-Base官方tokenizer配置 |

[E01]: https://github.com/UnbearableFate/LoopLora/blob/a255e814740ff85ab3b4ec4836c88f0a8f802b2d/reports/decisions/P1L.md
[E02]: https://github.com/UnbearableFate/LoopLora/blob/a255e814740ff85ab3b4ec4836c88f0a8f802b2d/reports/decisions/P5L.md
[E03]: https://github.com/UnbearableFate/LoopLora/blob/a255e814740ff85ab3b4ec4836c88f0a8f802b2d/reports/decisions/B8W.md
[E04]: https://github.com/UnbearableFate/LoopLora/blob/a255e814740ff85ab3b4ec4836c88f0a8f802b2d/docs/reports/RFX.md
[E05]: https://github.com/UnbearableFate/LoopLora/blob/a255e814740ff85ab3b4ec4836c88f0a8f802b2d/docs/reports/DPFP.md
[E06]: https://github.com/UnbearableFate/LoopLora/blob/a255e814740ff85ab3b4ec4836c88f0a8f802b2d/docs/reports/SQ2.md
[E07]: https://github.com/UnbearableFate/LoopLora/blob/a255e814740ff85ab3b4ec4836c88f0a8f802b2d/docs/reports/LSQT.md
[E08]: https://github.com/UnbearableFate/LoopLora/blob/a255e814740ff85ab3b4ec4836c88f0a8f802b2d/docs/reports/DPFP-X.md
[E09]: https://github.com/UnbearableFate/LoopLora/blob/a255e814740ff85ab3b4ec4836c88f0a8f802b2d/docs/reports/P1L-C.md
[E10]: https://github.com/UnbearableFate/LoopLora/blob/a255e814740ff85ab3b4ec4836c88f0a8f802b2d/docs/reports/CTC.md
[E11]: https://github.com/UnbearableFate/LoopLora/blob/a255e814740ff85ab3b4ec4836c88f0a8f802b2d/docs/reports/TSM.md
[E12]: https://github.com/UnbearableFate/LoopLora/blob/a255e814740ff85ab3b4ec4836c88f0a8f802b2d/docs/reports/LSR.md
[E13]: https://github.com/UnbearableFate/LoopLora/blob/a255e814740ff85ab3b4ec4836c88f0a8f802b2d/src/looplora/core/adapters.py
[E14]: https://github.com/UnbearableFate/LoopLora/blob/a255e814740ff85ab3b4ec4836c88f0a8f802b2d/src/looplora/core/wrapper.py
[E15]: https://github.com/UnbearableFate/LoopLora/blob/a255e814740ff85ab3b4ec4836c88f0a8f802b2d/src/looplora/llm/protocols.py
[E16]: https://github.com/UnbearableFate/LoopLora/blob/a255e814740ff85ab3b4ec4836c88f0a8f802b2d/tests/test_gradient_flow.py
[E17]: https://github.com/UnbearableFate/LoopLora/blob/a255e814740ff85ab3b4ec4836c88f0a8f802b2d/tests/test_llm_determinism.py
[E18]: https://github.com/UnbearableFate/LoopLora/blob/a255e814740ff85ab3b4ec4836c88f0a8f802b2d/src/looplora/core/memory.py
[E19]: https://github.com/UnbearableFate/LoopLora/blob/a255e814740ff85ab3b4ec4836c88f0a8f802b2d/src/looplora/llm/sq2.py
[E20]: https://github.com/UnbearableFate/LoopLora/blob/a255e814740ff85ab3b4ec4836c88f0a8f802b2d/scripts/submit.py
[E21]: https://github.com/UnbearableFate/LoopLora/blob/a255e814740ff85ab3b4ec4836c88f0a8f802b2d/src/looplora/llm/inject.py
[E22]: https://github.com/UnbearableFate/LoopLora/blob/a255e814740ff85ab3b4ec4836c88f0a8f802b2d/tests/test_precision.py
[D00]: https://chatgpt.com/library?file_id=libfile_da47142df1888191830a7d2f45a9430c&account_id=personal
[R01]: https://huggingface.co/Qwen/Qwen3-0.6B-Base/blob/main/config.json
[R02]: https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/qwen3/modeling_qwen3.py
[R03]: https://arxiv.org/abs/2006.16236
[R04]: https://arxiv.org/abs/2106.09685
[R05]: https://github.com/ybisk/PIQA
[R06]: https://github.com/google-research-datasets/boolean-questions
[R07]: https://arxiv.org/abs/1609.07843
[R08]: https://github.com/EleutherAI/lm-evaluation-harness
[R09]: https://arxiv.org/abs/2002.05202
[R10]: https://huggingface.co/Qwen/Qwen3-0.6B-Base/blob/main/tokenizer_config.json



