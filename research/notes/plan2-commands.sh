#!/usr/bin/env bash
# plan2（docs/plan2.md v2.1）作为 RB 线（残差旁路线）加入：6 个假设与 3 个实验契约（RB0、RBc、RB1）。
#
# 2026-11-02 在记录沙盒上按本文件原样执行过一次：
#   P=/work/xg24i002/x10041/bypass-lora-worktrees/plan2-sandbox bash research/notes/plan2-commands.sh
# 审核后在主项目重放：
#   P=/work/xg24i002/x10041/bypass-lora bash research/notes/plan2-commands.sh
#
# - 每条 rl 命令都带 --project "$P"；新 id 用 --json + jq 取得，后面的命令引用这些变量。
# - 契约的 Design 取自 research/notes/plan2-contracts/{RB0,RBc,RB1}.md（沙盒契约的副本）中
#   "## Design" 之后的部分；只替换 rl 刚写入的占位 Design。
# - 不创建任务，不封存契约。rl task create 不会封存契约：研究者开始前先运行 rl experiment seal E-x。
# - RB5（plan2 V5 的确认）现在不建契约，见 research/notes/plan2-extraction.md。
# - 需要 bash、jq、awk、grep。
set -euo pipefail

RL=${RL:-/work/xg24i002/x10041/smooth-research/dist/claude/plugins/smooth-research/bin/rl}
P=${P:?set P to the research project directory}
NOTES=$(cd "$(dirname "$0")" && pwd)
PLACEHOLDER='Edit freely until the first run, review or task references this contract'

need_id() {  # need_id <prefix> <value> <label>
  [[ $2 =~ ^$1-[0-9]{4,}$ ]] || { echo "unexpected id for $3: '$2'" >&2; exit 1; }
}

fill_design() {  # fill_design <contract path relative to $P> <file whose part after "## Design" is the Design>
  local f="$P/$1"
  grep -qF -- "$PLACEHOLDER" "$f" || { echo "refusing: $f no longer has the placeholder Design" >&2; exit 1; }
  { awk '{print} /^## Design$/{exit}' "$f"; awk 'f{print} /^## Design$/{f=1}' "$2"; } > "$f.tmp"
  mv "$f.tmp" "$f"
}

T_MAIN='残差旁路低维因果 linear attention 的实用价值（R_res+KernelMatrix S1，plan2 主线）'
T_HIST='残差旁路的收益依赖侧支自身的因果历史（同参数 T0 与冻结历史干预）'
T_LOOP='同一残差旁路内的低维重复读取有超出幅度、步长与逐 token 深度的增量（S_F*∈{2,4}）'
T_CONCAT='同一 KernelMatrix 状态下 FFN gate/up 条件融合（C*_RB）优于块输出残差写回'
T_IMPL='RB 线新实现正确：因果前缀 KernelMatrix、T0、四层 R_res/concat 与侧支历史干预'
T_PROBE='RB1 无信号时一次同布局容量或特征探针（r64/r128 或 raw/activation taps）'
T_RB0='RB0：残差旁路线实现与计时（因果前缀 KernelMatrix、T0、四层 R_res/concat、侧支历史干预）'
T_RBC='RBc：残差旁路线四个训练家族的学习率校准（LoRA4 r7、T0、R_res+KernelMatrix、C_dual+KernelMatrix）'
T_RB1='RB1：残差旁路低维因果 linear attention 首轮筛选（plan2 R1 八组 + C_up 核对）'

# 0. 重放保护：这些标题都不能已经存在。
EXISTING=$("$RL" --project "$P" graph)
for t in "$T_MAIN" "$T_HIST" "$T_LOOP" "$T_CONCAT" "$T_IMPL" "$T_PROBE" "$T_RB0" "$T_RBC" "$T_RB1"; do
  if grep -qF -- "] $t" <<<"$EXISTING"; then
    echo "refusing: a record titled '$t' already exists in $P" >&2
    exit 1
  fi
done

# 1. 假设：核心主张、关键机制与竞争解释、未证明的前提；远分支为 candidate。
H_MAIN=$("$RL" --project "$P" --json hypothesis add \
  --title "$T_MAIN" \
  --statement '保留原 attention 的 F 线上，在冻结 Qwen3-0.6B-Base 的四层 [6,13,20,27] 各加一个 r32 块输出残差旁路 Y=B(X)+Z·P_R（R_res），Z 由真实因果 KernelMatrix 低维 linear attention 单次读取（S1）产生；它在 PIQA+BoolQ 上的宏平均准确率比参数接近的标准 LoRA 至少高 1 pp，且每个任务的差不低于 −1 pp。' \
  --scope 'RB 线。筛选（RB1）：dev 每任务 1,000 条、单 seed，对照 Base、同参数 T0 与同四层 all-7 LoRA r7（630,784 参数，比 R_res 的 663,552 少 4.94%）。确认（RB5，尚未建契约）：新登记并封存的确认集、3 个新 seed，对照全 28 层 all-7 LoRA r1（630,784）。固定 qknorm_pre_rope、inclusive 先写一次再读、η=0.25、g=1；参数不完全相等，不宣称参数效率；同时报告延迟与状态字节；+1 pp 与每任务 −1 pp 是点估计门槛。' \
  --origin 'plan2 §0.1–0.3、§3.6、§6A（R1-02、R1-07）、§11.1（V5-01、V5-05）、§11.4、§13.4 族 1' \
  --derived-from H-0006 | jq -r .id)
need_id H "$H_MAIN" main; echo "main claim: $H_MAIN"

H_HIST=$("$RL" --project "$P" --json hypothesis add \
  --title "$T_HIST" \
  --statement '在 R_res+KernelMatrix 上，同参数、同深度、没有侧支历史的同源逐 token 核 T0 的宏平均至少低 1 pp；并且保留原 attention 与其 KV、只改侧支历史的冻结干预（episode 间交换历史状态、历史内打乱 key/value 配对）各使宏平均至少下降 1 pp。' \
  --scope 'RB 线、F 线。筛选：RB1-02 对 RB1-01（S1）及 RB1 的冻结干预（dev）；确认：RB5 的同深度 T0（RB5-08）与确认集上的干预。原 Q/K/V 特征本身含上下文，清空侧支历史不等于模型失去历史；干预可能带来分布偏移，单次 zero/shuffle 下降不作机制证明；T0 也改变了读函数，因此同时报告窗口、strict 与当前 token 读权重。不成立时按 plan2 §11.4 只称动态/非线性适配。' \
  --origin 'plan2 §0.3、§6A R1.2（R1-01）、§11.1（V5-08）、§11.3、§11.4、§13.4 族 2' \
  --derived-from "$H_MAIN" --derived-from H-0006 | jq -r .id)
need_id H "$H_HIST" history; echo "history: $H_HIST"

H_LOOP=$("$RL" --project "$P" --json hypothesis add \
  --title "$T_LOOP" \
  --statement 'R_res+KernelMatrix 的累加式循环 S_F*（2 或 4，由 RB1 dev 锁定）比 S1 的宏平均至少高 1 pp；该增量不能由重复同一读取的输出放大（S4 固定第 1 步 query）、累积总步长（S4 且 η=0.25/4）或同深度 T0 解释。' \
  --scope 'RB 线、F 线；S_F* 只由 F 线 train/dev 决定，不移植 M 线的 S_M*。筛选：RB1-03/04 对 RB1-02，RB1-05/06 为幅度与步长对照；确认：RB5-02 对 RB5-01 与 RB5-08。RB1 的 T0 只有 S1，不能排除 S4 的逐 token 深度解释；S4 若只等于固定 query 对照，记为暂无迭代查询证据（plan2 §17.2）；不宣称因果逐跳推理；报告读取次数、MAC 与实测延迟。' \
  --origin 'plan2 §0.2(2)、§3.5、§4.5–4.6、§6A（R1-03..06）、§11.1（V5-02、V5-08）、§17.2' \
  --derived-from "$H_MAIN" --derived-from H-0005 | jq -r .id)
need_id H "$H_LOOP" loop; echo "loop: $H_LOOP"

H_CONCAT=$("$RL" --project "$P" --json hypothesis add \
  --title "$T_CONCAT" \
  --statement '在相同的四层 r32 KernelMatrix 状态与相同 S 下，把 Z 送入原 FFN gate/up 的条件融合 C*_RB（C_dual 或 C_up，按 plan2 §7.4 在 RB1 选定）比 R_res 的宏平均至少高 1 pp；确认时该收益在匹配 C*_RB 参数的 LoRA、更宽的 R_res 与 C*_RB 自己的同深度 T0 对照下仍成立。' \
  --scope 'RB 线、F 线。筛选：RB1-08（C_dual）与 RB1-09（C_up）对 RB1-02；确认：RB5-03/04 对 RB5-01/02，RB5-06/07/09 作参数与历史对照。接口参数不同（C_dual 1,318,912，C_up 925,696，R_res 663,552），不宣称参数效率。V1 proxy 的结论（EV-0048、D-0004）不改写；concat 失败不影响残差主线（plan2 §6A R1.3）。' \
  --origin 'plan2 §0.2(3)、§4.4、§6A（R1-08）、§7.4、§11.1（V5-03/04/06/07/09）、§13.4 族 3–4' \
  --derived-from "$H_MAIN" --derived-from H-0002 --motivated-by EV-0048 | jq -r .id)
need_id H "$H_CONCAT" concat; echo "concat: $H_CONCAT"

H_IMPL=$("$RL" --project "$P" --json hypothesis add \
  --title "$T_IMPL" \
  --statement '向量化因果前缀 KernelMatrix 与已审核的逐 token 参考扫描在输出和梯度上一致；四层 R_res 与 concat 在零出口时与 Base 一致，R_res 只在本块原 FFN 之后加 Z·P_R 且原块每层只执行一次；四层两步梯度启动成立；T0 与侧支历史干预符合定义且不泄漏未来 token；参数计数与 plan2 §15 一致。' \
  --scope 'RB0（E0）；FP32 小张量用运行前锁定的严格容差，GPU BF16 沿用 E-0001/E-0004 锁定的容差；只检查本线的新路径，已有 V0 与记忆路径测试作回归；不产生任务性能结论。' \
  --origin 'plan2 §3.4–3.6、§4.1、§4.5、§6.3、§15.5、§16.1–16.3' \
  --derived-from H-0001 | jq -r .id)
need_id H "$H_IMPL" implementation; echo "implementation: $H_IMPL"

H_PROBE=$("$RL" --project "$P" --json hypothesis add \
  --title "$T_PROBE" \
  --statement '若 RB1 的残差与 concat 组都没有信号，而 LoRA 与 T0 正对照可学、实现检查通过，则一次同布局探针（同四层、同 KernelMatrix 与 S1，只把 r 改为 64 或 128；或在 r32 下比较 qknorm_pre_rope、raw 与 SiLU(raw) taps）能让 R_res 达到 Base+1 pp 以上，且不比同布局的近预算 LoRA 低 1 pp 以上。' \
  --scope '只在 RB1 规则 9 触发时释放，一次为限；样本预算取 plan2 §14.3 的 8M 容量/特征储备（例如三臂各 2M）；r 探针配匹配参数的同四层 LoRA；四层 r32 的负结果只界定该配置，不推广到所有低维残差旁路（plan2 §6A R1.3）。' \
  --origin 'plan2 §6A R1.3 末段、§12.1、§14.3' \
  --disposition candidate \
  --derived-from "$H_MAIN" | jq -r .id)
need_id H "$H_PROBE" probe; echo "probe (candidate): $H_PROBE"

# 2. 近期实验契约：RB0（E0）、RBc（E2）、RB1（E2）。RB5 不建契约。
OUT=$("$RL" --project "$P" --json experiment add \
  --title "$T_RB0" \
  --question 'plan2 主线需要的新 F 线路径是否正确、训练成本是否可承受：向量化因果前缀 KernelMatrix 与已审核的逐 token 参考扫描在输出与梯度上是否一致？T0、固定 query、四层 R_res 与 concat 的零出口、残差拓扑、两步梯度启动、因果、cache、padding 与侧支历史干预是否按定义工作？各 RB1 arm 的实测步时与 node·h 是多少？' \
  --hypotheses "$H_IMPL" \
  --stage E0 \
  --intervention '新增并测试（默认行为不变）：KernelMatrix 向量化因果前缀实现（impl=prefix）；T0 分支（branch=kernel_tokenwise，与 KernelMatrix 分支同参数）；AdapterConfig 的 fixed_query 与 impl；F 线推理期侧支历史干预（iv_off、iv_window1/8、iv_strict、iv_swap、iv_permute、当前 token 读权重）；RB 训练入口（arm 表与参数断言、residual_la_v21 种子、2M tokens/3 epochs 日程、LR 校准模式）；latency 脚本；协调者选定方案 A1/A2 时生成 RB 确认集切分清单（不做任何模型评估）。不训练正式模型' \
  --comparator '已审核的逐 token 参考扫描 process_sequence 与 KernelSlotReference；未修改的 Base（零出口）；E-0001/E-0004 已有的不变量测试作回归；V1 tokenwise arm 的实测 0.183–0.194 s/step（research/notes/T-0003-tables.md）作计时参照' \
  --metric '前向输出与对 q̄/k̄/v̄/Rq/Ro 的梯度的最大绝对、相对误差；四层零出口 logits 与 Base 的差；拓扑、因果、cache、padding、重排、写读计数、导出重载与干预恒等各项通过/失败；故意破坏的实现被拒绝；stored/trainable/有梯度参数；≥100 个稳定步的 s/step、tokens/s、峰值显存；按 plan2 §14.4 推算的每个 RB1 arm 与 RB5 run 的 node·h' \
  --scope 'F 线（context-retained）、冻结 Qwen3-0.6B-Base、四层 [6,13,20,27]、r32（另核对 r44、r64 参数计数）、qknorm_pre_rope；实现与计时检查，不产生任务性能结论；FP32 容差在运行前写进测试，GPU BF16 容差沿用 E-0001/E-0004 的锁定值，不看结果后放宽；计时作业在 2g.24gb 上 fail-fast，其它作业的 MIG guard 只记录（D-0019）' \
  --outcome-actions '全部通过 -> 提交 RB0 代码审核包，审核通过后释放 RBc；prefix 与参考不一致，或四层零出口、拓扑、梯度启动、因果检查失败 -> 修复后重测，不进入任何训练，不放宽已锁定的容差；prefix 稳定步时 > 0.4 s/step（约为 V1 tokenwise 的 2 倍）-> 先 profile 再定 RB1；按实测推算 RB1 > 4 node·h -> 报告协调者，统一缩短所有 arm 的样本预算或先优化，不让 arm 停在不同的样本预算；参数计数与 plan2 §15 不符 -> 查明原因后再继续' \
  --budget '≤ 1.0 node·h（MIG 按 1/4 计；估计约 0.5：PBS 中的测试与冒烟约 0.3，计时约 0.2）' \
  --stop '全部检查通过且计时表完成；或出现本契约内无法修复的实现问题时停下报告' \
  --code-from E-0004)
E_RB0=$(jq -r .id <<<"$OUT"); need_id E "$E_RB0" RB0
fill_design "$(jq -r .contract <<<"$OUT")" "$NOTES/plan2-contracts/RB0.md"
echo "RB0: $E_RB0"

OUT=$("$RL" --project "$P" --json experiment add \
  --title "$T_RBC" \
  --question '在 RB1 的数据、模板与优化配方下，四个训练家族（同四层 all-7 LoRA rank7；R_res+T0 S1；R_res+KernelMatrix S1；C_dual+KernelMatrix S1）的学习率按预登记规则应锁定为多少？' \
  --hypotheses "$H_MAIN" \
  --stage E2 \
  --intervention '每家族 seed 0 短训练，到首次累计达到 200,000 非 padding 输入 tokens 的完整步为止（同 V1、E-0013）；初始网格：LoRA 与 T0 为 {1e-4, 3e-4, 1e-3}，R_res+KernelMatrix 与 C_dual+KernelMatrix 为 {3e-4, 1e-3, 3e-3}；初始网格的原始最小 NLL 在上端或下端时追加一个端点（LoRA/T0：3e-3 或 3e-5；两个 KernelMatrix 家族：1e-2 或 1e-4），只扩一次' \
  --comparator '同家族候选互比。plan2 §6.4 的建议点（LoRA/逐 token {1e-4,3e-4}，记忆 {3e-4,3e-3}）都在初始网格内；E-0002（V1 三族 3e-4，停在网格上端）、E-0013 校准（三个 arm 都是 1e-3）与 E-0009（M 线 KernelMatrix 1e-2）只作先验参照，不沿用' \
  --metric 'RB calibration 子集上的 NLL（V1 定义：各任务 gold CE 按 token 求和后取任务均值，再取两任务算术平均）；非有限值；每个候选的步数、tokens、步时与 node·h' \
  --scope 'F 线；RB 数据（V1 切分 split map SHA256 47cf5c777d07863b8def6903487697eafc970b0cd5b5d80babb7f20ded98571b、模板、max_length 512；方案 A1 时 calibration 去掉 RB 确认集行，余每任务 1,048 条）；优化配方同 RB1；初始化与训练顺序用 derive_seed(20261101, "residual_la_v21", "V0", "init"|"train", 0)；不看 dev 与任何确认集；锁定结果用于 RB1：LoRA→RB1-07，T0→RB1-01，R_res+KernelMatrix→RB1-02..06，concat→RB1-08、RB1-09；共 2.4–3.2M tokens，超出 plan2 §6.4 的 2M 部分记入 §14.3 储备' \
  --outcome-actions '在最终有限候选中取 (NLL−最小 NLL)/最小 NLL ≤ 0.01 的最小 LR，写 lr-lock 文件（含各 manifest SHA256）并在 RB1 训练前锁定；某家族全部非有限 -> 回到 RB0 查实现，不扩网格；扩展后的最小值仍在新端点 -> 照规则选取并标注“网格边界”，不再扩展' \
  --budget '≤ 1.5 node·h（估计 12–16 个短 run × 约 0.05 node·h ≈ 0.6–0.8；E-0013 校准实测约 0.046 node·h/run，以 RB0 计时为准）' \
  --stop '四个家族的 LR 全部锁定并登记；或出现实现问题时停下报告' \
  --depends-on "$E_RB0" \
  --code-from "$E_RB0")
E_RBC=$(jq -r .id <<<"$OUT"); need_id E "$E_RBC" RBc
fill_design "$(jq -r .contract <<<"$OUT")" "$NOTES/plan2-contracts/RBc.md"
echo "RBc: $E_RBC"

OUT=$("$RL" --project "$P" --json experiment add \
  --title "$T_RB1" \
  --question '保留原 attention 的 F 线上，四层 r32 的块输出残差旁路 R_res + 真实因果 KernelMatrix（S1）是否带来任务收益（对 Base、同参数无侧支历史的 T0、近预算同四层 LoRA）？同一旁路里 S2/S4 是否有超出幅度与总步长解释的初步信号？在同一 KernelMatrix 状态上，FFN gate/up 条件融合（C_dual、C_up）是否改善写回？' \
  --hypotheses "$H_MAIN" --hypotheses "$H_HIST" --hypotheses "$H_LOOP" --hypotheses "$H_CONCAT" \
  --stage E2 \
  --intervention '九个训练 arm，seed 0：RB1-01 R_res+T0 S1；RB1-02 R_res+KernelMatrix S1；RB1-03 同 S2；RB1-04 同 S4；RB1-05 S4 且 query 固定为第 1 步；RB1-06 S4 且 η=0.25/4；RB1-07 同四层 all-7 LoRA rank7；RB1-08 C_dual+KernelMatrix S1；RB1-09 C_up+KernelMatrix S1（plan2 无此组，用于核对 V1 的 C*=C_up）。共同：四层 [6,13,20,27]、r32、qknorm_pre_rope、inclusive 先写一次再读、η=0.25 与 g=1 固定、readout 为循环增量、出口零初始化；Base 只评估。训练后对 RB1-02、S_F* 对应的 arm、RB1-08、RB1-09 做冻结侧支历史干预' \
  --comparator 'Base（只评估，评估代码不变时沿用 ATT-0024-1/EV-0022）；RB1-01（同参数、无侧支历史）；RB1-07（630,784 参数，比 R_res 少 4.94%）；循环对照 RB1-05/06；concat 对照 RB1-02' \
  --metric 'dev（每任务 1,000 条）上 PIQA、BoolQ 准确率与宏平均（主）及答案 NLL；25/50/75/100% 的 dev 曲线；冻结干预下的 dev 宏平均变化与当前 token 读权重；描述性配对分层 bootstrap 95% 区间（10,000 次，seed 20261101，各 arm 共用索引）；stored/trainable/有梯度参数；每 token 读写次数与原块调用数；峰值训练显存；prefill 吞吐、decode p50/p95；新增状态字节（M+c FP32 r32 为 4,224 字节/层/序列）；node·h' \
  --scope 'F 线、冻结 Qwen3-0.6B-Base、四层 r32、qknorm_pre_rope、KernelMatrix（不换 SoftmaxSlot）；V1 切分、模板与候选打分，max_length 512，按 E-0003 排除不可编码的 train 行；单 seed 筛选，不作确认；所有 arm 相同的样本预算与训练顺序 derive_seed(20261101, "residual_la_v21", "R1", "train", 0)：累计非 padding 输入 tokens 首次达到 2M 的完整步（或 3 epochs，先到为准）；LR 来自 RBc；最终 checkpoint；接口参数不等，不宣称参数效率；T0 只有 S1，不能排除 S4 的逐 token 深度解释；任何确认集都不开封' \
  --outcome-actions "按 Design 中运行前锁定的规则（dev 宏平均点估计，pp）：G0 正对照 RB1-07 ≥ Base+1 不成立 -> 停在实现与可学性检查；G0 且（GR：RB1-02 ≥ Base+1 且 ≥ RB1-07−1，或 GC：max(RB1-08, RB1-09) − RB1-02 ≥ +1）-> 按规则定 RB5 矩阵，起草 RB5（E3）契约并先做结论审核；GL（S_F* arm − RB1-02 ≥ +1）不成立而 GC 成立 -> 先做一次 C-S1/C-S4 配对 pilot（RB1p，各 1M tokens）；G0 成立而 GR、GC 都不成立 -> 查 T0 正对照与梯度启动，再至多一次容量或特征探针（${H_PROBE}），仍无信号则停止 RB 确认；全部训练 arm 在 Base±1 内 -> 只做实现与可学性检查。每个问题记一条 screening 证据，方向按 Design 规则 11，假设状态在结论审核后再改" \
  --budget '≤ 4 node·h（估计 2.6–3.2：九个 arm 按前缀路径约 0.23–0.28 node·h/arm，干预约 0.3，延迟约 0.1，冒烟约 0.2；RB0 实测后在训练前更新成本表）' \
  --stop '九个 arm 完成、重载一致、干预与延迟测量完成并分析；工程失败（OOM、非有限、重载不一致）时停下报告；按实测成本需要缩短时，所有 arm 统一缩短并在训练前登记' \
  --depends-on "$E_RBC" \
  --code-from "$E_RB0" \
  --compared-with E-0003)
E_RB1=$(jq -r .id <<<"$OUT"); need_id E "$E_RB1" RB1
fill_design "$(jq -r .contract <<<"$OUT")" "$NOTES/plan2-contracts/RB1.md"
echo "RB1: $E_RB1"

# 3. 查看
"$RL" --project "$P" graph
"$RL" --project "$P" view
echo "created: main $H_MAIN, history $H_HIST, loop $H_LOOP, concat $H_CONCAT, implementation $H_IMPL, probe $H_PROBE (candidate); RB0 $E_RB0, RBc $E_RBC, RB1 $E_RB1"
