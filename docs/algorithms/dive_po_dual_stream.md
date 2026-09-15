# DiVE-PO Dual-Stream Advantage 注入修复：前后公式与正确性分析

更新时间：2026-07-16

> 实现位置（2026-08-07 起）：本文所述修正版即生产默认实现，见
> `harborrl/algorithms/dive_po/rewards/dual_stream.py`（旧名
> `centered_gate.py`），由 `slime_train.sh` 的默认
> `--custom-reward-post-process-path` 接入；`DIVE_PO_CENTERED_GATE_ENABLED=0`
> 可回退到 `rewards/postprocess.py` 的基础实现。

## 1. 文档目的与修复边界

本文记录 DiVE-PO 当前实现中 intrinsic advantage 注入 DAPO/GRPO 的原始公式、存在的问题、建议修复公式及其正确性分析，便于后续独立迁移。

本次建议是保守的增量修复，不替换已经通过实验验证有效的主体结构。以下组件保持不变：

- SimHash-KNN episodic novelty；
- hierarchical decayed-count lifelong novelty；
- episodic 与 lifelong 的 NGU-lite 乘法融合；
- K6 beta-arm/UCB allocator；
- temperature ladder；
- verifier reward 的独立 group normalization。

本次只修复 intrinsic advantage stream 在完成 group normalization 后又被逐样本的 beta 和 gate 缩放，从而破坏组内零均值的问题。同时固定 beta 归一化分母，并采用保持零均值的 group-wise clip。

实验依据：过滤 Docker/server 空运行后，v0710 K6 DiVE-PO 在前 487 个共同有效 rollout-step 上的 raw reward 为 0.4018，SETA+DAPO baseline 为 0.3056，相对提升约 31.5%。因此不应在当前阶段大幅替换 estimator、乘法融合或 UCB 结构。

## 2. 记号

设同一个 prompt 的 rollout group 为 $G$，其中轨迹编号为 $i$，轨迹内 action/turn 编号为 $t$。

| 记号 | 含义 |
| --- | --- |
| $R_i$ | 轨迹 $i$ 的 verifier/task reward |
| $E_i$ | 轨迹级 episodic novelty |
| $L_i$ | 轨迹级 lifelong raw novelty |
| $M_i$ | lifelong novelty 产生的乘法 modifier |
| $I_i$ | episodic 与 lifelong 融合后的 intrinsic signal |
| $A_i^{\mathrm{ext}}$ | 独立归一化后的 verifier advantage |
| $A_i^{\mathrm{int}}$ | 独立归一化后的 intrinsic advantage |
| $\beta_i$ | 分配给轨迹 $i$ 的 beta arm 值 |
| $q_i$ | outcome-aware quality gate |
| $r_i$ | trust/status reliability gate |
| $h_i$ | lifelong eligibility indicator |
| $\lambda_t$ | 当前训练步的 intrinsic advantage 系数 |
| $c$ | intrinsic bonus 的绝对值上限 |

## 3. 修复前：完整公式

### 3.1 Episodic novelty

当前 K6 配置不包含 turn position。每个 action 的 episodic state 为：

$$
s_{i,t}
=
(\text{tool},\ \text{command signature},\ \text{observation fingerprint},\ \text{exit bucket}).
$$

SimHash-KNN novelty 为：

$$
e_{i,t}
=
\begin{cases}
1,
& \mathcal{N}(s_{i,t})=\varnothing, \\
\max\left(
e_{\min},
\dfrac{\bar d_K(s_{i,t})}{1+\bar d_K(s_{i,t})}
\right),
& \text{otherwise}.
\end{cases}
$$

当前 cosine distance 定义为：

$$
d(x,y)=\frac{1-\cos(x,y)}{2}.
$$

使用 mean reducer 得到轨迹级 episodic novelty：

$$
E_i=\frac{1}{T_i}\sum_{t=1}^{T_i}e_{i,t}.
$$

### 3.2 Lifelong novelty

对 lifelong key $k$，访问前的衰减 count 为：

$$
\widetilde c_k=c_k\delta^{\Delta n},
$$

其中当前配置为 $\delta=0.995$。

每一个层级 $\ell$ 的 raw novelty 为：

$$
L_i^{(\ell)}
=
\frac{1}{\left|K_i^{(\ell)}\right|}
\sum_{k\in K_i^{(\ell)}}
\frac{1}{\sqrt{\widetilde c_k+1}}.
$$

task、skill 和 global 三层通过加权平均融合：

$$
L_i
=
\frac{
w_{\mathrm{task}}L_i^{(\mathrm{task})}
+w_{\mathrm{skill}}L_i^{(\mathrm{skill})}
+w_{\mathrm{global}}L_i^{(\mathrm{global})}
}{
w_{\mathrm{task}}+w_{\mathrm{skill}}+w_{\mathrm{global}}
}.
$$

当前权重为：

$$
(w_{\mathrm{task}},w_{\mathrm{skill}},w_{\mathrm{global}})
=(0.50,0.35,0.15).
$$

使用当前轨迹写入前的历史运行统计量对 lifelong raw novelty 标准化。当历史样本数大于 1 且 $\sigma_L>10^{-8}$ 时：

$$
z_i^{\mathrm{life}}
=
\operatorname{clip}
\left(
\frac{L_i-\mu_L}{\sigma_L},
-z_{\max},z_{\max}
\right).
$$

否则当前实现显式令：

$$
z_i^{\mathrm{life}}=0.
$$

当前 standardized-softplus modifier 为：

$$
M_i
=
\operatorname{clip}
\left(
1+\operatorname{softplus}(z_i^{\mathrm{life}}),
1,M_{\max}
\right).
$$

因此当 $z_i^{\mathrm{life}}=0$ 时：

$$
M_i=1+\log 2\approx1.693.
$$

### 3.3 Episodic/lifelong 乘法融合

现有 NGU-lite intrinsic signal 为：

$$
I_i=E_iM_i.
$$

对应实现字段为 `explore_agent57_intrinsic_signal`。

当前实现需要注意：即使 `lifelong_eligible=False`，代码仍然会先计算并写出 $I_i$；eligibility 和 trust 只决定 diagnostic NGU bonus 是否启用，并不必然阻止 dual-stream 后处理器使用 $I_i$。

### 3.4 Verifier 与 intrinsic 的独立 group normalization

Verifier stream 为：

$$
A_i^{\mathrm{ext}}
=
\frac{R_i-\overline R_G}
{\operatorname{std}_G(R)+10^{-6}}.
$$

Intrinsic stream 为：

$$
A_i^{\mathrm{int}}
=
\frac{I_i-\overline I_G}
{\operatorname{std}_G(I)+10^{-6}}.
$$

因此在浮点误差范围内：

$$
\sum_{i\in G}A_i^{\mathrm{int}}=0.
$$

在 `dynamic_history` 模式下，当前实现先按照 `(group_index, sample.index)` 对完整轨迹去重，每条轨迹只参与一次 group statistics，然后将结果广播到该轨迹的所有 turn sample。这一部分的处理是正确的。

### 3.5 修复前的 beta arm 归一化

当前 `normalized_beta` 使用当前 batch 中实际出现的最大正 beta：

$$
a_i^{\mathrm{old}}
=
\frac{\max(0,\beta_i)}
{\max_{j\in\mathrm{current\ batch},\ \beta_j>0}\beta_j}.
$$

因此，即使轨迹 $i$ 使用相同 beta arm，只要某个 batch 恰好缺少更高 beta arm，$a_i^{\mathrm{old}}$ 就会变大。这使同一个 arm 的训练强度依赖 batch composition。

### 3.6 修复前的 outcome-aware gate

令归一化后的 verifier outcome 为 $y_i\in[0,1]$，状态 floor 为 $f_i$。v0710 K6 的配置近似为：

$$
f_i
=
\begin{cases}
0.55, & \text{completed},\\
0.12, & \text{truncated},\\
0, & \text{failed or aborted}.
\end{cases}
$$

Quality gate 为：

$$
q_i=f_i+(1-f_i)y_i.
$$

K6 使用 `quality_gate` 模式，因此最终 dual-stream gate 主要来自 $q_i$；lifelong trust 和 status intrinsic scale 没有同时进入这一分支。

### 3.7 修复前的 intrinsic advantage 注入

修复前的逐样本 intrinsic bonus 为：

$$
B_i^{\mathrm{old}}
=
\operatorname{clip}
\left(
\lambda_t a_i^{\mathrm{old}}q_iA_i^{\mathrm{int}},
-c,c
\right).
$$

最终训练 reward/advantage 为：

$$
A_i^{\mathrm{train,old}}
=
A_i^{\mathrm{ext}}
+B_i^{\mathrm{old}}
+P_i^{\mathrm{trunc}},
$$

其中 $P_i^{\mathrm{trunc}}$ 是独立的 truncation penalty。

## 4. 修复前的核心问题

### 4.1 beta/gate 缩放破坏 intrinsic stream 的零均值

虽然：

$$
\sum_{i\in G}A_i^{\mathrm{int}}=0,
$$

但在逐样本乘以 beta arm 和 quality gate 后，通常有：

$$
\sum_{i\in G}
a_i^{\mathrm{old}}q_iA_i^{\mathrm{int}}
\neq0.
$$

因此 intrinsic stream 会给整个 group 引入额外正漂移或负漂移。该漂移取决于 arm composition、outcome gate、轨迹长度及后续 PPO/DAPO clipping，不再只是一个组内相对探索 advantage。

### 4.2 逐样本 clip 会再次破坏零均值

即使 clip 前人为保证：

$$
\sum_{i\in G}D_i=0,
$$

通常仍有：

$$
\sum_{i\in G}\operatorname{clip}(D_i,-c,c)\neq0.
$$

### 4.3 当前 batch 的最大 beta 造成非平稳缩放

当前公式使用 batch-local denominator。同一个 beta arm 的有效权重会随其他 arm 是否出现在当前 batch 中而变化，这不是 UCB 决策本身希望表达的变化。

### 4.4 beta=0 control arm 不能通过普通去均值修复

如果简单地对 $a_iq_iA_i^{\mathrm{int}}$ 做普通 unweighted centering，beta=0 的 control arm 也会因为减去 group mean 而得到非零 bonus，从而破坏 control arm 的语义。因此需要采用保留零权重样本为零的加权中心化。

## 5. 修复后：完整公式

修复后保持以下公式不变：

$$
E_i,\quad L_i,\quad M_i,\quad I_i=E_iM_i,
\quad A_i^{\mathrm{int}},\quad A_i^{\mathrm{ext}}.
$$

K6 UCB arm 分配和 temperature ladder 也保持不变。

### 5.1 使用配置级固定 beta denominator

令配置中的完整 beta ladder 为 $\mathcal{B}_{\mathrm{cfg}}$，定义：

$$
\beta_{\max}^{\mathrm{cfg}}
=
\max_{\beta\in\mathcal{B}_{\mathrm{cfg}}}|\beta|.
$$

修复后的 arm weight 为：

$$
a_i
=
\frac{\max(0,\beta_i)}
{\max(\beta_{\max}^{\mathrm{cfg}},\epsilon)}.
$$

对于当前 K6 beta ladder：

$$
\mathcal{B}_{\mathrm{cfg}}
=
\{0,0.004,0.008,0.012,0.016,0.022\},
$$

所以：

$$
\beta_{\max}^{\mathrm{cfg}}=0.022.
$$

### 5.2 Reliability 与 quality gate 的可控融合

现有 trust/status reliability 定义为：

$$
r_i
=
\operatorname{clip}
\left(
\mathrm{trust}_i\cdot\mathrm{status\mbox{-}scale}_i,
0,1
\right).
$$

保留现有 quality gate：

$$
q_i=f_i+(1-f_i)y_i.
$$

令 lifelong eligibility indicator 为：

$$
h_i
=
\mathbf{1}[\mathrm{lifelong\ eligible}].
$$

定义 gate blend coefficient $\eta\in[0,1]$：

$$
g_i
=
h_i\left[(1-\eta)r_i+\eta q_i\right].
$$

其中：

- $\eta=1$：保持当前 K6 outcome quality gate；
- $\eta=0$：只使用 trust/status reliability；
- $0<\eta<1$：两者软融合。

`gate blend` 是可选优化项，不属于必须的数学正确性修复。为了最小化改动并清晰归因，第一轮正式实验建议使用：

$$
\boxed{\eta=1.0}.
$$

完成纯中心化修复的验证后，再单独消融：

$$
\eta\in\{1.0,0.75,0.5\}.
$$

### 5.3 beta 与 gate 的联合权重

定义：

$$
u_i=a_i g_i.
$$

### 5.4 beta/gate 缩放后的加权中心化

计算 prompt group 内的加权 intrinsic baseline：

$$
S_G=\sum_{j\in G}u_j.
$$

当 $S_G>\epsilon$ 时，定义：

$$
\mu_G^{\mathrm{int}}
=
\frac{
\sum_{j\in G}u_jA_j^{\mathrm{int}}
}{
S_G
}.
$$

修复后的 centered intrinsic advantage 为：

$$
C_i
=
u_i\left(A_i^{\mathrm{int}}-\mu_G^{\mathrm{int}}\right).
$$

如果 $S_G\le\epsilon$，则不执行除法，直接定义该 group 中所有 $C_i=0$。这里不能在加权均值分母中直接写成 $S_G+\epsilon$ 后仍声称严格零和；显式分支是保证后续零和不变量成立的必要条件。

### 5.5 保持零均值的 group-wise clip

先计算未裁剪 bonus：

$$
D_i=\lambda_tC_i.
$$

令 $D_G^{\max}=\max_{j\in G}|D_j|$。当 clip 已启用（$c>0$）且 $D_G^{\max}>0$ 时，定义整个 group 共享的缩放因子：

$$
\kappa_G
=
\min
\left(
1,
\frac{c}{D_G^{\max}}
\right).
$$

如果 clip 未启用或 $D_G^{\max}=0$，定义 $\kappa_G=1$。显式分支避免除零，同时与现有配置中“clip 值不大于 0 表示关闭裁剪”的语义保持一致。

修复后的最终 intrinsic bonus 为：

$$
B_i^{\mathrm{new}}=\kappa_GD_i.
$$

展开后：

$$
B_i^{\mathrm{new}}
=
\kappa_G\lambda_tu_i
\left(
A_i^{\mathrm{int}}
-
\frac{
\sum_{j\in G}u_jA_j^{\mathrm{int}}
}{
\sum_{j\in G}u_j
}
\right).
$$

上式只在 $S_G>\epsilon$ 时使用；否则整组 $B_i^{\mathrm{new}}=0$。

### 5.6 修复后的最终训练信号

$$
A_i^{\mathrm{train,new}}
=
A_i^{\mathrm{ext}}
+B_i^{\mathrm{new}}
+P_i^{\mathrm{trunc}}.
$$

Verifier 和 intrinsic 仍然分别使用独立 normalization statistics，intrinsic signal 不修改 verifier score。

## 6. 第一轮保守版本的最终推荐公式

第一轮实验使用 $\eta=1$，从而完全保留当前 K6 quality gate。此时：

$$
g_i=h_iq_i,
$$

$$
u_i=a_ih_iq_i.
$$

最终公式为：

$$
\boxed{
B_i^{\mathrm{new}}
=
\kappa_G\lambda_t
a_ih_iq_i
\left(
A_i^{\mathrm{int}}
-
\frac{
\sum_{j\in G}a_jh_jq_jA_j^{\mathrm{int}}
}{
\sum_{j\in G}a_jh_jq_j
}
\right)
}.
$$

该 boxed 公式同样只在 $\sum_{j\in G}a_jh_jq_j>\epsilon$ 时使用；否则整组 intrinsic bonus 置零。

这一版本只改变 advantage 注入的数学正确性，不改变 estimator、乘法融合、UCB 或采样温度。

## 7. 正确性不变量

### 7.1 轨迹级组内零和

在 $S_G>\epsilon$ 的分支中，由使用精确分母 $S_G$ 的 $\mu_G^{\mathrm{int}}$ 定义可得：

$$
\sum_{i\in G}C_i
=
\sum_{i\in G}u_iA_i^{\mathrm{int}}
-
\mu_G^{\mathrm{int}}\sum_{i\in G}u_i
=0.
$$

由于同一个 group 共享 $\kappa_G$ 和 $\lambda_t$：

$$
\boxed{
\sum_{i\in G}B_i^{\mathrm{new}}=0
}.
$$

该结论是按独立 trajectory 计算的，与 GRPO/DAPO 的 group baseline 语义一致。`dynamic_history` 展开后的多个 turn sample 只接收同一轨迹 bonus 的广播值，不重复参与该统计。

### 7.2 beta=0 control arm 保持为零

如果 $\beta_i=0$，则：

$$
a_i=0,
$$

$$
u_i=0,
$$

因此：

$$
\boxed{B_i^{\mathrm{new}}=0}.
$$

加权中心化不会给 control arm 引入人为 bonus。

### 7.3 Clip 上界

当 clip 已启用，即 $c>0$ 时，由 $\kappa_G$ 的定义：

$$
\boxed{
|B_i^{\mathrm{new}}|\le c
}.
$$

同时 group-wise scaling 不会像逐样本 clip 那样破坏零和性质。

### 7.4 Batch composition 不变性

因为 beta denominator 来自固定配置：

$$
\beta_{\max}^{\mathrm{cfg}}
=
\max_{\beta\in\mathcal{B}_{\mathrm{cfg}}}|\beta|,
$$

所以相同 beta arm 的 $a_i$ 不再因为某个高 beta arm 是否出现在当前 batch 中而变化。

### 7.5 数值边界

实现时需要满足以下约束：

- 所有输入数值先检查 `finite`，NaN/Inf 回退为 0 或安全默认值；
- beta denominator 使用 $\max(\beta_{\max}^{\mathrm{cfg}},\epsilon)$；
- 当 $S_G=\sum_i u_i\le\epsilon$ 时，不执行除法，整个 group 的 intrinsic bonus 置零；
- 当 group 内 intrinsic variance 为 0 时，$A_i^{\mathrm{int}}=0$；
- 当 clip 未启用或 $\max_i|D_i|=0$ 时，令 $\kappa_G=1$；
- normalization epsilon 与现有实现保持一致，使用 $10^{-6}$。

## 8. 梯度语义

Episodic novelty、lifelong novelty、UCB arm、gate 和 reward post-process 都是在 rollout/reward 阶段以 detached scalar 计算的，不参与模型反向传播。

训练阶段只通过最终 advantage 进入 policy-gradient objective。抽象地写为：

$$
\mathcal{L}_{\mathrm{policy}}
=
-\mathbb{E}_i
\left[
A_i^{\mathrm{train,new}}
\log\pi_\theta(\tau_i)
\right],
$$

因此不存在从 verifier advantage 反向流入 intrinsic estimator，或从 intrinsic estimator 反向流入 verifier 的计算图串扰。潜在耦合来自数值上的 gate 和最终 advantage 组合，而不是 autograd graph。

## 9. 正确性修复与实验优化的区分

以下属于高置信度正确性修复：

1. 使用配置级固定 $\beta_{\max}^{\mathrm{cfg}}$；
2. 在 beta/gate 缩放后重新做轨迹级加权中心化；
3. 保持 beta=0 control arm 的 bonus 为 0；
4. 使用 group-wise scale clip 同时保持零和与幅度上限；
5. 按 trajectory 去重，避免把多个 turn 当作多个 rollout；
6. 让 `lifelong_eligible` 真正控制乘法 intrinsic signal 是否进入训练。

以下属于需要消融验证的优化项，而非纯正确性修复：

1. 将 $\eta$ 从 1.0 降至 0.75 或 0.5；
2. 修改 SimHash bits、KNN K 或 multi-probe radius；
3. 修改 lifelong count decay；
4. 改变 standardized-softplus modifier 的中心值；
5. 替换或删除 UCB allocator；
6. 改变 task/skill/global lifelong weights。

为了保护当前已经观察到的 DiVE-PO 性能收益，以上实验优化项不应与中心化正确性修复同时进入第一轮实验。

## 10. 推荐的实验顺序

### 实验 A：纯正确性修复

保持 K6 的 estimator、product、UCB、temperature 和 quality gate 不变，仅启用：

$$
\eta=1.0,
$$

以及 fixed beta denominator、post-gate weighted centering 和 group-wise clip。

### 实验 B：Gate 软融合消融

只有实验 A 确认不回退后，再比较：

$$
\eta\in\{1.0,0.75,0.5\}.
$$

### 实验 C：Estimator/UCB 调参

SimHash、lifelong decay 和 UCB 的修改应分别进行单变量消融，不与 dual-stream 修复混合，以免无法判断性能变化来源。

## 11. 修复前后摘要

修复前：

$$
B_i^{\mathrm{old}}
=
\operatorname{clip}
\left(
\lambda_t
\frac{\beta_i}{\beta_{\max}^{\mathrm{batch}}}
q_iA_i^{\mathrm{int}},
-c,c
\right).
$$

修复后：

$$
B_i^{\mathrm{new}}
=
\kappa_G\lambda_tu_i
\left(
A_i^{\mathrm{int}}
-
\frac{
\sum_{j\in G}u_jA_j^{\mathrm{int}}
}{
\sum_{j\in G}u_j
}
\right),
$$

该式仅用于 $S_G=\sum_{j\in G}u_j>\epsilon$ 的分支；否则整组 bonus 为 0。

其中：

$$
u_i
=
\frac{\max(0,\beta_i)}
{\max(\beta_{\max}^{\mathrm{cfg}},\epsilon)}
h_i\left[(1-\eta)r_i+\eta q_i\right].
$$

第一轮保守实验取：

$$
\boxed{\eta=1.0}.
$$
