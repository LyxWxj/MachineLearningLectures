# Extended Hidden Markov Models, Hierarchical Gaussian Filters, and Kalman Filters

## 符号设定与基础

### Bayesian Theorem

$$
p(A|B) = \frac{p(B|A)p(A)}{p(B)}
$$
通常我们不管 $p(B)$。
因此：
$$
\underbrace{p(A|B)}_{\text{后验}}
\;\propto\;
\underbrace{p(B|A)}_{\text{似然}}
\underbrace{p(A)}_{\text{先验}}
$$

- **后验**：$p(A|B)$
- **似然**：$p(B|A)$
- **先验**：$p(A)$
- 省略的分母 $p(B)$ 是证据/归一化常数，对 $A$ 而言是常数，因此写成正比于 $\propto$.
- 决策时我们通常只关心不同参数后验的相对大小.

### 符号

| 简写 | 全写 |
| --- | --- |
| MM | Markov Model |
| HMM | Hidden Markov Model |
| HGF | Hierarchical Gaussian Filter |
| KF | Kalman Filter |
| $p(x_{1:t})$ | $p(x_1, x_2, \dots, x_t)$ |
| $p(\cdot\| x_{1:t})$ | $p(\cdot\|x_1, x_2, \dots, x_t)$ |
| $\mathcal{N}(x;\mu,\sigma^2)$ | $X \sim \mathcal{N}(\mu, \sigma^2)$时,X的概率密度函数$f$在$X=x$处的值 |

## HMM

### 1. 我们要解决什么问题？

HMM 用来描述这样的序列：系统内部有一个随时间变化、但不能直接观察的状态 $z_t$，我们只能看到由它产生的观测 $x_t$。

例如，$z_t$ 可以是天气、语音中的音素或动物所处的位置区域；$x_t$ 可以是空气湿度，风速、声音特征或传感器读数。我们观察到的是

$$
x_{1:T}=(x_1,x_2,\ldots,x_T),
$$

但真正关心的通常是隐藏状态

$$
z_{1:T}=(z_1,z_2,\ldots,z_T).
$$

因此 HMM 中有两类任务：

1. **状态推断**：给定模型和观测，估计当前或过去的隐藏状态；
2. **参数学习**：给定观测数据，学习状态如何转移以及状态如何产生观测。

前向滤波主要更新状态的后验分布；Baum--Welch/EM 等方法才更新模型参数。

### 2. 从马尔可夫链到 HMM

马尔可夫性质表示：给定当前状态后，过去对下一状态不再提供额外信息：

$$
p(z_t\mid z_{1:t-1})=p(z_t\mid z_{t-1}).
$$

这定义了一个马尔可夫链：

```text
z_1 → z_2 → z_3 → ... → z_T
```

HMM 在马尔可夫链下方增加观测节点，并作出条件独立假设：当前观测只依赖当前隐藏状态：

```text
隐藏状态:   z_1 → z_2 → z_3 → ... → z_T
               ↓     ↓     ↓        ↓
观测值:       x_1    x_2    x_3   ... x_T
```

$$
p(x_t\mid z_{1:T},x_{1:t-1})=p(x_t\mid z_t).
$$

因此：

$$
\text{HMM}=\text{马尔可夫状态转移}+\text{条件观测模型}.
$$

### 3. 模型的已知量、未知量和参数

设隐藏状态有 $K$ 个离散取值，观测序列长度为 $T$。一组 HMM 参数记为

$$
\theta=(\boldsymbol\pi,A,B).
$$

| 符号 | 定义 | 在模型中控制什么 |
| --- | --- | --- |
| $\pi_i$ | $p(z_1=i)$ | 序列一开始处于哪个状态 |
| $A_{ij}$ | $p(z_t=j\mid z_{t-1}=i)$ | 状态保持、切换以及时间尺度 |
| $B_{jk}$（离散观测） | $p(x_t=k\mid z_t=j)$ | 状态 $j$ 产生观测 $k$ 的倾向 |

其中 $A$ 的每一行和为 1；离散观测下 $B$ 的每一行和也为 1。若观测是连续变量，$B$ 不再是有限矩阵，而是每个状态的一组发射分布参数，例如

$$
x_t\mid z_t=j\sim\mathcal{N}(\mu_j,\Sigma_j).
$$

需要根据任务区分“已知”和“未知”：

| 场景 | 已知 | 通常需要求解 |
| --- | --- | --- |
| 已知模型做推断 | $x_{1:T}$ 和 $\theta$ | $z_t$ 的后验或最可能路径 |
| 有标签训练 | $x_{1:T}$ 和 $z_{1:T}$ | $\pi,A,B$ |
| 无标签训练 | 只有 $x_{1:T}$ | 同时估计参数和隐藏状态的概率分布 |

状态数 $K$ 通常是预先指定的模型结构或超参数。它控制模型可以表达多少种潜在状态，不是普通前向更新自动得到的量。

### 4. 预测、滤波、平滑和解码

这些名称针对不同的目标：

| 任务 | 使用的数据 | 目标 |
| --- | --- | --- |
| 观测预测 | $x_{1:t-1}$ | $p(x_t\mid x_{1:t-1},\theta)$ |
| 滤波 | $x_{1:t}$ | $p(z_t\mid x_{1:t},\theta)$，可在线计算 |
| 平滑 | $x_{1:T}$ | $p(z_t\mid x_{1:T},\theta)$，利用未来观测修正过去 |
| 最可能路径 | $x_{1:T}$ | $\arg\max_{z_{1:T}}p(z_{1:T}\mid x_{1:T},\theta)$，通常用 Viterbi 算法 |

这四个任务使用的数据范围不同：观测预测发生在 $x_t$ 到来之前；滤波在观测到来时在线更新；平滑要等整段序列结束后利用未来信息；解码则选择一条最可能的完整状态路径。滤波和平滑都假定 $\theta$ 已知，它们更新的是隐藏状态后验，不是模型参数。

### 5. 生成模型：参数如何产生数据

给定 $\theta$ 后，HMM 按以下顺序生成一条数据：

1. 从初始分类分布采样 $z_1\sim\operatorname{Categorical}(\boldsymbol\pi)$；
2. 对 $t=2,\ldots,T$，按转移分布采样 $z_t\sim p(z_t\mid z_{t-1})$；
3. 对每个时刻，按发射分布采样 $x_t\sim p(x_t\mid z_t)$。

对应的联合分布为

$$
p(z_{1:T},x_{1:T}\mid\theta)
=p(z_1\mid\theta)\prod_{t=2}^{T}p(z_t\mid z_{t-1},\theta)\prod_{t=1}^{T}p(x_t\mid z_t,\theta).
$$

这里的 $p(x_t\mid z_t)$ 叫**发射概率**（连续观测时叫发射概率密度）：它表示“状态如何产生观测”。它不是 $p(z_t\mid x_t)$；后者是看到观测后对状态的后验，需要通过贝叶斯公式计算。

### 6. 已知参数时：推断隐藏状态

现在假设 $\theta$ 已知，观测 $x_{1:T}$ 按时间到达。目标是计算

$$
p(z_t\mid x_{1:t},\theta),
$$

即看到当前及过去观测后，对当前状态的信念。这个过程称为**前向滤波**，每个时刻包含三个有明确区别的量。

#### 6.1 先预测隐状态，再预测观测

上一时刻的状态后验经过状态转移，得到当前状态先验：

$$
p(z_t\mid x_{1:t-1},\theta)
=\sum_{z_{t-1}}p(z_t\mid z_{t-1},\theta)
p(z_{t-1}\mid x_{1:t-1},\theta).
$$

这个式子预测的是**隐状态**，还没有使用当前观测 $x_t$。将状态先验通过发射模型映射到观测空间，才得到下一观测的预测分布：

$$
p(x_t\mid x_{1:t-1},\theta)
=\sum_{z_t}p(x_t\mid z_t,\theta)
p(z_t\mid x_{1:t-1},\theta).
$$

因此要区分：$p(z_t\mid x_{1:t-1})$ 是隐状态预测，$p(x_t\mid x_{1:t-1})$ 才是观测预测。

#### 6.2 观测到来后更新隐状态

当 $x_t$ 已经观察到时，观测值本身不再被“更新”；被更新的是对 $z_t$ 的信念：

$$
p(z_t\mid x_{1:t},\theta)
=\frac{p(x_t\mid z_t,\theta)
p(z_t\mid x_{1:t-1},\theta)}
{p(x_t\mid x_{1:t-1},\theta)}.
$$

也就是

$$
p(z_t\mid x_{1:t},\theta)
\propto p(x_t\mid z_t,\theta)
p(z_t\mid x_{1:t-1},\theta).
$$

这里的三个角色分别是：

| 量 | 角色 |
| --- | --- |
| $p(z_t\mid x_{1:t-1},\theta)$ | 状态预测先验 |
| $p(x_t\mid z_t,\theta)$ | 发射概率/观测似然 |
| $p(z_t\mid x_{1:t},\theta)$ | 更新后的状态后验 |

如果只写

$$
p(z_t\mid x_t)\propto p(x_t\mid z_t)p(z_t),
$$

那么只有在把 $p(z_t)$ 理解为已经包含历史观测的预测先验 $p(z_t\mid x_{1:t-1})$ 时，它才是前向更新的简写。使用裸的边缘先验会丢掉历史观测的信息。

#### 6.3 前向递推的计算形式

这一节先回顾没有观测的马尔可夫链，再说明 HMM 如何在此基础上加入观测信息。以下统一采用**行向量**表示状态分布。

##### 先回顾马尔可夫链：只做状态预测

设上一时刻对状态的概率分布为

$$
\mathbf q_{t-1}
=[p(z_{t-1}=1),\ldots,p(z_{t-1}=K)].
$$
$1\cdots K$是对 $K$ 个离散状态的枚举。

状态转移矩阵$A$的每个元素$A_{ij}$为状态$i$转移到状态$j$的概率：

$$
A_{ij}=p(z_t=j\mid z_{t-1}=i).
$$

如果只知道 $z_{t-1}$ 的分布，下一时刻处于状态 $j$ 的概率需要把所有可能的前一状态加起来：

$$
\begin{aligned}
\tilde q_t(j)
&=p(z_t=j)\\
&=\sum_{i=1}^{K}p(z_t=j\mid z_{t-1}=i)p(z_{t-1}=i)\\
&=\sum_{i=1}^{K}q_{t-1}(i)A_{ij}.
\end{aligned}
$$

这正是矩阵乘法

$$
\boxed{\tilde{\mathbf q}_t=\mathbf q_{t-1}A}.
$$

因为 $A$ 的每一行和为 1，$\tilde{\mathbf q}_t$ 仍然是一个概率分布。若连续若干步都没有新的观测，则

$$
\mathbf q_{t+n}=\mathbf q_tA^n.
$$

所以，马尔可夫链的矩阵乘法只回答一个问题：**如果暂时没有新的观测，状态分布会如何随时间传播？**

##### HMM 的一步更新：先预测，再用观测加权

HMM 比马尔可夫链多了发射模型。假设当前观测 $x_t$ 已经看到，对每个隐藏状态计算发射似然：

$$
\boldsymbol\alpha_t
=[\alpha_t(1),\ldots,\alpha_t(K)],\qquad
\alpha_t(j)=p(x_{1:t},z_t=j\mid\theta),
$$

$$
\mathbf b_t
=[p(x_t\mid z_t=1,\theta),\ldots,p(x_t\mid z_t=K,\theta)]^{\top},
\qquad
D_t=\operatorname{Diag}(\mathbf b_t).
$$

这里 $\mathbf b_t$ 是列向量；它的第 $j$ 个分量表示：假设当前状态是 $j$，看到 $x_t$ 的可能性有多大。$D_t$ 是由这些似然构成的对角矩阵。若离散观测 $x_t=k$，则 $\mathbf b_t$ 是发射矩阵 $B$ 的第 $k$ 列：$\mathbf b_t=B_{:,k}$。

先用马尔可夫链传播上一时刻的后验：

$$
\tilde{\mathbf q}_t=\mathbf q_{t-1}A.
$$

然后，对于每个候选状态 $j$，把预测概率乘以该状态对当前观测的发射似然：

$$
\hat q_t(j)
=\tilde q_t(j)p(x_t\mid z_t=j,\theta).
$$

逐个状态写成向量，就是

$$

\hat{\mathbf q}_t=\tilde{\mathbf q}_tD_t\text{ or } \tilde{\mathbf q}_t \odot \mathbf b_t
$$

乘完以后还不是概率分布，因为分量之和不一定为 1。把它归一化：

$$
\begin{aligned}
q_t(j)
&=\frac{\hat q_t(j)}{\sum_{r=1}^{K}\hat q_t(r)}\\
&=\frac{p(x_t\mid z_t=j,\theta)p(z_t=j\mid x_{1:t-1},\theta)}
{p(x_t\mid x_{1:t-1},\theta)}.
\end{aligned}
$$

分母正是当前观测的预测概率：

$$
p(x_t\mid x_{1:t-1},\theta)
=\sum_{r=1}^{K}p(x_t\mid z_t=r,\theta)
p(z_t=r\mid x_{1:t-1},\theta).
$$

因此，一步 HMM 前向更新可以先用标量形式理解为：

$$
\boxed{
\text{后验}
\;\propto\;
\text{发射似然}\times\text{状态预测先验}
}.
$$

##### 写成前向量矩阵形式

上面的“转移、乘似然、归一化”可以合并为

初始时

$$
\boldsymbol\alpha_1=\boldsymbol\pi \odot \mathbf b_1,
$$

之后每一步先乘转移矩阵，再乘发射对角矩阵：

$$
\boxed{\boldsymbol\alpha_t
=\boldsymbol\alpha_{t-1}A \odot \mathbf b_t},
\qquad t=2,\ldots,T.
$$

其中 $\boldsymbol\alpha_t$ 与 $\mathbf q_t$ 的区别是：$\boldsymbol\alpha_t$ 还保留了观测序列的联合概率，通常没有归一化；$\mathbf q_t$ 才是归一化后的状态后验。

归一化的滤波后验为

$$
\mathbf q_t
=p(z_t\mid x_{1:t},\theta)
=\frac{\boldsymbol\alpha_t}{\text{Sum}(\boldsymbol\alpha_t)}.
$$

前向向量在最后一个时刻给出整条观测序列的边际似然：

$$
p(x_{1:T}\mid\theta)=\text{Sum}(\boldsymbol\alpha_t).
$$

因此，参数学习中的目标函数可以由前向算法高效计算，而不需要枚举全部 $K$ 条隐藏状态路径。

在实现中通常使用对数概率或缩放因子，避免长序列中概率连乘造成数值下溢。

### 7. 学习模型参数：需要调整什么、如何调整

前面的前向算法假设 $\theta=(\pi,A,B)$ 已知，目标只是随观测更新状态后验。如果模型参数未知，就要根据训练数据调整它们，使观测序列的概率更大：

$$
\theta^*=\arg\max_{\theta}p(x_{1:T}\mid\theta).
$$

参数学习有两种情况：如果训练数据还提供了状态标签，可以直接统计频数；如果状态不可见，就需要先估计状态的概率，再用这些概率更新参数。

#### 7.1 隐状态已知：用计数估计参数

如果训练数据同时给出了隐藏状态 $z_t$，可以用 one-hot 向量把计数写成矩阵。设共有 $N$ 条序列，$e_i\in\mathbb R^K$ 是状态 $i$ 的 one-hot 列向量，$f_k\in\mathbb R^M$ 是离散观测 $k$ 的 one-hot 列向量。定义初始状态计数、转移计数和发射计数：

$$
\mathbf c^{\pi}=\sum_{n=1}^{N}e_{z_{n,1}},
$$

$$
C^{A}=\sum_{n=1}^{N}\sum_{t=2}^{T_n}
e_{z_{n,t-1}}e_{z_{n,t}}^{\top}\in\mathbb R^{K\times K},
$$

$$
C^{B}=\sum_{n=1}^{N}\sum_{t=1}^{T_n}
e_{z_{n,t}}f_{x_{n,t}}^{\top}\in\mathbb R^{K\times M}.
$$

令 $\mathbf 1_d$ 表示 $d$ 维全 1 列向量，$\operatorname{Diag}(\mathbf v)$ 表示以向量 $\mathbf v$ 为对角线的矩阵。逐行归一化后得到

$$
\boldsymbol\pi^{\top}=\frac{\mathbf c^{\pi}}{N},
$$

$$
\boxed{A=\operatorname{Diag}(C^{A}\mathbf 1_K)^{-1}C^{A}},
\qquad
\boxed{B=\operatorname{Diag}(C^{B}\mathbf 1_M)^{-1}C^{B}}.
$$

其中 $C^A_{ij}$ 是 $i\to j$ 的转移次数，$C^B_{jk}$ 是状态 $j$ 产生观测 $k$ 的次数；左乘对角逆矩阵就是把每一行除以该行计数总和。实际应用中可先加 Dirichlet 伪计数矩阵，例如 $C^A+\lambda_A\mathbf 1_K\mathbf 1_K^{\top}$ 和 $C^B+\lambda_B\mathbf 1_K\mathbf 1_M^{\top}$，避免未出现事件得到严格的零概率。

#### 7.2 隐状态未知：Baum--Welch（EM）

通常只有观测序列 $x_{1:T}$，并不知道真实的 $z_{1:T}$。Baum--Welch 是 HMM 上的 EM 算法：E 步计算隐藏状态和转移的后验软计数，M 步根据这些期望计数重估参数。

**E 步：前向--后向矩阵递推。** 仍采用行向量约定。对每个时刻定义发射对角矩阵 $D_t$，其对角元素为 $p(x_t\mid z_t=j,\theta)$。前向向量为

$$
\boldsymbol\alpha_1=\boldsymbol\pi \odot \mathbf b_1,\qquad
\boldsymbol\alpha_t=\boldsymbol\alpha_{t-1}A \odot \mathbf b_t.
$$

定义列向量 $\boldsymbol\beta_t$，其中第 $i$ 个分量为 $p(x_{t+1:T}\mid z_t=i,\theta)$。反向递推为

$$
\boldsymbol\beta_T=\mathbf 1_K,\qquad
\boldsymbol\beta_{t-1}=A D_t\boldsymbol\beta_t.
$$

令 $L=p(x_{1:T}\mid\theta)=\boldsymbol\alpha_T\mathbf 1_K$。第 $t$ 时刻的状态软计数向量（行向量）为

$$
\boldsymbol\gamma_t
=p(z_t\mid x_{1:T},\theta)
=\frac{\boldsymbol\alpha_t\odot\boldsymbol\beta_t^\top}{L}.
$$

其中 $\odot$ 表示逐元素乘法。相邻时刻的转移软计数矩阵为

$$
\Xi_t
=\left[\xi_t(i,j)\right]_{i,j=1}^{K}
=\frac{\operatorname{Diag}(\boldsymbol\alpha_{t-1})
A D_t\operatorname{Diag}(\boldsymbol\beta_t)}{L},
\qquad t=2,\ldots,T.
$$

$\gamma_t(j)$ 是时刻 $t$ 属于状态 $j$ 的后验概率；$\Xi_t(i,j)$ 是 $t-1$ 时刻处于 $i$ 且 $t$ 时刻转移到 $j$ 的联合后验概率。这些量是软计数：不确定的状态归属会按概率分摊。

**M 步：对期望计数逐行归一化。** 对单条序列，初始分布更新为

$$
\boldsymbol\pi^{\text{new}}=\boldsymbol\gamma_1.
$$

期望转移计数矩阵为

$$
C^A=\sum_{t=2}^{T}\Xi_t.
$$

第 $i$ 行除以从状态 $i$ 出发的期望次数，得到转移矩阵：

$$
\boxed{A^{\text{new}}
=\operatorname{Diag}(C^A\mathbf 1_K)^{-1}C^A.}
$$

对离散观测，令 $Y\in\{0,1\}^{T\times M}$ 为观测 one-hot 矩阵（$Y_{tk}=1$ 表示 $x_t=k$），令 $\Gamma\in\mathbb R^{T\times K}$ 的第 $t$ 行为 $\boldsymbol\gamma_t$。期望发射计数矩阵为

$$
C^B=\Gamma^\top Y\in\mathbb R^{K\times M},
$$

于是

$$
\boxed{B^{\text{new}}
=\operatorname{Diag}(C^B\mathbf 1_M)^{-1}C^B.}
$$

这里 $C^B_{jk}$ 是状态 $j$ 发射观测 $k$ 的期望次数。对多条独立序列，先将每条序列的 $C^A,C^B$ 相加，并将各序列的初始后验相加后除以序列数。

若观测是 $d$ 维连续向量，将观测按行堆叠成 $X\in\mathbb R^{T\times d}$。第 $j$ 个状态的软权重列向量记为 $\mathbf w_j=\Gamma_{:,j}$，则高斯发射参数的加权矩阵更新为

$$
\boldsymbol\mu_j^{\text{new}}
=\frac{X^\top\mathbf w_j}{\mathbf 1_T^\top\mathbf w_j},
$$

$$
\Sigma_j^{\text{new}}
=\frac{(X-\mathbf 1_T(\boldsymbol\mu_j^{\text{new}})^\top)^\top
\operatorname{Diag}(\mathbf w_j)
(X-\mathbf 1_T(\boldsymbol\mu_j^{\text{new}})^\top)}
{\mathbf 1_T^\top\mathbf w_j}.
$$

对多条观测序列重复 E、M 步，直到总对数似然基本不再增加。EM 一般收敛到局部最优，因此实践中常用多个初始化比较结果。

#### 7.3 在线参数更新

如果希望模型随新数据逐渐适应环境，可以维护转移和发射的期望计数，在每批数据到来后更新参数。这属于在线 HMM 或自适应 HMM；标准 HMM 并不要求参数随每个观测改变。在线更新会提高对环境变化的适应性，但也可能使模型把短期噪声误认为参数变化，因此通常需要遗忘因子、先验或较小的更新步长。

### 8. 一条完整的信息流

令 $\mathbf q_{t-1}=p(z_{t-1}\mid x_{1:t-1},\theta)$ 为行向量，$\mathbf b_t$ 为各状态对当前观测的发射似然列向量，$D_t=\operatorname{Diag}(\mathbf b_t)$。一整个时刻的推断可以写成三个矩阵/向量步骤：

$$
\underbrace{\tilde{\mathbf q}_t}_{\text{状态预测先验}}
=\mathbf q_{t-1}A,
\qquad
\underbrace{p(x_t\mid x_{1:t-1},\theta)}_{\text{观测预测概率}}
=\tilde{\mathbf q}_t\mathbf b_t,
$$

观测 $x_t$ 到来后，用发射似然逐状态加权并归一化，得到新的状态后验：

$$
\boxed{\mathbf q_t
=p(z_t\mid x_{1:t},\theta)
=\frac{\tilde{\mathbf q}_tD_t}
{\tilde{\mathbf q}_t\mathbf b_t}}
\;=\;
\frac{\mathbf q_{t-1}A D_t}
{\mathbf q_{t-1}A\mathbf b_t}.
$$

分母是标量，正是观测预测概率，确保 $\mathbf q_t$ 的分量和为 1。离散观测 $x_t=k$ 时，$\mathbf b_t=B_{:,k}$；连续观测时，$\mathbf b_t$ 由各状态的发射概率密度在实际 $x_t$ 处取值构成。

若 $A,B$ 已知，这个循环只更新 $\mathbf q_t$。若参数未知，则对整段数据累计后验软计数：

$$
C^A=\sum_{t=2}^{T}\Xi_t,\qquad
C^B=\Gamma^\top Y,
$$

再按行归一化得到新的 $A,B$。所以 HMM 中的两种“更新”可以简明地区分为：前向滤波用矩阵更新状态信念 $\mathbf q_t$；Baum--Welch/EM 用软计数矩阵更新模型参数 $\theta=(\pi,A,B)$。因此：

| 问题 | 固定什么 | 调整什么 |
| --- | --- | --- |
| 前向滤波 | $\theta$ 固定 | 每一步的状态后验 $q_t$ |
| Baum--Welch/EM | 当前参数与软状态分布交替固定 | 通过 E 步和 M 步逐步调整 $\theta$ |
| 在线自适应 HMM | 允许参数随数据变化 | 用新数据的软计数和遗忘机制调整 $\theta_t$ |

这也是 HMM 与后面 HGF、Kalman filter 的共同骨架：先根据动力学模型预测隐状态，再用观测模型计算似然，最后用贝叶斯规则更新隐状态信念；不同模型主要区别在于状态空间、噪声分布以及参数是否固定。

## Kalman Filter (KF)

卡尔曼滤波可以看作 HMM 在连续状态、线性转移和高斯噪声条件下的特例。HMM 中我们维护离散状态的概率；卡尔曼滤波中我们维护连续状态的高斯后验分布。这里跳过无偏估计和最优估计的前置知识，重点回答两个问题：
- 现实中的真值和观测值是什么关系
- 知道概率分布之后，如何把它落实为每一步的数值计算？

### 1. 现实中的真值和观测值

#### 1.1 真值不能直接获得

设系统在时刻 $t$ 的真实状态为 $\mathbf s_t$。它可以是位置、速度、温度，也可以是多个物理量组成的向量。真实状态是系统实际处于的状态，但传感器通常不能直接读出它，只能返回一个带噪声的观测 $\mathbf y_t$。

最简单的一维直接测量是

$$
y_t=s_t+v_t,
\qquad v_t\sim\mathcal N(0,r).
$$

这里 $s_t$ 是真值，$v_t$ 是观测噪声，$r$ 是观测噪声方差。等价地，在真值 $s_t$ 给定时，观测值服从一个以真值为中心的高斯分布：

$$
\boxed{y_t\mid s_t\sim\mathcal N(s_t,r)},
$$

也可以写成概率密度

$$
p(y_t\mid s_t)=\mathcal N(y_t;s_t,r).
$$

这句话的含义是：真值固定为 $s_t$ 时，重复测量得到的 $y_t$ 会在 $s_t$ 附近随机散布。$r$ 越小，观测点越集中，传感器越可靠；$r$ 越大，观测点越分散。

需要区分“真值”和“观测样本”：$s_t$ 是我们想知道但没有直接看到的隐变量，$y_t$ 是从上述条件分布中实际抽到的一个数。连续变量的 $p(y_t\mid s_t)$ 是密度，不是某个精确点的离散概率。

#### 1.2 观测不一定直接等于整个状态

如果状态是位置和速度

$$
\mathbf s_t=\begin{bmatrix}p_t\\v_t\end{bmatrix},
$$

传感器可能只测位置、只测速度，或者同时测量两者。因此一般观测模型写成

$$
\boxed{\mathbf y_t=H\mathbf s_t+\mathbf v_t,\qquad
\mathbf v_t\sim\mathcal N(\mathbf 0,R)},
$$

即

$$
p(\mathbf y_t\mid\mathbf s_t)
=\mathcal N(\mathbf y_t;H\mathbf s_t,R).
$$

$H$ 是从状态空间到观测空间的映射：

$$
H=\begin{bmatrix}1&0\end{bmatrix}
$$

表示只测位置，

$$
H=\begin{bmatrix}0&1\end{bmatrix}
$$

表示只测速度。$R$ 是观测噪声协方差；它的非对角元素还可以表示多个传感器噪声之间的相关性。

#### 1.3 真值本身也会随时间变化

除了观测有噪声，系统运动规律也通常不是完全准确的。用线性动力学模型表示真实状态的演化：

$$
\boxed{\mathbf s_t=F\mathbf s_{t-1}+\mathbf w_t,\qquad
\mathbf w_t\sim\mathcal N(\mathbf 0,Q)}.
$$

给定上一时刻真值时，下一时刻真值也服从高斯分布：

$$
p(\mathbf s_t\mid\mathbf s_{t-1})
=\mathcal N(\mathbf s_t;F\mathbf s_{t-1},Q).
$$

$F$ 描述理想动力学，$Q$ 描述模型没有解释的变化。例如匀速运动的状态转移矩阵是

$$
F=\begin{bmatrix}1&\Delta t\\0&1\end{bmatrix},
\qquad
\begin{bmatrix}p_t\\v_t\end{bmatrix}
=F\begin{bmatrix}p_{t-1}\\v_{t-1}\end{bmatrix}+\mathbf w_t.
$$

因此，$Q$ 和 $R$ 的含义不同：$Q$ 是“系统按照模型运动时有多不确定”，$R$ 是“传感器测量时有多不确定”。

这对应 HMM 中的两类概率模型：HMM 的离散状态转移 $p(z_t\mid z_{t-1})$，在这里变成连续状态的高斯转移 $p(\mathbf s_t\mid\mathbf s_{t-1})$；HMM 的发射概率 $p(x_t\mid z_t)$，在这里变成观测似然 $p(\mathbf y_t\mid\mathbf s_t)$。矩阵 $F,Q$ 描述状态如何传播，$H,R$ 描述状态如何产生观测。

### 2. 我们究竟在计算什么分布？

卡尔曼滤波的目标不是恢复一个确定的真值，而是维护看到数据后的状态分布：

$$
p(\mathbf s_t\mid\mathbf y_{1:t}).
$$

在线滤波只使用当前及过去观测。在线性高斯模型下，这个后验始终是高斯分布，因此只需要保存两个数值对象：

$$
\boxed{p(\mathbf s_t\mid\mathbf y_{1:t})
=\mathcal N(\mathbf s_t;\boldsymbol\mu_t,P_t).}
$$

- $\boldsymbol\mu_t$ 是当前状态的最佳均值估计；
- $P_t$ 是估计不确定性的协方差矩阵；
- $P_t$ 的对角线是各状态分量的方差，非对角线是它们的误差相关性。

这就是从“计算一个概率分布”转为“计算均值和协方差”：不需要在连续状态空间上列出所有可能的 $\mathbf s_t$，因为高斯分布已经完全由 $\boldsymbol\mu_t,P_t$ 表示。

在 HMM 中，我们维护的是离散后验 $p(z_t\mid x_{1:t})$；在卡尔曼滤波中，维护的是连续后验 $p(\mathbf s_t\mid\mathbf y_{1:t})$。两者要解决的是同一个问题，只是状态空间从有限个类别换成了连续向量。

### 3. 从概率运算到数值运算：预测分布

假设上一时刻已经得到

$$
\mathbf s_{t-1}\mid\mathbf y_{1:t-1}
\sim\mathcal N(\boldsymbol\mu_{t-1},P_{t-1}).
$$

在没有看到当前观测之前，根据状态方程

$$
\mathbf s_t=F\mathbf s_{t-1}+\mathbf w_t
$$

计算当前状态的预测分布

$$
p(\mathbf s_t\mid\mathbf y_{1:t-1})
=\mathcal N(\mathbf s_t;\boldsymbol\mu_t^-,P_t^-).
$$

从概率图的角度，预测分布仍然是把上一时刻的后验沿状态转移传播：

$$
p(\mathbf s_t\mid\mathbf y_{1:t-1})
=\int p(\mathbf s_t\mid\mathbf s_{t-1})
p(\mathbf s_{t-1}\mid\mathbf y_{1:t-1})\,d\mathbf s_{t-1}.
$$

这正是 HMM 中

$$
p(z_t\mid x_{1:t-1})
=\sum_{z_{t-1}}p(z_t\mid z_{t-1})
p(z_{t-1}\mid x_{1:t-1})
$$

的连续版本：离散状态对所有 $z_{t-1}$ 求和，连续状态对所有 $\mathbf s_{t-1}$ 积分。线性高斯假设使这个积分可以只通过均值和协方差来计算。

#### 3.1 预测均值如何得到

对状态方程取条件期望：

$$
\begin{aligned}
\boldsymbol\mu_t^-
&=\mathbb E[\mathbf s_t\mid\mathbf y_{1:t-1}]\\
&=\mathbb E[F\mathbf s_{t-1}+\mathbf w_t\mid\mathbf y_{1:t-1}]\\
&=F\boldsymbol\mu_{t-1}+\mathbb E[\mathbf w_t]\\
&=F\boldsymbol\mu_{t-1}.
\end{aligned}
$$

所以预测均值就是把上一时刻的均值代入动力学模型。这里不是把随机状态变成确定状态，而是计算随机状态分布的中心。

#### 3.2 预测协方差如何得到

预测误差为

$$
\mathbf e_t^-
=\mathbf s_t-\boldsymbol\mu_t^-
=F(\mathbf s_{t-1}-\boldsymbol\mu_{t-1})+\mathbf w_t.
$$

于是

$$
\begin{aligned}
P_t^-
&=\mathbb E[\mathbf e_t^-\mathbf e_t^{-\top}]\\
&=F P_{t-1}F^\top+Q.
\end{aligned}
$$

展开乘积时会出现过程噪声与上一时刻估计误差的交叉项。卡尔曼模型假设新的过程噪声 $\mathbf w_t$ 与过去信息独立且均值为零，所以交叉项为零，剩下两部分：

$$
\boxed{P_t^-=FP_{t-1}F^\top+Q.}
$$

第一项是旧不确定性经过动力学传播后的结果，第二项是模型误差新增的不确定性。因此只做预测时，$Q$ 通常会使方差增大。

### 4. 从概率运算到数值运算：观测更新

当前观测 $\mathbf y_t$ 到来后，需要把预测分布和观测似然相乘：

$$
p(\mathbf s_t\mid\mathbf y_{1:t})
\propto
p(\mathbf y_t\mid\mathbf s_t)
p(\mathbf s_t\mid\mathbf y_{1:t-1}).
$$

完整的贝叶斯公式是

$$
p(\mathbf s_t\mid\mathbf y_{1:t})
=\frac{p(\mathbf y_t\mid\mathbf s_t)
p(\mathbf s_t\mid\mathbf y_{1:t-1})}
{p(\mathbf y_t\mid\mathbf y_{1:t-1})}.
$$

这与 HMM 的前向更新完全同构：

$$
p(z_t\mid x_{1:t})
\propto p(x_t\mid z_t)p(z_t\mid x_{1:t-1}).
$$

区别只在于，HMM 的状态是离散的，可以逐个状态相乘再求和归一化；卡尔曼滤波的状态是连续的，需要计算积分，但高斯共轭性把这个积分化成均值、协方差和矩阵运算。

分母是观测预测密度：

$$
p(\mathbf y_t\mid\mathbf y_{1:t-1})
=\int p(\mathbf y_t\mid\mathbf s_t)
p(\mathbf s_t\mid\mathbf y_{1:t-1})\,d\mathbf s_t.
$$

卡尔曼滤波的关键是：这个积分不需要数值积分。两个高斯分布相乘后仍是高斯分布，可以直接由均值和协方差计算其参数。

#### 随机观测变量和实际采样点

在公式中，$\mathbf Y_t$ 可以理解为“尚未观测时的随机观测变量”，而 $\mathbf y_t^{\mathrm{obs}}$ 是传感器实际返回的一个采样点。观测模型先描述随机变量的分布：

$$
\mathbf Y_t\mid\mathbf s_t
\sim\mathcal N(H\mathbf s_t,R).
$$

当传感器返回具体数值 $\mathbf y_t^{\mathrm{obs}}$ 后，更新中使用的是这个点处的似然函数：

$$
\mathbf s_t\longmapsto
p(\mathbf Y_t=\mathbf y_t^{\mathrm{obs}}\mid\mathbf s_t)
=\mathcal N(\mathbf y_t^{\mathrm{obs}};H\mathbf s_t,R).
$$

因此，卡尔曼滤波并不是把整个观测分布采样很多次再取平均，而是：先用分布参数 $H\boldsymbol\mu_t^-$、$S_t$ 描述“观测大概会落在哪里”，再把真实采样点与预测中心的差

$$
\boldsymbol\nu_t=\mathbf y_t^{\mathrm{obs}}-H\boldsymbol\mu_t^-
$$

代入条件高斯公式，计算一个具体的后验均值和协方差。

### 5. 一维高斯融合：先看清楚数值是怎样出现的

先考虑最简单的直接测量：

$$
s_t\mid y_{1:t-1}\sim\mathcal N(\mu_t^-,P_t^-),
\qquad
y_t\mid s_t\sim\mathcal N(s_t,r).
$$

这里实际拿到的是一个具体的观测样本 $y_t=y_t^{\mathrm{obs}}$。后验密度作为 $s_t$ 的函数为

$$
p(s_t\mid y_{1:t})\propto
\exp\left[-\frac12\left(
\frac{(s_t-\mu_t^-)^2}{P_t^-}
+\frac{(y_t^{\mathrm{obs}}-s_t)^2}{r}
\right)\right].
$$

把平方项展开并配方，后验仍可写成

$$
p(s_t\mid y_{1:t})=\mathcal N(s_t;\mu_t,P_t),
$$

其中精度（方差的倒数）满足

$$
\boxed{\frac1{P_t}=\frac1{P_t^-}+\frac1r},
$$

均值满足信息加权平均：

$$
\boxed{\mu_t=P_t\left(\frac{\mu_t^-}{P_t^-}+\frac{y_t^{\mathrm{obs}}}{r}\right).}
$$

把它改写成更直观的残差形式。定义

$$
K_t=\frac{P_t^-}{P_t^-+r},
$$

则

$$
\boxed{\mu_t=\mu_t^-+K_t(y_t^{\mathrm{obs}}-\mu_t^-)},
\qquad
\boxed{P_t=(1-K_t)P_t^-}.
$$

这里 $y_t^{\mathrm{obs}}-\mu_t^-$ 是观测和预测之间的创新。$P_t^-$ 大，说明预测不确定，$K_t$ 大；$r$ 大，说明传感器不可靠，$K_t$ 小。

从 HMM 的角度看，$p(x_t\mid z_t=j)$ 是离散状态 $j$ 的发射似然；这里的 $p(y_t^{\mathrm{obs}}\mid s_t)$ 是连续状态 $s_t$ 的发射概率密度。HMM 会为每个候选状态计算一个似然并重新归一化，卡尔曼滤波则利用高斯乘积的闭式结果，直接得到后验的均值和方差。

例如，若 $\mu_t^-=10$、$P_t^-=4$，实际观测为 $y_t^{\mathrm{obs}}=12$、观测方差 $r=9$，则

$$
K_t=\frac4{4+9}=\frac4{13},
\qquad
\mu_t=10+\frac4{13}(12-10)\approx10.615,
$$

$$
P_t=\left(1-\frac4{13}\right)4=\frac{36}{13}\approx2.769.
$$

这个例子展示了“从分布到数值”的过程：先验分布由 $(\mu_t^-,P_t^-)$ 表示，观测样本只通过创新项进入均值更新，观测噪声方差进入增益和后验方差。

### 6. 多维情形：用联合高斯分布计算条件分布

对一般的线性观测模型

$$
\mathbf y_t=H\mathbf s_t+\mathbf v_t,
\qquad \mathbf v_t\sim\mathcal N(\mathbf0,R),
$$

已知预测分布 $\mathbf s_t\sim\mathcal N(\boldsymbol\mu_t^-,P_t^-)$，可以直接写出状态和观测的联合高斯分布：

$$
\begin{bmatrix}\mathbf s_t\\\mathbf y_t\end{bmatrix}
\Bigg|\mathbf y_{1:t-1}
\sim
\mathcal N\left(
\begin{bmatrix}\boldsymbol\mu_t^-\\H\boldsymbol\mu_t^-\end{bmatrix},
\begin{bmatrix}
P_t^- & P_t^-H^\top\\
HP_t^- & HP_t^-H^\top+R
\end{bmatrix}
\right).
$$

这个联合分布中的每一项都有明确含义：

- 预测观测均值：$H\boldsymbol\mu_t^-$；
- 创新协方差：

  $$
  S_t=HP_t^-H^\top+R;
  $$

- 状态与观测的交叉协方差：$P_t^-H^\top$。

联合高斯的条件分布公式给出

$$
\boxed{K_t=P_t^-H^\top S_t^{-1}},
$$

$$
\boxed{\boldsymbol\mu_t
=\boldsymbol\mu_t^-+K_t(\mathbf y_t^{\mathrm{obs}}-H\boldsymbol\mu_t^-)},
$$

$$
\boxed{P_t
=P_t^- -P_t^-H^\top S_t^{-1}HP_t^-}
= (I-K_tH)P_t^-.
$$

这就是矩阵卡尔曼更新。它没有另造一个经验性的“权重公式”：$K_t$ 来自联合高斯条件分布中状态和观测的交叉协方差。实现时通常不显式计算 $S_t^{-1}$，而是解线性方程 $S_tX=(P_t^-H^\top)^\top$，数值上更稳定。

这里的条件高斯公式承担了 HMM 中“观测加权并归一化”的工作。HMM 通过枚举离散状态得到后验概率向量；卡尔曼滤波通过状态—观测的交叉协方差 $P_t^-H^\top$，一次性计算连续状态后验的均值和协方差。

### 7. 完整的数值递推

给定初始分布

$$
p(\mathbf s_0)=\mathcal N(\boldsymbol\mu_0,P_0),
$$

每个时刻执行以下步骤：

**预测：**

$$
\boldsymbol\mu_t^-=F\boldsymbol\mu_{t-1},
\qquad
P_t^-=FP_{t-1}F^\top+Q.
$$

**把预测分布映射到观测空间：**

$$
\hat{\mathbf y}_t=H\boldsymbol\mu_t^-,
\qquad
\boldsymbol\nu_t=\mathbf y_t^{\mathrm{obs}}-\hat{\mathbf y}_t,
\qquad
S_t=HP_t^-H^\top+R.
$$

$\boldsymbol\nu_t$ 是实际采样点相对于预测观测分布中心的偏差，即创新。

**更新：**

$$
K_t=P_t^-H^\top S_t^{-1},
$$

$$
\boldsymbol\mu_t=\boldsymbol\mu_t^-+K_t\boldsymbol\nu_t,
\qquad
P_t=(I-K_tH)P_t^-.
$$

于是“概率分布计算”已经完全转化为一组数值运算：传播均值和协方差、计算观测样本的残差、计算创新协方差、用矩阵增益修正均值和协方差。观测样本影响 $\boldsymbol\mu_t$ 的具体位置；$P_t$ 由模型协方差决定，在标准线性高斯模型中与观测样本的具体数值无关。

一维直接观测时，这些公式退化为

$$
K_t=\frac{P_t^-}{P_t^-+r},
\qquad
\mu_t=\mu_t^-+K_t(y_t^{\mathrm{obs}}-\mu_t^-),
\qquad
P_t=(1-K_t)P_t^-.
$$

## Hierarchical Gaussian Filter (HGF)

### 1. HGF 想解决什么问题？

前面的 HMM 和卡尔曼滤波都假设我们已经知道状态如何变化、观测有多嘈杂。HGF 关注的是更难的一类学习问题：环境中的规律本身也在变化，而且学习者不知道变化得有多快。

最简单的强化学习更新是

$$
\text{新估计}=\text{旧估计}+\alpha\times\text{预测误差}.
$$

固定学习率 $\alpha$ 有明显限制：环境稳定时，较小的 $\alpha$ 能抑制噪声；环境突然改变时，较小的 $\alpha$ 又会反应太慢。理想的贝叶斯学习可以解决这个问题，但每个试次都要计算隐藏状态的积分，层级一多还会遇到高维积分，难以实时运行。

Mathys 等人的 HGF 试图同时解决三个问题：

1. 用概率模型表示环境状态、感知噪声和环境波动；
2. 从数据中推断波动性，让有效学习率随环境变化；
3. 用变分贝叶斯把难以在线计算的积分近似成逐试次的解析更新，并允许不同个体拥有不同的学习参数。

这里的“个体差异”不是简单地给每个人设置一个固定学习率。HGF 让个体参数控制层级之间的耦合，例如一个人是否相信环境很快会改变，以及高层波动性信念对低层学习的影响有多强。

### 2. 层级生成模型：状态、波动性和观测

用试次 $k$ 表示时间。以二元结果为例，$u_k\in\{0,1\}$ 是观测，$x_{1,k}$ 是当前结果背后的二元状态，$x_{2,k}$ 表示结果为 1 的倾向，$x_{3,k}$ 表示这种倾向变化得有多快。

#### 2.1 最底层：从倾向产生结果

第二层使用 log-odds 表示概率：

$$
x_{1,k}\mid x_{2,k}
\sim\operatorname{Bernoulli}(\sigma(x_{2,k})),
\qquad
\sigma(x)=\frac{1}{1+e^{-x}}.
$$

如果 $x_{2,k}=0$，则结果为 1 的概率是 $0.5$；$x_{2,k}$ 越大，结果为 1 越可能；$x_{2,k}$ 越小，结果为 0 越可能。最简单的任务中观测直接等于第一层状态：$u_k=x_{1,k}$。

$$
u_k\mid x_{1,k}\sim\mathcal N(x_{1,k},\pi_u^{-1}),
$$

其中 $\pi_u$ 是观测精度。精度越高，观测越可靠；精度越低，观测对信念的影响越弱。

#### 2.2 更高层：用波动性控制低层随机游走

第二层不是固定不变的，而是作高斯随机游走：

$$
x_{2,k}\mid x_{2,k-1},x_{3,k}
\sim\mathcal N\left(x_{2,k-1},v_{2,k}\right),
\qquad
v_{2,k}=\exp(\kappa x_{3,k}+\omega_2).
$$

第三层本身也会变化：

$$
x_{3,k}\mid x_{3,k-1}
\sim\mathcal N(x_{3,k-1},\vartheta).
$$


- $\kappa$：层级耦合强度，决定 $x_3$ 对 $x_2$ 波动性的影响；
- $\omega_2$：第二层的 tonic volatility，即不随试次改变的基础波动性；
- $\vartheta$ 或 $\omega_3$：顶层波动性，决定环境波动性本身变化得多快。

$x_{3,k}$ 不是“结果是什么”，而是“结果倾向会不会很快改变”。它越大，$v_{2,k}$ 越大，模型越不相信旧的 $x_2$；它越小，模型越相信环境稳定。

### 3. 真正的目标：推断后验，而不是只拟合一个平均值

给定观测 $u_{1:k}$ 和参数 $\chi=\{\kappa,\omega_2,\omega_3,\vartheta,\ldots\}$，HGF 想得到

$$
p(x_{1,k},x_{2,k},x_{3,k}\mid u_{1:k},\chi).
$$

精确推断需要把上一时刻的联合后验沿随机游走传播，再乘当前观测的似然：

$$
p(x_k\mid u_{1:k},\chi)
\propto p(u_k\mid x_{1,k})
\int p(x_k\mid x_{k-1},\chi)
p(x_{k-1}\mid u_{1:k-1},\chi)\,dx_{k-1}.
$$

层级和非线性使这个积分通常没有简单闭式解。HGF 使用均值场变分贝叶斯近似：

$$
q(x_k)\approx q(x_{1,k})q(x_{2,k})q(x_{3,k}),
$$

其中二层和三层的近似后验用均值和方差表示：

$$
q(x_{i,k})=\mathcal N(\mu_{i,k},\sigma_{i,k}^2),\qquad i=2,3.
$$

变分方法最小化

$$
\mathcal F(q)=\mathbb E_q[\log q(x_k)-\log p(u_k,x_k\mid u_{1:k-1},\chi)],
$$

它相当于在可计算的近似分布中寻找最接近真实后验的结果。直观地说，HGF 不再保存完整的高维密度，而是保存每一层的中心 $\mu_i$ 和不确定性 $\sigma_i^2$。

### 4. 一次试次中如何更新？

#### 4.1 先做随机游走预测

对每一层，先把上一试次的后验传播成当前先验：

$$
\hat\mu_{i,k}=\mu_{i,k-1},
\qquad
\hat\sigma_{i,k}^2=\sigma_{i,k-1}^2+v_{i,k}.
$$

其中 $v_{2,k}=\exp(\kappa\mu_{3,k-1}+\omega_2)$，顶层使用固定的 $\vartheta$ 或 $\exp(\omega_3)$。这一步和卡尔曼滤波的“预测方差 = 旧方差 + 过程噪声”是同一种概率操作。

#### 4.2 底层预测误差

在二元任务中，看到 $u_k$ 之前，第二层对结果为 1 的预测是

$$
\hat m_{1,k}=\sigma(\hat\mu_{2,k}).
$$

因此底层 value prediction error（VAPE）可以写成

$$
\delta_{1,k}=u_k-\hat m_{1,k}.
$$

无论是二元还是连续输入，意思都是“实际输入与预测输入的差”。

在高斯观测的特例下，底层更新可以直接写成信息加权形式：

$$
\mu_{1,k}=\hat\mu_{1,k}
+\frac{\pi_u}{\hat\pi_{1,k}+\pi_u}\delta_{u,k},
\qquad
\sigma_{1,k}^2=\frac{1}{\hat\pi_{1,k}+\pi_u},
$$

其中 $\pi=1/\sigma^2$ 是精度。观测越可靠，$\pi_u$ 越大，底层更新越多。

#### 4.3 高层更新：预测误差乘以动态学习率

对任意高层 $i>1$，2014 年论文给出的均值更新（等间隔输入）可写为

$$
\boxed{\mu_{i,k}
=\hat\mu_{i,k}
+\underbrace{\frac12\kappa_{i-1}v_{i-1,k}
\frac{\hat\pi_{i-1,k}}{\pi_{i,k}}}_{\text{动态学习率}}
\underbrace{\delta_{i-1,k}}_{\text{下层预测误差}}.}
$$

这里先假设等间隔输入；不等间隔时还要把相应时间间隔乘入随机游走方差。$\hat\pi_{i-1,k}$ 是下层预测精度，$\pi_{i,k}=1/\sigma_{i,k}^2$ 是本层更新后的精度。公式显示，高层更新的有效步长由层级耦合 $\kappa$、下层的随机游走方差 $v$ 以及两层精度共同决定。2020 年论文将这一结构概括为“precision-weighted prediction error”。

高层的误差不是简单的结果误差，而是**波动性预测误差**（volatility prediction error, VOPE）。一个直观的近似写法是

$$
\delta_{i,k}
=\frac{\sigma_{i,k}^2+(\mu_{i,k}-\hat\mu_{i,k})^2}
{\hat\sigma_{i,k}^2}-1.
$$

分子是看到新数据后实际需要解释的总不确定性，分母是更新前预测的不确定性：

- $\delta_{i,k}>0$：实际变化比预期大，模型低估了波动性，应提高更高层的波动性信念；
- $\delta_{i,k}<0$：实际变化比预期小，模型高估了波动性，应降低更高层的波动性信念。

因此 HGF 的学习率不是外部手工指定的数字，而是由“我对下层预测有多确定”和“我认为环境会变化多快”共同计算出来的。

### 5. HGF 相比固定学习率多学了什么？

设第二层的后验方差为 $\sigma_{2,k}^2$。如果 $x_3$ 上升，$v_{2,k}$ 增大，预测精度下降，第二层的后验不确定性通常增大，于是新的预测误差更容易改变 $\mu_{2,k}$。这就是环境变得不稳定时自动提高学习率。

反过来，如果连续结果都符合预测，VOPE 会支持较低的波动性，$v_{2,k}$ 变小，模型逐渐相信环境稳定，新证据的影响减弱。这里要区分两种不确定性：

- **估计不确定性**：我不知道当前状态到底是多少，例如对 $x_2$ 的后验方差；
- **环境不确定性**：当前状态本身可能正在改变，例如随机游走方差 $v_{2,k}$。

HGF 把第二种不确定性放到更高层来推断，所以能够解释“我不确定”与“世界正在变”之间的区别。2014 年论文还将输入到达时间间隔 $\Delta t_k$ 乘到随机游走方差上：间隔越长，允许状态积累的变化越多。

### 6. HGF 中哪些量逐试次更新，哪些量是模型参数？

可以把 HGF 分成两层来看：每来一个新观测，模型更新“我现在相信环境是什么样”；分析一整段行为数据时，研究者再估计“这个人通常怎样更新信念”的参数。前者是状态推断，后者是参数拟合，二者不是同一次更新。

#### 6.1 每个试次更新：对环境状态的信念

在二元结果的三层 HGF 中，第一层是二元变量，后两层是连续变量。模型逐试次维护这些后验分布：

$$
q(x_{1,k})=\operatorname{Bernoulli}(\mu_{1,k}),\qquad
q(x_{i,k})=\mathcal N(\mu_{i,k},\sigma_{i,k}^2),\quad i=2,3.
$$

所以每次看到新的 $u_k$ 后，更新的状态信念包括：

| 逐试次数值 | 它回答的问题 |
| --- | --- |
| $\mu_{1,k}$ | 当前观测/结果是什么？ |
| $\mu_{2,k}$ | 当前结果倾向或建议者可靠性是多少？ |
| $\mu_{3,k}$ | 这个倾向正在变化得多快？ |
| $\sigma_{i,k}^2$ | 对相应状态估计有多不确定？ |

例如社会建议任务中，$\mu_{2,k}$ 是参与者在看过第 $k$ 次建议及反馈后，对建议者当前可靠性的信念。它会随试次改变，并不是模型训练完后固定不动的一个可靠性参数。

#### 6.2 控制更新方式：HGF 参数

模型参数决定上述信念如何更新。典型的三层 HGF 可以用下面这组参数概括：

$$
\chi=\{\kappa,\omega_2,\omega_3,\pi_u,
\mu_{2,0},\mu_{3,0},\sigma_{2,0}^2,\sigma_{3,0}^2\}.
$$

| 参数 | 控制什么 |
| --- | --- |
| $\kappa$ | 高层波动性对低层状态变化方差的影响强度 |
| $\omega_2$ | 第二层的基础波动性，决定即使没有高层波动信号时仍会有多快的变化 |
| $\omega_3$ | 第三层波动性本身的变化速度；某些写法用顶层方差 $\vartheta$ 表示 |
| $\pi_u$ | 观测精度；越大表示越相信当前输入 |
| $\mu_{i,0},\sigma_{i,0}^2$ | 初始信念及其不确定性 |

因此参数不会直接说“第 $k$ 次建议者有多可靠”；它们规定的是学习规则。例如，$\kappa$ 较大时，模型更愿意让对变化速度的判断影响可靠性信念；$\omega_2$ 较大时，即使没有明显波动证据，可靠性也允许较快变化。

参数符号随论文和模型版本会有差异。2011 年模型用顶层随机游走方差 $\vartheta$；2014 和 2020 年任务模型也可写成 $\exp(\omega_3)$。两种记法都表示顶层状态变化速度，但具体参数化要以所用模型为准。

#### 6.3 参数在什么时候估计？

如果 $\chi$ 已知，给定观测 $u_{1:K}$ 后，HGF 逐试次运行并得到 $q(x_{i,k})$。这是**状态推断/滤波**：参数固定，后验信念随观测变化。

在人类行为研究中，研究者还会看到参与者的选择 $y_k$，但看不到其内部信念。因此需要在 HGF 外再加一个决策模型，把信念转换成选择概率：

$$
p(y_k\mid \mathbf m_k,\mathbf s_k,\zeta),
$$

其中 $\mathbf m_k,\mathbf s_k$ 收集各层的后验均值和不确定性，$\zeta$ 表示决策参数，例如决策噪声或社会建议权重。然后用整段输入和选择估计个体参数：

$$
(\chi^*,\zeta^*)
=\arg\max_{\chi,\zeta}
\left[
\sum_{k=1}^{K}\log p(y_k\mid \mathbf m_k(u_{1:k},\chi),\mathbf s_k(u_{1:k},\chi),\zeta)
+\log p(\chi,\zeta)
\right].
$$

实际计算中，候选参数 $\chi$ 决定 HGF 每一试次的后验轨迹；决策模型据此计算观察到的选择有多可能；参数估计算法再调整 $\chi,\zeta$，寻找能解释整段行为的值。2014 年论文称这类“从被试的选择反推被试内部贝叶斯模型”的流程为 model inversion / observing the observer。

| 任务 | 已知 | 更新或估计什么 |
| --- | --- | --- |
| 逐试次 HGF 滤波 | HGF 参数 $\chi$ 和新观测 $u_k$ | 状态后验 $\mu_{i,k},\sigma_{i,k}^2$ |
| 个体参数拟合 | 整段观测 $u_{1:K}$ 和行为 $y_{1:K}$ | 个体参数 $\chi$ 和决策参数 $\zeta$ |

标准离线分析通常不会在每个试次都改写 $\kappa,\omega_2,\omega_3$；而是先用一组固定参数跑完整段 HGF，再根据整体行为拟合这些参数。只有明确采用在线参数学习扩展时，参数才会随新数据继续改变。

#### 6.4 用社会建议任务核对这些量

2020 年论文把三层状态解释为：

$$
x_{1,k}=\text{当前这条建议是否准确},\qquad
x_{2,k}=\text{建议者提供有效建议的倾向},\qquad
x_{3,k}=\text{该倾向变化速度的 log-volatility}.
$$

所以每次收到建议和彩票反馈后，参与者更新的是 $\mu_{2,k}$（当前对建议者可靠性的看法）以及 $\mu_{3,k}$（对可靠性是否正在改变的看法）。整段选择数据拟合出的 $\kappa,\omega_2,\omega_3$ 则描述参与者的学习风格。模型还有响应参数：$\zeta$ 控制社会建议相对非社会线索的权重，$\beta$ 控制选择的确定程度。注意这些响应参数不等于 HGF 的隐藏状态。

### 7. HGF 的一句话总结

HGF 把“看到一个意外结果”拆成三层问题：结果本身错了多少，底层规律是否正在变化，以及这种变化速度是否也在改变。运行 HGF 时逐试次更新的是环境状态的后验信念；拟合行为数据时估计的是控制这些信念如何变化的个体参数。变分贝叶斯让前者可以高效递推，同时保留强化学习式预测误差更新的直觉。

## Momentum Learning：情绪作为奖励动量的表征

### 1. 这篇论文要解决什么问题？

Eldar 等人的论文讨论的是一个不同于 SGD 的“动量”问题：为什么情绪会被最近的奖励结果影响，而情绪又会反过来改变我们对下一次奖励的判断？论文把已有现象组织成一个计算假说：**情绪是近期奖励变化趋势的内部表征**。

这解决的是标准强化学习的一个局限。若每个状态的奖励独立，经典更新足够：

$$
V_{s,t+1}=V_{s,t}+\alpha_t(r_t-V_{s,t}).
$$

但自然环境往往有相关性。春天的果实逐渐增多时，这一次高于预期的奖励不仅说明“这棵树不错”，还暗示附近树木或下一时刻的奖励可能也在上升。如果仍把每个结果当作相互独立，学习会滞后。

论文是一篇理论性综述与模型观点文章，不是用一个新实验单独证明全部机制。它把已有的情绪、奖励预测误差和行为研究与一个概率状态空间模型连接起来。

### 2. 从奖励预测误差到情绪

奖励预测误差为

$$
\delta_t=r_t-V_t.
$$

已有研究显示，主观幸福感更依赖“结果比预期好还是坏”，而不是简单依赖累计得到多少钱。因此可以用一个带遗忘的误差平均表示情绪/动量：

$$
m_{t+1}=m_t+\alpha'_t(\delta_t-m_t).
$$

这里 $m_t$ 是近期预测误差的运行平均：连续正误差使 $m_t$ 上升，连续负误差使它下降；$\alpha'_t$ 控制情绪对新误差的记忆速度。它不是把每次奖励简单相加，而是在估计“最近的奖励变化整体偏向哪个方向”。

### 3. 用概率模型表示奖励的趋势

论文 Box 2 区分了三种环境：

1. 各状态相互独立，学习每个状态自己的奖励均值；
2. 多个状态受一个共同环境因素影响，某个状态的误差可以迁移给相关状态；
3. 即使只有一个状态，奖励均值也可能沿着一个持续的趋势变化。

对第三种情况，令 $V_t$ 为当前奖励基线，$m_t$ 为奖励均值的动量，建立局部线性趋势模型：

$$
r_t\mid V_t\sim\mathcal N(V_t,\sigma_r^2),
$$

$$
V_t=V_{t-1}+m_{t-1}+\epsilon_t,
\qquad
m_t=m_{t-1}+\zeta_t,
$$

其中 $\epsilon_t\sim\mathcal N(0,q_V)$、$\zeta_t\sim\mathcal N(0,q_m)$。$m_t>0$ 表示奖励均值预计继续上升，$m_t<0$ 表示预计下降；$q_m$ 控制趋势本身改变得有多快。

写成状态向量 $\mathbf h_t=[V_t,m_t]^\top$：

$$
\mathbf h_t
=\underbrace{\begin{bmatrix}1&1\\0&1\end{bmatrix}}_{F_m}
\mathbf h_{t-1}
+\begin{bmatrix}\epsilon_t\\\zeta_t\end{bmatrix},
\qquad
r_t=\underbrace{\begin{bmatrix}1&0\end{bmatrix}}_{H_m}\mathbf h_t+\eta_t.
$$

这正是一个二维线性高斯状态空间模型：奖励是观测，基线和趋势是不可直接观察的状态。因此 Kalman filter 可以同时估计“现在的奖励水平”和“奖励正在往哪里走”。

### 4. 情绪如何帮助学习，而不是只描述情绪

如果 $m_t$ 只被当作一个内部标签，模型还没有说明它为什么有用。论文提出，情绪会偏置后续结果的主观感知：

$$
\tilde r_t=r_t+f_t m_t,
$$

再把主观结果放回价值更新：

$$
\begin{aligned}
V_{s,t+1}
&=V_{s,t}+\alpha_t(\tilde r_t-V_{s,t})\\
&=V_{s,t}+\alpha_t\left[(r_t-V_{s,t})+f_t m_t\right].
\end{aligned}
$$

当趋势为正时，后续奖励被主观上看得稍好，价值估计向上追赶正在上升的环境；当趋势为负时，结果被看得稍差，价值估计更快向下追赶。这样，情绪相当于把“其他相关状态或相邻时刻的变化”迁移到当前学习中，弥补独立状态 RL 的滞后。

参数 $f_t$ 控制动量偏置的强度。它可以理解为环境相关性、状态相似性和个体情绪影响之间的综合系数。若状态实际上互不相关，或趋势已经反转，过强的 $f_t$ 就会把错误的历史趋势带入当前判断。

### 5. 适应性与可能的失灵

在奖励确实有持续趋势的环境中，动量机制是有益的：正向趋势让估计快速跟上资源增加，负向趋势让行为快速适应资源减少。论文用“春天资源增加、冬天资源减少”的例子说明，情绪可以帮助动物少用几次试错就捕捉到季节变化。

但同一个反馈环也可能失灵：

- 情绪偏置过弱，学习仍把相关结果当成独立结果，适应趋势太慢；
- 对负面结果学习过慢，可能形成过度乐观预期，之后不断遭遇负预测误差；
- 情绪偏置过强或持续过久，正反馈可能形成“坏心情使结果显得更差，结果更差又进一步降低心情”的循环；
- 正向循环也可能把情绪和预期推得过高，随后因现实不及过高预期而反向摆动。

因此论文把抑郁和双相情绪波动作为可能的计算失灵方向进行讨论，而不是声称单一的动量变量能够诊断这些疾病。

### 6. 和 HGF、Kalman filter 的关系

Momentum 模型和 HGF 都在回答“环境结构是否在变化”，但关注的隐状态不同：

- HGF 的高层状态主要估计**变化的波动性和不确定性**，从而调节学习率；
- Momentum 模型增加了**变化的方向**，即奖励基线正在上升还是下降；
- 这篇论文的奖励趋势模型可以直接写成 Kalman 状态空间模型，因此 Kalman filter 是估计 $V_t,m_t$ 的自然算法；
- 这里的 Momentum 不是神经网络优化器中的 SGD Momentum。

参考：Eldar et al. (2016), *Mood as Representation of Momentum*（本地文件名 `Eldar et al.(2015).pdf`；文件对应论文的 2016 年卷期）。
