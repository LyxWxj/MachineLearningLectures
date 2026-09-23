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
| $p(m_{1:t})$ | $p(m_1, m_2, \dots, m_t)$ |
| $p(\cdot\| m_{1:t})$ | $p(\cdot\|m_1, m_2, \dots, m_t)$ |
| $\mathcal{N}(x;\mu,\sigma^2)$ | $X \sim \mathcal{N}(\mu, \sigma^2)$时,X的概率密度函数$f$在$X=x$处的值 |


本节把贝叶斯更新、马尔可夫状态转移和预测误差学习连成一条脉络：HMM 用离散隐状态解释观测；HGF 进一步推断环境状态及其变化的不确定性；动量模型解释奖励为何随时间相关，以及情绪如何帮助学习者追踪趋势；Kalman filter 则给出线性高斯状态空间模型的递归贝叶斯解。

## HMM

### Markov Model

马尔可夫模型假设当前状态只依赖于前一时刻的状态：

$$
p(z_t\mid z_{1:t-1})=p(z_t\mid z_{t-1}).
$$

状态转移矩阵 $A$ 的元素是 $A_{ij}=p(z_t=j\mid z_{t-1}=i)$，每一行之和为 1。若 $\boldsymbol\pi_{t-1}$ 是行向量形式的状态分布，则一步预测为 $\boldsymbol\pi_{t|t-1}=\boldsymbol\pi_{t-1}A$。

### Hidden Markov Model

HMM 在马尔可夫链上增加观测模型：隐状态按转移分布演化，而观测只依赖当前隐状态。

$$
p(z_{1:T},x_{1:T})=p(z_1)\prod_{t=2}^{T}p(z_t\mid z_{t-1})\prod_{t=1}^{T}p(x_t\mid z_t).
$$

生成模型描述由状态产生观测的方向；推断则由观测反推状态。前向递推先预测，再用新观测更新：

$$
p(z_t\mid x_{1:t})\propto p(x_t\mid z_t)\sum_{z_{t-1}}p(z_t\mid z_{t-1})p(z_{t-1}\mid x_{1:t-1}).
$$

这就是离散状态下的递归贝叶斯更新。HGF 保留“状态转移预测、观测似然校正”结构，但使用连续高斯变量，并把环境变化本身也纳入隐状态。

## Hierarchical Gaussian Filter (HGF)

### 动机：学习环境，也学习环境变化得有多快

前面介绍的 delta-rule / Rescorla–Wagner 学习可写成

$$
\text{新估计}=\text{旧估计}+\text{学习率}\times\text{预测误差}.
$$

固定学习率在稳定环境中可以抑制噪声，但环境突然改变时又会适应得太慢。HGF 的动机是让学习者根据证据推断环境是否稳定，并据此调整有效学习率。它把单层的“状态估计”扩展为层级模型：低层表示当前事件的倾向，更高层表示低层状态如何变化，再高层表示这种变化的波动性。

### 二元输入下的层级生成模型

令 $u_t\in\{0,1\}$ 为观测，$x_{1,t}$ 为第一层二元状态，$x_{2,t}$ 为第二层连续状态。简化的确定性观测映射下 $u_t=x_{1,t}$，而二层状态决定第一层取 1 的概率：

$$
x_{1,t}\mid x_{2,t}\sim\operatorname{Bernoulli}(\sigma(x_{2,t})),\qquad u_t=x_{1,t},\qquad
\sigma(x)=\frac{1}{1+e^{-x}}.
$$

$x_2$ 是概率的 log-odds 尺度：$x_2=0$ 表示两种结果等可能，正值提高 $u=1$ 的概率。二层按高斯随机游走变化，但其方差由第三层控制：

$$
x_{2,t}\mid x_{2,t-1},x_{3,t}\sim\mathcal{N}\!\left(x_{2,t-1},\exp(\kappa x_{3,t}+\omega)\right),\qquad
x_{3,t}\mid x_{3,t-1}\sim\mathcal{N}(x_{3,t-1},\vartheta).
$$

$x_3$ 编码 $x_2$ 的 log-volatility（对数波动性）。$x_3$ 越大，$x_2$ 的一步预测方差越大，模型就认为环境可能快速改变；$x_3$ 较低则意味着环境较稳定。$\kappa,\omega,\vartheta$ 是模型参数。层级还可以向上扩展，让更高层解释下一层的波动性。

### 预测误差与自适应学习率

设观测前对第二层的近似后验为 $q(x_{2,t})=\mathcal{N}(\mu_{2,t},\sigma^2_{2,t})$。先按随机游走传播：

$$
\hat\mu_{2,t}=\mu_{2,t-1},\qquad
\hat\sigma^2_{2,t}=\sigma^2_{2,t-1}+\exp(\kappa\mu_{3,t-1}+\omega).
$$

观测带来的第一层误差可记作 $\delta_{1,t}=\hat m_{1,t}-\sigma(\hat\mu_{2,t})$，其中 $\hat m_{1,t}$ 是看到 $u_t$ 后对第一层状态的后验均值。第二层均值更新具有熟悉的 delta-rule 形式：

$$
\mu_{2,t}=\hat\mu_{2,t}+\sigma^2_{2,t}\delta_{1,t},\qquad
\sigma^2_{2,t}=\left[(\hat\sigma^2_{2,t})^{-1}+\hat\pi_{1,t}\right]^{-1}.
$$

$\hat\pi_{1,t}$ 是第一层预测的精度（方差的倒数）。因此 $\sigma^2_{2,t}$ 扮演**有效学习率**：对 $x_2$ 越不确定，观测误差影响越大；对输入预测越不确定，同样大小的误差就越不值得相信。高层状态 $x_3$ 通过预测方差影响第二层不确定性，进而调节学习率。HGF 的要点不是简单加一个固定误差项，而是让学习者在线推断误差应获得多大权重。

精确贝叶斯推断通常涉及难以解析的积分；HGF 采用变分贝叶斯对各层后验作高斯近似，从而得到高效的逐试次更新。完整 HGF 的高层方差更新还含更完整的二阶预测误差项；这里的式子突出均值更新和精度加权机制，不代表所有 HGF 变体的完整实现。

### 与 HMM 及强化学习的关系

- HMM 的状态是有限个离散类别；HGF 用连续高斯信念表示倾向、置信度和波动性。两者都执行“状态转移预测，再由观测似然修正”。
- 固定学习率 delta-rule 只看当前预测误差；HGF 还根据后验不确定性和环境波动推断该误差应有多大影响。
- HGF 可用于概率学习、社会信息学习和计算精神病学。2020 年本地论文将它用于社会建议任务，分析参与者如何学习建议者的可靠性，并比较不同个体的信念更新特征。

参考：Mathys et al. (2011), [A Bayesian foundation for individual learning under uncertainty](https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2011.00039/full)；Mathys et al. (2014), [Uncertainty in perception and the Hierarchical Gaussian Filter](https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2014.00825/full)；Diaconescu et al. (2020), *Hierarchical Bayesian Models of Social Inference for Probing Persecutory Delusional Ideation*（本地文件 `2020-57436-003.pdf`）。

## Momentum Learning：情绪作为奖励动量的表征

### 动机：相邻结果往往不是独立的

前面的强化学习更新常写为

$$
V_{s,t+1}=V_{s,t}+\alpha(r_t-V_{s,t}).
$$

它适合按状态累积奖励预测误差，但许多自然环境有持续趋势：春天果实逐渐增多时，一次高于预期的结果意味着后续奖励也可能更高。若奖励变化具有时间相关性，只估计每个状态的均值就会浪费这种信息。Eldar 等人提出，情绪可以表征近期奖励预测误差的总体动量，帮助学习者推断共同趋势并用于随后判断。这里的 Momentum 指环境奖励趋势，不是优化器里的 SGD Momentum。

### “均值 + 趋势”的状态空间模型

令 $V_t$ 表示当前平均奖励，$m_t$ 表示其变化趋势：

$$
r_t\mid V_t\sim\mathcal{N}(V_t,\sigma_r^2),\qquad
V_t=V_{t-1}+m_{t-1}+\epsilon_t,\qquad
m_t=m_{t-1}+\zeta_t,
$$

其中 $\epsilon_t\sim\mathcal{N}(0,q_V)$、$\zeta_t\sim\mathcal{N}(0,q_m)$。正的 $m_t$ 表示奖励均值倾向继续上升，负值表示倾向下降；随机游走允许趋势缓慢改变。等价的向量形式为

$$
\mathbf{s}_t=\begin{bmatrix}V_t\\m_t\end{bmatrix}
=\begin{bmatrix}1&1\\0&1\end{bmatrix}\begin{bmatrix}V_{t-1}\\m_{t-1}\end{bmatrix}
+\begin{bmatrix}\epsilon_t\\\zeta_t\end{bmatrix},\qquad
r_t=\begin{bmatrix}1&0\end{bmatrix}\mathbf{s}_t+\eta_t,\quad \eta_t\sim\mathcal{N}(0,\sigma_r^2).
$$

奖励 $r_t$ 是带噪声的观测，潜在状态同时包含基线和趋势。Kalman filter 因而可以从奖励序列在线估计二者：连续高于预测的奖励会推动基线上升；若这种偏差持续出现，也支持正向趋势估计。趋势还可帮助预测相关状态或相邻时刻的奖励。

### 情绪如何影响价值学习

若情绪状态 $m_t$ 汇总了近期误差，学习者可把它作为对新结果评价的偏置：

$$
\tilde r_t=r_t+f m_t,\qquad
V_{s,t+1}=V_{s,t}+\alpha_t(\tilde r_t-V_{s,t})
=V_{s,t}+\alpha_t\big[(r_t-V_{s,t})+f m_t\big].
$$

$f$ 控制趋势迁移到当前奖励判断的强度。正向动量会使结果主观上更好，负向动量使其更差，帮助多个相关状态的价值更快跟上共同环境变化。若状态之间实际无关、趋势已经反转，或情绪估计过强，这种迁移也会造成偏差。该论文提出的是解释情绪与学习交互的计算理论，不表示每种情绪变化都等同于精确的 Kalman 状态估计。

**与 HGF 的联系**：二者都试图推断环境结构，而不是假定固定学习率和独立误差。HGF 的高层主要表示低层变化的波动性（变化有多不确定/剧烈）；动量模型增加了变化方向（奖励正在上升还是下降）。它们是互补思路，并非同一个模型。

参考：Eldar et al. (2016), [Mood as Representation of Momentum](https://doi.org/10.1016/j.tics.2015.12.010)（本地文件名为 `Eldar et al.(2015).pdf`，论文发表卷期为 2016 年）。

## Kalman Filter

### 直觉：在预测和带噪测量之间折中

想象用传感器追踪一个移动目标。上一刻的状态和运动规律给出目标现在大概位置的**预测**；传感器给出一个有噪声的**测量**。只信预测会忽略新信息，只信测量又会被噪声带着跳。Kalman filter 按两者的不确定性加权：预测越不确定，越听测量；测量越嘈杂，越听预测。它是线性高斯状态空间模型的递归贝叶斯解，也是前面 HMM“先预测、再更新”在连续状态下的对应形式。

### 一维模型：预测一步，再用观测纠偏

令隐藏状态 $s_t$ 按线性动力学变化，观测 $y_t$ 是状态的带噪测量：

$$
s_t=a s_{t-1}+w_t,\quad w_t\sim\mathcal{N}(0,q),\qquad
y_t=h s_t+v_t,\quad v_t\sim\mathcal{N}(0,r).
$$

若上一步后验为 $s_{t-1}\mid y_{1:t-1}\sim\mathcal{N}(\mu_{t-1},P_{t-1})$，先传播得到当前先验：

$$
\mu_t^-=a\mu_{t-1},\qquad P_t^-=a^2P_{t-1}+q.
$$

$\mu_t^-$ 是未看新读数时的状态预测；$P_t^-$ 是预测方差，过程噪声 $q$ 增加不确定性。预测观测为 $h\mu_t^-$，创新（残差）$e_t=y_t-h\mu_t^-$，其方差 $S_t=h^2P_t^-+r$。Kalman 增益和后验为

$$
K_t=\frac{P_t^-h}{h^2P_t^-+r},\qquad
\mu_t=\mu_t^-+K_t e_t,\qquad
P_t=(1-K_th)P_t^-.
$$

当 $h=1$ 时，$K_t=P_t^-/(P_t^-+r)$，后验均值就是预测与读数的加权平均。$r$ 大（传感器不可靠）时 $K_t$ 小，修正少；$P_t^-$ 大（状态预测不可靠）时 $K_t$ 大，更采纳当前读数。$P_t$ 则表示更新后还剩多少不确定性。因此 Kalman filter 不只是平滑观测值，而是维护状态后验及其不确定性，并用它决定下一步该相信谁。

### 向量形式与应用边界

多维状态和观测可以写为

$$
\mathbf{s}_t=F\mathbf{s}_{t-1}+\mathbf{w}_t,\quad \mathbf{w}_t\sim\mathcal{N}(0,Q),\qquad
\mathbf{y}_t=H\mathbf{s}_t+\mathbf{v}_t,\quad \mathbf{v}_t\sim\mathcal{N}(0,R).
$$

递推公式为

$$
\begin{aligned}
\boldsymbol\mu_t^-&=F\boldsymbol\mu_{t-1}, &P_t^-&=FP_{t-1}F^\top+Q,\\
K_t&=P_t^-H^\top(HP_t^-H^\top+R)^{-1}, &\boldsymbol\mu_t&=\boldsymbol\mu_t^-+K_t(\mathbf y_t-H\boldsymbol\mu_t^-),\\
&&P_t&=(I-K_tH)P_t^-.
\end{aligned}
$$

$Q$ 表示动力学不确定性，$R$ 表示测量噪声；增益矩阵 $K_t$ 按二者的相对大小决定状态修正。线性高斯假设成立时，这个后验均值和协方差是精确的。Kalman **filter** 只使用当前与过去观测作在线估计；利用整段序列回头修正过去状态则称为 Kalman smoother。非线性系统通常采用 EKF/UKF 等近似，明显非高斯问题可考虑粒子滤波。

**联系总结**：HMM 是离散状态的递归贝叶斯滤波；Kalman filter 是线性高斯连续状态下的闭式递推；HGF 则把层级状态和波动性纳入模型，通常用变分近似推断。三者共享“预测—观测更新”的逻辑，区别在于状态空间和可用的推断方法。

参考：本地第 8 章《马尔可夫过程与卡尔曼滤波》的 1D 信息加权直觉；Eldar et al. (2016) 对“基线奖励 + 动量”概率 Kalman-filter 模型的讨论。
