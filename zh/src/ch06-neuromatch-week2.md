# Neuromatch 笔记本 — 第 2 周

线性系统 (Linear Systems) · 生物神经元模型 (Biological Neuron Models) · 动力系统 (Dynamical Systems)

---

## 概述 (Overview)

第 2 周聚焦于**动力系统与神经模型 (dynamical systems and neural models)** ——从线性系统到生物神经元模型，再到网络动力学：

| 天数     | 主题                                      | 核心技能                                                                                                                     |
| -------- | ----------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| **W2D3** | 线性系统 (Linear Systems)                 | 欧拉积分 (Euler integration)、振荡 (Oscillations)、随机游走 (Random walks)、OU 过程、自回归模型 (AR models)                  |
| **W2D4** | 生物神经元模型 (Biological Neuron Models) | 泄漏积分发放神经元 (LIF neuron)、电导突触 (Conductance synapses)、短期可塑性 (STP)、脉冲时间依赖可塑性 (STDP)                |
| **W2D5** | 动力系统 (Dynamical Systems)              | 放电频率模型 (Firing rate models)、Wilson-Cowan 模型、相平面分析 (Phase plane)、雅可比矩阵 (Jacobian)、极限环 (Limit cycles) |

**统一主题**：神经元和网络如何随时间演化，以及我们如何用数学方法对其动力学进行建模？

---

## W2D3：线性系统 (Linear Systems)

---

### Tutorial 1：一维微分方程 (One-Dimensional Differential Equations)

最简单的动力系统：$\dot{x} = ax$

**解析解 (Analytical solution)**：$x(t) = x_0 e^{at}$

| $a$                  | 行为 (Behavior)                                      |
| -------------------- | ---------------------------------------------------- |
| $a < 0$              | 指数衰减 → 0 (Exponential decay → 0)                 |
| $a > 0$              | 指数增长 → ∞ (Exponential growth → ∞)                |
| $a = \text{complex}$ | 振荡（伴随增长/衰减）(Oscillation with growth/decay) |

**前向欧拉积分 (Forward Euler integration)**（数值解 (numerical solution)）：

$$
x(t_i) = x(t_{i-1}) + \dot{x}(t_{i-1}) \cdot dt
$$

对于 $\dot{x} = ax$ 具体形式：$x[k] = x[k-1] + a \cdot x[k-1] \cdot dt$

**实现细节**：使用 `dtype=complex` 处理复数 $a$（振荡动力学需要）

---

### Tutorial 1：复数 $a$ 与振荡动力学

当 $a$ 为复数时（$a = \text{real} + i \cdot \text{imag}$），系统产生振荡：

$$
x(t) = x_0 e^{(\text{real} + i \cdot \text{imag})t} = x_0 e^{\text{real} \cdot t} \cdot [\cos(\text{imag} \cdot t) + i \sin(\text{imag} \cdot t)]
$$

#### 为什么复数 $a$ 导致振荡？——欧拉公式的几何直觉

核心在于 **欧拉公式**：

$$
e^{i\theta} = \cos\theta + i\sin\theta
$$

**几何含义**：乘以 $e^{i\theta}$ 等价于在复平面上**旋转** $\theta$ 弧度。

当系统的微分方程 $\dot{x} = ax$ 中的 $a$ 为纯虚数（即 $a = i\omega$）时：

$$
x(t) = x_0 e^{i\omega t} = x_0 [\cos(\omega t) + i\sin(\omega t)]
$$

在复平面上，这是以 $\omega$ 为角速度的**匀速圆周运动**——其在实轴上的投影是 $\cos(\omega t)$，虚轴上的投影是 $\sin(\omega t)$，两者都是振荡的。

**为什么是 e 的指数形式？**

$\dot{x} = ax$ 的解来自微分方程的基本性质——导数等于自身乘以常数。唯一满足这一性质的函数是指数函数 $e^{at}$：

$$
\frac{d}{dt}\big(e^{at}\big) = a \cdot e^{at}
$$

当 $a$ 是实数时，$e^{at}$ 是**单调增长或衰减**；当 $a$ 是虚数时，$e^{i\omega t}$ 是**纯振荡**（幅值不变）；当 $a$ 是复数时，$e^{(\sigma + i\omega)t} = e^{\sigma t} \cdot e^{i\omega t}$ 是**振荡 + 包络增长/衰减**。

**可视化**：

![欧拉公式与复平面振荡](../../assets/complex_oscillation.png)

图中展示了：

- **左图（复平面轨迹）**：红色圆点为 $t=0$ 时刻的初始点，蓝色轨迹随时间沿单位圆逆时针旋转（$a = i\omega$），轨迹在实轴和虚轴上的投影分别为余弦和正弦
- **右图（实部/虚部时域图）**：实部为余弦振荡（蓝色），虚部为正弦振荡（橙色）
- **复数 $a$ 的一般情况**：当 $a = \sigma + i\omega$ 时，轨迹变为螺旋线（$e^{\sigma t}$ 控制螺旋半径的收缩或扩张）

**关键洞察 (Key insight)**：

- **实部 (Real part)** → 增长/衰减率（振幅包络）
- **虚部 (Imaginary part)** → 振荡频率

**稳定振荡条件**：设实部 = 0，虚部 = $2\pi f$

例如：产生 0.5 Hz 的稳定振荡 → 虚部 = $2\pi \times 0.5 = \pi \approx 3.14$

**增长振荡条件**：实部 > 0 且 虚部 ≠ 0

---

### Tutorial 1：二维线性系统 (Two-Dimensional Linear Systems)

扩展到二维：$\dot{\mathbf{x}} = \mathbf{A}\mathbf{x}$

$$
\begin{bmatrix} \dot{x}_1 \\ \dot{x}_2 \end{bmatrix} = \begin{bmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \end{bmatrix}
$$

**数值求解**：使用 `scipy.integrate.solve_ivp`（而非手动欧拉法）

**流线图 (Stream plot)**：在网格上计算 $\mathbf{A}\mathbf{x}$，箭头显示状态变化方向

**特征向量 (Eigenvectors)**：$\mathbf{A}\mathbf{x}$ 与 $\mathbf{x}$ 平行的方向（不变方向）

**特征值 (Eigenvalues)**：$\mathbf{A}\mathbf{x}$ 沿特征向量方向的拉伸/压缩因子

**稳定性分类 (Stability classification)**：

| 特征值类型 | 行为                                 |
| ---------- | ------------------------------------ |
| 均为负实数 | 稳定节点 (Stable node)（收敛到原点） |
| 均为正实数 | 不稳定节点 (Unstable node)（发散）   |
| 符号相反   | 鞍点 (Saddle point)                  |
| 复数       | 振荡/螺旋 (Oscillation / Spiral)     |

---

### Tutorial 2：马尔可夫过程 (Markov Processes)

**马尔可夫性质**：当前状态完全决定下一状态的转移（无记忆性）

换句话说，也就是下一状态完全依赖当前状态。

**电报过程 (Telegraph process)**：两态离子通道模型

- 状态：关闭 (0) 和 打开 (1)
- 转移概率：$P(0 \to 1 | x=0) = \mu_{c2o}$，$P(1 \to 0 | x=1) = \mu_{o2c}$

**泊松过程**：以恒定速率 $\lambda$ 发生的事件序列，是描述随机事件计数的最基本模型。

#### 定义与三条件

计数过程 $\{N(t), t \geq 0\}$ 被称为速率为 $\lambda$ 的泊松过程，当且仅当：

1. **独立增量**：在不相交的时间区间内发生的事件数相互独立
2. **平稳增量**：在任意长度为 $s$ 的区间内，事件数的分布只与长度 $s$ 有关，与起点无关
3. **普通性**：在极短时间 $\Delta t$ 内，发生多于一个事件的概率是 $\Delta t$ 的高阶无穷小：
   $$P(N(\Delta t) \geq 2) = o(\Delta t)$$

#### 泊松分布

在长度为 $t$ 的区间内恰好发生 $k$ 个事件的概率服从泊松分布：

$$P(N(t) = k) = \frac{(\lambda t)^k}{k!} e^{-\lambda t}, \quad k = 0, 1, 2, \ldots$$

- 均值：$\mathbb{E}[N(t)] = \lambda t$
- 方差：$\text{Var}[N(t)] = \lambda t$
- 均值和方差相等是泊松分布的重要特征

#### 等待时间与指数分布

相邻事件之间的间隔时间 $T_i$（等待时间）服从**指数分布**：

$$f_T(t) = \lambda e^{-\lambda t}, \quad t \geq 0$$

推导思路：
$$P(T_1 > t) = P(N(t) = 0) = e^{-\lambda t}$$

这意味着无记忆性：
$$P(T > s + t \mid T > s) = P(T > t)$$

**与马尔可夫过程的联系**：指数分布的无记忆性是泊松过程具有马尔可夫性质的根源。

#### 泊松过程的模拟

```python
import numpy as np

# 方法 1：对等待时间采样（利用指数分布）
def poisson_process_exponential(rate, T, seed=42):
    """用指数等待时间模拟泊松过程到时间 T"""
    rng = np.random.default_rng(seed)
    events = []
    t = 0
    while t < T:
        # 采样下一个等待时间 Exp(rate)
        t += rng.exponential(1 / rate)
        if t < T:
            events.append(t)
    return np.array(events)

# 方法 2：对计数采样（利用泊松分布）
def poisson_process_counting(rate, T, delta=0.01, seed=42):
    """用固定时间片内的泊松计数模拟"""
    rng = np.random.default_rng(seed)
    n_bins = int(T / delta)
    # 每个 bin 内的事件数 ~ Poisson(rate * delta)
    counts = rng.poisson(rate * delta, n_bins)
    return counts

rate = 2.0  # 每秒平均 2 个事件
events = poisson_process_exponential(rate, T=10)
print(f"在 10 秒内发生了 {len(events)} 个事件 (期望: {rate * 10})")
```

#### 泊松过程与二项分布的关系

泊松分布可以看作二项分布的极限情况：将 $[0, t]$ 划分为 $n$ 个小区间，每个区间内发生事件的概率为 $p = \lambda t / n$，当 $n \to \infty$ 时：

$$\lim_{n \to \infty} \binom{n}{k} p^k (1-p)^{n-k} = \frac{(\lambda t)^k}{k!} e^{-\lambda t}$$

#### 与 Neuron 放电的联系

在神经科学中，如果忽略不应期，可以近似认为神经元的放电时刻构成一个泊松过程：
- 每个时间 bin 内放电的概率为 $\lambda \Delta t$（$\lambda$ 是放电率）
- 放电率 $\lambda$ 受刺激强度调制 → **非齐次泊松过程**
- ISI（脉冲间间隔）的直方图呈指数递减（与真实神经元中的伽马分布对比）

**状态转移矩阵 (State transition matrix)**：

$$
\begin{bmatrix} C \\ O \end{bmatrix}_{k+1} = \begin{bmatrix} 1-\mu_{c2o} & \mu_{o2c} \\ \mu_{c2o} & 1-\mu_{o2c} \end{bmatrix} \begin{bmatrix} C \\ O \end{bmatrix}_k
$$

- 每列之和 = 1（概率守恒）
- 矩阵元素含义：
  - $1 - \mu_{c2o}$：保持关闭的概率
  - $\mu_{c2o}$：从关闭转为打开的概率
  - $\mu_{o2c}$：从打开转为关闭的概率
  - $1 - \mu_{o2c}$：保持打开的概率

**概率传播算法**：$\mathbf{x}_{k+1} = \mathbf{A} \cdot \mathbf{x}_k$（矩阵 - 向量乘法）

**平衡态分析 (Equilibrium analysis)**：

- 特征值 = 1 对应**稳定平衡**特征向量
- 其他特征值对应瞬态衰减
- 平衡特征向量需归一化（元素之和 = 1）
- 打开的平衡概率：$\frac{\mu_{c2o}}{\mu_{c2o} + \mu_{o2c}}$

---

### Tutorial 3：随机游走与扩散 (Random Walks and Diffusion)

**随机游走 (Random walk)**：每一步以等概率移动 $\Delta x = \pm 1$

**位置更新**：$x_{k+1} = x_k + \Delta x$

**高斯步长的随机游走**：步骤从 $\mathcal{N}(\mu, \sigma)$ 采样

**高效向量化实现**：

```python
def random_walk_simulator(N, T, mu=0, sigma=1):
    steps = np.random.normal(mu, sigma, size=(N, T))
    sim = np.cumsum(steps, axis=1)
    return sim
```

**扩散过程性质 (Properties of diffusive process)**：

- 均值保持在 0 附近（与时间无关）
- **方差随时间线性增长**：$\text{Var} \propto t$（具体为 $\text{Var} = \sigma^2 t$）
- 分布随时间变宽但中心不变

---

### Tutorial 3：确定性衰减与 OU 过程

**基本衰减**：$x_{k+1} = \lambda x_k$，解：$x_k = x_0 \lambda^k$（$|\lambda| < 1$ 时衰减）

**带目标的衰减**：$x_{k+1} = x_\infty + \lambda(x_k - x_\infty)$

**解析解**：$x_k = x_\infty(1 - \lambda^k) + x_0 \lambda^k$

当 $k \to \infty$：$x_k \to x_\infty$

**Ornstein-Uhlenbeck (OU) 过程 / 漂移扩散模型 (Drift-Diffusion Model)**：

$$
x_{k+1} = x_\infty + \lambda(x_k - x_\infty) + \sigma \eta
$$

其中 $\eta \sim \mathcal{N}(0,1)$（标准正态分布）

**两个组成部分**：

- **漂移项 (Drift)**：$x_\infty + \lambda(x_k - x_\infty)$，将 $x$ 拉向 $x_\infty$
- **扩散项 (Diffusion)**：$\sigma \eta$，添加随机噪声

**平衡方差 (Equilibrium variance)**（关键结果）：

$$
\text{Var}_{eq} = \frac{\sigma^2}{1 - \lambda^2}
$$

**性质**：

- 仅依赖于 $\lambda$ 和 $\sigma$，**不依赖** $x_0$ 或 $x_\infty$
- 当 $\lambda \to 1$：方差发散（接近纯随机游走）
- 当 $\lambda \to 0$：方差趋近 $\sigma^2$（每步独立）

**经验方差计算**：运行长时间 $T$ 的模拟，取**后半段**的方差（假设系统已稳定）

```python
x[-round(T/2):].var()
```

**关键观察**：

- OU 过程的均值精确遵循确定性解
- 方差达到平衡（不像随机游走那样无限增长）
- 恢复漂移力防止了方差的无限增长

---

### Tutorial 4：自回归模型 (Autoregressive Models)

**视角转换**：给定数据，学习其动力学（逆向问题）

**一阶自回归 AR(1)**：$x_{k+1} = \lambda x_k + \eta$

**回归公式化**：$\mathbf{x}_2 = \lambda \mathbf{x}_1$

- $\mathbf{x}_1 = x[0:T-1]$（过去值）
- $\mathbf{x}_2 = x[1:T]$（未来值，偏移 1）

**最小二乘求解**：

```python
p, res, rnk, s = np.linalg.lstsq(x1, x2, rcond=None)
```

**添加截距项**：在 x1 前添加一列 1

```python
x1 = x1[:, np.newaxis]**[0, 1]  # 添加列：常数项和线性项
```

回归系数 $p[1]$ 即为估计的 $\hat{\lambda}$

**残差分析 (Residual analysis)**：

- 残差 = 数据 - 预测值：$\text{res} = x_2 - (p[0] + \hat{\lambda} \cdot x_1[:, 1])$
- 残差标准差应近似等于 $\sigma$（噪声参数）
- 残差直方图应近似正态分布

---

### Tutorial 4：高阶自回归模型

**r 阶 AR 模型**：$x_{k+1} = \alpha_0 + \alpha_1 x_k + \alpha_2 x_{k-1} + \dots + \alpha_r x_{k-r}$

共 $r+1$ 个系数需要拟合（包括截距 $\alpha_0$）

**时间延迟矩阵构建 (build_time_delay_matrices)**：

- $\mathbf{x}_1$：大小为 $[(r+1) \times (n-r)]$ 的矩阵
  - 第 0 行：全 1（截距项）
  - 第 1 行：$x[0:T-r]$（滞后 1）
  - 第 2 行：$x[1:T-r+1]$（滞后 2，通过 `np.roll` 实现）
  - … 直到滞后 $r$
- $\mathbf{x}_2$：向量 $x[r:]$（要预测的值）

**np.roll 技巧**：`xprime = np.roll(xprime, -1)` 将数组左移 1 位

**预测与分类**：

- 对于二值 (+1/-1) 数据：预测 = $\text{sign}(\mathbf{x}_1^T \cdot \mathbf{p})$
- 错误率 = $\text{count}(x_2 \neq \text{prediction}) / \text{len}(x_2)$
- 随机猜测基线：错误率 = 0.5

**过拟合观察**：

- 扫描 AR 阶数从 r=1 到 r=20
- 存在**最佳点**（约 r=6 对于人类生成数据）
- 低 r：欠拟合（错过模式）
- 高 r：过拟合（拟合训练噪声，测试表现差）
- 体现了偏差 - 方差权衡 (bias-variance tradeoff)

**人类随机性 vs 机器随机性**：

- 人类在生成随机序列方面表现很差（可检测的模式）
- AR 模型可以利用这些模式进行优于随机的预测
- 机器生成的随机整数真正不可预测（错误率 ≈ 0.5）
- 二值编码：'0' → -1，'1' → +1（通过 `x*2 - 1`）

---

## W2D4：生物神经元模型 (Biological Neuron Models)

---

### Tutorial 1：泄漏积分发放模型 (LIF Model)

**核心膜电位方程（阈下动力学）**：

$$
\tau_m \frac{dV}{dt} = -(V - E_L) + \frac{I}{g_L}
$$

其中 $\tau_m = C_m / g_L$ 是膜时间常数，$g_L$ 是漏电导，$E_L$ 是静息电位

**脉冲与复位规则 (Spike-and-reset rule)**：

$$
\text{if } V(t_{sp}) \geq V_{th}: \quad V(t) = V_{reset} \text{ for } t \in (t_{sp}, t_{sp} + \tau_{ref}]
$$

**默认参数**：

| 参数        | 值     | 含义       |
| ----------- | ------ | ---------- |
| $V_{th}$    | -55 mV | 脉冲阈值   |
| $V_{reset}$ | -75 mV | 复位电位   |
| $E_L$       | -75 mV | 静息电位   |
| $\tau_m$    | 10 ms  | 膜时间常数 |
| $g_L$       | 10 nS  | 漏电导     |
| $t_{ref}$   | 2 ms   | 不应期     |
| $dt$        | 0.1 ms | 时间步长   |

**欧拉积分实现 (run_LIF)**：

```python
for it in range(Lt - 1):
    if tr > 0:                          # 不应期
        v[it] = V_reset
        tr = tr - 1
    elif v[it] >= V_th:                 # 脉冲！
        rec_spikes.append(it)
        v[it] = V_reset
        tr = tref / dt
    # 计算膜电位增量
    dv = (dt / tau_m) * (-(v[it] - E_L) + Iinj[it] / g_L)
    # 更新膜电位
    v[it + 1] = v[it] + dv
```

---

### Tutorial 1：不同类型的输入电流

**直流电流 (DC)**：恒定电流，产生规则脉冲（CV_ISI ≈ 0）

**高斯白噪声 (GWN)**：

$$
I_{gwn} = \mu + \sigma \cdot \frac{\xi(t)}{\sqrt{dt/1000}}
$$

其中 $\xi(t) \sim \mathcal{N}(0,1)$，除以 $\sqrt{dt/1000}$ 将离散时间噪声转换为正确的连续时间缩放（单位转换为秒）

**Ornstein-Uhlenbeck (OU) 过程（有色噪声）**：

$$
\tau_\eta \frac{d\eta}{dt} = -\eta(t) + \sigma_\eta \sqrt{2\tau_\eta} \xi(t)
$$

**性质**：

- $\mathbb{E}[\eta(t)] = \mu$
- 自协方差：$\text{Cov}[\eta(t), \eta(t+\tau)] = \sigma_\eta^2 e^{-|t-\tau|/\tau_\eta}$

**欧拉实现**：

```python
I_ou[it+1] = I_ou[it] + (dt/tau_ou)*(mu - I_ou[it]) + sqrt(2*dt/tau_ou)*sig*noise[it+1]
```

---

### Tutorial 1：放电频率与脉冲不规则性

**频率 - 电流曲线 (F-I curve)**：输出放电频率作为输入电流的函数

**脉冲间隔变异系数 (CV of ISI)**：

$$
\text{CV}_{\text{ISI}} = \frac{\text{std}(\text{ISI})}{\text{mean}(\text{ISI})}
$$

| CV 值 | 含义                     |
| ----- | ------------------------ |
| 0     | 完全规则（时钟般）       |
| 1     | 泊松过程（最大不规则性） |

**关键发现**：

- DC 输入产生规则脉冲（CV ≈ 0）
- GWN 输入产生不规则脉冲；更高的 $\sigma$ 增加 CV_ISI
- 增加 $\sigma$ 使 F-I 曲线变平滑
- 增加均值 $\mu$ 同时保持 $\sigma$ 固定会降低 CV_ISI（高频时更规则）

---

### Tutorial 2：相关输入与相关性转移

**相关输入模型**：

$$
\frac{I_i}{g_L} = \mu_i + \sigma_i (\sqrt{1-c}\,\xi_i + \sqrt{c}\,\xi_c)
$$

**逐项解释**：

| 项 | 含义 | 为什么这样设计 |
|---|---|---|
| $I_i$ | 第 $i$ 个神经元的**总突触输入电流** | LIF 模型的驱动项，决定膜电位变化 |
| $g_L$ | **漏电导** | 除以 $g_L$ 后可将力学与电位解耦 |
| $\mu_i$ | **漂移项（确定性的 DC 成分）** | 控制平均输入强度，类比施加的刺激强度 |
| $\sigma_i$ | **噪声幅度** | 控制输入波动的幅度（方差的大小） |
| $\xi_i \sim \mathcal{N}(0,1)$ | **第 $i$ 个神经元的独立噪声** | 模拟非共享的突触输入波动 |
| $\xi_c \sim \mathcal{N}(0,1)$ | **所有神经元共享的共同噪声** | 模拟全局状态变化或公共输入 |
| $c \in [0,1]$ | **相关强度** | $c=0$ 时完全独立，$c=1$ 时完全相关 |
| $\sqrt{1-c}$ | 独立噪声的缩放因子 | 使得 $\text{Var}(\sqrt{1-c}\xi_i) = 1-c$ |
| $\sqrt{c}$ | 共享噪声的缩放因子 | 使得 $\text{Var}(\sqrt{c}\xi_c) = c$ |

**方差守恒**：$\text{Var}(\sqrt{1-c}\xi_i + \sqrt{c}\xi_c) = (1-c) + c = 1$

即无论 $c$ 取何值，每个神经元的总噪声方差保持为 $\sigma_i^2$——变化的只是独立部分和共享部分的比例，而非总方差。这确保我们单独研究相关性而不引入方差混淆。

**样本相关系数 (Pearson)**：

$$
r_{ij} = \frac{\text{cov}(I_i, I_j)}{\sqrt{\text{var}(I_i)} \sqrt{\text{var}(I_j)}}
$$

- $\text{cov}(I_i, I_j)$：$I_i$ 和 $I_j$ 的协方差，衡量两者共同变化的程度
- $\text{var}(I_i)$：$I_i$ 的方差（自身波动幅度）
- 分子归一化后，$r_{ij} \in [-1, 1]$：
  - $r_{ij} = 1$：完全正相关（同步变化）
  - $r_{ij} = 0$：不相关
  - $r_{ij} = -1$：完全负相关（反向变化）

对于相关输入模型，理论相关系数为 $r_{ij} = c$（当 $\sigma_i = \sigma_j$ 时）。验证：$\text{cov}(I_i, I_j) = \sigma_i\sigma_j \cdot c$，$\text{var}(I_i) = \sigma_i^2$，所以 $r_{ij} = c$。

**Corr Poisson 生成方法**：

```python
# 为每个神经元生成相关泊松脉冲序列
def generate_corr_Poisson(rate, corr, T, dt, n_neurons):
    """生成 n 个神经元的相关系列为 corr 的泊松脉冲"""
    # 1. 生成"母序列"：频率为 rate/corr 的泊松序列
    #    母序列的脉冲密度更高，作为公共"素材池"
    mother = np.random.rand(int(T/dt)) < (rate / corr) * dt

    # 2. 每个子神经元独立采样母序列中比例为 corr 的脉冲
    #    通过从母序列的脉冲时刻中随机抽取实现
    mother_times = np.where(mother)[0]
    n_shared = int(len(mother_times) * corr)
    spikes = np.zeros((n_neurons, int(T/dt)))
    for i in range(n_neurons):
        idx = np.random.choice(mother_times, n_shared, replace=False)
        spikes[i, idx] = 1.0
    return spikes
```

为什么这样生成？从母序列中采样而不是直接生成相关噪声，是因为泊松过程的相关性不能像高斯那样直接线性叠加。通过"公共父过程 + 子采样"的方式，保证每对神经元的输出相关性恰好为 $c$。

**Campbell 定理（泊松输入的突触电流均值和方差）**：

$$
\mu_{\rm syn} = \lambda J \int P(t) dt
$$

$$
\sigma_{\rm syn}^2 = \lambda J^2 \int P(t)^2 dt
$$

| 项 | 含义 | 直觉 |
|---|---|---|
| $\lambda$ | **泊松率**：每秒脉冲数 | 脉冲到达越密集，电流越大 |
| $J$ | **PSP 幅度**：单个脉冲引起的突触电导跳变 | 每个脉冲的"权重" |
| $P(t)$ | **突触后电流核**：脉冲引起的电流随时间演化的波形 | 突触传递的"形状" |
| $\int P(t) dt$ | 核函数的积分（时间×幅度） | 单脉冲的总电荷注入 |
| $\int P(t)^2 dt$ | 核函数平方的积分 | 控制电流波动的平方幅度 |

**推导直觉**：
- **均值** $\mu_{\rm syn} \propto \lambda J$：脉冲率越高、幅度越大，平均电流越大
- **方差** $\sigma_{\rm syn}^2 \propto \lambda J^2$：方差对 $J$ 比对 $\lambda$ 更敏感（$J^2$ vs $\lambda$），说明增大单个脉冲的幅度比增大脉冲率更能放大波动

**关键发现**：

- 输出相关性 **总是小于** 输入相关性（LIF 充当"相关性滤波器"）
- 相关性转移函数近似线性
- 更高的均值 $\mu$ 和更高的 $\sigma$ 都会增加转移函数的斜率（更好的相关性传递）
- 更高的放电率导致更好的相关性传递

---

### Tutorial 3：基于电导的突触 (Conductance-Based Synapses)

**突触电导动力学**：

$$
\frac{dg_{\rm syn}(t)}{dt} = \bar{g}_{\rm syn} \sum_k \delta(t-t_k) - \frac{g_{\rm syn}(t)}{\tau_{\rm syn}}
$$

- $\bar{g}_{\rm syn}$：每个脉冲引起的最大电导变化（突触权重）
- $\tau_{\rm syn}$：突触时间常数（控制衰减速度）

**欧姆定律（电导转电流）**：

$$
I_{\rm syn}(t) = g_{\rm syn}(t)(V(t) - E_{\rm syn})
$$

- $E_E = 0$ mV（兴奋性反转电位，去极化）
- $E_I = -80$ mV（抑制性反转电位，超极化）

**总突触电流**：

$$
I_{\rm syn} = -g_E(t)(V - E_E) - g_I(t)(V - E_I)
$$

**电导 LIF 膜电位方程**：

$$
\tau_m \frac{dV}{dt} = -(V - E_L) - \frac{g_E(t)}{g_L}(V - E_E) - \frac{g_I(t)}{g_L}(V - E_I) + \frac{I_{\rm inj}}{g_L}
$$

**欧拉更新电导 (run_LIF_cond)**：

```python
gE[it+1] = gE[it] - (dt/tau_syn_E)*gE[it] + gE_bar * spike_train_ex[it+1]
gI[it+1] = gI[it] - (dt/tau_syn_I)*gI[it] + gI_bar * spike_train_in[it+1]
```

**默认突触参数**：

- 兴奋性：$g_E = 2.4$ nS，$E_E = 0$ mV，$\tau_E = 2$ ms
- 抑制性：$g_I = 2.4$ nS，$E_I = -80$ mV，$\tau_I = 5$ ms
- 80 个兴奋性、20 个抑制性突触前神经元，频率 10 Hz

**自由膜电位 (Free Membrane Potential, FMP)**：去除脉冲阈值的膜电位（人为设定 $V_{th} = \infty$）

- 平均 FMP > 阈值：**均值驱动体制** (Mean-driven regime)（规则放电，低 CV_ISI）
- 平均 FMP < 阈值：**波动驱动体制** (Fluctuation-driven regime)（不规则放电，高 CV_ISI）
- 兴奋/抑制平衡决定放电模式
- 突触输入是**有色噪声**（指数核滤波），不是白噪声

---

### Tutorial 3：短期突触可塑性 (Short-Term Plasticity, STP)

**三个动态变量模型**：

$$
\frac{du_E}{dt} = -\frac{u_E}{\tau_f} + U_0(1-u_E^-)\delta(t-t_{sp})
$$

$$
\frac{dR_E}{dt} = \frac{1-R_E}{\tau_d} - u_E^+ R_E^- \delta(t-t_{sp})
$$

$$
\frac{dg_E}{dt} = -\frac{g_E}{\tau_E} + \bar{g}_E u_E^+ R_E^- \delta(t-t_{sp})
$$

#### 逐项详解

**方程 1：释放概率 $u$（使用率，Utilization）**

$$
\frac{du_E}{dt} = \underbrace{-\frac{u_E}{\tau_f}}_{\text{衰减项}} + \underbrace{U_0(1-u_E^-)\delta(t-t_{sp})}_{\text{脉冲触发项}}
$$

| 项 | 含义 | 为什么这样设计 |
|---|---|---|
| $\frac{du_E}{dt}$ | 释放概率 $u$ 随时间的变化率 | 钙离子浓度的动态模型 |
| $-\frac{u_E}{\tau_f}$ | **指数衰减项**：$u$ 以 $\tau_f$ 为时间常数回归 0 | 脉冲之间钙离子被逐步泵出，释放概率回落 |
| $U_0$ | **基线释放概率**：单次脉冲后的初始增幅 | 这是可调参数，$U_0$ 大 → 初始释放率高 |
| $(1-u_E^-)$ | **可用增幅空间**：$u_E^-$ 是脉冲到达前瞬间的 $u$ 值 | $u$ 不能超过 1，剩余空间 $(1-u)$ 越大，一次脉冲的提升幅度越大 |
| $\delta(t-t_{sp})$ | **Dirac delta 函数**：只在脉冲到达时刻 $t_{sp}$ 生效 | 表示脉冲是离散事件，在瞬间产生影响 |
| $U_0(1-u_E^-)\delta(t-t_{sp})$ | **脉冲触发项**：脉冲到达时 $u$ 跳跃式增加 | 模拟钙内流的快速过程 |

**直觉**：
- 脉冲到达 → 钙离子内流 → 释放概率 $u$ 瞬间跳升
- 脉冲之间 → 钙离子衰减 → $u$ 指数回落
- $U_0$ 控制单次脉冲的跳升幅度，$\tau_f$ 控制回落速度
- **当 $\tau_f$ 很大时**，$u$ 在脉冲之间几乎不衰减 → 连续脉冲累积效应 → **短期易化 (STF)**

**方程 2：可释放资源池 $R$（Recovery）**

$$
\frac{dR_E}{dt} = \underbrace{\frac{1-R_E}{\tau_d}}_{\text{恢复项}} - \underbrace{u_E^+ R_E^- \delta(t-t_{sp})}_{\text{消耗项}}
$$

| 项 | 含义 | 为什么这样设计 |
|---|---|---|
| $\frac{1-R_E}{\tau_d}$ | **恢复项**：资源向 1 恢复，时间常数 $\tau_d$ | 神经递质在脉冲间重新合成和转运 |
| $1-R_E$ | **可恢复的空间** | $R$ 越接近 0，恢复驱动力越大 |
| $u_E^+ R_E^- \delta(t-t_{sp})$ | **消耗项**：脉冲到达时资源被消耗 | 释放概率 $u$ 越大，消耗的资源越多 |
| $u_E^+$ | **脉冲到达后的释放概率** | 先更新 $u$ 再用新 $u$ 计算消耗（顺序重要！） |
| $R_E^-$ | **脉冲到达前的资源量** | 消耗的是脉冲前的资源存量 |

**直觉**：
- 脉冲到达 → $u \cdot R$ 的资源被消耗 → $R$ 跳跃式下降
- 脉冲之间 → 资源逐步恢复 → $R$ 指数回归 1
- $\tau_d$ 控制恢复速度，$\tau_d$ 越大 → 恢复越慢
- **当 $\tau_d$ 很大时**，高频脉冲下资源来不及恢复 → $R$ 持续走低 → **短期抑制 (STD)**

**方程 3：突触电导 $g$（Conductance）**

$$
\frac{dg_E}{dt} = \underbrace{-\frac{g_E}{\tau_E}}_{\text{衰减项}} + \underbrace{\bar{g}_E u_E^+ R_E^- \delta(t-t_{sp})}_{\text{产生项}}
$$

| 项 | 含义 | 为什么这样设计 |
|---|---|---|
| $-\frac{g_E}{\tau_E}$ | **指数衰减项**：电导以 $\tau_E$ 衰减 | 突触后受体关闭，电导自然消退 |
| $\bar{g}_E$ | **最大电导** | 突触传递的幅度上限 |
| $u_E^+ R_E^- \delta(t-t_{sp})$ | **脉冲触发电导跳变** | 实际释放量 = 释放概率 × 可用资源 |
| $\bar{g}_E u_E^+ R_E^-$ | **电导增量** | 电导跳变正比于 $u \times R$ |

**直觉**：
- 实际突触强度 = $\bar{g}_E \times u \times R$（三个因素的乘积）
- $u$ 控制"释放概率多大"，$R$ 控制"有多少可释放"，$\bar{g}_E$ 控制"最大能有多大"
- 脉冲之间 $g$ 指数衰减到 0

**为什么需要三个变量？**

突触传递不是一个简单的开/关过程，而是多个时间尺度因素的相互作用：
- $u$（快/中时间尺度）：钙动力学 → 易化
- $R$（慢时间尺度）：递质回收 → 抑制
- $g$（中时间尺度）：突触后响应 → 输出

**三个变量的相互作用时序**（脉冲到达时的更新顺序至关重要）：

```
① 先计算 u 的跳跃:   u ← u + U₀(1-u)      // 用脉冲前的 u 值
② 再计算 R 的消耗:   R ← R - u⁺·R          // 用刚更新的 u⁺
③ 最后计算 g 的增加:  g ← g + ḡ · u⁺·R⁻    // 用新的 u⁺ 和旧的 R⁻
```

为什么是这个顺序？因为实际生物过程是：钙内流（$u$）→ 触发囊泡释放（消耗 $R$）→ 打开离子通道（增加 $g$）。如果先消耗 $R$ 再用新的 $R$ 算 $g$，就会错误地使用"已经减少后的资源"来计算电导，不符合生物实际。

**STP 计算示例**（两次脉冲）：

```
初始状态: u=0.1, R=1.0, g=0

第1个脉冲到达:
  u = 0.1 + 0.5×(1-0.1) = 0.55
  R = 1.0 - 0.55×1.0 = 0.45
  g = 0 + 1.0×0.55×1.0 = 0.55
  → 然后 u、R、g 指数衰减...

第2个脉冲到达（在 R 恢复之前）:
  u = u' + 0.5×(1-u')    ← u' 是恢复到今的值
  R = R' - u⁺×R'          ← 资源比第一次少，输出更弱
  g = g' + ḡ·u⁺·R⁻
  → 如果 R 还没恢复，第二次的 g 增量更小 → 这就是 STD
```

**短期抑制 (STD) vs 短期易化 (STF) 参数**：

**短期抑制 (STD) vs 短期易化 (STF) 参数**：

| 参数     | STD                 | STF                 |
| -------- | ------------------- | ------------------- |
| $U_0$    | 0.5（高初始释放率） | 0.2（低初始释放率） |
| $\tau_d$ | 100 ms              | 100 ms              |
| $\tau_f$ | 50 ms（快速恢复）   | 750 ms（慢速衰减）  |

**STD 机制**：

- 高频输入时资源来不及恢复，电导持续减小
- $g_{10}/g_1$ 随输入率单调递减

**STF 机制**：

- $\tau_f$ 大时，$u$ 在脉冲间衰减慢，累积效应明显
- $g_{10}/g_1$ 随输入率非单调变化（先增后减）

---

### Tutorial 4：脉冲时间依赖可塑性 (STDP)

#### STDP 的基本原理：Hebbian 学习的时间版本

> "Fire together, wire together" —— 但 STDP 说"精确的顺序和时机"才是关键。

STDP 是 Hebbian 学习规则的时间精确版本：突触权重的变化取决于**突触前和突触后脉冲的相对时间**。

#### STDP 权重变化规则（双相指数衰减）

$$
\Delta W = \begin{cases}
A_+ e^{\, (t_{pre}-t_{post})/\tau_+} & \text{if } t_{post} > t_{pre} \text{ (LTP)} \\
-A_- e^{\, -(t_{pre}-t_{post})/\tau_-} & \text{if } t_{post} < t_{pre} \text{ (LTD)}
\end{cases}
$$

**逐项解释**：

| 项 | 含义 | 为什么这样设计 |
|---|---|---|
| $\Delta W$ | 突触权重的**改变量** | 正值 = 增强（LTP），负值 = 减弱（LTD） |
| $t_{pre}$ | **突触前脉冲到达时间** | 上游神经元发放的时刻 |
| $t_{post}$ | **突触后脉冲发放时间** | 下游神经元发放的时刻 |
| $t_{pre} - t_{post}$ | **脉冲时间差**：$\Delta t$ | $>0$ 表示 pre 先于 post（因果关系），$<0$ 表示 post 先于 pre |
| $A_+$ | LTP 的最大变化幅度 | 控制权重增强的强度 |
| $A_-$ | LTD 的最大变化幅度 | 控制权重减弱的强度（通常 $A_- > A_+$，即 LTD 主导） |
| $\tau_+$ | LTP 的时间窗口常数 | 决定 pre-post 间隔多大时还能触发 LTP |
| $\tau_-$ | LTD 的时间窗口常数 | 决定 post-pre 间隔多大时还能触发 LTD |
| $e^{\Delta t/\tau_+}$ | LTP 的指数衰减 | $\Delta t > 0$ 时，间隔越大增强越弱 |
| $e^{-\Delta t/\tau_-}$ | LTD 的指数衰减 | $\Delta t < 0$ 时，间隔越大减弱越弱 |

**为什么是不对称的双相指数？**

```
权重变化 ΔW
    ↑ A₊                       ↱ LTP 区域 (pre 先于 post)
    |                    ／
    |               ／
    |          ／
    |     ／
    ────┼────────────────────────────→ Δt = t_pre - t_post
    |     ＼                           （pre 领先为正）
    |        ＼
    |           ＼              ↳ LTD 区域 (post 先于 pre)
    |              ＼
    |                 ＼
    ↓ -A₋
```

- **LTP 侧**（$\Delta t > 0$）：pre 先于 post → pre 可能"预测"或"导致"post → 增强该突触
- **LTD 侧**（$\Delta t < 0$）：post 先于 pre → post 的发放与 pre 无关 → 削弱该突触（不浪费资源）
- **指数衰减**：时间差越大，因果关系的可信度越低 → 权重变化越小
- **不对称性**（$A_- > A_+$）：如果 LTP 和 LTD 平衡，不相关输入会导致净零变化，但 $A_-$ 略大使得不相关输入的整体趋势是 LTD —— 只有**持续**的 pre-post 因果关系才能维持权重

**默认参数的含义**：

| 参数 | 值 | 含义 |
|---|---|---|
| $A_+ = 0.008$ | LTP 幅度 | 一次 pre-post 配对最多增强 0.008（相对于最大电导 $\bar{g}_{max}$） |
| $A_- = 1.10 \times A_+$ | LTD 幅度略大 | 不相关的 pre-post 时序导致净 LTD（竞争性学习） |
| $\tau_{\rm stdp} = 20$ ms | 时间窗口 | 间隔 > 20ms 时，权重变化衰减到 $e^{-1} \approx 37\%$ |

#### 高效 STDP 实现：追踪变量 $P(t)$ 和 $M(t)$

标准 STDP 规则需要对每个 pre-post 事件对计算时间差——在大量神经元和持续脉冲的情况下，直接计算效率极低。

**解决方案**：引入两个追踪变量 $P$（positive trace）和 $M$（negative trace），把 STDP 转化为**事件触发的变量更新**。

对于每个突触前神经元 $i$：

$$
\tau_+ \frac{dP}{dt} = -P
$$

突触前脉冲到达时：$P(t) = P(t) + A_+$

对于每个突触后神经元：

$$
\tau_- \frac{dM}{dt} = -M
$$

突触后脉冲到达时：$M(t) = M(t) - A_-$

**$P$ 和 $M$ 的直观理解**：

```
P(t)  ↑  A₊                        突触前脉冲在 P 上留下"标记":
     |    \                        - P 在 pre 脉冲时跳升 A₊
     |     \__        跳升 A₊       - 然后指数衰减到 0
     |        \_                   - P 的正值表示"最近有 pre 脉冲"
     └────────────────────────→ 时间

M(t)  ◄──────────────────────── 时间
      _/
     _/    -A₋
     |    /                       突触后脉冲在 M 上留下"标记":
     |   /                         - M 在 post 脉冲时跳减 -A₋
     |  /                          - 然后指数衰减到 0
     ↓ -A₋                         - M 的负值表示"最近有 post 脉冲"
```

**为什么这样转化是等价的？**

- 当突触后发放时，$P(t)$ 的值正好等于 $\sum_{\text{recent pre}} A_+ e^{-(t_{post}-t_{pre})/\tau_+}$——这正是所有最近 pre 脉冲对权重变化的 LTP 贡献之和！
- 当突触前发放时，$M(t)$ 的值正好等于 $\sum_{\text{recent post}} -A_- e^{-(t_{pre}-t_{post})/\tau_-}$——这是所有最近 post 脉冲对权重变化的 LTD 贡献之和。

**使用追踪变量的权重更新规则**：

当突触前神经元 $i$ 发放时（执行 LTD）：

$$
\bar{g}_i \leftarrow \bar{g}_i + M(t) \cdot \bar{g}_{max}
$$

- $M(t) < 0$（跳变为负后逐渐恢复）：所以权重**减小**
- $\bar{g}_{max}$ 将相对变化量转换为绝对电导变化
- 若 $\bar{g}_i < 0$，钳制为 0（电导不能为负）
- **意义**：如果最近有 post 脉冲（$M$ 很负），但当前 pre 又发放了，这个 pre 的贡献似乎"多余"了，应削弱

当突触后神经元发放时（执行 LTP）：

$$
\bar{g}_i \leftarrow \bar{g}_i + P_i(t) \cdot \bar{g}_{max} \quad \forall i
$$

- $P_i(t) > 0$（跳变为正后逐渐恢复）：所以权重**增大**
- 对所有突触前 $i$ 的权重同时更新
- **意义**：对每个 pre 连接，如果它最近有脉冲（$P_i$ 还很大），说明是它帮助触发了 post 发放，应增强

```
事件流示意图：

突触前脉冲 @ t_pre:
  ├─ 立即: g ← g + M(t) × ḡ_max    (LTD: 如果post最近发放过，削弱)
  └─ 更新: P ← P + A₊               (为将来可能的LTP做标记)

突触后脉冲 @ t_post:
  ├─ 立即: 对所有i: ḡ_i ← ḡ_i + P_i(t) × ḡ_max  (LTP: 增强所有"有功"突触)
  └─ 更新: M ← M - A₋               (为将来可能的LTD做标记)
```

**参数释放的生物学对应**：

| STDP 参数 | 生物学对应 |
|---|---|
| $\tau_{\rm stdp} = 20$ ms | NMDA 受体的钙信号时间常数 |
| $A_+ > 0$ | CaMKII 激活（增强 AMPA 受体数量） |
| $A_- > 0$ | 去磷酸化酶激活（减少 AMPA 受体数量） |
| $A_- > A_+$ | LTD 更容易触发（维持竞争平衡） |

**带 STDP 突触的 LIF 膜电位方程**：

$$
\tau_m \frac{dV}{dt} = -(V - E_L) - g_E(t)(V - E_E)
$$

其中 $g_E(t) = \sum_i g_i(t)$，每个 $g_i(t)$ 使用动态更新的 $\bar{g}_i$

**默认突触参数（STDP 模拟）**：

- $\bar{g}_E = 0.024$ nS（每个突触的最大电导）
- $g_{E,init} = 0.014 - 0.024$ nS（初始电导）
- $E_E = 0$ mV，$\tau_E = 5$ ms
- $N = 300$ 个突触前神经元，频率 10-15 Hz，$dt = 1$ ms

**关键发现**：

- 不相关泊松输入时，许多突触随时间减弱（LTD 主导，因为 $A_- > A_+$）
- 权重分布随时间演化；出现双峰分布（许多权重接近 0，一些接近 $g_{max}$）
- 相关输入时：相关突触前神经元维持其权重（更高的 pre-before-post 配对机会），不相关突触抑制
- STDP 实现**无监督学习**：携带相关/相关信息的突触被选择性增强

---

## W2D5：动力系统 (Dynamical Systems)

---

### Tutorial 1：单群放电频率模型

**前馈放电频率动力学 (Eq. 1)**：

$$
\tau \frac{dr}{dt} = -r + F(I_{\rm ext})
$$

**逐项解释**：

| 项 | 含义 | 为什么这样设计 |
|---|---|---|
| $r$ | 群体的**平均放电频率** | 用连续量描述一群神经元的集体活动 |
| $\frac{dr}{dt}$ | 放电频率随时间的变化率 | 动力系统的标准形式 |
| $\tau$ | **时间常数** | 控制神经元对输入响应的快慢 |
| $-r$ | **衰减项**（向 0 回归的泄漏项） | 如果没有输入，放电频率应指数衰减到 0 |
| $F(x)$ | **Sigmoid 传递函数**（F-I 曲线） | 将输入电流转换为放电频率的非线性映射 |
| $I_{\rm ext}$ | **外部输入电流** | 来自其他脑区或刺激的驱动 |

**直觉**：
- 方程形式上等价于 $\tau \frac{dr}{dt} = -r + \text{input}$，是一个**一阶低通滤波器**
- 输入 $I_{\rm ext}$ 经 $F$ 非线性变换后，驱动 $r$ 向目标值趋近
- 时间常数 $\tau$ 控制趋近的快慢

---

**Sigmoid 传递函数 / F-I 曲线 (Eq. 2)**：

$$
F(x; a, \theta) = \frac{1}{1 + e^{-a(x-\theta)}} - \frac{1}{1 + e^{a\theta}}
$$

**逐项解释**：

| 项 | 含义 | 为什么这样设计 |
|---|---|---|
| $x$ | **净输入**（电流强度） | 传递给函数的自变量 |
| $a$ | **增益 (gain)** | 控制 F-I 曲线的斜率，越大则从"关闭"到"饱和"的过渡越陡峭 |
| $\theta$ | **阈值 (threshold)** | 曲线拐点的位置，决定多少输入才能让神经元开始明显放电 |
| $\frac{1}{1 + e^{-a(x-\theta)}}$ | **标准 sigmoid** | 经典 S 形曲线，输出范围 $(0,1)$ |
| $-\frac{1}{1 + e^{a\theta}}$ | **减法修正项** | **确保 $F(0; a, \theta) = 0$**——零输入时输出必须为零 |

**为什么需要减法修正？**

标准 sigmoid $1/(1+e^{-a(x-\theta)})$ 在 $x=0$ 时输出为 $1/(1+e^{a\theta})$：
- 如果 $\theta=2.8$，$a=1.2$，则 $1/(1+e^{3.36}) \approx 0.033$
- 这意味着即使没有输入，神经元也有少量"自发活动"
- 减去 $1/(1+e^{a\theta})$ 后，$F(0; a, \theta) = 0$，物理上合理

**F-I 曲线的形状**：

```
r = F(x)
   ↑
1  |        _______________
   |       /
   |      /
   |     /
   |    /
0  |___/______________________→ x
      θ（阈值）
```

- $x < \theta$：输出接近 0（神经元不放电）
- $x \approx \theta$：输出迅速上升（阈值附近）
- $x > \theta$：输出接近 1（饱和）

---

**递归网络动力学 (Eq. 3)**：

$$
\tau \frac{dr}{dt} = -r + F(w \cdot r + I_{\rm ext})
$$

| 项 | 含义 | 为什么这样设计 |
|---|---|---|
| $w \cdot r$ | **递归输入**：群体自身放电的反馈 | 突触权重 $w$ 乘以当前放电率 $r$ |
| $w$ | **递归突触权重** | $w > 0$ 为兴奋性反馈，$w < 0$ 为抑制性反馈 |
| $w \cdot r + I_{\rm ext}$ | **总驱动输入** | 外部驱动 + 自身反馈的叠加 |

**自反馈的效应**：
- $w > 0$（自兴奋）：正反馈 → 活动在输入驱动下放大
- $w = 0$：无反馈 → 纯前馈响应
- $w < 0$（自抑制）：负反馈 → 活动被抑制

---

**$w = 0$ 时的解析解**：

$$
r(t) = r(0) + [F(I_{\rm ext}; a, \theta) - r(0)](1 - e^{-t/\tau})
$$

| 项 | 含义 |
|---|---|
| $r(0)$ | 初始放电频率 |
| $F(I_{\rm ext})$ | 稳态目标值（$t \to \infty$ 时的 $r$） |
| $r(0) - F(I_{\rm ext})$ | 初始状态与稳态的偏差 |
| $1 - e^{-t/\tau}$ | 从初始到稳态的指数趋近过程 |

当 $w=0$ 时，这是一个标准的一阶线性微分方程，解是指数趋近。

---

### Tutorial 1：不动点与稳定性

**不动点条件 (Eq. 4)**：

$$
-r^* + F(w \cdot r^* + I_{\rm ext}; a, \theta) = 0
$$

**逐项解释**：

| 项 | 含义 | 为什么这样设计 |
|---|---|---|
| $r^*$ | **不动点 (fixed point)** | 系统达到平衡时的放电频率 |
| $-r^*$ | 衰减项在不动点处的取值 | 必须与驱动项平衡 |
| $F(w r^* + I_{\rm ext})$ | 驱动项在不动点处的取值 | 输入产生的驱动力 |
| 整个方程 = 0 | **不动点 = 零条件** | $\frac{dr}{dt}=0$ 的等价表达 |

**几何意义**：不动点就是 $r = F(w r + I_{\rm ext})$ 的解——输出恰好等于输入。

---

**Sigmoid 传递函数的导数 (Eq. 5)**：

$$
\frac{dF}{dx} = a \cdot e^{-a(x-\theta)} \cdot (1 + e^{-a(x-\theta)})^{-2}
$$

**为什么需要这个导数？**

$F'(x)$ 衡量的是传递函数的**增益**——输入微小变化 $\delta x$ 引起的输出变化为 $F'(x) \cdot \delta x$。F-I 曲线越陡峭的地方，$F'$ 越大。

**推导**：

$$
F(x) = \frac{1}{1 + e^{-ax + a\theta}} - C \quad (\text{其中 $C$ 是常数修正项})
$$

令 $u = -a(x-\theta)$，则 $1 + e^{u} = (1+e^{u})$：

$$
\frac{dF}{dx} = \frac{a e^{-a(x-\theta)}}{(1 + e^{-a(x-\theta)})^2}
$$

这个形式等价于 $F'(x) = a \cdot F(x) \cdot (1 - F(x))$（标准 sigmoid 导数）。

---

**特征值（稳定性分析）(Eq. 4 in Bonus)**：

$$
\lambda = \frac{-1 + w \cdot F'(w \cdot r^* + I_{\rm ext}; a, \theta)}{\tau}
$$

**特征值的逐项解释**：

| 项 | 含义 | 生物学意义 |
|---|---|---|
| $\lambda$ | **特征值** | 决定不动点附近的局部动力学 |
| $-1$ | **泄漏项**的导数 | 固有衰减倾向 |
| $w \cdot F'$ | 反馈环路的"环路增益" | 递归连接通过传递函数的导数产生的净反馈 |
| $w$ | 连接强度 | 反馈的幅度 |
| $F'(x)$ | 传递函数在驱动点的导数 | 将电流变化转换为频率变化的效率 |
| $\tau$ | 时间常数 | 归一化因子 |

**为什么特征值决定稳定性？**

在不动点附近做线性展开：设 $r(t) = r^* + \delta r(t)$：

$$
\tau \frac{d}{dt}\delta r = (-1 + w F') \cdot \delta r
$$

解为 $\delta r(t) = \delta r(0) e^{(-1 + w F')t/\tau} = \delta r(0) e^{\lambda t}$

- $\lambda < 0$：扰动 $\delta r$ 指数衰减 → **稳定**
- $\lambda > 0$：扰动 $\delta r$ 指数增长 → **不稳定**

**何时 $\lambda$ 最大？** 当 $F'$ 最大时（F-I 曲线最陡峭处），即 $x \approx \theta$ 时。这意味着在阈值附近，反馈环路的增益最高。

| $\lambda$     | 稳定性         | 动力学行为 |
| ------------- | -------------- | ---------- |
| $\lambda < 0$ | 稳定（吸引）   | 扰动被拉回不动点 |
| $\lambda > 0$ | 不稳定（排斥） | 扰动被放大，离开不动点 |

---

### Tutorial 1：OU 噪声输入

**OU 过程**：

$$
\tau_\eta \frac{d\eta}{dt} = -\eta(t) + \sigma_\eta \sqrt{2\tau_\eta} \, \xi(t)
$$

**逐项解释**：

| 项 | 含义 | 为什么这样设计 |
|---|---|---|
| $\eta$ | **有色噪声** | 模拟具有有限时间相关的噪声输入（比白噪声更真实） |
| $\tau_\eta$ | **噪声的时间常数** | 控制噪声的相关时间尺度 |
| $-\eta(t)$ | **漂移恢复项** | 把 $\eta$ 拉向 0，防止随机游走发散 |
| $\sigma_\eta$ | **噪声幅度** | 控制波动的规模 |
| $\sqrt{2\tau_\eta}$ | **方差守恒因子** | 确保 $\eta$ 的**平衡方差**恰好为 $\sigma_\eta^2$，与 $\tau_\eta$ 无关 |
| $\xi(t)$ | **白噪声** | $\mathbb{E}[\xi(t)\xi(t')] = \delta(t-t')$，无记忆的高斯噪声 |

**推导：为什么需要 $\sqrt{2\tau_\eta}$？**

若令 $dx = -\frac{x}{\tau} dt + g \, dW$（$dW$ 是维纳过程），其平衡方差为：

$$
\text{Var}(x) = \frac{g^2 \tau}{2}
$$

要使得 $\text{Var}(x) = \sigma^2$，则 $g = \sigma \cdot \sqrt{2/\tau} = \sigma \sqrt{2\tau} / \tau$

代入原式即得 $\tau_\eta \frac{d\eta}{dt} = -\eta + \sigma_\eta \sqrt{2\tau_\eta} \, \xi(t)$

---

### Tutorial 2：Wilson-Cowan 模型

**两个耦合群（兴奋 + 抑制）(Eq. 1)**：

$$
\tau_E \frac{dr_E}{dt} = -r_E + F_E(w_{EE}r_E - w_{EI}r_I + I_E^{\rm ext}; a_E, \theta_E)
$$

$$
\tau_I \frac{dr_I}{dt} = -r_I + F_I(w_{IE}r_E - w_{II}r_I + I_I^{\rm ext}; a_I, \theta_I)
$$

**逐项解释——每个方程的结构都是 $\tau \frac{dr}{dt} = -r + F(\text{net input})$**

**兴奋群 $r_E$ 方程**：

| 项 | 含义 | 为什么这样设计 |
|---|---|---|
| $-r_E$ | 固有衰减 | 无输入时放电率归零 |
| $w_{EE}r_E$ | **兴奋性自反馈**（$E \to E$） | $w_{EE} > 0$，E 群自兴奋 |
| $-w_{EI}r_I$ | **抑制性输入**（$I \to E$） | $w_{EI} > 0$，但前面有负号 → I 抑制 E |
| $I_E^{\rm ext}$ | 外部输入到 E 群 | 来自其他区域的刺激 |

**抑制群 $r_I$ 方程**：

| 项 | 含义 | 为什么这样设计 |
|---|---|---|
| $-r_I$ | 固有衰减 | 同上 |
| $w_{IE}r_E$ | **兴奋性驱动**（$E \to I$） | E 群兴奋 I 群（$w_{IE} > 0$） |
| $-w_{II}r_I$ | **抑制性自反馈**（$I \to I$） | I 群自抑制（$w_{II} > 0$） |
| $I_I^{\rm ext}$ | 外部输入到 I 群 | |

**整体结构**：

```
E 群: τ_E dr_E/dt = -r_E + F_E( +w_EE·r_E - w_EI·r_I + I_E )
                      ↑衰减     ↑自兴奋    ↑I抑制E    ↑外部输入

I 群: τ_I dr_I/dt = -r_I + F_I( +w_IE·r_E - w_II·r_I + I_I )
                      ↑衰减     ↑E兴奋I    ↑自抑制    ↑外部输入
```

**为什么是两个耦合方程而不是一个？**

兴奋-抑制（E/I）平衡是皮层网络的核心特征。单个方程只能描述一种类型的群体，而 E 和 I 之间的相互耦合产生了丰富的动力学：振荡、多稳态、分岔等。

---

**欧拉更新**：

```python
r_E[k+1] = r_E[k] + (dt/τ_E)*(-r_E[k] + F(w_EE*r_E[k] - w_EI*r_I[k] + I_ext_E, a_E, θ_E))
r_I[k+1] = r_I[k] + (dt/τ_I)*(-r_I[k] + F(w_IE*r_E[k] - w_II*r_I[k] + I_ext_I, a_I, θ_I))
```

**默认参数**：

| 参数       | 值     | 含义         |
| ---------- | ------ | ------------ |
| $\tau_E$   | 1.0 ms | E 群时间常数 |
| $\tau_I$   | 2.0 ms | I 群时间常数 |
| $a_E$      | 1.2    | E 群增益     |
| $a_I$      | 1.0    | I 群增益     |
| $\theta_E$ | 2.8    | E 群阈值     |
| $\theta_I$ | 4.0    | I 群阈值     |
| $w_{EE}$   | 9.0    | E→E 连接强度 |
| $w_{EI}$   | 4.0    | I→E 连接强度 |
| $w_{IE}$   | 13.0   | E→I 连接强度 |
| $w_{II}$   | 11.0   | I→I 连接强度 |

---

### Tutorial 2：零线 (Nullclines)

**零线定义**：系统中某个变量变化率为 0 的曲线。零线将相平面划分为不同流向的区域。

**E 零线 ($\frac{dr_E}{dt} = 0$, Eq. 2)**：

$$
-r_E + F_E(w_{EE}r_E - w_{EI}r_I + I_E^{\rm ext}; a_E, \theta_E) = 0
$$

**逐项解释**：

E 零线是 $(r_E, r_I)$ 空间中所有 $r_E$ 变化率为零的点的集合。方程形式为 $\text{衰减} + \text{驱动} = 0$，即：

$$
r_E = F_E(w_{EE}r_E - w_{EI}r_I + I_E^{\rm ext})
$$

- 左边：衰减项。要求在不动点处 $r_E$ 等于驱动产生的目标值
- 右边：驱动项。$r_E$ 被 $F_E$ 映射后的结果

**I 零线 ($\frac{dr_I}{dt} = 0$, Eq. 3)**：

$$
-r_I + F_I(w_{IE}r_E - w_{II}r_I + I_I^{\rm ext}; a_I, \theta_I) = 0
$$

同理，I 零线是 $r_I$ 变化率为零的曲线。

---

**零线显式表达 (Eqs. 4-5)**：

$$
\text{E 零线：} \quad r_I = \frac{1}{w_{EI}}[w_{EE}r_E - F_E^{-1}(r_E; a_E, \theta_E) + I_E^{\rm ext}]
$$

$$
\text{I 零线：} \quad r_E = \frac{1}{w_{IE}}[w_{II}r_I + F_I^{-1}(r_I; a_I, \theta_I) - I_I^{\rm ext}]
$$

**推导过程**：

从 E 零线条件 $-r_E + F_E(\cdots) = 0$ 出发：

1. $F_E(w_{EE}r_E - w_{EI}r_I + I_E) = r_E$
2. $w_{EE}r_E - w_{EI}r_I + I_E = F_E^{-1}(r_E)$ （两边取 $F^{-1}$）
3. $-w_{EI}r_I = F_E^{-1}(r_E) - w_{EE}r_E - I_E$
4. $r_I = (w_{EE}r_E - F_E^{-1}(r_E) + I_E) / w_{EI}$

---

**逆传递函数 (Eq. 6)**：

$$
F^{-1}(x; a, \theta) = -\frac{1}{a} \ln\left[\frac{1}{x + \frac{1}{1+e^{a\theta}}} - 1\right] + \theta
$$

**逐项解释**：

| 项 | 含义 | 为什么这样设计 |
|---|---|---|
| $F^{-1}(x)$ | 传递函数的**反函数** | 给定输出 $x$，问"需要多大的输入才能产生这个输出？" |
| $-\frac{1}{a} \ln[\cdots]$ | 反转 sigmoid 的代数运算 | 从 $y = 1/(1+e^{-a(x-\theta)})$ 反解 $x$ |
| $x + \frac{1}{1+e^{a\theta}}$ | **修正项**补偿 $F$ 的减法 | 必须抵消 $F$ 中的 $-\frac{1}{1+e^{a\theta}}$ 项 |
| $\cdots - 1$ | 反转 $\frac{1}{1+e^{-z}}$ | 标准 sigmoid 反函数的内部运算 |

**为什么需要逆函数？**

在相平面分析中，零线的显式形式让我们可以直接绘制零线：对于每个 $r_E$，计算对应的 $r_I$ 值。

---

**零线的性质**：

- E 零线将相平面分为 $\frac{dr_E}{dt} > 0$（零线上方/下方）和 $\frac{dr_E}{dt} < 0$ 两个区域
- I 零线将相平面分为 $\frac{dr_I}{dt} > 0$ 和 $\frac{dr_I}{dt} < 0$ 两个区域
- 两条零线的交点是系统的**不动点**
- 零线的形状（斜率）决定了不动点的稳定性

---

### Tutorial 2：向量场 (Vector Field)

**向量场定义**：在相平面每个点上显示 $(\frac{dr_E}{dt}, \frac{dr_I}{dt})$ 的箭头

```python
def EIderivs(rE, rI, tau_E, a_E, theta_E, wEE, wEI, I_ext_E,
             tau_I, a_I, theta_I, wIE, wII, I_ext_I, **other_pars):
    drEdt = (-rE + F(wEE*rE - wEI*rI + I_ext_E, a_E, theta_E)) / tau_E
    drIdt = (-rI + F(wIE*rE - wII*rI + I_ext_I, a_I, theta_I)) / tau_I
    return drEdt, drIdt
```

**关键观察**：

- 轨迹遵循向量场方向
- 不同轨迹最终到达两个不动点之一（取决于初始条件）
- 轨迹收敛的点是零线曲线的交点

---

### Tutorial 3：雅可比矩阵与稳定性

对于二维系统，不动点的稳定性由雅可比矩阵的特征值决定。雅可比矩阵是系统在不动点附近的一阶泰勒展开（线性近似）。

**系统重写**：

$$
\frac{dr_E}{dt} = G_E(r_E, r_I) = \frac{1}{\tau_E}[-r_E + F_E(w_{EE}r_E - w_{EI}r_I + I_E^{\rm ext}; a, \theta)]
$$

$$
\frac{dr_I}{dt} = G_I(r_E, r_I) = \frac{1}{\tau_I}[-r_I + F_I(w_{IE}r_E - w_{II}r_I + I_I^{\rm ext}; a, \theta)]
$$

---

**雅可比矩阵 (Jacobian, Eq. 7)**：

$$
J = \begin{bmatrix} \frac{\partial G_E}{\partial r_E} & \frac{\partial G_E}{\partial r_I} \\ \frac{\partial G_I}{\partial r_E} & \frac{\partial G_I}{\partial r_I} \end{bmatrix}
$$

**雅可比矩阵的几何含义**：

$J$ 描述了不动点附近每个方向上的"力"。如果 $J$ 的特征值实部都 < 0，所有方向的扰动都被拉回不动点 → 稳定。如果有任何一个方向的特征值实部 > 0，扰动在该方向上被放大 → 不稳定。

---

**雅可比矩阵元素 (Eqs. 8-11)**：

$$
J[0,0] = \frac{\partial G_E}{\partial r_E} = \frac{1}{\tau_E}[-1 + w_{EE} F_E'(w_{EE}r_E^* - w_{EI}r_I^* + I_E^{\rm ext})]
$$

**逐项解释——$J[0,0]$（E 群对自身的影响）**：

| 项 | 含义 | 为什么这样设计 |
|---|---|---|
| $-1$ | 泄漏项的导数 | $r_E$ 增加 → $(-r_E)$ 项使 $\frac{dr_E}{dt}$ 减少 |
| $w_{EE} F_E'$ | 自兴奋反馈的增益 | $r_E$ 增加 → $w_{EE}r_E$ 增加 → $F_E$ 增加 → 正反馈 |
| $\frac{1}{\tau_E}$ | 时间常数归一化 | 时间常数越大，导数越小（响应越慢） |

- 如果 $J[0,0] > 0$：E 群自兴奋太强 → 局部不稳定
- 如果 $J[0,0] < 0$：泄漏占主导 → 局部稳定

---

$$
J[0,1] = \frac{\partial G_E}{\partial r_I} = \frac{1}{\tau_E}[-w_{EI} F_E'(w_{EE}r_E^* - w_{EI}r_I^* + I_E^{\rm ext})]
$$

**$J[0,1]$（抑制群对兴奋群的影响）**：
- $w_{EI} > 0$，前面有负号 → 整体为负
- $r_I$ 增加 → $(-w_{EI}r_I)$ 更负 → $F_E$ 减小 → $\frac{dr_E}{dt}$ 减小
- 这就是"抑制"的数学表达

---

$$
J[1,0] = \frac{\partial G_I}{\partial r_E} = \frac{1}{\tau_I}[w_{IE} F_I'(w_{IE}r_E^* - w_{II}r_I^* + I_I^{\rm ext})]
$$

**$J[1,0]$（兴奋群对抑制群的影响）**：
- $w_{IE} > 0$ → 整体为正
- $r_E$ 增加 → $w_{IE}r_E$ 增加 → $F_I$ 增加 → $\frac{dr_I}{dt}$ 增加
- 这就是"兴奋驱动抑制"的数学表达

---

$$
J[1,1] = \frac{\partial G_I}{\partial r_I} = \frac{1}{\tau_I}[-1 - w_{II} F_I'(w_{IE}r_E^* - w_{II}r_I^* + I_I^{\rm ext})]
$$

**$J[1,1]$（抑制群对自身的影响）**：
- 泄漏项 $(-1)$ + 自抑制 $(-w_{II}F')$ → 两项都是负的
- 与 $J[0,0]$ 不同（后者可能是正），$J[1,1]$ **始终为负**
- 这意味着抑制群天然稳定（自抑制使其不会无限激发）

---

**矩阵符号表示**：

$$
J = T^{-1}(F W - I)
$$

其中：

- $T = \begin{bmatrix} \tau_E & 0 \\ 0 & \tau_I \end{bmatrix}$：**时间常数矩阵**（对角线，代表各群的时间常数）
- $F = \begin{bmatrix} F_E' & 0 \\ 0 & F_I' \end{bmatrix}$：**增益导数矩阵**（对角线，代表各群 F-I 曲线的斜率）
- $W = \begin{bmatrix} w_{EE} & -w_{EI} \\ w_{IE} & -w_{II} \end{bmatrix}$：**连接矩阵**（包含所有突触权重）

**为什么用矩阵形式？**

$J = T^{-1}(FW - I)$ 一次性捕获了系统的全部稳定性信息：

```
J = [时间尺度逆] × ( [增益] × [连接] - [单位阵] )
```

- $FW$ 是"有效连接矩阵" —— 连接权重被传递函数在不动点处的斜率缩放
- 减去 $I$ 表示"固有的泄漏衰减"
- 乘以 $T^{-1}$ 表示"时间常数越大的群变化越慢"

**稳定性准则**：

- 对于二维系统，$\det(J) > 0$ 且 $\text{tr}(J) < 0$ 时系统稳定
- $\det(J) > 0$ 保证特征值同号（两个都正或两个都负）
- $\text{tr}(J) < 0$ 保证特征值之和为负

$$
\det(FW - I) = (F_E' w_{EI})(F_I' w_{IE}) - (F_I' w_{II} + 1)(F_E' w_{EE} - 1) > 0
$$

| 项 | 含义 |
|---|---|
| $(F_E' w_{EI})(F_I' w_{IE})$ | E→I→E 环路的"环路增益" |
| $(F_I' w_{II} + 1)$ | I 固有衰减 + I 自抑制 |
| $(F_E' w_{EE} - 1)$ | E 自兴奋减去固有衰减 |
| 整个式子 > 0 | I 对 E 的抑制反馈必须足够强才能稳定 |

**实现**：

```python
def get_eig_Jacobian(fp, tau_E, a_E, theta_E, wEE, wEI, I_ext_E,
                     tau_I, a_I, theta_I, wIE, wII, I_ext_I, **other_pars):
    rE, rI = fp
    J = np.zeros((2, 2))
    J[0, 0] = (-1 + wEE * dF(wEE*rE - wEI*rI + I_ext_E, a_E, theta_E)) / tau_E
    J[0, 1] = (-wEI * dF(wEE*rE - wEI*rI + I_ext_E, a_E, theta_E)) / tau_E
    J[1, 0] = (wIE * dF(wIE*rE - wII*rI + I_ext_I, a_I, theta_I)) / tau_I
    J[1, 1] = (-1 - wII * dF(wIE*rE - wII*rI + I_ext_I, a_I, theta_I)) / tau_I
    evals = np.linalg.eig(J)[0]
    return evals
```

---

### Tutorial 3：零线斜率分析 (Nullcline Slope Analysis)

**E 零线斜率 (Eq. 12)**：

$$
\left(\frac{dr_I}{dr_E}\right)_{\text{E零线}} = \frac{F_E' w_{EE} - 1}{F_E' w_{EI}}
$$

**I 零线斜率 (Eq. 13)**：

$$
\left(\frac{dr_I}{dr_E}\right)_{\text{I零线}} = \frac{F_I' w_{IE}}{F_I' w_{II} + 1}
$$

**性质**：

- I 零线斜率始终为正
- E 零线斜率的符号取决于 $(F_E' w_{EE} - 1)$

**结论 1**：在稳定不动点处，I 零线比 E 零线更陡峭

**结论 2**：向抑制群添加输入时

- E 零线保持不变
- I 零线向左平移 $\delta I_I^{\rm ext} / w_{IE}$

---

### Tutorial 3：极限环与振荡 (Limit Cycles and Oscillations)

**振荡产生的条件**：特征值变为**复数**

**振荡参数**：$w_{EE}=6.4$，$w_{EI}=4.8$，$w_{IE}=6.0$，$w_{II}=1.2$，$I_E^{\rm ext}=0.8$

- 轨迹在相平面中形成**极限环 (limit cycle)**
- 兴奋 (E) 和抑制 (I) 群交替活跃
- 频率由特征值的虚部决定
- 振荡稳定性由特征值实部决定（正实部 → 振荡增长，负实部 → 振荡衰减）

**分岔 (Bifurcation)**：随着参数变化，系统行为发生剧烈变化

- 改变 $\tau_I$ 可以在稳态与振荡之间切换
- 零线保持不变，但向量场发生变化
- 直觉：$\tau_I$ 较小时，抑制活动变化快于兴奋活动，导致振荡

---

### Tutorial 3：抑制稳定网络 (Inhibition-Stabilized Network, ISN)

**基于 $\frac{\partial G_E}{\partial r_E}$ 的两种模式**：

$$
\frac{\partial G_E}{\partial r_E} = \frac{1}{\tau_E}[-1 + w_{EE} F_E'] = \frac{1}{\tau_E}(F_E' w_{EE} - 1)
$$

| 模式                 | 条件                  | E 零线斜率 | 行为                           |
| -------------------- | --------------------- | ---------- | ------------------------------ |
| **非 ISN (non-ISN)** | $F_E' w_{EE} - 1 < 0$ | 负         | 增加对 I 的抑制 → E 减少       |
| **ISN**              | $F_E' w_{EE} - 1 > 0$ | 正         | 增加对 I 的抑制 → E 矛盾地增加 |

**ISN 在皮层中很常见**：强的反复性兴奋 ($w_{EE}$ 较大) 创造了一种需要抑制来维持稳定的模式

**ISN 的矛盾行为**：

- 正常情况：抑制 I → E 增加（减少抑制）
- ISN 情况：抑制 I → E 也减少（因为 E 的自兴奋太强，需要 I 来稳定）

---

### Tutorial 3：工作记忆：持续活动 (Working Memory: Persistent Activity)

**机制**：多个不动点 + 噪声

1. 系统从低活动不动点开始
2. 短暂脉冲将状态推过不稳定不动点
3. 系统在高活动不动点稳定下来
4. 这代表了对刺激的 " 记忆 "

**实现**：OU 噪声 + 短暂电流脉冲

```python
def my_inject(pars, t_start, t_lag=10.):
    I = np.zeros(Lt)
    N_start = int(t_start / dt)
    N_lag = int(t_lag / dt)
    I[N_start:N_start + N_lag] = 1.
    return I
```

**关键参数**：

- 脉冲幅度 $S_E$ 决定是否触发转换
- 临界脉冲幅度：刚好足够将状态推过不稳定不动点
- 足够大的脉冲：系统切换到持续活动
- 脉冲结束后：系统保持在高活动状态（工作记忆）

---

## 总结 (Summary)

---

### 第 2 周：核心概念 (Key Concepts)

### W2D3：线性系统 (Linear Systems)

- 欧拉积分 (Euler integration)
- 特征值分析 (Eigenvalue analysis)
- 马尔可夫过程与状态转移矩阵 (Markov processes & state transition matrices)
- 随机游走与扩散过程 (Random walks & diffusion processes)
- OU 过程与平衡方差 (OU process & equilibrium variance)
- 自回归模型与时间延迟矩阵 (AR models & time-delay matrices)

### W2D4：神经元模型 (Neuron Models)

- LIF 神经元动力学与欧拉积分 (LIF neuron dynamics & Euler integration)
- DC/GWN/OU 输入类型 (DC/GWN/OU input types)
- 相关输入与相关性转移 (Correlated inputs & correlation transfer)
- 基于电导的突触 (Conductance-based synapses)
- 自由膜电位与放电体制 (FMP & firing regimes)
- 短期可塑性：抑制与易化 (STP: depression & facilitation)
- STDP 学习规则与权重更新 (STDP learning rule & weight updates)
- P/M 追踪变量 (P/M trace variables)

### W2D5：网络动力学 (Network Dynamics)

- 放电频率模型与 sigmoid 传递函数 (Firing rate model & sigmoid transfer function)
- 不动点与特征值稳定性 (Fixed points & eigenvalue stability)
- Wilson-Cowan 模型与 E/I 耦合 (Wilson-Cowan model & E/I coupling)
- 零线与向量场 (Nullclines & vector fields)
- 雅可比矩阵与线性化 (Jacobian matrix & linearization)
- 零线斜率分析 (Nullcline slope analysis)
- 极限环与分岔 (Limit cycles & bifurcations)
- 抑制稳定网络 (Inhibition-stabilized network)
- 工作记忆与持续活动 (Working memory & persistent activity)

---

### 关键公式汇总 (Key Formulas)

$$
\tau_m \frac{dV}{dt} = -(V-E_L) + \frac{I}{g_L} \quad \text{(LIF neuron)}
$$

$$
\tau_m \frac{dV}{dt} = -(V-E_L) - \frac{g_E}{g_L}(V-E_E) - \frac{g_I}{g_L}(V-E_I) + \frac{I_{\rm inj}}{g_L} \quad \text{(Conductance-based LIF)}
$$

$$
x_{k+1} = x_\infty + \lambda(x_k - x_\infty) + \sigma\eta \quad \text{(OU process)}
$$

$$
\text{Var}_{eq} = \frac{\sigma^2}{1-\lambda^2} \quad \text{(OU equilibrium variance)}
$$

$$
\tau \frac{dr}{dt} = -r + F(w \cdot r + I_{\rm ext}) \quad \text{(Firing rate model)}
$$

$$
F(x; a, \theta) = \frac{1}{1+e^{-a(x-\theta)}} - \frac{1}{1+e^{a\theta}} \quad \text{(Sigmoid transfer function)}
$$

$$
\tau_E \frac{dr_E}{dt} = -r_E + F_E(w_{EE}r_E - w_{EI}r_I + I_E^{\rm ext}) \quad \text{(Wilson-Cowan E)}
$$

$$
\tau_I \frac{dr_I}{dt} = -r_I + F_I(w_{IE}r_E - w_{II}r_I + I_I^{\rm ext}) \quad \text{(Wilson-Cowan I)}
$$

$$
\lambda = \frac{-1 + w \cdot F'(w \cdot r^* + I_{\rm ext})}{\tau} \quad \text{(Eigenvalue/stability)}
$$

$$
J = T^{-1}(FW - I) \quad \text{(Jacobian matrix)}
$$

$$
\frac{dr_I}{dr_E}\bigg|_{\text{E零线}} = \frac{F_E' w_{EE} - 1}{F_E' w_{EI}} \quad \text{(E nullcline slope)}
$$

$$
\frac{dr_I}{dr_E}\bigg|_{\text{I零线}} = \frac{F_I' w_{IE}}{F_I' w_{II} + 1} \quad \text{(I nullcline slope)}
$$

$$
\Delta W = \begin{cases} A_+ e^{\Delta t/\tau_+} & \Delta t < 0 \text{ (LTP)} \\ -A_- e^{-\Delta t/\tau_-} & \Delta t > 0 \text{ (LTD)} \end{cases} \quad \text{(STDP rule)}
$$

---

### 教程之间的逻辑联系

| 教程    | 模型           | 维度 | 关键分析                          |
| ------- | -------------- | ---- | --------------------------------- |
| W2D3 T1 | $\dot{x} = ax$ | 1D   | 欧拉积分，特征值                  |
| W2D3 T2 | 马尔可夫过程   | 2D   | 状态转移矩阵，平衡态              |
| W2D3 T3 | OU 过程        | 1D   | 随机游走，漂移扩散，平衡方差      |
| W2D3 T4 | 自回归模型     | 1D   | 时间延迟矩阵，回归拟合            |
| W2D4 T1 | LIF 神经元     | 1D   | 膜电位动力学，F-I 曲线，CV_ISI    |
| W2D4 T2 | 相关 LIF       | 2×1D | 相关输入，相关性转移              |
| W2D4 T3 | 电导 LIF + STP | 1D   | 突触电导，u-R-g 动力学            |
| W2D4 T4 | LIF + STDP     | N×1D | 权重更新，无监督学习              |
| W2D5 T1 | 单群放电频率   | 1D   | F-I 曲线，不动点，特征值稳定性    |
| W2D5 T2 | Wilson-Cowan   | 2D   | 零线，向量场，相平面              |
| W2D5 T3 | WC + 分析      | 2D   | 雅可比矩阵，极限环，ISN，工作记忆 |

**渐进关系**：从单群单特征值，到双群 2×2 雅可比矩阵（两个特征值可为实数或复数），实现更丰富的动力学（振荡和双稳态）。
