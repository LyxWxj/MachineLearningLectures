# 深度学习基础 — 第一周

PyTorch · 梯度下降 · 多层感知机 · 优化

---

## 总览

第一周是深度学习的**地基**——从张量操作到梯度下降，从线性模型到多层感知机，最后落脚于优化算法：

| 日       | 主题               | 核心技能                                                             |
| -------- | ------------------ | -------------------------------------------------------------------- |
| **W1D1** | PyTorch 基础       | 张量创建与操作、GPU 加速、数据加载、构建第一个神经网络               |
| **W1D2** | 线性深度学习       | 梯度下降、计算图、反向传播、PyTorch Autograd、nn.Module 训练循环     |
| **W1D3** | 多层感知机 (MLP)   | 万能逼近定理、ReLU 基函数、MLP 架构、交叉熵损失、螺旋数据集分类     |
| **W1D4** | 优化               | SGD、动量、RMSprop、病态条件、非凸性、过参数化、小批量训练           |

**贯穿主题**：深度学习 = 参数化函数族 + 损失函数 + 基于梯度的优化。

---

## W1D1：PyTorch 基础

---

### 1. 张量：深度学习的基本数据结构

**张量**（Tensor）是 NumPy `ndarray` 的 GPU 加速版本，同时支持自动微分。

```python
import torch

# 从 Python 列表创建
a = torch.tensor([1, 2, 3])

# 从 NumPy 数组创建（共享内存！）
import numpy as np
b = torch.tensor(np.ones([2, 3]))

# 常用构造器
c = torch.zeros(3, 4)       # 全零
d = torch.ones(2, 3)        # 全一
e = torch.empty(5)           # 未初始化（快，但值无意义）
f = torch.arange(0, 10, 2)  # 类似 range
g = torch.linspace(0, 1, 5) # 等间距

# 随机张量
h = torch.rand(3, 4)    # 均匀分布 U(0,1)
i = torch.randn(3, 4)   # 标准正态 N(0,1)
```

**可复现性**（Reproducibility）：设置随机种子保证实验可重复：

```python
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)
```

---

### 2. 张量操作

#### 2.1 基本索引与切片

与 NumPy 完全一致：

```python
x = torch.arange(12).reshape(3, 4)
# tensor([[ 0,  1,  2,  3],
#         [ 4,  5,  6,  7],
#         [ 8,  9, 10, 11]])

x[0, :]       # 第 0 行 → tensor([0, 1, 2, 3])
x[:, 1]       # 第 1 列 → tensor([1, 5, 9])
x[1:3, 0:2]   # 子矩阵
x[-1]          # 最后一行（负索引）
```

#### 2.2 形状变换

```python
x = torch.arange(12)

x.reshape(3, 4)   # 重塑（元素总数必须匹配）
x.view(3, 4)      # 同上，但要求内存连续
x.flatten()        # 展平为 1D
x.unsqueeze(0)     # 增加一个维度：(12,) → (1, 12)
x.unsqueeze(-1)    # 增加最后一个维度：(12,) → (12, 1)
y = torch.randn(1, 10)
y.squeeze(0)       # 去掉大小为 1 的维度：(1, 10) → (10,)
```

**`view` vs `reshape`**：`view` 要求张量在内存中连续（contiguous），否则报错；`reshape` 会在需要时自动复制数据。实践中推荐用 `reshape`，除非你明确需要共享内存。

#### 2.3 维度置换与转置

```python
# 图像: (C, H, W) → (H, W, C)
img = torch.rand(3, 48, 64)
img_hwc = img.permute(1, 2, 0)   # 任意维度重排

# 2D 转置
M = torch.randn(3, 5)
M.T            # (5, 3)
M.transpose(0, 1)  # 等价
```

#### 2.4 拼接与分割

```python
a = torch.randn(2, 3)
b = torch.randn(4, 3)

torch.cat([a, b], dim=0)  # 沿行拼接 → (6, 3)
torch.cat([a.T, b.T], dim=1)  # 沿列拼接

# stack 在新维度上拼接
c = torch.stack([a, a], dim=0)  # (2, 2, 3)
```

---

### 3. 爱因斯坦求和约定 (Einsum) 与 einops

#### 3.1 `torch.einsum` —— 统一的张量运算语言

爱因斯坦求和约定的核心规则：
- **重复索引 = 求和**
- **自由索引 = 输出维度**

```python
# 向量点积: c = Σ_i a_i b_i
torch.einsum('i,i->', a, b)

# 矩阵-向量乘法: y_i = Σ_j A_ij x_j
torch.einsum('ij,j->i', A, x)

# 矩阵乘法: C_ij = Σ_k A_ik B_kj
torch.einsum('ik,kj->ij', A, B)
# 等价于 A @ B

# 批量矩阵乘法: C_bij = Σ_k A_bik B_bkj
torch.einsum('bik,bkj->bij', A, B)

# 转置
torch.einsum('ij->ji', A)

# 对角线提取
torch.einsum('ii->i', A)

# 迹 (trace)
torch.einsum('ii->', A)

# 张量缩并 (contract)
# 对两个张量的某些维度求和
torch.einsum('bijk,bilm->bjkm', T1, T2)

# 外积
torch.einsum('i,j->ij', a, b)
```

#### 3.2 `einops` 库 —— 更可读的张量操作

`einops` 提供了比 `reshape`/`view`/`permute` 更直观的 API：

```python
from einops import rearrange, reduce, repeat

x = torch.randn(2, 3, 4)

# rearrange: 重排维度
rearrange(x, 'b c h -> b h c')          # 等价于 permute(0, 2, 1)
rearrange(x, 'b c h -> (b c) h')        # 合并前两维
rearrange(x, 'b (c1 c2) h -> b c1 c2 h', c1=3)  # 拆分维度

# reduce: 聚合操作
reduce(x, 'b c h -> b c', 'mean')       # 对 h 维求均值
reduce(x, 'b c h -> b', 'max')          # 对 c,h 维求最大值
reduce(x, 'b c h -> ()', 'sum')         # 全部求和

# repeat: 重复
repeat(x, 'b c h -> b c h w', w=5)      # 在新维度上重复
repeat(x[0], 'c h -> b c h', b=4)       # 复制 batch
```

在 Transformer 中 einops 尤其有用：

```python
# 多头注意力中的维度变换
q = rearrange(q, 'b t (h k) -> b h t k', h=num_heads)  # 拆分头
out = rearrange(out, 'b h t k -> b t (h k)')            # 合并头
```

---

### 4. 矩阵乘法的维度匹配规则

#### 4.1 基本规则

`torch.matmul` (或 `@`) 的维度匹配：

```
(A) @ (B) → (C)

规则：最后一个维度 of A == 倒数第二个维度 of B

例如：
(3, 4) @ (4, 5) → (3, 5)
```

#### 4.2 批量矩阵乘法 (Batched MatMul)

当两个张量都有额外的前导维度时，PyTorch 会**自动广播**：

```python
# 情况 1: 两个 3D 张量
A = torch.randn(8, 3, 4)   # batch=8, 矩阵 3×4
B = torch.randn(8, 4, 5)   # batch=8, 矩阵 4×5
C = A @ B                   # (8, 3, 5) — 逐 batch 矩阵乘法

# 情况 2: 一个 3D，一个 2D — 广播！
A = torch.randn(8, 3, 4)   # batch=8, 矩阵 3×4
B = torch.randn(4, 5)      # 矩阵 4×5（无 batch 维）
C = A @ B                   # (8, 3, 5) — B 被广播到每个 batch

# 情况 3: 两个 4D 张量
Q = torch.randn(8, 6, 10, 64)  # batch=8, heads=6, seq=10, dim=64
K = torch.randn(8, 6, 10, 64)
# 计算注意力分数: Q @ K^T
scores = torch.einsum('bhik,bhjk->bhij', Q, K)  # (8,6,10,10)
# 或者
scores = Q @ K.transpose(-2, -1)  # 等价
```

#### 4.3 `torch.bmm` —— 严格的批量矩阵乘法

```python
# bmm 要求严格的 3D: (batch, n, m) @ (batch, m, p) → (batch, n, p)
A = torch.randn(32, 10, 64)
B = torch.randn(32, 64, 20)
C = torch.bmm(A, B)  # (32, 10, 20)
```

**选择建议**：`@` 运算符支持广播，更灵活；`bmm` 更严格，有时更清晰。

---

### 5. GPU 与设备管理

```python
# 检查 GPU 是否可用
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 创建时指定设备
x = torch.randn(3, 4, device=device)

# 移动到 GPU
x = x.to('cuda')
x = x.cuda()

# 移动回 CPU
x = x.cpu()
x = x.to('cpu')

# 注意：不同设备的张量不能运算！
a = torch.randn(3, device='cuda')
b = torch.randn(3, device='cpu')
# a + b → RuntimeError!

# 代码模式：设备无关
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
model = MyModel().to(DEVICE)
data = data.to(DEVICE)
```

---

### 6. 数据集与 DataLoader

```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 加载 CIFAR10
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_set = datasets.CIFAR10(root='./data', train=True,
                             download=True, transform=transform)

# DataLoader: 自动分批、打乱、多进程加载
train_loader = DataLoader(train_set, batch_size=64,
                          shuffle=True, num_workers=2)

# 迭代
for images, labels in train_loader:
    images = images.to(DEVICE)  # (64, 3, 32, 32)
    labels = labels.to(DEVICE)  # (64,)
    # ... 训练 ...
```

---

### 7. 构建第一个神经网络

```python
import torch.nn as nn
import torch.nn.functional as F

class SimpleNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 展平
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

model = SimpleNet(784, 128, 10).to(DEVICE)
```

---

## W1D2：梯度下降与 Autograd

---

### 1. 梯度的方向

对于函数 $f(\mathbf{x}): \mathbb{R}^d \to \mathbb{R}$，梯度

$$
\nabla f(\mathbf{x}) = \left[\frac{\partial f}{\partial x_1}, \ldots, \frac{\partial f}{\partial x_d}\right]^\top
$$

总是指向**最速上升**方向。因此 $-\nabla f$ 指向最速下降方向。

---

### 2. 梯度下降算法

$$
\mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} - \eta \, \nabla_{\mathbf{w}} \mathcal{L}\left(\mathbf{w}^{(t)}\right)
$$

其中 $\eta > 0$ 是学习率，$\mathcal{L}$ 是损失函数。

**收敛条件**（凸函数）：学习率足够小时，梯度下降保证收敛到全局最优。

**非凸情况**（神经网络）：只能保证收敛到**局部最优**或**鞍点**——但实践中这通常足够好。

---

### 3. 计算图与反向传播

#### 3.1 链式法则

对于复合函数 $F(x) = g(h(x))$：

$$
F'(x) = g'(h(x)) \cdot h'(x)
$$

反向传播就是链式法则在计算图上的系统应用。

#### 3.2 计算图示例

对于 $f = \tanh(\ln(1 + z \cdot \frac{2x}{\sin y}))$：

前向传播：依次计算中间变量 $a, b, c, d, e$

反向传播：从 $f$ 出发，沿箭头反向计算每个中间变量的梯度，最终得到 $\frac{\partial f}{\partial x}, \frac{\partial f}{\partial y}, \frac{\partial f}{\partial z}$。

---

### 4. PyTorch Autograd

```python
# requires_grad=True 告诉 PyTorch 追踪这个张量上的所有操作
w = torch.tensor([2.0], requires_grad=True)
b = torch.tensor([1.0], requires_grad=True)

x = torch.tensor([3.0])
y_true = torch.tensor([10.0])

# 前向传播
y_pred = w * x + b
loss = (y_true - y_pred) ** 2

# 反向传播：自动计算梯度
loss.backward()

print(w.grad)  # dL/dw
print(b.grad)  # dL/db

# 重要：梯度会累积！每次 backward 前要清零
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

#### 4.1 detach 与 no_grad

```python
# .detach(): 从计算图中分离，不再追踪梯度
y = model(x).detach()  # 用于绘图或存储

# torch.no_grad(): 上下文管理器，禁用梯度计算（节省内存）
with torch.no_grad():
    predictions = model(test_data)
```

---

### 5. nn.Module 训练循环

```python
model = SimpleNet(784, 128, 10).to(DEVICE)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)

        optimizer.zero_grad()          # 1. 清零梯度
        output = model(batch_x)        # 2. 前向传播
        loss = loss_fn(output, batch_y) # 3. 计算损失
        loss.backward()                # 4. 反向传播
        optimizer.step()               # 5. 更新参数
```

---

## W1D3：多层感知机 (MLP)

---

### 1. 万能逼近定理

**定理**（Cybenko 1989, Hornik 1991）：一个具有**单个隐藏层**和足够多神经元的前馈网络，可以以任意精度逼近任何连续函数。

**直觉**：ReLU 神经元是一个"分段线性基函数"——通过线性组合足够多的 ReLU，可以逼近任何光滑函数。

$$
f(x) \approx \sum_{i=1}^{N} \alpha_i \, \text{ReLU}(x - b_i)
$$

每个 ReLU 在位置 $b_i$ 产生一个"折点"，$\alpha_i$ 控制斜率变化。

---

### 2. MLP 架构

$$
\mathbf{h}^{(l)} = \sigma\left(\mathbf{W}^{(l)} \mathbf{h}^{(l-1)} + \mathbf{b}^{(l)}\right)
$$

其中 $\sigma$ 是激活函数（ReLU、tanh、sigmoid 等）。

**PyTorch 实现**：

```python
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.view(x.size(0), -1))
```

---

### 3. 激活函数

| 函数         | 公式                                           | 特点                              |
| ------------ | ---------------------------------------------- | --------------------------------- |
| **ReLU**     | $\max(0, x)$                                   | 简单高效，梯度消失问题较轻        |
| **Leaky ReLU** | $\max(0, x) + \alpha \min(0, x)$             | 解决 ReLU 的"死神经元"问题        |
| **GELU**     | $x \cdot \Phi(x)$                              | Transformer 中常用，平滑          |
| **Sigmoid**  | $\frac{1}{1+e^{-x}}$                           | 输出 (0,1)，容易梯度消失          |
| **Tanh**     | $\tanh(x)$                                     | 输出 (-1,1)，零中心化             |

---

### 4. 损失函数

#### 4.1 交叉熵损失（分类）

对于 $N$ 个样本、$C$ 个类别：

$$
\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N} \log \frac{\exp(x_i[y_i])}{\sum_{j=1}^{C} \exp(x_i[j])}
$$

其中 $x_i$ 是第 $i$ 个样本的 logits，$y_i$ 是真实标签。

**PyTorch 实现**：

```python
loss_fn = nn.CrossEntropyLoss()  # 内部已包含 softmax
loss = loss_fn(logits, labels)   # logits: (N, C), labels: (N,)
```

**注意**：`nn.CrossEntropyLoss` 内部已包含 softmax，输入应该是 **logits** 而非概率。

#### 4.2 均方误差（回归）

$$
\mathcal{L} = \frac{1}{N}\sum_{i=1}^{N} \|y_i - \hat{y}_i\|^2
$$

```python
loss_fn = nn.MSELoss()
```

---

### 5. 生物神经元的启示

**漏积分发放 (LIF) 模型**：

$$
C_m \frac{dV}{dt} = -\frac{V}{R_m} + I
$$

当 $V \geq V_{\text{th}}$ 时发放脉冲并重置。

**与 ReLU 的联系**：在极限 $R_m \to \infty$（无泄漏）且 $\tau_{\text{ref}} \to 0$（无不应期）时，LIF 神经元的频率-电流关系趋近于 ReLU：

$$
f(I) \propto \max(0, I - V_{\text{th}})
$$

---

## W1D4：优化

---

### 1. 随机梯度下降 (SGD)

全批量梯度下降在大数据集上太慢。SGD 每次只用一个小批量：

$$
\mathbf{w} \leftarrow \mathbf{w} - \eta \, \frac{1}{|B|}\sum_{i \in B} \nabla_{\mathbf{w}} \mathcal{L}_i
$$

**方差 vs 偏差权衡**：小批量引入梯度噪声，但这种噪声有时有助于逃离局部最优。

---

### 2. 动量 (Momentum)

解决 SGD 在"峡谷"地形中的振荡问题：

$$
\mathbf{v}_t = \beta \, \mathbf{v}_{t-1} + \eta \, \nabla \mathcal{L}(\mathbf{w}_t)
$$
$$
\mathbf{w}_{t+1} = \mathbf{w}_t - \mathbf{v}_t
$$

**直觉**：在一致的梯度方向上积累"速度"，在振荡方向上相互抵消。

---

### 3. 自适应学习率方法

#### 3.1 RMSprop

$$
v_t = \alpha \, v_{t-1} + (1 - \alpha) \, (\nabla \mathcal{L})^2
$$
$$
\mathbf{w}_{t+1} = \mathbf{w}_t - \frac{\eta}{\sqrt{v_t + \epsilon}} \, \nabla \mathcal{L}
$$

**核心思想**：对每个参数维护一个自适应的学习率——梯度历史大的参数用小学习率，反之亦然。

#### 3.2 Adam

结合动量和 RMSprop：

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) \nabla \mathcal{L} \quad \text{(一阶矩估计)}
$$
$$
v_t = \beta_2 v_{t-1} + (1-\beta_2) (\nabla \mathcal{L})^2 \quad \text{(二阶矩估计)}
$$
$$
\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t} \quad \text{(偏差校正)}
$$
$$
\mathbf{w}_{t+1} = \mathbf{w}_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t
$$

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
```

---

### 4. 病态条件 (Poor Conditioning)

当损失函数在不同方向上的曲率差异很大时，梯度下降在平坦方向上极慢，在陡峭方向上振荡。

**动量如何帮助**：动量在一致方向上积累速度，加速了平坦方向的收敛，同时抑制了陡峭方向的振荡。

---

### 5. 非凸性与过参数化

神经网络的损失函数是**非凸**的——存在大量局部最优和鞍点。

**过参数化假说**：当网络参数远多于训练样本时，损失景观中的"好解"变得非常密集，使得优化更容易找到好的解。

---

### 6. 小批量 (Minibatch) 训练

**批量大小的权衡**：
- 大批量 → 梯度估计更准确，但每次更新更慢
- 小批量 → 每次更新快，但梯度噪声大

**实践建议**：通常 32-256 是一个好的起点。大批量需要更大学习率。

---

### 7. 学习率调度

```python
# 阶梯衰减
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# 余弦退火
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

# 每个 epoch 结束后调用
for epoch in range(num_epochs):
    train(...)
    scheduler.step()
```

---

## 综合练习：多头缩放点积注意力

这是一个综合性的练习，涵盖张量操作、矩阵乘法、维度变换等多个知识点。

### 数学公式

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
$$

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O
$$

其中 $\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$

### 逐步实现框架

**步骤 1**：线性投影（拆分头）

输入: $Q \in \mathbb{R}^{B \times T \times d_{\text{model}}}$

```python
# 投影: (B, T, d_model) → (B, T, h * d_k)
Q_proj = W_Q(Q)  # nn.Linear(d_model, h * d_k, bias=False)
K_proj = W_K(K)
V_proj = W_V(V)

# 拆分头: (B, T, h * d_k) → (B, h, T, d_k)
Q_heads = rearrange(Q_proj, 'b t (h k) -> b h t k', h=num_heads)
K_heads = rearrange(K_proj, 'b t (h k) -> b h t k', h=num_heads)
V_heads = rearrange(V_proj, 'b t (h k) -> b h t k', h=num_heads)
```

**步骤 2**：计算注意力分数

```python
# scores: (B, h, T, T)
scores = Q_heads @ K_heads.transpose(-2, -1) / math.sqrt(d_k)
```

**步骤 3**：掩码（可选，用于因果注意力）

```python
# 上三角掩码：防止位置 i 注意到位置 j > i
mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
scores = scores.masked_fill(mask, float('-inf'))
```

**步骤 4**：Softmax + 加权求和

```python
weights = F.softmax(scores, dim=-1)  # (B, h, T, T)
out = weights @ V_heads              # (B, h, T, d_k)
```

**步骤 5**：合并头 + 输出投影

```python
# 合并头: (B, h, T, d_k) → (B, T, h * d_k)
out = rearrange(out, 'b h t k -> b t (h k)')

# 输出投影: (B, T, h * d_k) → (B, T, d_model)
out = W_O(out)  # nn.Linear(h * d_k, d_model)
```

### 练习要点

- 注意 `rearrange` 如何将 `(B, T, h*d_k)` 拆分为 `(B, h, T, d_k)`
- 理解为什么除以 $\sqrt{d_k}$（防止 softmax 饱和）
- 掩码操作中 `masked_fill` 的用法
- 最终输出的维度必须与输入一致

---

## 参考资料

- [PyTorch 官方教程](https://pytorch.org/tutorials/)
- [Deep Learning Book (Goodfellow et al.)](https://www.deeplearningbook.org/)
- [einops 文档](https://einops.rocks/)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
