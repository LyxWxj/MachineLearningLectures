# 深度学习进阶 — 第二周

正则化 · 卷积神经网络 · 生成模型 · 扩散模型 · 序列与语言

---

## 总览

第二周将视野从全连接网络扩展到**更丰富的架构和任务**——卷积网络处理图像、生成模型创造新样本、序列模型处理语言：

| 日       | 主题                 | 核心技能                                                         |
| -------- | -------------------- | ---------------------------------------------------------------- |
| **W2D1** | 正则化               | 过拟合、Frobenius 范数、早停、数据增强、记忆化                   |
| **W2D2** | 卷积神经网络 (CNN)   | 卷积操作、池化、经典架构、特征可视化                             |
| **W2D3** | 生成模型             | 自编码器、PCA、变分自编码器 (VAE)、GAN 基础                      |
| **W2D4** | 扩散生成模型         | 前向扩散、Score 函数、反向 SDE、去噪分数匹配                     |
| **W2D5** | 序列与语言           | 词嵌入 (Word2Vec/FastText)、文本分类、RNN 基础                   |

**贯穿主题**：如何让模型**泛化**而非记忆？如何**生成**新数据？如何处理**序列**数据？

---

## W2D1：正则化

---

### 1. 过拟合问题

**过拟合**：模型在训练集上表现很好，但在测试集上表现差。

**根本原因**：模型容量（参数数量）相对于训练数据量过大，导致模型"记住"了训练数据中的噪声。

**经典表现**：训练 loss 持续下降，但验证 loss 先降后升。

---

### 2. Frobenius 范数与权重衰减

**Frobenius 范数**衡量权重矩阵的"大小"：

$$
\|A\|_F = \sqrt{\sum_{i,j} |a_{ij}|^2}
$$

**L2 正则化**（权重衰减）：在损失函数中添加权重的 Frobenius 范数惩罚：

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + \lambda \sum_l \|\mathbf{W}^{(l)}\|_F^2
$$

**效果**：鼓励权重变小，降低模型复杂度，提高泛化能力。

```python
# PyTorch 中的 L2 正则化
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)
```

---

### 3. 早停 (Early Stopping)

**思想**：在验证集性能不再提升时停止训练。

```python
best_val_acc = 0
patience = 10
wait = 0

for epoch in range(max_epochs):
    train(...)
    val_acc = evaluate(...)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model = copy.deepcopy(model)
        wait = 0
    else:
        wait += 1

    if wait > patience:
        print(f"Early stopping at epoch {epoch}")
        break
```

**超参数**：`patience`（容忍多少个 epoch 不提升）

---

### 4. 记忆化实验

**惊人的发现**：即使标签完全随机，足够大的网络也能达到 100% 训练准确率！

这说明训练准确率**不是**衡量模型好坏的指标——必须看验证/测试集。

---

### 5. 数据增强 (Data Augmentation)

通过对训练数据施加变换来"创造"更多样本：

| 变换         | 适用场景                 |
| ------------ | ------------------------ |
| 水平翻转     | 自然图像                 |
| 随机裁剪     | 图像分类                 |
| 颜色抖动     | 图像分类                 |
| 随机旋转     | 旋转不变的任务           |
| Mixup        | 两张图片线性插值         |
| CutMix       | 将一张图的区域贴到另一张 |

```python
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
])
```

---

## W2D2：卷积神经网络 (CNN)

---

### 1. 为什么需要 CNN？

全连接网络处理图像的问题：
- 参数太多：$28 \times 28 = 784$ 像素，隐藏层 1000 → $784 \times 1000 = 784,000$ 参数
- 不具备平移不变性

**CNN 的核心思想**：
- **局部连接**：每个神经元只看输入的一个小区域
- **参数共享**：同一个卷积核在整个输入上滑动
- **平移等变性**：物体在图像中平移，特征图也平移

---

### 2. 卷积操作

#### 2.1 1D 卷积

$$
y[i] = \sum_{k=0}^{K-1} w[k] \cdot x[i+k]
$$

#### 2.2 2D 卷积

$$
y[i,j] = \sum_{m=0}^{M-1}\sum_{n=0}^{N-1} w[m,n] \cdot x[i+m, j+n]
$$

```python
# PyTorch 卷积层
conv = nn.Conv2d(
    in_channels=3,      # 输入通道数（RGB = 3）
    out_channels=16,    # 输出通道数（卷积核数量）
    kernel_size=3,      # 卷积核大小
    stride=1,           # 步长
    padding=1           # 填充（保持空间尺寸）
)

x = torch.randn(1, 3, 32, 32)  # (batch, channels, height, width)
y = conv(x)  # (1, 16, 32, 32)
```

#### 2.3 输出尺寸计算

$$
H_{\text{out}} = \left\lfloor\frac{H_{\text{in}} + 2 \cdot \text{padding} - \text{kernel\_size}}{\text{stride}}\right\rfloor + 1
$$

---

### 3. 池化 (Pooling)

**最大池化**：取局部区域的最大值

$$
y[i,j] = \max_{(m,n) \in \mathcal{R}_{i,j}} x[m,n]
$$

```python
pool = nn.MaxPool2d(kernel_size=2, stride=2)
# (1, 16, 32, 32) → (1, 16, 16, 16)
```

**作用**：降低空间分辨率，增大感受野，提供一定的平移不变性。

---

### 4. 经典 CNN 架构

| 架构      | 年份  | 关键创新                           |
| --------- | ----- | ---------------------------------- |
| **LeNet** | 1998  | 卷积 + 池化 + 全连接的经典结构     |
| **AlexNet** | 2012 | ReLU、Dropout、GPU 训练          |
| **VGG**   | 2014  | 小卷积核 (3×3) 堆叠               |
| **GoogLeNet** | 2014 | Inception 模块（多尺度）       |
| **ResNet** | 2015 | 残差连接（跳跃连接）             |
| **DenseNet** | 2017 | 密集连接                       |

---

### 5. 残差连接 (Residual Connection)

ResNet 的核心创新：

$$
\mathbf{y} = F(\mathbf{x}) + \mathbf{x}
$$

**为什么有效**：
- 梯度可以通过跳跃连接直接流过，缓解梯度消失
- 网络只需要学习"残差" $F(\mathbf{x}) = \mathbf{y} - \mathbf{x}$
- 恒等映射很容易学习，所以额外层不会损害性能

```python
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        residual = x
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        return F.relu(x + residual)  # 跳跃连接
```

---

## W2D3：生成模型

---

### 1. 生成模型 vs 判别模型

|            | 判别模型           | 生成模型           |
| ---------- | ------------------ | ------------------ |
| **目标**   | $p(y \mid x)$      | $p(x)$ 或 $p(x, y)$ |
| **任务**   | 分类、回归         | 生成新样本、密度估计 |
| **例子**   | MLP, CNN           | VAE, GAN, Diffusion |

---

### 2. 自编码器 (Autoencoder)

**结构**：编码器 + 瓶颈层 + 解码器

$$
\mathbf{x} \xrightarrow{\text{encode}} \mathbf{h} \xrightarrow{\text{decode}} \hat{\mathbf{x}}
$$

**损失函数**：重建误差

$$
\mathcal{L} = \|\mathbf{x} - \hat{\mathbf{x}}\|^2
$$

**线性自编码器 ≈ PCA**：当编码器和解码器都是线性映射时，自编码器学到的子空间与 PCA 相同。

---

### 3. 变分自编码器 (VAE)

VAE 将自编码器的概率化版本——编码器输出的不是一个点，而是一个**分布**。

#### 3.1 核心思想

编码器（识别模型）：$q_\phi(\mathbf{z} \mid \mathbf{x}) = \mathcal{N}(\boldsymbol{\mu}_\phi(\mathbf{x}), \boldsymbol{\sigma}^2_\phi(\mathbf{x}))$

解码器（生成模型）：$p_\theta(\mathbf{x} \mid \mathbf{z})$

#### 3.2 ELBO 损失

$$
\mathcal{L}_{\text{ELBO}} = \underbrace{\mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}[\log p_\theta(\mathbf{x}|\mathbf{z})]}_{\text{重建项}} - \underbrace{D_{\text{KL}}(q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))}_{\text{正则项}}
$$

- **重建项**：解码器能否从 $\mathbf{z}$ 重建 $\mathbf{x}$
- **正则项**：编码器的分布 $q_\phi$ 与先验 $p(\mathbf{z}) = \mathcal{N}(0, I)$ 的接近程度

#### 3.3 重参数化技巧

为了让梯度能通过采样操作：

$$
\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(0, I)
$$

```python
def reparameterize(mu, log_var):
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return mu + eps * std
```

---

### 4. GAN 基础

**生成对抗网络**：两个网络对抗训练

- **生成器** $G$：从噪声 $\mathbf{z}$ 生成假样本
- **判别器** $D$：区分真假样本

$$
\min_G \max_D \; \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}}[\log D(\mathbf{x})] + \mathbb{E}_{\mathbf{z} \sim p(\mathbf{z})}[\log(1 - D(G(\mathbf{z})))]
$$

---

## W2D4：扩散生成模型

---

### 1. 核心思想

**前向过程**：逐步向数据添加高斯噪声，将数据变成纯噪声

$$
\mathbf{x}_t = \mathbf{x}_0 + \sigma_t \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(0, I)
$$

**反向过程**：从噪声出发，逐步去噪，恢复数据

**关键**：学习 Score 函数 $\mathbf{s}(\mathbf{x}, t) = \nabla_{\mathbf{x}} \log p_t(\mathbf{x})$ 来指导反向过程。

---

### 2. Score 函数

$$
\mathbf{s}(\mathbf{x}) = \nabla_{\mathbf{x}} \log p(\mathbf{x})
$$

**几何直觉**：Score 指向概率密度增长最快的方向，其大小反映离高密度区域的远近。

对于高斯混合模型，Score 是各分量 Score 的加权平均：

$$
\mathbf{s}(\mathbf{x}) = \sum_i \pi_i(\mathbf{x}) \cdot \mathbf{s}_i(\mathbf{x})
$$

其中 $\pi_i(\mathbf{x})$ 是第 $i$ 个分量在 $\mathbf{x}$ 处的"责任"。

---

### 3. 反向扩散 SDE

如果前向过程是：

$$
d\mathbf{x} = g(t) \, d\mathbf{w}
$$

则反向过程为：

$$
d\mathbf{x} = -g^2(t) \, \nabla_{\mathbf{x}} \log p_t(\mathbf{x}) \, dt + g(t) \, d\mathbf{w}
$$

**离散化**（采样公式）：

$$
\mathbf{x}_{t-\Delta t} = \mathbf{x}_t + g(t)^2 \, \mathbf{s}(\mathbf{x}_t, t) \, \Delta t + g(t) \sqrt{\Delta t} \, \mathbf{z}_t
$$

---

### 4. 去噪分数匹配 (Denoising Score Matching)

**目标**：通过去噪来学习 Score 函数

$$
\mathcal{L} = \mathbb{E}_{t} \mathbb{E}_{\mathbf{x}_0} \mathbb{E}_{\mathbf{z}} \left[\left\|\sigma_t \, \mathbf{s}_\theta(\mathbf{x}_0 + \sigma_t \mathbf{z}, t) + \mathbf{z}\right\|^2\right]
$$

**步骤**：
1. 从训练数据采样 $\mathbf{x}_0$
2. 采样噪声 $\mathbf{z} \sim \mathcal{N}(0, I)$
3. 采样时间 $t$，创建带噪数据 $\tilde{\mathbf{x}} = \mathbf{x}_0 + \sigma_t \mathbf{z}$
4. 用神经网络预测 Score $\mathbf{s}_\theta(\tilde{\mathbf{x}}, t)$
5. 最小化 $\|\sigma_t \mathbf{s}_\theta + \mathbf{z}\|^2$

---

### 5. 训练与采样流程

**训练**：

```python
for x0 in dataloader:
    t = torch.rand(batch_size)           # 随机时间
    z = torch.randn_like(x0)             # 随机噪声
    sigma = sigma_t(t)                   # 噪声标准差
    x_t = x0 + sigma[:, None] * z        # 加噪
    score = model(x_t, t)                # 预测 Score
    loss = ((score * sigma[:, None] + z) ** 2).mean()
    loss.backward()
    optimizer.step()
```

**采样**：

```python
x_T = sigma_T * torch.randn(...)  # 从纯噪声开始
dt = 1.0 / num_steps
for i in range(num_steps):
    t = 1.0 - i * dt
    score = model(x_T, t)
    x_T = x_T + g(t)**2 * score * dt + g(t) * sqrt(dt) * torch.randn_like(x_T)
```

---

## W2D5：序列与语言

---

### 1. 词嵌入 (Word Embeddings)

将离散的词映射到连续的向量空间。

#### 1.1 Word2Vec

**核心思想**：出现在相似上下文中的词，含义相似。

**Skip-gram 模型**：给定中心词，预测上下文词

$$
P(w_{\text{context}} | w_{\text{center}}) = \frac{\exp(\mathbf{v}_{\text{context}} \cdot \mathbf{v}_{\text{center}})}{\sum_{w} \exp(\mathbf{v}_w \cdot \mathbf{v}_{\text{center}})}
$$

#### 1.2 词嵌入的有趣性质

**类比关系**：

$$
\mathbf{v}_{\text{king}} - \mathbf{v}_{\text{man}} + \mathbf{v}_{\text{woman}} \approx \mathbf{v}_{\text{queen}}
$$

**余弦相似度**：

$$
\text{sim}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|}
$$

---

### 2. 文本分类 pipeline

```
文本 → 分词 (Tokenization) → 词嵌入 (Embedding) → 聚合 (Pooling/Attention) → 分类器
```

```python
class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, pretrained_embeddings):
        super().__init__()
        self.embedding = nn.EmbeddingBag.from_pretrained(pretrained_embeddings)
        self.fc1 = nn.Linear(embed_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)  # 平均词嵌入
        x = F.relu(self.fc1(embedded))
        return self.fc2(x)
```

---

### 3. RNN 基础

处理序列数据的天然选择：

$$
\mathbf{h}_t = \sigma(\mathbf{W}_{hh} \mathbf{h}_{t-1} + \mathbf{W}_{xh} \mathbf{x}_t + \mathbf{b})
$$

**问题**：
- 梯度消失/爆炸：长序列中梯度会指数级衰减或增长
- 串行计算：无法并行化

**LSTM 解决方案**：引入门控机制（遗忘门、输入门、输出门）和细胞状态

**GRU 简化版**：只有两个门（重置门、更新门），参数更少

---

### 4. 上下文无关 vs 上下文相关嵌入

| 类型           | 代表          | 特点                                    |
| -------------- | ------------- | --------------------------------------- |
| **上下文无关** | Word2Vec, FastText | 每个词有固定的向量                   |
| **上下文相关** | ELMo, BERT    | 同一个词在不同上下文中有不同的向量      |

**上下文相关嵌入的优势**：能区分多义词（如 "bank" 在 "river bank" 和 "bank account" 中的不同含义）。

---

## 参考资料

- [Deep Learning Book, Chapter 7 (Regularization)](https://www.deeplearningbook.org/contents/regularization.html)
- [Deep Learning Book, Chapter 9 (CNNs)](https://www.deeplearningbook.org/contents/convnets.html)
- [Deep Learning Book, Chapter 20 (Generative Models)](https://www.deeplearningbook.org/contents/generative.html)
- [What are Diffusion Models? (Lil'Log)](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
- [The Illustrated Word2Vec](https://jalammar.github.io/illustrated-word2vec/)
