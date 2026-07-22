# 深度学习前沿 — 第三周

注意力机制 · Transformer · 自监督学习 · 强化学习

---

## 总览

第三周进入深度学习的**前沿领域**——Transformer 架构统治了 NLP 和 CV，自监督学习消除了对标注数据的依赖，强化学习让智能体学会决策：

| 日       | 主题                                   | 核心技能                                                        |
| -------- | -------------------------------------- | --------------------------------------------------------------- |
| **W3D1** | 注意力与 Transformer                   | Query-Key-Value、多头注意力、位置编码、Transformer 架构         |
| **W3D2** | DL 讨论 2：架构与多模态                | 数据增强、预训练-微调、典型相关分析 (CCA)                       |
| **W3D3** | 无监督与自监督学习                     | 对比学习、聚类、自监督预训练                                    |
| **W3D4** | 基础强化学习                           | MDP、Q 值、值迭代、策略迭代                                    |
| **W3D5** | 高级 RL 与 DL 讨论 3                   | 上下文学习、记忆、多信息源、语言与机器人                        |

**贯穿主题**：如何让模型理解**关系**（注意力）？如何从**无标注数据**中学习？如何让智能体**自主决策**？

---

## W3D1：注意力与 Transformer

---

### 1. 注意力机制的动机

RNN 的问题：
- **串行计算**：$\mathbf{h}_t$ 依赖于 $\mathbf{h}_{t-1}$，无法并行
- **长距离依赖**：信息需要经过很多步才能从序列一端传到另一端
- **遗忘**：早期的信息容易被后续的隐藏状态"稀释"

**注意力的核心思想**：让序列中的每个位置**直接**关注其他所有位置，不需要通过递归传递。

---

### 2. Query-Key-Value 注意力

**类比数据库查询**：
- **Query (查询)**：我在找什么？
- **Key (键)**：数据库中每条记录的索引
- **Value (值)**：数据库中每条记录的内容

**注意力计算**：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
$$

**直觉**：
1. $QK^\top$ 计算 Query 和每个 Key 的相似度
2. 除以 $\sqrt{d_k}$ 防止点积过大导致 softmax 饱和
3. Softmax 将相似度转化为概率分布
4. 用概率分布对 Value 加权求和

---

### 3. 缩放点积注意力的数学细节

#### 3.1 为什么要除以 $\sqrt{d_k}$？

假设 $Q$ 和 $K$ 的每个元素独立同分布，均值为 0，方差为 1。则：

$$
\text{Var}(Q \cdot K) = d_k
$$

当 $d_k$ 很大时，$QK^\top$ 的值会很大，导致 softmax 输出接近 one-hot（梯度接近 0）。除以 $\sqrt{d_k}$ 使方差回到 1。

#### 3.2 Softmax 的数值稳定性

$$
\text{softmax}(\mathbf{z})_i = \frac{\exp(z_i)}{\sum_j \exp(z_j)}
$$

实践中先减去最大值防止溢出：

$$
\text{softmax}(\mathbf{z})_i = \frac{\exp(z_i - \max(\mathbf{z}))}{\sum_j \exp(z_j - \max(\mathbf{z}))}
$$

---

### 4. 多头注意力 (Multi-Head Attention)

**思想**：不同的注意力头可以关注不同类型的关系（如语法关系、语义关系）。

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O
$$

其中 $\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$

**参数**：
- $W_i^Q \in \mathbb{R}^{d_{\text{model}} \times d_k}$
- $W_i^K \in \mathbb{R}^{d_{\text{model}} \times d_k}$
- $W_i^V \in \mathbb{R}^{d_{\text{model}} \times d_v}$
- $W^O \in \mathbb{R}^{hd_v \times d_{\text{model}}}$

通常 $d_k = d_v = d_{\text{model}} / h$。

---

### 5. 自注意力 (Self-Attention)

在自注意力中，$Q$、$K$、$V$ 都来自同一个序列：

$$
Q = XW^Q, \quad K = XW^K, \quad V = XW^V
$$

**含义**：序列中的每个位置关注序列中的所有其他位置，学习它们之间的关系。

**复杂度**：$O(n^2 d)$，其中 $n$ 是序列长度，$d$ 是维度。这是 Transformer 的主要瓶颈。

---

### 6. 位置编码 (Positional Encoding)

自注意力对位置不敏感——打乱词的顺序，输出不变。需要额外注入位置信息。

**正弦位置编码**（Vaswani et al., 2017）：

$$
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$
$$
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$

**性质**：
- 每个位置有唯一编码
- $PE_{pos+k}$ 可以表示为 $PE_{pos}$ 的线性函数（便于学习相对位置）
- 不需要学习，可以推广到更长的序列

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
```

---

### 7. Transformer 编码器块

每个编码器块包含：

```
输入 → 多头自注意力 → 残差连接 + LayerNorm → FFN → 残差连接 + LayerNorm → 输出
```

$$
\mathbf{z}' = \text{LayerNorm}(\mathbf{x} + \text{MultiHead}(\mathbf{x}, \mathbf{x}, \mathbf{x}))
$$
$$
\mathbf{z} = \text{LayerNorm}(\mathbf{z}' + \text{FFN}(\mathbf{z}'))
$$

**FFN (前馈网络)**：

$$
\text{FFN}(\mathbf{x}) = \text{ReLU}(\mathbf{x} W_1 + b_1) W_2 + b_2
$$

通常 $d_{\text{ff}} = 4 \times d_{\text{model}}$。

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # 自注意力
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        # FFN
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x
```

---

### 8. Layer Normalization

$$
\text{LayerNorm}(\mathbf{x}) = \gamma \odot \frac{\mathbf{x} - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$

其中 $\mu$ 和 $\sigma^2$ 在**特征维度**上计算（而非 batch 维度）。

**与 BatchNorm 的区别**：
- BatchNorm：对每个特征跨 batch 归一化（依赖 batch 大小）
- LayerNorm：对每个样本跨特征归一化（不依赖 batch 大小）

Transformer 中使用 LayerNorm，因为序列长度可变，batch 统计不稳定。

---

### 9. Transformer 的完整架构

**编码器-解码器结构**（原始论文）：

```
编码器: N 个编码器块堆叠
解码器: N 个解码器块堆叠（额外有交叉注意力层）
```

**解码器中的三种注意力**：
1. **自注意力**：输出序列内部的注意力（带因果掩码）
2. **交叉注意力**：输出关注输入（$Q$ 来自解码器，$K, V$ 来自编码器）
3. **编码器自注意力**：输入序列内部的注意力

---

### 10. 因果掩码 (Causal Mask)

在自回归生成中，位置 $i$ 只能关注位置 $\leq i$：

$$
\text{mask}[i, j] = \begin{cases} 0 & \text{if } j \leq i \\ -\infty & \text{if } j > i \end{cases}
$$

```python
# 上三角掩码
mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
scores = scores.masked_fill(mask, float('-inf'))
```

---

### 11. 伦理：语言模型中的偏见

预训练语言模型会继承训练数据中的偏见（性别、种族、社会经济地位等）。

**检测方法**：CrowS-Pairs 等数据集通过对比句子对来量化偏见。

**缓解方法**：
- 数据去偏
- 训练目标中加入公平性约束
- 后处理校准

---

## W3D2：DL 讨论 2——架构设计与多模态

---

### 1. 数据增强策略

当数据不足时，通过变换创造更多训练样本：

**图像增强**：
- 几何变换：翻转、旋转、裁剪、缩放
- 颜色变换：亮度、对比度、饱和度
- 噪声注入：高斯噪声、遮挡

**关键原则**：增强应该保持标签不变。对数字 "6" 做水平翻转会变成 "9"——这不合适。

---

### 2. 预训练与微调 (Pre-training & Fine-tuning)

**场景**：目标领域数据很少（如医学图像），但有大量通用数据（如 ImageNet）。

**流程**：
1. 在大数据集上预训练（学习通用特征）
2. 替换最后一层（适配目标任务）
3. 在小数据集上微调（可冻结底层，只训练顶层）

```python
# 加载预训练模型
model = torchvision.models.resnet18(pretrained=True)

# 替换最后一层
model.fc = nn.Linear(512, num_classes)

# 冻结前面的层
for param in model.parameters():
    param.requires_grad = False

# 只训练最后一层
for param in model.fc.parameters():
    param.requires_grad = True
```

---

### 3. 典型相关分析 (CCA)

**场景**：两个不同模态的数据（如脑成像和视频），想找到它们的共享信息。

**核心思想**：找到两个线性投影，使投影后的相关性最大。

$$
\max_{\mathbf{u}, \mathbf{v}} \text{corr}(\mathbf{u}^\top \mathbf{X}_1, \mathbf{v}^\top \mathbf{X}_2)
$$

**Deep CCA**：用神经网络替代线性投影，学习非线性的共享表示。

---

## W3D3：无监督与自监督学习

---

### 1. 为什么需要无监督/自监督学习？

- 标注数据昂贵且稀缺
- 无标注数据几乎无限（互联网上的文本、图片、视频）
- 学到的表示可以迁移到下游任务

---

### 2. 对比学习 (Contrastive Learning)

**核心思想**：让相似样本的表示靠近，不相似样本的表示远离。

**InfoNCE 损失**：

$$
\mathcal{L} = -\log \frac{\exp(\text{sim}(\mathbf{z}_i, \mathbf{z}_j) / \tau)}{\sum_{k=1}^{2N} \mathbb{1}_{[k \neq i]} \exp(\text{sim}(\mathbf{z}_i, \mathbf{z}_k) / \tau)}
$$

其中 $\mathbf{z}_i$ 和 $\mathbf{z}_j$ 是同一图像的两个不同增强版本，$\tau$ 是温度参数。

**SimCLR 框架**：
1. 对每张图像做两次随机增强 → 正样本对
2. 同 batch 中其他图像的增强 → 负样本
3. 最小化 InfoNCE 损失

---

### 3. 自监督预训练任务

| 方法           | 任务                                     | 领域   |
| -------------- | ---------------------------------------- | ------ |
| **BERT**       | 掩码语言模型 (MLM)                       | NLP    |
| **GPT**        | 下一词预测                               | NLP    |
| **SimCLR**     | 对比学习                                 | CV     |
| **MAE**        | 掩码自编码器                             | CV     |
| **DINO**       | 自蒸馏                                   | CV     |

---

## W3D4：基础强化学习

---

### 1. 强化学习的基本框架

**智能体 (Agent)** 在 **环境 (Environment)** 中：
1. 观察当前 **状态** $s$
2. 选择一个 **动作** $a$
3. 收到 **奖励** $r$
4. 转移到新状态 $s'$

**目标**：学习一个 **策略** $\pi(a|s)$，最大化累积奖励。

---

### 2. 马尔可夫决策过程 (MDP)

MDP 由五元组 $(S, A, P, R, \gamma)$ 定义：

| 符号           | 含义                       |
| -------------- | -------------------------- |
| $S$            | 状态集合                   |
| $A$            | 动作集合                   |
| $P(s' \mid s, a)$ | 转移概率               |
| $R(s, a)$      | 奖励函数                   |
| $\gamma$       | 折扣因子（$0 \leq \gamma \leq 1$）|

**马尔可夫性**：下一个状态只取决于当前状态和动作，与历史无关。

---

### 3. 值函数与 Q 函数

**状态值函数**：

$$
V^\pi(s) = \mathbb{E}_\pi\left[\sum_{t=0}^{\infty} \gamma^t r_t \mid s_0 = s\right]
$$

**动作值函数 (Q 函数)**：

$$
Q^\pi(s, a) = \mathbb{E}_\pi\left[\sum_{t=0}^{\infty} \gamma^t r_t \mid s_0 = s, a_0 = a\right]
$$

**最优 Q 函数**：

$$
Q^*(s, a) = \max_\pi Q^\pi(s, a)
$$

---

### 4. 贝尔曼方程

$$
Q^*(s, a) = R(s, a) + \gamma \sum_{s'} P(s' | s, a) \max_{a'} Q^*(s', a')
$$

**直觉**：最优 Q 值 = 即时奖励 + 折扣后的未来最优 Q 值。

---

### 5. 值迭代 (Value Iteration)

**算法**：

```
初始化 Q(s, a) = 0 对所有 (s, a)
重复直到收敛：
    对每个 (s, a):
        Q(s, a) ← R(s, a) + γ Σ_{s'} P(s'|s,a) max_{a'} Q(s', a')
提取策略: π(s) = argmax_a Q(s, a)
```

**收敛性**：在有限 MDP 中，值迭代保证收敛到最优 Q 函数。

---

### 6. 策略迭代 (Policy Iteration)

```
初始化随机策略 π
重复直到 π 不变：
    策略评估: 计算 Q^π(s, a)
    策略改进: π(s) ← argmax_a Q^π(s, a)
```

**特点**：通常比值迭代收敛更快（更少的迭代次数）。

---

### 7. GridWorld 示例

```
* * * * *
*     g *    g = 目标
*       *    * = 墙壁
*       *
* * * * *
```

**状态**：每个非墙壁格子
**动作**：上下左右
**奖励**：到达目标 +1，其他 0
**目标**：找到最短路径

---

## W3D5：高级 RL 与 DL 讨论 3

---

### 1. 上下文学习 (In-Context Learning)

**现象**：大语言模型在 prompt 中给出几个示例后，能"学会"新任务——无需更新参数。

**形式化**：给定示例序列 $(x_1, y_1, \ldots, x_k, y_k, x_{k+1})$，模型预测 $y_{k+1}$。

**理论解释**：
- Transformer 的注意力机制隐式地实现了梯度下降
- 预训练使模型学会了"学习如何学习"（元学习）

---

### 2. 记忆系统

| 记忆类型     | 定义                 | DL 对应                    |
| ------------ | -------------------- | -------------------------- |
| **情景记忆** | 具体事件的记忆       | 外部存储（Neural Turing Machine） |
| **语义记忆** | 一般知识             | 网络权重                   |
| **程序记忆** | 技能和程序           | 策略网络                   |

**Neural Turing Machine (NTM)**：用注意力机制读写外部记忆矩阵，可通过梯度下降训练。

---

### 3. 多信息源与多模态

**多模态大语言模型 (MLLM)**：同时处理文本、图像、音频等多种模态。

**挑战**：
- 如何对齐不同模态的表示？
- 如何融合来自不同模态的信息？
- 不同模态的信息可能互补也可能冲突

---

### 4. 语言与机器人

**思想**：用自然语言描述任务，让 LLM 将语言分解为机器人可以执行的子任务。

**示例**：
- 用户："把盘子放到洗碗机里"
- LLM：分解为 "拿起盘子" → "走到洗碗机" → "打开门" → "放下盘子" → "关上门"
- 每个子任务映射到机器人的动作空间

---

## 总结：深度学习的全景

```
第一周：基础
  张量操作 → 梯度下降 → MLP → 优化

第二周：架构与生成
  正则化 → CNN → VAE/GAN → 扩散模型 → RNN/词嵌入

第三周：前沿
  Transformer → 自监督学习 → 强化学习 → 上下文学习
```

**关键洞察**：
1. **表示学习**是深度学习的核心——好的表示让下游任务变得简单
2. **归纳偏置**很重要——CNN 的局部性、RNN 的序列性、Transformer 的全局性
3. **数据效率**是关键挑战——预训练、数据增强、自监督学习都是解决方案
4. **优化不等于学习**——过拟合、泛化、分布偏移都需要专门处理

---

## 参考资料

- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
- [BERT: Pre-training of Deep Bidirectional Transformers (Devlin et al., 2019)](https://arxiv.org/abs/1810.04805)
- [A Simple Framework for Contrastive Learning (Chen et al., 2020)](https://arxiv.org/abs/2002.05709)
- [Reinforcement Learning: An Introduction (Sutton & Barto)](http://incompleteideas.net/book/the-book-2nd.html)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [Neural Turing Machines (Graves et al., 2014)](https://arxiv.org/abs/1410.5401)
