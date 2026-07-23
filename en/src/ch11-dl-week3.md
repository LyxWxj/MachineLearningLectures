# Deep Learning Frontiers — Week 3

Attention Mechanism · Transformer · Self-Supervised Learning · Reinforcement Learning

---

## Overview

Week 3 enters the **frontier areas** of deep learning — the Transformer architecture dominates NLP and CV, self-supervised learning eliminates the dependency on labeled data, and reinforcement learning enables agents to learn decision-making:

| Day      | Topic                                   | Core Skills                                                     |
| -------- | --------------------------------------- | --------------------------------------------------------------- |
| **W3D1** | Attention and Transformer               | Query-Key-Value, Multi-Head Attention, Positional Encoding, Transformer Architecture |
| **W3D2** | DL Discussion 2: Architecture and Multimodal | Data Augmentation, Pre-training & Fine-tuning, Canonical Correlation Analysis (CCA) |
| **W3D3** | Unsupervised and Self-Supervised Learning | Contrastive Learning, Clustering, Self-Supervised Pre-training  |
| **W3D4** | Fundamentals of Reinforcement Learning   | MDP, Q-values, Value Iteration, Policy Iteration                |
| **W3D5** | Advanced RL and DL Discussion 3          | In-Context Learning, Memory, Multiple Information Sources, Language and Robotics |

**Recurring Themes**: How to make models understand **relationships** (attention)? How to learn from **unlabeled data**? How to enable agents to make **autonomous decisions**?

---

## W3D1: Attention and Transformer

---

### 1. Motivation for Attention Mechanisms

Problems with RNNs:
- **Sequential computation**: $\mathbf{h}_t$ depends on $\mathbf{h}_{t-1}$, cannot be parallelized
- **Long-range dependencies**: information needs to pass through many steps to travel from one end of the sequence to the other
- **Forgetting**: early information can easily be "diluted" by subsequent hidden states

**Core idea of attention**: let each position in a sequence **directly** attend to all other positions, without needing to pass through recurrence.

---

### 2. Query-Key-Value Attention

**Analogy with database queries**:
- **Query**: What am I looking for?
- **Key**: The index of each record in the database
- **Value**: The content of each record in the database

**Attention computation**:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
$$

**Intuition**:
1. $QK^\top$ computes the similarity between the Query and each Key
2. Dividing by $\sqrt{d_k}$ prevents the dot product from becoming too large, which would cause softmax saturation
3. Softmax converts similarities into a probability distribution
4. The probability distribution is used to compute a weighted sum of Values

---

### 3. Mathematical Details of Scaled Dot-Product Attention

#### 3.1 Why divide by $\sqrt{d_k}$?

Assume each element of $Q$ and $K$ is independently and identically distributed with mean 0 and variance 1. Then:

$$
\text{Var}(Q \cdot K) = d_k
$$

When $d_k$ is large, the values of $QK^\top$ will be large, causing the softmax output to approach one-hot (with gradients near 0). Dividing by $\sqrt{d_k}$ brings the variance back to 1.

#### 3.2 Numerical Stability of Softmax

$$
\text{softmax}(\mathbf{z})_i = \frac{\exp(z_i)}{\sum_j \exp(z_j)}
$$

In practice, subtract the maximum value first to prevent overflow:

$$
\text{softmax}(\mathbf{z})_i = \frac{\exp(z_i - \max(\mathbf{z}))}{\sum_j \exp(z_j - \max(\mathbf{z}))}
$$

---

### 4. Multi-Head Attention

**Idea**: Different attention heads can focus on different types of relationships (e.g., syntactic relationships, semantic relationships).

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O
$$

where $\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$

**Parameters**:
- $W_i^Q \in \mathbb{R}^{d_{\text{model}} \times d_k}$
- $W_i^K \in \mathbb{R}^{d_{\text{model}} \times d_k}$
- $W_i^V \in \mathbb{R}^{d_{\text{model}} \times d_v}$
- $W^O \in \mathbb{R}^{hd_v \times d_{\text{model}}}$

Typically $d_k = d_v = d_{\text{model}} / h$.

---

### 5. Self-Attention and Cross-Attention

#### 5.1 Self-Attention

In self-attention, $Q$, $K$, and $V$ **all come from the same sequence**:

$$
Q = XW^Q, \quad K = XW^K, \quad V = XW^V
$$

**Meaning**: Each position in the sequence attends to all other positions in the sequence, learning the relationships between them.

```
Input X: (B, T, d_model)
    ↓
Q = X @ W_Q,  K = X @ W_K,  V = X @ W_V    ← all from X
    ↓
Attention(Q, K, V) = softmax(QK^T / √d_k) @ V
    ↓
Output: (B, T, d_model)   ← each position "sees" the entire sequence
```

**Applications**:
- **BERT**: Bidirectional self-attention, each token attends to all tokens left and right
- **GPT**: Causal self-attention (with mask), each token only attends to tokens on its left
- **ViT**: Self-attention between image patches

**Complexity**: $O(n^2 d)$, where $n$ is the sequence length and $d$ is the dimension. This is the main bottleneck of the Transformer.

#### 5.2 Cross-Attention

In cross-attention, $Q$ comes from one sequence, while $K$ and $V$ come from **another sequence**:

$$
Q = X_{\text{query}} W^Q, \quad K = X_{\text{key}} W^K, \quad V = X_{\text{key}} W^V
$$

**Meaning**: Each position in one sequence attends to all positions in another sequence, enabling cross-sequence information fusion.

```
Query seq X_q: (B, T_q, d_model)    Key-Value seq X_kv: (B, T_kv, d_model)
    ↓                                         ↓
Q = X_q @ W_Q                          K = X_kv @ W_K,  V = X_kv @ W_V
    ↓                                         ↓
    └───────────── Attention(Q, K, V) ──────┘
                        ↓
              Output: (B, T_q, d_model)   ← output shape follows query sequence
```

**Applications**:
- **Encoder-Decoder Transformer**: Decoder attends to encoder output (in translation, target language attends to source language)
- **Stable Diffusion**: Text conditioning injected into image generation via cross-attention
- **Multimodal models**: Image features attend to text descriptions

#### 5.3 Self-Attention vs Cross-Attention

| | Self-Attention | Cross-Attention |
|---|---|---|
| **Q source** | Sequence X | Sequence X (query side) |
| **K, V source** | Sequence X (same) | Sequence Y (another) |
| **Output shape** | (B, T_x, d) | (B, T_x, d) |
| **Core purpose** | Model intra-sequence relations | Fuse information across sequences |
| **Typical use** | BERT, GPT, ViT | Decoder-Encoder, conditional generation |

```python
# Self-attention: Q, K, V all from x
self_attn = nn.MultiheadAttention(d_model, num_heads)
out, _ = self_attn(x, x, x)  # query=x, key=x, value=x

# Cross-attention: Q from decoder, K/V from encoder
cross_attn = nn.MultiheadAttention(d_model, num_heads)
out, _ = cross_attn(decoder_out, encoder_out, encoder_out)  # Q=decoder, KV=encoder
```

---

### 6. Positional Encoding

Self-attention is position-insensitive — shuffling the order of words leaves the output unchanged. Positional information must be injected externally.

**Sinusoidal Positional Encoding** (Vaswani et al., 2017):

$$
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$
$$
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$

**Properties**:
- Each position has a unique encoding
- $PE_{pos+k}$ can be expressed as a linear function of $PE_{pos}$ (facilitating learning of relative positions)
- Does not require learning and can generalize to longer sequences

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

### 7. Transformer Encoder Block

Each encoder block contains:

```
Input → Multi-Head Self-Attention → Residual Connection + LayerNorm → FFN → Residual Connection + LayerNorm → Output
```

$$
\mathbf{z}' = \text{LayerNorm}(\mathbf{x} + \text{MultiHead}(\mathbf{x}, \mathbf{x}, \mathbf{x}))
$$
$$
\mathbf{z} = \text{LayerNorm}(\mathbf{z}' + \text{FFN}(\mathbf{z}'))
$$

**FFN (Feed-Forward Network)**:

$$
\text{FFN}(\mathbf{x}) = \text{ReLU}(\mathbf{x} W_1 + b_1) W_2 + b_2
$$

Typically $d_{\text{ff}} = 4 \times d_{\text{model}}$.

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
        # Self-attention
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

where $\mu$ and $\sigma^2$ are computed over the **feature dimension** (not the batch dimension).

**Difference from BatchNorm**:
- BatchNorm: normalizes each feature across the batch (depends on batch size)
- LayerNorm: normalizes each sample across features (does not depend on batch size)

Transformer uses LayerNorm because sequence length is variable and batch statistics are unstable.

---

### 9. The Complete Transformer Architecture

**Encoder-Decoder Structure** (original paper):

```
Encoder: N encoder blocks stacked
Decoder: N decoder blocks stacked (with additional cross-attention layers)
```

**Three types of attention in the decoder**:
1. **Self-attention**: attention within the output sequence (with causal mask)
2. **Cross-attention**: output attends to input ($Q$ comes from the decoder, $K, V$ come from the encoder)
3. **Encoder self-attention**: attention within the input sequence

---

### 10. Causal Mask

In autoregressive generation, position $i$ can only attend to positions $\leq i$:

$$
\text{mask}[i, j] = \begin{cases} 0 & \text{if } j \leq i \\ -\infty & \text{if } j > i \end{cases}
$$

```python
# Upper triangular mask
mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
scores = scores.masked_fill(mask, float('-inf'))
```

---

### 11. Ethics: Bias in Language Models

Pre-trained language models inherit biases from the training data (gender, race, socioeconomic status, etc.).

**Detection methods**: Datasets like CrowS-Pairs quantify biases by contrasting sentence pairs.

**Mitigation methods**:
- Data debiasing
- Fairness constraints in the training objective
- Post-processing calibration

---

### 12. Thinking Exercise: The Attention Sink Phenomenon

> **Question**: Research the Attention Sink phenomenon online, and consider:
> 1. What is Attention Sink? Why do LLMs concentrate so much attention on the first token?
> 2. How does this phenomenon affect model performance?
> 3. What solutions exist?

**Hint**: Read the following paper to understand the Attention Sink phenomenon:
- Xiao et al. (2023). "Efficient Streaming Language Models with Attention Sinks". [arxiv:2309.17453](https://arxiv.org/abs/2309.17453)

**Discussion points**:

Attention Sink refers to the phenomenon where large language models **assign a large portion of attention weights to the first token in the sequence** (usually `[BOS]` or `<s>`), regardless of the input content. This token acts as an "attention sink" — it has no special semantic meaning, but the model habitually dumps "excess" attention into it.

**Why does it happen?**
- Softmax normalization constraint: attention weights must sum to 1. Even when a position doesn't need to attend to anything, it must distribute its attention somewhere
- The first token is the only token **visible to all positions** (in causal attention), making it the "default receiver"
- During training, the model discovers that dumping "useless" attention to the first token is an effective "cheating" strategy

**Impact**:
- In long-sequence inference, discarding the first token's KV cache causes dramatic performance degradation — even though the first token itself carries no important information
- This limits KV cache compression and streaming inference efficiency

**Solutions** (to research):
- Preserve the initial token's KV cache (StreamingLLM)
- Use special placeholder tokens instead of `[BOS]`
- Introduce sink-aware pruning strategies in attention computation

---

### 1. Data Augmentation Strategies

When data is insufficient, create more training samples through transformations:

**Image augmentation**:
- Geometric transformations: flipping, rotation, cropping, scaling
- Color transformations: brightness, contrast, saturation
- Noise injection: Gaussian noise, occlusion

**Key principle**: augmentations should preserve labels. Horizontally flipping the digit "6" turns it into "9" — this is inappropriate.

---

### 2. Pre-training and Fine-tuning

**Scenario**: Very little data in the target domain (e.g., medical images), but abundant general data (e.g., ImageNet).

**Process**:
1. Pre-train on a large dataset (learn general features)
2. Replace the last layer (adapt to the target task)
3. Fine-tune on the small dataset (can freeze lower layers, only train the top layer)

```python
# Load pre-trained model
model = torchvision.models.resnet18(pretrained=True)

# Replace the last layer
model.fc = nn.Linear(512, num_classes)

# Freeze earlier layers
for param in model.parameters():
    param.requires_grad = False

# Only train the last layer
for param in model.fc.parameters():
    param.requires_grad = True
```

---

### 3. Canonical Correlation Analysis (CCA)

**Scenario**: Data from two different modalities (e.g., brain imaging and video), wanting to find their shared information.

**Core idea**: Find two linear projections such that the correlation after projection is maximized.

$$
\max_{\mathbf{u}, \mathbf{v}} \text{corr}(\mathbf{u}^\top \mathbf{X}_1, \mathbf{v}^\top \mathbf{X}_2)
$$

**Deep CCA**: Use neural networks instead of linear projections to learn non-linear shared representations.

---

## W3D3: Unsupervised and Self-Supervised Learning

---

### 1. Why Unsupervised/Self-Supervised Learning?

- Labeled data is expensive and scarce
- Unlabeled data is nearly unlimited (text, images, videos on the internet)
- Learned representations can transfer to downstream tasks

---

### 2. Contrastive Learning

**Core idea**: Make representations of similar samples close together, and representations of dissimilar samples far apart.

**InfoNCE Loss**:

$$
\mathcal{L} = -\log \frac{\exp(\text{sim}(\mathbf{z}_i, \mathbf{z}_j) / \tau)}{\sum_{k=1}^{2N} \mathbb{1}_{[k \neq i]} \exp(\text{sim}(\mathbf{z}_i, \mathbf{z}_k) / \tau)}
$$

where $\mathbf{z}_i$ and $\mathbf{z}_j$ are two different augmented versions of the same image, and $\tau$ is the temperature parameter.

**SimCLR Framework**:
1. Apply two random augmentations to each image → positive pair
2. Augmentations of other images in the same batch → negative samples
3. Minimize the InfoNCE loss

---

### 3. Self-Supervised Pre-training Tasks

| Method         | Task                                     | Domain |
| -------------- | ---------------------------------------- | ------ |
| **BERT**       | Masked Language Model (MLM)              | NLP    |
| **GPT**        | Next Token Prediction                    | NLP    |
| **SimCLR**     | Contrastive Learning                     | CV     |
| **MAE**        | Masked Autoencoder                       | CV     |
| **DINO**       | Self-Distillation                        | CV     |

---

## W3D4: Fundamentals of Reinforcement Learning

---

### 1. The Basic Framework of Reinforcement Learning

An **Agent** interacts with an **Environment**:
1. Observes the current **state** $s$
2. Selects an **action** $a$
3. Receives a **reward** $r$
4. Transitions to a new state $s'$

**Goal**: Learn a **policy** $\pi(a|s)$ that maximizes cumulative reward.

---

### 2. Markov Decision Process (MDP)

An MDP is defined by a 5-tuple $(S, A, P, R, \gamma)$:

| Symbol         | Meaning                    |
| -------------- | -------------------------- |
| $S$            | Set of states              |
| $A$            | Set of actions             |
| $P(s' \mid s, a)$ | Transition probability |
| $R(s, a)$      | Reward function            |
| $\gamma$       | Discount factor ($0 \leq \gamma \leq 1$) |

**Markov property**: The next state depends only on the current state and action, not on history.

---

### 3. Value Function and Q-Function

**State Value Function**:

$$
V^\pi(s) = \mathbb{E}_\pi\left[\sum_{t=0}^{\infty} \gamma^t r_t \mid s_0 = s\right]
$$

**Action Value Function (Q-Function)**:

$$
Q^\pi(s, a) = \mathbb{E}_\pi\left[\sum_{t=0}^{\infty} \gamma^t r_t \mid s_0 = s, a_0 = a\right]
$$

**Optimal Q-Function**:

$$
Q^*(s, a) = \max_\pi Q^\pi(s, a)
$$

---

### 4. Bellman Equation

$$
Q^*(s, a) = R(s, a) + \gamma \sum_{s'} P(s' | s, a) \max_{a'} Q^*(s', a')
$$

**Intuition**: Optimal Q-value = immediate reward + discounted future optimal Q-value.

---

### 5. Value Iteration

**Algorithm**:

```
Initialize Q(s, a) = 0 for all (s, a)
Repeat until convergence:
    For each (s, a):
        Q(s, a) ← R(s, a) + γ Σ_{s'} P(s'|s,a) max_{a'} Q(s', a')
Extract policy: π(s) = argmax_a Q(s, a)
```

**Convergence**: In a finite MDP, value iteration is guaranteed to converge to the optimal Q-function.

---

### 6. Policy Iteration

```
Initialize a random policy π
Repeat until π does not change:
    Policy evaluation: Compute Q^π(s, a)
    Policy improvement: π(s) ← argmax_a Q^π(s, a)
```

**Characteristics**: Usually converges faster than value iteration (fewer iterations).

---

### 7. GridWorld Example

```
* * * * *
*     g *    g = goal
*       *    * = wall
*       *
* * * * *
```

**States**: Each non-wall cell
**Actions**: Up, Down, Left, Right
**Reward**: +1 for reaching the goal, 0 otherwise
**Goal**: Find the shortest path

---

## W3D5: Advanced RL and DL Discussion 3

---

### 1. In-Context Learning

**Phenomenon**: Large language models can "learn" new tasks after being given a few examples in the prompt — without updating parameters.

**Formalization**: Given an example sequence $(x_1, y_1, \ldots, x_k, y_k, x_{k+1})$, the model predicts $y_{k+1}$.

**Theoretical explanation**:
- The Transformer's attention mechanism implicitly implements gradient descent
- Pre-training enables the model to learn "how to learn" (meta-learning)

---

### 2. Memory Systems

| Memory Type    | Definition                  | DL Correspondence               |
| -------------- | --------------------------- | ------------------------------- |
| **Episodic**   | Memory of specific events   | External memory (Neural Turing Machine) |
| **Semantic**   | General knowledge           | Network weights                 |
| **Procedural** | Skills and procedures       | Policy networks                 |

**Neural Turing Machine (NTM)**: Uses attention mechanisms to read and write external memory matrices, trainable via gradient descent.

---

### 3. Multiple Information Sources and Multimodal

**Multimodal Large Language Models (MLLMs)**: Process text, images, audio, and other modalities simultaneously.

**Challenges**:
- How to align representations across different modalities?
- How to fuse information from different modalities?
- Information from different modalities may be complementary or conflicting

---

### 4. Language and Robotics

**Idea**: Use natural language to describe tasks, and let the LLM decompose language into sub-tasks that the robot can execute.

**Example**:
- User: "Put the plate in the dishwasher"
- LLM: Decomposes into "Pick up the plate" → "Walk to the dishwasher" → "Open the door" → "Put down the plate" → "Close the door"
- Each sub-task is mapped to the robot's action space

---

## Summary: The Full Landscape of Deep Learning

```
Week 1: Foundations
  Tensor operations → Gradient descent → MLP → Optimization

Week 2: Architectures and Generative Models
  Regularization → CNN → VAE/GAN → Diffusion Models → RNN/Word Embeddings

Week 3: Frontiers
  Transformer → Self-Supervised Learning → Reinforcement Learning → In-Context Learning
```

**Key Insights**:
1. **Representation learning** is the core of deep learning — good representations make downstream tasks simple
2. **Inductive bias** matters — the locality of CNNs, the sequentiality of RNNs, the globality of Transformers
3. **Data efficiency** is a key challenge — pre-training, data augmentation, and self-supervised learning are all solutions
4. **Optimization does not equal learning** — overfitting, generalization, and distribution shift all require specialized handling

---

## References

- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
- [BERT: Pre-training of Deep Bidirectional Transformers (Devlin et al., 2019)](https://arxiv.org/abs/1810.04805)
- [A Simple Framework for Contrastive Learning (Chen et al., 2020)](https://arxiv.org/abs/2002.05709)
- [Reinforcement Learning: An Introduction (Sutton & Barto)](http://incompleteideas.net/book/the-book-2nd.html)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [Neural Turing Machines (Graves et al., 2014)](https://arxiv.org/abs/1410.5401)
