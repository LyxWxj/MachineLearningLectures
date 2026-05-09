---
theme: seriph
themeConfig:
  primary: '#5d8392'
title: "Lecture 1: Machine Learning Fundamentals"
transition: slide-left
math: true
---

# Lecture 1: Machine Learning Fundamentals

<div class="text-gray-400 mt-2">
What is ML? · Learning Paradigms · Workflow & Terminology
</div>

<Toc minDepth="2" maxDepth="2" />

---
layout: center
---

## Review: Lecture 0

<v-click>

| Subject | Role in ML |
|------|---------|
| **Calculus** | Optimization — gradient descent updates parameters |
| **Linear Algebra** | Representation — data as vectors, computation as matrices |
| **Probability** | Modeling — distributions, Bayes' theorem, uncertainty |

</v-click>

<v-click>

<div class="mt-4 p-4 bg-blue-900/20 rounded-lg text-center">

**ML = Representation + Modeling + Optimization**

</div>

</v-click>

<v-click>

Now we know the **tools**. Let's ask: **what exactly is machine learning?**

</v-click>

---

## Part 1: What is ML?

### Traditional Programming vs Machine Learning

<div class="grid grid-cols-2 gap-8">
<div>

**Traditional Programming**

$$\text{Rules} + \text{Data} \xrightarrow{\text{program}} \text{Output}$$

<v-click>

Human writes explicit rules:

```python
if temperature > 30:
    return "hot"
elif temperature > 20:
    return "warm"
else:
    return "cold"
```

</v-click>

</div>
<div>

<v-click>

**Machine Learning**

$$\text{Data} + \text{Output} \xrightarrow{\text{learning}} \text{Rules (Model)}$$

</v-click>

<v-click>

Machine discovers rules from examples:

| Temperature | Label |
|---|---|
| 35°C | hot |
| 22°C | warm |
| 10°C | cold |

$\rightarrow$ learns the mapping automatically

</v-click>

</div>
</div>

---

### Formal Definition

Machine learning is about learning a **function** $f$ from data:

$$f: \mathbf{x} \rightarrow y$$

<v-click>

where:

- $\mathbf{x}$: input (features / observations)
- $y$: output (prediction / label / structure)
- $f$: the **model** — a hypothesis function from a hypothesis space $\mathcal{H}$

</v-click>

<v-click>

The learning process:

1. Choose a hypothesis space $\mathcal{H}$ (e.g., linear functions, neural networks)
2. Define a **loss function** to measure how wrong $f$ is
3. Use an **optimization algorithm** to find the best $f \in \mathcal{H}$

</v-click>

<v-click>

<div class="mt-4 p-4 bg-blue-900/20 rounded-lg">

This directly connects Lecture 0: $f$ is **represented** with linear algebra, **modeled** with probability, and **optimized** with calculus.

</div>

</v-click>

---

## Part 2: Three Learning Paradigms

### The Unifying Perspective

All machine learning paradigms share one goal: **learn the underlying data distribution**.

<v-click>

| Paradigm | What to learn | Data available |
|------|------|------|
| **Supervised** | $P(y \mid \mathbf{x})$ — label given input | $(\mathbf{x}, y)$ pairs |
| **Unsupervised** | $P(\mathbf{x})$ — structure of input | $\mathbf{x}$ only, no labels |
| **Reinforcement** | $\pi(a \mid s)$ — action given state | reward signal from environment |

</v-click>

<v-click>

In all three cases, we are estimating or leveraging a **probability distribution** — the central theme from Lecture 0.

</v-click>

---
layout: center
---

### Supervised Learning

---

### Supervised Learning

Given labeled data $\{(\mathbf{x}_i, y_i)\}_{i=1}^{N}$, learn a mapping $f: \mathbf{x} \to y$.

<div class="grid grid-cols-2 gap-8">
<div>

**Regression**: $y \in \mathbb{R}$ (continuous)

<v-click>

Predict house price from features:

| Size (m²) | Rooms | Price ($\times 10^4$) |
|---|---|---|
| 80 | 2 | 150 |
| 120 | 3 | 230 |
| ? | ? | **?** |

</v-click>

<v-click>

$$\hat{y} = f(\mathbf{x}) = \mathbf{w}^T\mathbf{x} + b$$

</v-click>

</div>
<div>

**Classification**: $y \in \{1, 2, \ldots, C\}$ (discrete)

<v-click>

Classify email as spam or not:

| Features | Label |
|---|---|
| contains "free", "winner" | spam |
| contains "meeting", "report" | not spam |
| ? | **?** |

</v-click>

<v-click>

$$P(y = c \mid \mathbf{x}) = \text{softmax}(\mathbf{w}_c^T\mathbf{x} + b_c)$$

</v-click>

</div>
</div>

---

### Supervised Learning: Algorithm Examples

| Algorithm | Type | Idea |
|------|------|------|
| **Linear Regression** | Regression | Fit a line/plane to data: $\hat{y} = \mathbf{w}^T\mathbf{x} + b$ |
| **Logistic Regression** | Classification | Sigmoid maps linear output to probability: $P(y{=}1) = \sigma(\mathbf{w}^T\mathbf{x})$ |
| **Decision Tree** | Both | Split feature space into regions with if-else rules |
| **Neural Network** | Both | Stack layers of linear + nonlinear transformations |

<v-click>

<div class="mt-4 p-4 bg-blue-900/20 rounded-lg">

**Common thread**: every supervised algorithm defines a function class $\mathcal{H}$, a loss function, and an optimization method.

This is exactly what Lecture 2 will formalize: **Metric, Loss, Optimization**.

</div>

</v-click>

---
layout: center
---

### Unsupervised Learning

---

### Unsupervised Learning

Given unlabeled data $\{\mathbf{x}_i\}_{i=1}^{N}$, discover hidden structure.

<v-click>

No labels $y$ — the model must find patterns on its own.

</v-click>

<div class="grid grid-cols-2 gap-8 mt-4">
<div>

**Clustering**: group similar samples

<v-click>

Customer segmentation:

- Group A: young, low income, frequent buyer
- Group B: middle-age, high income, occasional buyer
- Group C: ...

</v-click>

</div>
<div>

**Dimensionality Reduction**: compress features

<v-click>

1000-dimensional gene expression $\to$ 2D visualization

Preserve the most informative structure.

</v-click>

</div>
</div>

---

### Unsupervised Learning: Algorithm Examples

| Algorithm | Task | Idea |
|------|------|------|
| **K-Means** | Clustering | Iterate: assign points to nearest centroid, update centroids |
| **PCA** | Dim. Reduction | Find directions of maximum variance via eigendecomposition |
| **Autoencoder** | Dim. Reduction | Neural network: compress $\mathbf{x} \to \mathbf{z} \to \hat{\mathbf{x}}$ |
| **GMM** | Clustering | Model data as mixture of Gaussians, fit with EM algorithm |

<v-click>

<div class="mt-4 p-4 bg-blue-900/20 rounded-lg">

**Key insight**: unsupervised learning estimates $P(\mathbf{x})$ or its structure. K-Means implicitly assumes $P(\mathbf{x})$ is a mixture of spherical clusters; GMM explicitly models it as $\sum_k \pi_k \mathcal{N}(\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$.

</div>

</v-click>

---
layout: center
---

### Reinforcement Learning

---

### Reinforcement Learning

An **agent** interacts with an **environment**, takes **actions**, receives **rewards**.

<div class="grid grid-cols-2 gap-8">
<div>

**Agent-Environment Loop**:

<v-click>

1. Agent observes state $s_t$
2. Agent chooses action $a_t$ via policy $\pi(a \mid s)$
3. Environment returns reward $r_t$ and new state $s_{t+1}$
4. Repeat — goal: maximize cumulative reward

</v-click>

</div>
<div>

<v-click>

**Key concepts**:

| Concept | Meaning |
|------|------|
| **State** $s$ | Current situation |
| **Action** $a$ | What the agent does |
| **Reward** $r$ | Feedback signal |
| **Policy** $\pi(a \mid s)$ | Strategy: state $\to$ action |
| **Value** $V(s)$ | Expected future reward from state $s$ |

</v-click>

</div>
</div>

<v-click>

The agent learns a **policy** $\pi$ that maximizes $E\left[\sum_{t=0}^{\infty} \gamma^t r_t\right]$, where $\gamma \in (0,1)$ discounts future rewards.

</v-click>

---

### Reinforcement Learning: Algorithm Examples

| Algorithm | Type | Idea |
|------|------|------|
| **Q-Learning** | Value-based | Learn $Q(s, a)$ = expected reward; act greedily: $a = \arg\max_a Q(s, a)$ |
| **DQN** | Value-based | Q-Learning with neural network as function approximator |
| **Policy Gradient** | Policy-based | Directly optimize $\pi(a \mid s)$ by gradient ascent on expected reward |
| **PPO** | Policy-based | Stable policy gradient with clipped objective |

<v-click>

<div class="mt-4 p-4 bg-blue-900/20 rounded-lg">

**Famous applications**:

- AlphaGo (Go) — RL + MCTS defeated world champion
- Robotics — robot learns to walk, grasp objects
- ChatGPT fine-tuning — RLHF aligns language model with human preferences

</div>

</v-click>

---
layout: center
---

## Part 3: ML Workflow

---

### ML Workflow: Train, Validate, Test

A standard machine learning project follows this pipeline:

<v-click>

```mermaid
graph LR
    A[Data Collection] --> B[Data Split]
    B --> C[Train Model]
    C --> D[Evaluate]
    D --> E{Good enough?}
    E -->|No| C
    E -->|Yes| F[Deploy]
```

</v-click>

<v-click>

**Data split** — three sets with distinct roles:

| Set | Purpose | Analogy |
|------|------|------|
| **Training set** (~70%) | Learn model parameters $\mathbf{w}$ | Study material |
| **Validation set** (~15%) | Tune hyperparameters, select model | Practice exam |
| **Test set** (~15%) | Final unbiased evaluation | Final exam |

</v-click>

<v-click>

**Critical rule**: the test set must be **untouched** until the final evaluation. Peeking at it leads to overfitting to the test set.

</v-click>

---

### Training vs Inference

<div class="grid grid-cols-2 gap-8">
<div>

**Training** (learning phase)

<v-click>

- Input: training data + loss function + optimizer
- Process: iteratively update $\mathbf{w}$ to minimize loss
- Output: learned parameters $\mathbf{w}^*$

$$\mathbf{w}^* = \arg\min_{\mathbf{w}} \frac{1}{N}\sum_{i=1}^{N} L(f(\mathbf{x}_i; \mathbf{w}), y_i)$$

</v-click>

</div>
<div>

**Inference** (prediction phase)

<v-click>

- Input: new unseen data $\mathbf{x}_{\text{new}}$
- Process: forward pass through trained model
- Output: prediction $\hat{y} = f(\mathbf{x}_{\text{new}}; \mathbf{w}^*)$

No gradient computation, no weight updates.

</v-click>

</div>
</div>

<v-click>

<div class="mt-4 p-4 bg-blue-900/20 rounded-lg">

This equation — $\arg\min_{\mathbf{w}} \sum L(\cdot)$ — is exactly what Lecture 2 will dissect: **Loss** defines what to minimize, **Optimization** defines how to minimize it.

</div>

</v-click>

---
layout: center
---

## Part 4: Key Terminology

---

### Features, Labels, and Models

| Term | Symbol | Meaning |
|------|------|------|
| **Feature** | $\mathbf{x} = [x_1, \ldots, x_d]^T$ | Input variables describing a sample |
| **Label** | $y$ | Target output (supervised learning only) |
| **Sample** | $(\mathbf{x}_i, y_i)$ | One data point |
| **Dataset** | $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^N$ | Collection of $N$ samples |
| **Model** | $f(\mathbf{x}; \mathbf{w})$ | Function parameterized by $\mathbf{w}$ |
| **Parameters** | $\mathbf{w}$ | Learned from data (weights, biases) |
| **Hyperparameters** | — | Set before training (learning rate, layers, etc.) |

<v-click>

**Example**: image classification

- $\mathbf{x}$: pixel values of an image (e.g., $224 \times 224 \times 3$)
- $y$: class label (e.g., "cat" = 0, "dog" = 1)
- $\mathbf{w}$: millions of weights in a CNN
- hyperparameters: learning rate $\eta$, number of layers, batch size

</v-click>

---

### Overfitting and Underfitting

<div class="grid grid-cols-3 gap-6">
<div class="p-4 bg-gray-800/50 rounded-lg">

**Underfitting**

<v-click>

Model is too **simple** to capture the pattern.

- High bias, low variance
- Poor on training AND test data

Example: fitting a line to curved data.

</v-click>

</div>
<div class="p-4 bg-gray-800/50 rounded-lg">

**Good Fit**

<v-click>

Model captures the true pattern.

- Balanced bias and variance
- Good on training AND test data

This is the goal.

</v-click>

</div>
<div class="p-4 bg-gray-800/50 rounded-lg">

**Overfitting**

<v-click>

Model is too **complex** — memorizes noise.

- Low bias, high variance
- Great on training, poor on test data

Example: high-degree polynomial fitting every point.

</v-click>

</div>
</div>

<v-click>

<div class="mt-4 text-center">

**Model complexity** $\longrightarrow$

| Underfitting | Sweet spot | Overfitting |
|---|---|---|
| $\leftarrow$ too simple | **just right** | too complex $\rightarrow$ |

</div>

</v-click>

---

### Generalization

**Generalization**: the ability to perform well on **unseen** data.

<v-click>

This is the ultimate goal of machine learning — not to memorize training data, but to learn patterns that transfer to new inputs.

</v-click>

<v-click>

**Why does overfitting happen?**

- Model has too many parameters relative to training samples
- Training data has noise that the model memorizes
- Training and test distributions differ

</v-click>

<v-click>

**How to improve generalization?**

| Strategy | How it helps |
|------|------|
| More training data | Harder to memorize, forces learning real patterns |
| Regularization (L1, L2) | Penalize complex models, prefer simpler $f$ |
| Cross-validation | Better estimate of test performance |
| Early stopping | Stop training before overfitting kicks in |

</v-click>

---

### Bias-Variance Tradeoff (from Lecture 0)

Recall:

$$E[(y - \hat{f}(\mathbf{x}))^2] = \text{Bias}^2 + \text{Variance} + \text{Irreducible Noise}$$

<v-click>

| | Bias | Variance |
|------|------|------|
| **Definition** | Error from wrong assumptions | Error from sensitivity to training data |
| **Underfitting** | High | Low |
| **Overfitting** | Low | High |
| **Goal** | Minimize | Minimize |

</v-click>

<v-click>

The tradeoff: reducing bias (more complex model) tends to increase variance, and vice versa.

The optimal model sits at the minimum of the total error curve.

</v-click>

<v-click>

<div class="mt-4 p-4 bg-blue-900/20 rounded-lg">

This tradeoff is why we need a principled way to measure error — which brings us to **Loss functions** and **Metrics**.

</div>

</v-click>

---
layout: center
---

## Summary & Transition

---

### Summary

<div class="grid grid-cols-3 gap-6">
<div class="p-4 bg-gray-800/50 rounded-lg">

### What is ML?

<v-click>

- Learn a function $f$ from data
- Data + Output $\to$ Model
- Replace hand-crafted rules

</v-click>

</div>
<div class="p-4 bg-gray-800/50 rounded-lg">

### Learning Paradigms

<v-click>

- **Supervised**: learn from $(\mathbf{x}, y)$ pairs
- **Unsupervised**: discover structure in $\mathbf{x}$
- **Reinforcement**: learn from reward signals

</v-click>

</div>
<div class="p-4 bg-gray-800/50 rounded-lg">

### Key Concepts

<v-click>

- Train / Validation / Test split
- Overfitting vs Underfitting
- Bias-Variance tradeoff
- Generalization

</v-click>

</div>
</div>

---

### What's Next: Lecture 2

We now know **what** ML does — learn $f$ from data.

<v-click>

But two questions remain:

1. **How to measure how good $f$ is?** $\to$ **Loss function** $L(\hat{y}, y)$
2. **How to find the best $f$?** $\to$ **Optimization** (gradient descent and beyond)

</v-click>

<v-click>

<div class="mt-6 p-4 bg-blue-900/30 rounded-lg text-center">

**Lecture 2: Metric, Loss, and Optimization**

- How to define what "good" means (metrics)
- How to turn "good" into a differentiable objective (loss)
- How to minimize the loss efficiently (optimization)

</div>

</v-click>

<v-click>

<div class="mt-4 text-center text-gray-400">

Lecture 0 gave us the **tools**.
Lecture 1 gave us the **big picture**.
Lecture 2 will give us the **engine**.

</div>

</v-click>
