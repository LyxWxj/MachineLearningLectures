---
theme: seriph
themeConfig:
  primary: '#5d8392'
title: "Lecture 0: Math Terminology for Machine Learning"
transition: slide-left
math: true
---

# Lecture 0: Math Foundations for Machine Learning

<div class="text-gray-400 mt-2">
Calculus · Linear Algebra · Probability
</div>

<Toc minDepth="2" maxDepth="2" />

---

layout: center
---

## Overview

<v-click>

Three main components:

| Subject | Usage |
|------|---------|
| **Calculus** | Optimization — updating model parameters via gradient descent |
| **Linear Algebra** | Representation — storing and computing data as vectors/matrices |
| **Probability** | Modeling — describing uncertainty with probability distributions |

</v-click>

<v-click>

Go through all these three **components**, dive into machine learning as quickly as possible.

</v-click>

---

## Part 1: Calculus

### Derivative-Definition

For a function $y=f(x)$, the derivative at point $x$ is:

$$f'(x) = \lim_{\Delta x \to 0} \frac{f(x + \Delta x) - f(x)}{\Delta x} = \frac{dy}{dx}$$

<v-click>

<div class="mt-4">

**Example**: $f(x) = x^3/3 - x$

$$f'(x) = \frac{d}{dx}\left(\frac{x^3}{3} - x\right) = x^2 - 1$$

</div>

</v-click>

<v-click>

<div class="mt-4">

The sign of $\frac{df}{dx}$ indicates if $f$ increases by an increasing $x$.

</div>
</v-click>

---

### Graphical Interpretation

<div class="grid grid-cols-2 gap-8">
<div style="height: 320px;">

<DerivativeChart :tangent-x="1" />

</div>
<div>

In the chart:

- **Blue curve**: $f(x) = \frac{x^3}{3} - x$
- **Red dashed line**: tangent at $x=1$
- Tangent slope = $f'(1) = 1^2 - 1 = 0$

<v-click>

**Key point**: points where $f'(x) = 0$ are extrema (max or min).

**The Basis of Optimization**

</v-click>

</div>
</div>

---

### Common Differentiation Rules

| Function $y$ | Derivative $\dfrac{dy}{dx}$ | Differential $dy$ | Rule |
|----------|----------------------|-----------|---------|
| $y = cf(x)$ | $c\dfrac{df}{dx}$ | $dy=c\, df=cf'(x)dx$ | Constant multiple |
| $y = f + g$ | $\dfrac{df}{dx} + \dfrac{dg}{dx}$ | $dy=df + dg=(f'(x)+g'(x))dx$ | Sum |
| $y = f(g(x))$ | $\dfrac{df}{dg} \cdot \dfrac{dg}{dx}$ | $dy=f'(g)\, dg=f'(g(x))g'(x)dx$ | **Chain rule** |

<v-click>

<div class="mt-4 p-4 bg-blue-900/20 rounded-lg">

The **chain rule** is critical for backpropagation in neural networks. It computes derivatives of composite functions: propagating gradients backward layer by layer.

$$\frac{\partial L}{\partial w_1} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial z} \cdot \frac{\partial z}{\partial w_1}$$

</div>

</v-click>

---

### Gradient Descent Visualization

<div class="grid grid-cols-2 gap-8">
<div style="height: 320px;">

<GradientDescent />

</div>
<div>

**Goal**: find the minimum of $f(x) = (x-2)^2 + 1$

<v-click>

**Gradient descent process**:

1. Start at $x_0 = -0.5$
2. Compute $f'(x) = 2(x-2)$
3. Update $x_{n+1} = x_n - \eta \cdot f'(x_n)$
4. Repeat until convergence

</v-click>

<v-click>

**Parameters**:

- Learning rate $\eta = 0.3$
- Red dots show each step
- Converges to $x = 2$ (the minimum)

</v-click>

<v-click>

In ML, model parameters are optimized iteratively this way.

</v-click>

</div>
</div>
---

### Partial Derivative

For a multivariable function $f(x_1, x_2, \ldots, x_n)$, the partial derivative is the rate of change when **only one variable** varies:

$$\frac{\partial f}{\partial x_i} = \lim_{\Delta x_i \to 0} \frac{f(x_1, \ldots, x_i + \Delta x_i, \ldots, x_n) - f(x_1, \ldots, x_n)}{\Delta x_i}$$

<v-click>

Example:

$$
y=f(x,y,z),\frac{\partial f}{\partial x}=\lim_{\Delta x_i}\frac{f(x+\Delta x_i,y,z)}{\Delta x_i}
$$

</v-click>

<v-click>

The **gradient** is the vector of all partial derivatives:

$$\nabla f = \left[\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \ldots, \frac{\partial f}{\partial x_n}\right]$$

</v-click>

<v-click>

The gradient points in the direction of **steepest ascent**. In ML, we update parameters in the **opposite** direction to minimize the loss:

$$\mathbf{w} \leftarrow \mathbf{w} - \eta \nabla L(\mathbf{w})$$

where $\eta$ is the learning rate.

</v-click>

---

### Integral

<script setup>
import { ref } from 'vue'
const n = ref(8)
</script>

A **definite integral** is the inverse of the derivative, representing the accumulated quantity over $[a, b]$:

$$\int_a^b f(x)\, dx = F(a)-F(b)=\left[F(x)\right]^{b}_{a}=\lim_{n \to \infty} \sum_{i=1}^{n} f(x_i) \Delta x$$

<div class="grid grid-cols-2 gap-8">
<div style="height: 280px;">

<div class="flex items-center justify-center gap-4 mb-2">
  <button class="px-3 py-1 rounded border border-gray-500 hover:bg-gray-400/20 font-mono" @click="n = Math.max(1, n - 1)">−</button>
  <span class="font-mono text-lg">n = {{ n }}</span>
  <button class="px-3 py-1 rounded border border-gray-500 hover:bg-gray-400/20 font-mono" @click="n += 1">+</button>
</div>
<v-click>

**Exact calculation** using the Fundamental Theorem of Calculus:

$$\int_{0.5}^{2} \left(\frac{x^2}{2} + 0.3\right) dx = \left[\frac{x^3}{6} + 0.3x\right]_{0.5}^{2}$$

$$= \underbrace{\left(\frac{2^3}{6} + 0.3 \times 2\right)}_{F(2) = \frac{29}{15} \approx 1.933} - \underbrace{\left(\frac{0.5^3}{6} + 0.3 \times 0.5\right)}_{F(0.5) = \frac{41}{240} \approx 0.171} = \frac{141}{80} \approx 1.7625$$

</v-click>
</div>
<div style="height: 280px;">

<RiemannChart :a="0.5" :b="2" :n="n" />

</div>
</div>

---

### Indefinite Integral

The **indefinite integral** is the antiderivative — the family of all functions whose derivative is $f(x)$:

$$\int f(x)\, dx = F(x) + C$$

<v-click>

where $F'(x) = f(x)$ and $C$ is an arbitrary constant (since $(F(x)+C)' = f(x)$).

</v-click>

<v-click>

**Common integration formulas**:

| $f(x)$ | $\int f(x)\, dx$ |
|------|------|
| $x^n$ ($n \neq -1$) | $\dfrac{x^{n+1}}{n+1} + C$ |
| $\dfrac{1}{x}$ | $\ln\|x\| + C$ |
| $e^x$ | $e^x + C$ |

</v-click>

---

### Example:
<v-click>
$$
\int (3x^2 + 2x)\, dx = x^3 + x^2 + C
$$

Check by differentiation: $\frac{d}{dx}(x^3 + x^2 + C) = 3x^2 + 2x$ ✓

</v-click>

<v-click>

<div class="mt-4 p-4 bg-blue-900/20 rounded-lg">

**Connection**: definite integrals are computed via indefinite integrals — find $F(x)$, then evaluate $F(b) - F(a)$. In ML, integration appears in probability (computing $P(a \leq X \leq b)$) and in normalizing constants.

</div>

</v-click>

---

## Part 2: Linear Algebra

### Basic Data Structures

<div class="grid grid-cols-2 gap-8">
<div>

**Scalar**: a single number

$$x = 5, \quad x \in \mathbb{R}$$

<v-click>

**Vector**: an ordered list of numbers

$$\mathbf{v} = \begin{bmatrix} v_1 \\ v_2 \\ v_3 \end{bmatrix}, \quad \mathbf{v} \in \mathbb{R}^3$$

</v-click>

<v-click>

**Matrix**: a 2D array

$$\mathbf{A} = \begin{bmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \\ a_{31} & a_{32} \end{bmatrix}, \quad \mathbf{A} \in \mathbb{R}^{3 \times 2}$$

</v-click>

</div>
<div>

<v-click>

**Tensor**: a generalization to $n$ dimensions

- 0th-order tensor = scalar
- 1st-order tensor = vector
- 2nd-order tensor = matrix
- 3rd-order and above = higher-order tensors

</v-click>

<v-click>

**Dimension notation**: $\mathbb{R}^{m \times n}$ denotes a real matrix with $m$ rows and $n$ columns.

</v-click>

<v-click>

In ML frameworks (PyTorch, TensorFlow), all data is stored and computed as tensors.

</v-click>

</div>
</div>

---

### Geometric Representation of Vectors

<div class="grid grid-cols-2 gap-8">
<div style="height: 320px;">

<VectorChart />

</div>
<div>

A vector can be represented as a **directed line segment** from the origin:

$$\mathbf{v} = \begin{bmatrix} 3 \\ 1 \end{bmatrix}, \quad \mathbf{u} = \begin{bmatrix} 1 \\ 2 \end{bmatrix}$$

<v-click>

**Norm (length)**:

$$\|\mathbf{v}\| = \sqrt{v_1^2 + v_2^2 + \cdots + v_n^2}$$

</v-click>

<v-click>

**Unit vector**: a vector with norm 1, $\hat{\mathbf{v}} = \frac{\mathbf{v}}{\|\mathbf{v}\|}$

</v-click>

<v-click>

In ML, a data sample (e.g. an image, a user profile) is typically represented as a high-dimensional vector.

</v-click>

</div>
</div>

---

### Vector Operations

<div class="grid grid-cols-2 gap-8">
<div>

**Addition**: component-wise sum

$$\mathbf{u} + \mathbf{v} = \begin{bmatrix} u_1 + v_1 \\ u_2 + v_2 \\ \vdots \end{bmatrix}$$

<v-click>

**Scalar multiplication**: multiply each component by the scalar

$$c\mathbf{v} = \begin{bmatrix} cv_1 \\ cv_2 \\ \vdots \end{bmatrix}$$

</v-click>

</div>
<div>

<v-click>

**Dot product (inner product)**:

$$\mathbf{u} \cdot \mathbf{v} = \sum_{i=1}^{n} u_i v_i = \|\mathbf{u}\|\|\mathbf{v}\|\cos\theta$$

</v-click>

<v-click>

Geometric interpretation of the dot product:

- $\mathbf{u} \cdot \mathbf{v} > 0$: angle < 90° (roughly same direction)
- $\mathbf{u} \cdot \mathbf{v} = 0$: **orthogonal** (perpendicular)
- $\mathbf{u} \cdot \mathbf{v} < 0$: angle > 90° (roughly opposite)

</v-click>

<v-click>

In ML, a single layer of a neural network is essentially a **dot product** of the input and weight vectors.

</v-click>

</div>
</div>

---

### Matrix Multiplication

For $\mathbf{A} \in \mathbb{R}^{m \times k}$ and $\mathbf{B} \in \mathbb{R}^{k \times n}$, the product $\mathbf{C} = \mathbf{A}\mathbf{B} \in \mathbb{R}^{m \times n}$ is:

$$C_{ij} = \sum_{p=1}^{k} A_{ip} B_{pj}$$

<v-click>

**Example**:

$$\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} \begin{bmatrix} 5 & 7 & 9 \\ 6 & 8 & 10 \end{bmatrix} = \begin{bmatrix} 1{\times}5+2{\times}6 & 1{\times}7+2{\times}8 & 1{\times}9+2{\times}10 \\ 3{\times}5+4{\times}6 & 3{\times}7+4{\times}8 & 3{\times}9+4{\times}10 \end{bmatrix} = \begin{bmatrix} 17 & 23 & 29 \\ 39 & 53 & 67 \end{bmatrix}$$

</v-click>

<v-click>

<div class="mt-4 p-4 bg-blue-900/20 rounded-lg">

**Forward propagation** in a neural network is a sequence of matrix multiplications and nonlinear activations:

$$\mathbf{h} = \sigma(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1), \quad \mathbf{y} = \mathbf{W}_2 \mathbf{h} + \mathbf{b}_2$$

where $\mathbf{W}$ is the weight matrix, $\mathbf{b}$ is the bias vector, and $\sigma$ is the activation function.

</div>

</v-click>

---

### Other Important Matrix Operations

Let $\mathbf{A} = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$, $\mathbf{B} = \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix}$, $c = 2$

| Operation | Definition | Example |
|------|------|------|
| **Transpose** | $(A^T)_{ij} = A_{ji}$ | $\begin{bmatrix} 1 & 3 \\ 2 & 4 \end{bmatrix}$ |
| **Matrix Addition** | $(A+B)_{ij} = A_{ij} + B_{ij}$ | $\begin{bmatrix} 6 & 8 \\ 10 & 12 \end{bmatrix}$ |
| **Scalar Addition** | $(A+c)_{ij} = A_{ij} + c$ | $\begin{bmatrix} 3 & 4 \\ 5 & 6 \end{bmatrix}$ |

---

### Matrix Operations (cont.)

| Operation | Definition | Example |
|------|------|------|
| **Element-wise (Hadamard)** | $(A \odot B)_{ij} = A_{ij} \cdot B_{ij}$ | $\begin{bmatrix} 5 & 12 \\ 21 & 32 \end{bmatrix}$ |
| **Matrix Multiplication** | $(AB)_{ij} = \sum_k A_{ik}B_{kj}$ | $\begin{bmatrix} 19 & 22 \\ 43 & 50 \end{bmatrix}$ |

<v-click>

<div class="mt-4 p-4 bg-blue-900/20 rounded-lg">

**In deep learning**:

- **Matrix multiplication** — forward/backward pass through linear layers: $\mathbf{y} = \mathbf{W}\mathbf{x} + \mathbf{b}$
- **Element-wise ops** — activation functions, dropout, batch norm applied per element
- **Transpose** — backpropagation of linear layers: $\frac{\partial L}{\partial \mathbf{W}} = \frac{\partial L}{\partial \mathbf{y}} \mathbf{x}^T$

</div>

</v-click>

---

## Part 3: Probability

### Random Variable

A **random variable** is a mapping from outcomes of a random event to real numbers: $X: \Omega \to \mathbb{R}$.

<div class="grid grid-cols-2 gap-8 mt-4">
<div>

**Discrete random variable**: takes finitely many or countably many values

<v-click>

Example: rolling a die, $X \in \{1, 2, 3, 4, 5, 6\}$

</v-click>

<v-click>

Described by a **PMF**: $P(X = x_k) = p_k$

satisfying $\sum_k p_k = 1$

</v-click>

</div>
<div>

**Continuous random variable**: takes values in a real interval

<v-click>

Example: measurement error $X \in (-\infty, +\infty)$

</v-click>

<v-click>

Described by a **PDF**: $p(x) \geq 0$

satisfying $\int_{-\infty}^{+\infty} p(x)\, dx = 1$

</v-click>

<v-click>

Note: $P(X = a) = 0$; only interval probabilities are meaningful:

$$P(a \leq X \leq b) = \int_a^b p(x)\, dx$$

</v-click>

</div>
</div>

---

### Common Distributions

#### Discrete

| Distribution | PMF | Typical Use |
|------|-----|---------|
| **Bernoulli** $X \sim \text{Bernoulli}(p)$ | $P(X=1)=p, \; P(X=0)=1-p$ | Binary labels |
| **Binomial** $X \sim \text{Bin}(n,p)$ | $P(X=k)=\binom{n}{k}p^k(1-p)^{n-k}$ | Number of successes in $n$ trials |

<v-click>

#### Continuous

| Distribution | PDF | Typical Use |
|------|-----|---------|
| **Uniform** $X \sim U(a,b)$ | $p(x) = \frac{1}{b-a}$ | Random initialization |
| **Normal** $X \sim \mathcal{N}(\mu, \sigma^2)$ | $p(x) = \frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}$ | Noise modeling, weight priors |

</v-click>

---

#### Normal Distribution

The normal (Gaussian) distribution is the most commonly used continuous distribution in ML:

$$X \sim \mathcal{N}(\mu, \sigma^2) \quad \Rightarrow \quad p(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)$$

<div class="grid grid-cols-2 gap-8">
<div style="height: 300px;">

<NormalDistribution :mu="0" :sigma="1" />

</div>
<div>

Parameters:

- **$\mu$ (mean)**: center of the distribution
- **$\sigma$ (std dev)**: width / spread
- **$\sigma^2$ (variance)**: squared standard deviation

<v-click>

**68-95-99.7 rule**:

- $P(\mu - \sigma \leq X \leq \mu + \sigma) \approx 68\%$
- $P(\mu - 2\sigma \leq X \leq \mu + 2\sigma) \approx 95\%$
- $P(\mu - 3\sigma \leq X \leq \mu + 3\sigma) \approx 99.7\%$

</v-click>

</div>
</div>

---

#### Effect of Normal Distribution Parameters

<div class="grid grid-cols-2 gap-8">
<div style="height: 300px;">

<NormalDistribution :mu="0" :sigma="0.5" />

</div>
<div style="height: 300px;">

<NormalDistribution :mu="0" :sigma="2" />

</div>
</div>

<v-click>

- **Left** $\sigma = 0.5$: narrow and tall, data concentrated near the mean
- **Right** $\sigma = 2$: wide and short, data widely spread

$\mu$ controls location, $\sigma$ controls spread. Adjusting these two parameters describes different data distributions.

</v-click>

---

#### Joint Distribution and Conditional Probability

For two random variables $X$ and $Y$:

<v-click>

**Joint distribution**: $P(X = x, Y = y)$ describes the probability of both taking specific values simultaneously.

</v-click>

<v-click>

**Conditional probability**: the distribution of $X$ given $Y = y$

$$P(X = x \mid Y = y) = \frac{P(X = x, Y = y)}{P(Y = y)}$$

</v-click>

<v-click>

**Bayes' theorem**:

$$P(Y = y \mid X = x) = \frac{P(X = x \mid Y = y) \cdot P(Y = y)}{P(X = x)}$$

</v-click>

---

#### Prior, Likelihood, and Posterior

Bayes' theorem has a powerful interpretation in ML classification. With sample $x$ and label $y$:

$$\underbrace{P(y \mid x)}_{\text{posterior}} = \frac{\overbrace{P(x \mid y)}^{\text{likelihood}} \cdot \underbrace{P(y)}_{\text{prior}}}{P(x)}$$

<v-click>

| Term | Meaning | Example |
|------|---------|---------|
| **Prior** $P(y)$ | Probability of class $y$ **before** seeing $x$ | In a dataset: 60% cats, 40% dogs |
| **Likelihood** $P(x \mid y)$ | How likely is sample $x$ given class $y$? | Given it's a cat, how likely is this fur pattern? |
| **Posterior** $P(y \mid x)$ | Updated belief about $y$ **after** seeing $x$ | After seeing the image: 90% cat |

</v-click>
---

<v-click>

**Intuition**: posterior $\propto$ likelihood $\times$ prior

- With **little data**, the posterior is dominated by the prior
- With **lots of data**, the likelihood dominates and the prior "washes out"

</v-click>

<v-click>

<div class="mt-4 p-4 bg-blue-900/20 rounded-lg">

This is exactly how **Naive Bayes** works: estimate $P(y)$ from class frequencies, estimate $P(x \mid y)$ from feature statistics, then compute $P(y \mid x)$ for prediction.

</div>

</v-click>

---

#### Example: Coin Flip

Is a coin fair? Let $\theta$ = probability of heads.

<div class="grid grid-cols-2 gap-8">
<div>

<v-click>

**Prior**: we believe the coin is roughly fair

$$\theta \sim \text{Beta}(5, 5)$$

This encodes: "most likely near 0.5"

</v-click>

<v-click>

**Data**: flip 20 times, get 15 heads

**Likelihood**:

$$P(D \mid \theta) = \binom{20}{15} \theta^{15}(1-\theta)^5$$

</v-click>

</div>
<div>

<v-click>

**Posterior**:

$$P(\theta \mid D) \propto \theta^{15}(1-\theta)^5 \cdot \theta^4(1-\theta)^4$$

$$= \text{Beta}(5+15, 5+5) = \text{Beta}(20, 10)$$

</v-click>

<v-click>

**Result**: posterior mean = $\frac{20}{30} \approx 0.67$

Shifted from prior (0.5) toward the data (0.75), balancing both sources of information.

</v-click>

</div>
</div>

<v-click>

<div class="mt-4 p-4 bg-blue-900/20 rounded-lg">

**In ML**: Maximum Likelihood Estimation (MLE) ignores the prior; Maximum A Posteriori (MAP) includes it. Bayesian methods keep the full posterior distribution.

</div>

</v-click>

---

#### Independence and Conditional Independence

**Independent**: $X$ and $Y$ are independent iff

$$P(X, Y) = P(X) \cdot P(Y)$$

<v-click>

Equivalently, $P(X \mid Y) = P(X)$ — knowing $Y$ does not change the prediction of $X$.

</v-click>

<v-click>

**Conditionally independent**: given $Z$, $X$ and $Y$ are independent

$$P(X, Y \mid Z) = P(X \mid Z) \cdot P(Y \mid Z)$$

</v-click>

<v-click>

In the Naive Bayes classifier, the core assumption is: **given the class label $y$, all features are conditionally independent**.

</v-click>

<v-click>

If $\mathbf{x} = [x_1, x_2, \ldots, x_n]$ are $n$ features, the likelihood decomposes:

$$P(\mathbf{x} \mid y) = P(x_1, x_2, \ldots, x_n \mid y) = \prod_{i=1}^{n} P(x_i \mid y)$$

</v-click>


---
layout: center
---

### Expectation, Variance, and Covariance

---

### Expectation

The expectation is the **weighted average** of a random variable, reflecting the "center" of the distribution:

<v-click>

**Discrete**:

$$E[X] = \sum_x x \cdot P(X = x)$$

</v-click>

<v-click>

**Continuous**:

$$E[X] = \int_{-\infty}^{+\infty} x \cdot p(x)\, dx$$

</v-click>

<v-click>

**Properties of expectation**:

| Property | Formula |
|------|------|
| Linearity | $E[aX + b] = aE[X] + b$ |
| Sum | $E[X + Y] = E[X] + E[Y]$ |
| Product (independent) | $E[XY] = E[X] \cdot E[Y]$ |

</v-click>

<v-click>

**Example**: fair die, $E[X] = 1 \cdot \frac{1}{6} + 2 \cdot \frac{1}{6} + \cdots + 6 \cdot \frac{1}{6} = 3.5$

</v-click>

---

### Variance and Standard Deviation

Variance measures the **spread** of a random variable around its mean:

$$\text{Var}(X) = E[(X - \mu)^2] = E[X^2] - (E[X])^2$$

<v-click>

**Standard deviation**: $\sigma = \sqrt{\text{Var}(X)}$, same unit as the data, easier to interpret.

</v-click>

<v-click>

**Properties of variance**:

$$\text{Var}(aX + b) = a^2 \text{Var}(X)$$

A constant shift $b$ does not affect variance; scaling by $a$ scales variance by $a^2$.

</v-click>

<v-click>

<div class="mt-4 p-4 bg-blue-900/20 rounded-lg">

**Bias-variance tradeoff**:

$$E[(y - \hat{f}(x))^2] = \underbrace{\text{Bias}^2[\hat{f}(x)]}_{\text{bias}^2} + \underbrace{\text{Var}[\hat{f}(x)]}_{\text{variance}} + \underbrace{\sigma^2}_{\text{irreducible noise}}$$

Increasing model complexity decreases bias but increases variance. The optimal model balances the two.

</div>

</v-click>

---

### Covariance and Correlation

**Covariance** measures the **linear relationship** between two random variables:

$$\text{Cov}(X, Y) = E[(X - \mu_X)(Y - \mu_Y)] = E[XY] - E[X]E[Y]$$

<v-click>

| $\text{Cov}(X,Y)$ | Meaning |
|---|------|
| $> 0$ | Positive correlation: $Y$ tends to increase with $X$ |
| $= 0$ | Uncorrelated (in the linear sense) |
| $< 0$ | Negative correlation: $Y$ tends to decrease with $X$ |

</v-click>

<v-click>

**Correlation coefficient** (normalized covariance):

$$\rho(X, Y) = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y}, \quad \rho \in [-1, 1]$$

</v-click>

---

### Expectations and Variances of Common Distributions

| Distribution | $E[X]$ | $\text{Var}(X)$ |
|------|------------|---------------------|
| Bernoulli $\text{Bernoulli}(p)$ | $p$ | $p(1-p)$ |
| Binomial $\text{Bin}(n, p)$ | $np$ | $np(1-p)$ |
| Uniform $U(a, b)$ | $\frac{a+b}{2}$ | $\frac{(b-a)^2}{12}$ |
| Normal $\mathcal{N}(\mu, \sigma^2)$ | $\mu$ | $\sigma^2$ |
| Exponential $\text{Exp}(\lambda)$ | $\frac{1}{\lambda}$ | $\frac{1}{\lambda^2}$ |

<v-click>

These are the most commonly used distributions in ML. Knowing their expectations and variances helps understand model behavior.

</v-click>

---

### Summary

<div class="grid grid-cols-3 gap-6">
<div class="p-4 bg-gray-800/50 rounded-lg">

### Calculus

<v-click>

- **Derivative**: rate of change, foundation of gradient descent
- **Partial derivative / gradient**: direction for multivariate optimization
- **Integral**: accumulation, foundation of probability

</v-click>

</div>
<div class="p-4 bg-gray-800/50 rounded-lg">

### Linear Algebra

<v-click>

- **Vector**: representation of data
- **Matrix**: batch computation, core of neural networks
- **Matrix multiplication**: linear transformation, forward propagation

</v-click>

</div>
<div class="p-4 bg-gray-800/50 rounded-lg">

### Probability

<v-click>

- **Random variables & distributions**: describing uncertainty
- **Expectation & variance**: two core metrics of a distribution
- **Bayes' theorem**: updating beliefs from data

</v-click>

</div>
</div>

<v-click>

<div class="mt-8 p-4 bg-blue-900/30 rounded-lg text-center">

**ML = Linear Algebra (representation) + Probability (modeling) + Calculus (optimization)**

</div>

</v-click>

---
layout: two-cols
---

## Next Steps

- Machine-learning concepts
- Few tools
- Some simple codes
- Get hands dirty
