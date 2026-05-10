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

| Distribution                               | PMF                                 | Typical Use                       |
| ------------------------------------------ | ----------------------------------- | --------------------------------- |
| **Bernoulli** $X \sim \text{Bernoulli}(p)$ | $P(X=1)=p, \; P(X=0)=1-p$           | Binary labels                     |
| **Binomial** $X \sim \text{Bin}(n,p)$      | $P(X=k)=\binom{n}{k}p^k(1-p)^{n-k}$ | Number of successes in $n$ trials |

<v-click>

#### Continuous

| Distribution                                   | PDF                                                                  | Typical Use                   |
| ---------------------------------------------- | -------------------------------------------------------------------- | ----------------------------- |
| **Uniform** $X \sim U(a,b)$                    | $p(x) = \frac{1}{b-a}$                                               | Random initialization         |
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

| Term                         | Meaning                                        | Example                                           |
| ---------------------------- | ---------------------------------------------- | ------------------------------------------------- |
| **Prior** $P(y)$             | Probability of class $y$ **before** seeing $x$ | In a dataset: 60% cats, 40% dogs                  |
| **Likelihood** $P(x \mid y)$ | How likely is sample $x$ given class $y$?      | Given it's a cat, how likely is this fur pattern? |
| **Posterior** $P(y \mid x)$  | Updated belief about $y$ **after** seeing $x$  | After seeing the image: 90% cat                   |

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

| Property              | Formula                   |
| --------------------- | ------------------------- |
| Linearity             | $E[aX + b] = aE[X] + b$   |
| Sum                   | $E[X + Y] = E[X] + E[Y]$  |
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

| $\text{Cov}(X,Y)$ | Meaning                                              |
| ----------------- | ---------------------------------------------------- |
| $> 0$             | Positive correlation: $Y$ tends to increase with $X$ |
| $= 0$             | Uncorrelated (in the linear sense)                   |
| $< 0$             | Negative correlation: $Y$ tends to decrease with $X$ |

</v-click>

<v-click>

**Correlation coefficient** (normalized covariance):

$$\rho(X, Y) = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y}, \quad \rho \in [-1, 1]$$

</v-click>

---

### Expectations and Variances of Common Distributions

| Distribution                        | $E[X]$              | $\text{Var}(X)$       |
| ----------------------------------- | ------------------- | --------------------- |
| Bernoulli $\text{Bernoulli}(p)$     | $p$                 | $p(1-p)$              |
| Binomial $\text{Bin}(n, p)$         | $np$                | $np(1-p)$             |
| Uniform $U(a, b)$                   | $\frac{a+b}{2}$     | $\frac{(b-a)^2}{12}$  |
| Normal $\mathcal{N}(\mu, \sigma^2)$ | $\mu$               | $\sigma^2$            |
| Exponential $\text{Exp}(\lambda)$   | $\frac{1}{\lambda}$ | $\frac{1}{\lambda^2}$ |

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