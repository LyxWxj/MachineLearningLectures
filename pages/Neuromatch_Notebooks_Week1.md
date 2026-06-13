---
theme: seriph
themeConfig:
  primary: "#5d8392"
title: "Neuromatch Notebooks - Week 1"
transition: slide-left
math: true
---

# Neuromatch Notebooks — Week 1

<div class="text-gray-400 mt-2">
Model Fitting · Linear Regression · GLMs · Logistic Regression · Bootstrapping
</div>

<Toc minDepth="2" maxDepth="2" />

---
layout: center
---

## Overview

<v-click>

Week 1 focuses on **model fitting** — how to find parameters that explain data, and how confident we should be in those parameters:

| Day      | Topic                      | Core Skill                               |
| -------- | -------------------------- | ---------------------------------------- |
| **W1D1** | Model Types                | Descriptive, Mechanistic, Why models     |
| **W1D2** | Model Fitting              | MSE, MLE, Bootstrapping, Polynomial      |
| **W1D3** | GLMs                       | Linear-Gaussian, Poisson GLM, Logistic   |

</v-click>

<v-click>

<div class="mt-4 p-4 bg-blue-900/20 rounded-lg text-center">

**The unifying theme**: given data, find the best model parameters, and quantify how sure we are.

</div>

</v-click>

---
layout: center
---

## W1D1: Model Types

---

### Three Types of Models

W1D1 introduces a framework for thinking about models in neuroscience:

<div class="grid grid-cols-3 gap-6">
<div class="p-4 bg-gray-800/50 rounded-lg text-center">

### "What" Models

Descriptive — characterize the data

Example: ISI histogram shape

</div>
<div class="p-4 bg-gray-800/50 rounded-lg text-center">

### "How" Models

Mechanistic — simulate the process

Example: LIF neuron simulation

</div>
<div class="p-4 bg-gray-800/50 rounded-lg text-center">

### "Why" Models

Teleological — explain the purpose

Example: entropy maximization

</div>
</div>

---

### "What" Models: Exploring Neural Data

The Steinmetz dataset: 734 neurons recorded with Neuropixels in mice.

<v-click>

**Key quantities to compute**:

```python
spike_counts = np.array([len(spikes[i]) for i in range(n_neurons)])
mean_count = np.mean(spike_counts)
median_count = np.median(spike_counts)
```

</v-click>

<v-click>

**ISI (Inter-Spike Interval)**: time between consecutive spikes

```python
isis = np.diff(spike_times)  # differences between consecutive spike times
```

ISI distributions are typically right-skewed — many short intervals, few long ones.

</v-click>

<v-click>

**Fitting by hand**: use sliders to tune parameters of exponential, inverse, and linear functions to match the ISI histogram. This builds intuition for what "fitting a model" means.

</v-click>

---

### "How" Models: LIF Neuron Simulation

Build a simple neuron model and compare its output to real data:

<v-click>

**Linear IF**: $dV = \alpha \cdot I$, threshold at 1, reset to 0

**Leaky IF**: $dV = -\beta V + \alpha \cdot I$ (adds leakage)

**Input**: Poisson spikes — `exc = scipy.stats.poisson.rvs(lambda_exc, size=T)`

</v-click>

<v-click>

```python
for i in range(1, T):
    dv = -beta * v[i-1] + alpha * (exc[i] - inh[i])
    v[i] = v[i-1] + dv
    if v[i] >= 1:
        v[i] = 0
        spike_times.append(i)
```

</v-click>

<v-click>

**Key finding**: balanced excitation/inhibition + leakage → ISI distribution closer to exponential (matching real data).

</v-click>

---

### "Why" Models: Entropy and Information

**Shannon entropy**: measures uncertainty in a distribution:
$H(X) = -\sum_x p(x) \log_2 p(x) \quad \text{(bits)}$

<v-click>

| Distribution | Entropy |
| ------------ | ------- |
| Deterministic (always same value) | 0 bits |
| Uniform over $N$ values | $\log_2 N$ bits |
| Exponential | Maximum for fixed mean |

</v-click>

<v-click>

**Key insight**: exponential ISI distributions maximize entropy for a fixed mean firing rate — they encode the most information per spike for a given energy budget.

</v-click>

<v-click>

**Code**:

```python
def entropy(pmf):
    pmf = pmf[pmf > 0]           # remove zeros (log(0) is undefined)
    # or pmf = pmf + 0.000001 : laplace smoothing
    return -np.sum(pmf * np.log2(pmf))
```

</v-click>

---
layout: center
---

## W1D2: Model Fitting

---

### Linear Regression with MSE

The simplest model fitting problem: find the slope $\theta$ that best fits $y = \theta x + \epsilon$.

<v-click>

**Mean Squared Error** (objective function):

$$\text{MSE}(\theta) = \frac{1}{N}\sum_{i=1}^N (y_i - \theta x_i)^2$$

</v-click>

<v-click>

**Analytic solution** (set derivative to zero):

$$\hat{\theta} = \frac{\mathbf{x}^T \mathbf{y}}{\mathbf{x}^T \mathbf{x}} = \frac{\sum x_i y_i}{\sum x_i^2}$$

</v-click>

<v-click>

**Code**:

```python
def solve_normal_eqn(x, y):
    return (x @ y) / (x @ x)

def mse(x, y, theta):
    y_hat = theta * x
    return np.mean((y - y_hat)**2)
```

</v-click>


---

### Visual: Fitting a Line to Data

<div style="height: 300px;">

<LinearFitChart />

</div>

<v-click>

Red line = $\hat{\theta}$ from the normal equation. It minimizes the sum of squared vertical distances from each blue dot to the line.

</v-click>

---

### Data as Vectors

Each scatter point is one $(x_i, y_i)$ pair. Collect all points into vectors:

$$\mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_N \end{bmatrix}, \quad \mathbf{y} = \begin{bmatrix} y_1 \\ y_2 \\ \vdots \\ y_N \end{bmatrix}$$

<v-click>

**Example** (5 data points):

```python
x = np.array([1.2, 2.5, 3.1, 0.8, 4.3])   # input features
y = np.array([2.1, 4.0, 5.2, 1.5, 6.8])   # targets
```

</v-click>

<v-click>

**Compute the fit**:

```python
theta_hat = (x @ y) / (x @ x)             # = sum(x_i * y_i) / sum(x_i^2)
# theta_hat ≈ 1.56
y_hat = theta_hat * x                       # predictions on the line
residuals = y - y_hat                       # errors (vertical distances)
mse = np.mean(residuals**2)                 # mean squared error
```

</v-click>
---

<v-click>

| Symbol | Shape | Meaning |
| ------ | ----- | ------- |
| $\mathbf{x}$ | $(N,)$ | input features |
| $\mathbf{y}$ | $(N,)$ | observed targets |
| $\hat{\theta}$ | scalar | fitted slope |
| $\hat{\mathbf{y}} = \hat{\theta}\mathbf{x}$ | $(N,)$ | predictions (on the line) |
| $\mathbf{y} - \hat{\mathbf{y}}$ | $(N,)$ | residuals (errors) |

</v-click>

---

### Linear Regression with MLE

Same problem, probabilistic perspective: assume $y_i \sim \mathcal{N}(\theta x_i, \sigma^2)$.

<v-click>

**Likelihood**:

$$L(\theta) = \prod_{i=1}^N \frac{1}{\sqrt{2\pi\sigma^2}} \exp\!\left(-\frac{(y_i - \theta x_i)^2}{2\sigma^2}\right)$$

</v-click>

<v-click>

**Log-likelihood**:

$$\log L(\theta) = -\frac{N}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^N (y_i - \theta x_i)^2$$

</v-click>

<v-click>

**Key result**: maximizing log-likelihood $\Leftrightarrow$ minimizing MSE. They give the same $\hat{\theta}$!

The probabilistic view adds the ability to compute confidence intervals and do Bayesian inference.

</v-click>

<v-click>

```python
from scipy.stats import norm

def likelihood(theta, x, y):
    return np.prod(norm.pdf(y, loc=theta*x, scale=1))
```

</v-click>

---

### From Deterministic to Probabilistic

**MSE view**: $y = \theta x + \epsilon$, noise is just a nuisance.

**Probabilistic view**: noise is part of the model. Treat $y$ as a **random variable**:

$$y \sim \mathcal{N}(\theta x,\; \sigma^2)$$

<v-click>

This means: for a given $x$ and $\theta$, the response $y$ is not deterministic — it follows a Gaussian distribution centered at $\theta x$.

$$p(y \mid x, \theta) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\!\left(-\frac{(y - \theta x)^2}{2\sigma^2}\right)$$

</v-click>

<v-click>

**Why this matters**: instead of just finding one "best" $\hat{\theta}$, we can now:
- Compute **how likely** each $\hat{\theta}$ is given the data
- Build **confidence intervals**
- Do **Bayesian inference**

</v-click>

---

### Probabilistic Model: Geometric Interpretation

For each $x$ value, $y$ is drawn from a Gaussian centered at $\theta x$:

<v-click>

**At $x = 3$**: $y \sim \mathcal{N}(3\theta, \sigma^2)$. The peak of the Gaussian is at $3\theta$.

**At $x = 7$**: $y \sim \mathcal{N}(7\theta, \sigma^2)$. The peak shifts to $7\theta$.

</v-click>

<v-click>

The Gaussian "tube" around the line $y = \theta x$ represents the probability density of $y$ at each $x$. Points closer to the line are more likely; points far from the line are unlikely.

</v-click>

<v-click>

**Code to generate data**:

```python
np.random.seed(121)
theta_true = 1.2
n_samples = 30
x = 10 * np.random.rand(n_samples)    # uniform in [0, 10)
noise = np.random.randn(n_samples)     # N(0, 1)
y = theta_true * x + noise             # y ~ N(1.2x, 1)
```

</v-click>

---

### Single-Point Likelihood

Given one data point $(x, y)$, the **likelihood** of parameter $\hat{\theta}$ is:

$$\mathcal{L}(\hat{\theta} \mid x, y) = p(y \mid x, \hat{\theta}) = \frac{1}{\sqrt{2\pi}} \exp\!\left(-\frac{(y - \hat{\theta} x)^2}{2}\right)$$

<v-click>

**Example**: $x = 2.1$, $y = 3.7$, test $\hat{\theta} = 1.0$:

$$\mathcal{L}(1.0 \mid 2.1, 3.7) = \frac{1}{\sqrt{2\pi}} \exp\!\left(-\frac{(3.7 - 1.0 \times 2.1)^2}{2}\right) = \frac{1}{\sqrt{2\pi}} e^{-1.28} \approx 0.11$$

</v-click>

<v-click>

**Code**:

```python
def likelihood(theta_hat, x, y, sigma=1):
    return (1 / np.sqrt(2 * np.pi * sigma**2)) * np.exp(-(y - theta_hat * x)**2 / (2 * sigma**2))

likelihood(1.0, 2.1, 3.7)  # ≈ 0.113
```

</v-click>

<v-click>

**Interpretation**: if $\hat{\theta} = 1.0$, the probability of observing $y = 3.7$ at $x = 2.1$ is about 11.3%. Not very high — maybe $\hat{\theta} = 1.0$ isn't the best fit?

</v-click>

---

### Joint Likelihood: From One Point to All Data

We have $N$ data points. Assuming noise is **independent** across observations:

$$\mathcal{L}(\hat{\theta} \mid \mathbf{x}, \mathbf{y}) = \prod_{i=1}^N p(y_i \mid x_i, \hat{\theta}) = \prod_{i=1}^N \frac{1}{\sqrt{2\pi}} \exp\!\left(-\frac{(y_i - \hat{\theta} x_i)^2}{2}\right)$$

<v-click>

**Problem**: multiplying $N$ small probabilities → **numerical underflow**.

Example: $N = 30$, each likelihood $\approx 0.3$ → product $\approx 0.3^{30} \approx 10^{-16}$, which rounds to zero.

</v-click>

<v-click>

**Solution**: take the log

$$\log \mathcal{L}(\hat{\theta}) = \sum_{i=1}^N \log p(y_i \mid x_i, \hat{\theta}) = -\frac{N}{2}\log(2\pi) - \frac{1}{2}\sum_{i=1}^N (y_i - \hat{\theta} x_i)^2$$

</v-click>

<v-click>

**Key property**: $\log$ is monotonically increasing, so $\arg\max \mathcal{L} = \arg\max \log \mathcal{L}$. The $\hat{\theta}$ that maximizes the likelihood also maximizes the log-likelihood.

</v-click>

---

### Comparing Different $\hat{\theta}$ via Log-Likelihood

| $\hat{\theta}$ | $\log \mathcal{L}$ | Quality |
| --------------- | ------------------- | ------- |
| 0.5             | $-198.3$            | Poor — line too flat |
| 1.0             | $-42.1$             | Better — closer to truth |
| 1.2 (true)      | $-38.7$             | Best — matches data generation |

<v-click>

**Code**:

```python
theta_hats = [0.5, 1.0, 2.2]
for th in theta_hats:
    ll = np.sum(np.log(likelihood(th, x, y)))
    print(f"theta={th}, log-likelihood={ll:.2f}")
```

</v-click>

<v-click>

**Visual intuition**: for $\hat{\theta} = 0.5$, the Gaussian "tube" is too flat — most data points are far from the center, giving low likelihood. For $\hat{\theta} = 1.2$, the tube aligns with the data — high likelihood.

</v-click>

---

### MLE Derivation: From Log-Likelihood to Formula

Maximize the log-likelihood by taking the derivative and setting to zero:

<v-click>

$$\log \mathcal{L}(\theta) = -\frac{N}{2}\log(2\pi) - \frac{1}{2}\sum_{i=1}^N (y_i - \theta x_i)^2$$

</v-click>

<v-click>

$$\frac{\partial \log \mathcal{L}}{\partial \theta} = \sum_{i=1}^N (y_i - \theta x_i) x_i = 0$$

</v-click>

<v-click>

Expand: $\sum x_i y_i - \theta \sum x_i^2 = 0$

</v-click>

<v-click>

$$\boxed{\;\hat{\theta}_{\text{MLE}} = \frac{\sum x_i y_i}{\sum x_i^2} = \frac{\mathbf{x}^T \mathbf{y}}{\mathbf{x}^T \mathbf{x}}\;}$$

</v-click>

<v-click>

**This is the same formula as MSE!** Minimizing MSE and maximizing likelihood give identical $\hat{\theta}$ when the noise is Gaussian with constant variance.

The probabilistic view doesn't change the answer — it changes what we can **do** with the answer (confidence intervals, Bayesian updates, model comparison).

</v-click>

---

### Notation Reference

| Symbol | Meaning |
| ------ | ------- |
| $x$ | input (independent variable) |
| $y$ | response (dependent variable) |
| $\epsilon \sim \mathcal{N}(0, \sigma^2)$ | Gaussian noise |
| $\theta$ | true parameter |
| $\hat{\theta}$ | estimated parameter |
| $p(y \mid x, \theta)$ | probability of $y$ given $x$ and $\theta$ |
| $\mathcal{L}(\theta \mid x, y)$ | likelihood of $\theta$ given data $(x, y)$ |
| $\hat{\theta}_{\text{MLE}}$ | maximum likelihood estimate |

<v-click>

**Key distinction**: $p(y \mid x, \theta)$ and $\mathcal{L}(\theta \mid x, y)$ use the **same formula** but ask different questions:
- $p(y \mid x, \theta)$: "how likely is $y$?" (data varies, $\theta$ fixed)
- $\mathcal{L}(\theta \mid x, y)$: "how good is $\hat{\theta}$?" ($\theta$ varies, data fixed)

</v-click>

---

### Bootstrapping: Quantifying Uncertainty

How confident are we in $\hat{\theta}$? **Bootstrapping** estimates uncertainty without distributional assumptions.

<v-click>

**Algorithm**:
1. Resample $N$ data points **with replacement** from the original dataset
2. Compute $\hat{\theta}$ on the resampled data
3. Repeat $B$ times (e.g., $B = 2000$)
4. The distribution of $\hat{\theta}^*$ values estimates the sampling distribution

</v-click>

<v-click>

**95% confidence interval**: the 2.5th and 97.5th percentiles of the bootstrap distribution.

</v-click>

<v-click>

```python
def bootstrap_estimates(x, y, n=2000):
    estimates = []
    for _ in range(n):
        idx = np.random.choice(len(x), size=len(x), replace=True)
        estimates.append(solve_normal_eqn(x[idx], y[idx]))
    return np.array(estimates)

theta_boots = bootstrap_estimates(x, y)
ci_95 = np.percentile(theta_boots, [2.5, 97.5])
```

</v-click>

---

### Multiple Linear Regression

Generalize to multiple features: $\mathbf{y} = X\boldsymbol{\theta} + \boldsymbol{\epsilon}$

<v-click>

**Design matrix** $X$: each row is one observation, each column is one feature.

**OLS estimator**:

$$\hat{\boldsymbol{\theta}} = (X^T X)^{-1} X^T \mathbf{y}$$

</v-click>

<v-click>

**Polynomial regression**: features are powers of $x$

$$X = \begin{bmatrix} 1 & x_1 & x_1^2 \\ 1 & x_2 & x_2^2 \\ \vdots & \vdots & \vdots \end{bmatrix}$$

</v-click>

<v-click>

```python
def make_design_matrix(x, order):
    X = np.ones((len(x), 1))         # bias column
    for p in range(1, order + 1):
        X = np.hstack([X, x.reshape(-1, 1)**p])
    return X
def ordinary_least_squares(X, y):
    return np.linalg.inv(X.T @ X) @ X.T @ y
```

</v-click>

<v-click>

**Model selection**: compare MSE for different polynomial orders. Higher order = lower training MSE but risk of overfitting.

</v-click>

---
layout: center
---

## W1D3: Generalized Linear Models

---

### From Linear to GLM

Linear regression: $\hat{y} = X\boldsymbol{\theta}$ — output is unbounded.

<v-click>

**Problem**: neural spike counts are non-negative integers. Firing rates are positive. Binary choices are 0/1.

**Solution**: apply a **link function** $g$ to transform the linear output:

$$g(\hat{y}) = X\boldsymbol{\theta} \quad \Leftrightarrow \quad \hat{y} = g^{-1}(X\boldsymbol{\theta})$$

</v-click>

<v-click>

| Model | Link function $g$ | Inverse $g^{-1}$ | Output type |
| ----- | ----------------- | ----------------- | ----------- |
| Linear-Gaussian | identity | identity | continuous |
| Poisson GLM | $\log$ | $\exp$ | positive counts |
| Logistic | $\log\frac{p}{1-p}$ | sigmoid | probability $[0,1]$ |

</v-click>

---

### Key Concept: Design Matrix

The design matrix organizes raw data into a format that the model can use.

<v-click>

**Definition**: The design matrix $\mathbf{X}$ is a $T \times d$ matrix where each row is a **feature vector** for one time point, and each column is a feature.

</v-click>

<v-click>

**In neuroscience**: We want to know "how do the stimuli over the past $d$ time steps influence the current spike?" The design matrix arranges the past $d$ stimulus values into a row:

$$\mathbf{X} = \begin{bmatrix} \text{stim}[t_0 - d+1] & \cdots & \text{stim}[t_0 - 1] & \text{stim}[t_0] \\ \text{stim}[t_T - d+1] & \cdots & \text{stim}[t_T - 1] & \text{stim}[t_T] \end{bmatrix}$$

</v-click>

<v-click>

**Zero-padding**: For the earliest time points, we don't have a full $d$ history — pad with zeros:

```python
padded_stim = np.concatenate([np.zeros(d - 1), stim])
# padded_stim = [0, 0, ..., 0, stim[0], stim[1], ..., stim[T-1]]
```

</v-click>

<v-click>

**Sliding window extraction**: For each time point $t$, take a window of length $d$ and reverse it:

```python
for t in range(T):
    X[t] = padded_stim[t:t + d][::-1]  # [::-1] reverses so earliest time comes first
```

</v-click>

---

### Key Concept: Observations

Observations are the target variable $\mathbf{y}$ we want to predict.

<v-click>

**In this experiment**:

| Variable | Meaning | Shape |
|----------|---------|-------|
| $\text{stim}[t]$ | Screen luminance (stimulus) at time $t$ | $(T,)$ |
| $\text{spikes}[t]$ | Spike count (response) at time $t$ | $(T,)$ |
| $\mathbf{X}[t]$ | Design matrix row (features) at time $t$ | $(d,)$ |

</v-click>

<v-click>

**Key insight**:

- Each row of $\mathbf{X}$ = "what happened over the past $d$ time steps" (input features)
- Each value of $\mathbf{y}$ = "what happened now" (observation to predict)
- The model learns: **what input feature patterns lead to what outputs**

</v-click>

<v-click>

**Analogy**:

| Prediction Task | Input $\mathbf{X}$ | Output $\mathbf{y}$ |
|-----------------|---------------------|----------------------|
| House price | Area, location, age... | Price |
| Weather | Past 7 days of temp, humidity... | Tomorrow's temperature |
| **Neural spike prediction** | **Past 25 time bins of stimulus** | **Current spike count** |

</v-click>

---

### Key Concept: Poisson Distribution

The Poisson distribution is the core tool for modeling **count data**.

<v-click>

**Probability Mass Function (PMF)**:

$$P(Y = k) = \frac{\lambda^k \cdot e^{-\lambda}}{k!}, \quad k = 0, 1, 2, \ldots$$

where $\lambda > 0$ is the **rate parameter** — the average number of events per unit time/space.

</v-click>

<v-click>

**Key properties**:

| Property | Formula | Meaning |
|----------|---------|---------|
| Mean | $\mathbb{E}[Y] = \lambda$ | Average count |
| Variance | $\text{Var}(Y) = \lambda$ | Variance equals mean |
| Support | $k \in \{0, 1, 2, \ldots\}$ | Non-negative integers |

</v-click>

---

### Poisson Distribution: Intuition and Applications

The Poisson distribution is typically used to model the following types of problems:

<v-click>

**1. Rare event counts**

- Number of phone calls received per day
- Number of typos on a page
- Number of customers arriving at a bank per hour
- **Number of spikes a neuron fires per time bin**

</v-click>

<v-click>

**2. Conditions for Poisson modeling**

The Poisson distribution applies when these conditions hold:

| Condition | Meaning | Neuroscience analogue |
|-----------|---------|----------------------|
| Independence | Events are independent of each other | Spikes are approximately independent |
| Uniformity | Event rate is constant | Rate is approximately constant in short windows |
| Sparsity | At most one event per instant | At most 1–2 spikes per bin |

</v-click>

---

<v-click>

**3. Why are neural spikes well-modeled by Poisson?**

- Spikes are non-negative integers (0, 1, 2, ...)
- Spikes are sparse (mostly 0)
- Spike variance ≈ mean ($\text{Var} \approx \text{mean}$)
- Spikes are approximately independent (over short timescales)

</v-click>

<v-click>

**4. Shape of the Poisson distribution**

```python
import numpy as np
from scipy.stats import poisson

k = np.arange(0, 15)
for lam in [1, 3, 5, 10]:
    pmf = poisson.pmf(k, lam)
    # Small λ → right-skewed, concentrated near 0
    # Large λ → approximately symmetric (CLT)
```

</v-click>

---

### LNP Model: Full Derivation from Linear to Poisson

The Linear-Nonlinear-Poisson (LNP) model is one of the most commonly used GLMs in neuroscience.

<v-click>

**Goal**: Given the past $d$ time bins of stimulus $\mathbf{x}_t = [\text{stim}[t-d+1], \ldots, \text{stim}[t]]$, predict the spike count $y_t$ at time $t$.

</v-click>

<v-click>

**Step 1: Linear component**

Compute the weighted sum of stimulus and weights:

$$z_t = \mathbf{x}_t^\top \boldsymbol{\theta} + b = \sum_{j=1}^{d} \theta_j \cdot \text{stim}[t-d+j] + b$$

where $\boldsymbol{\theta}$ is the temporal filter and $b$ is the bias.

**Meaning**: $z_t$ represents "the combined drive from the past $d$ time bins of stimulus."

</v-click>
---

<v-click>

**Step 2: Nonlinear component**

Map the linear output to a positive firing rate via the exponential function:

$$\lambda_t = \exp(z_t) = \exp(\mathbf{x}_t^\top \boldsymbol{\theta} + b)$$

**Why exp?**

| Problem | Solution |
|---------|----------|
| Linear output $z_t$ can be negative | $\exp(z_t) > 0$, guarantees positive rate |
| Firing rate should increase with drive | $\exp$ is monotonically increasing |
| Small changes in drive produce multiplicative effects | $\exp$ converts addition to multiplication |

</v-click>

---

### LNP Model: Likelihood and Parameter Estimation

<v-click>

**Step 3: Poisson observation model**

Assume spike counts follow a Poisson distribution:

$$y_t \mid \mathbf{x}_t, \boldsymbol{\theta} \sim \text{Poisson}(\lambda_t)$$

Probability mass function:

$$P(y_t \mid \mathbf{x}_t, \boldsymbol{\theta}) = \frac{\lambda_t^{y_t} \cdot e^{-\lambda_t}}{y_t!}$$

</v-click>

<v-click>

**Step 4: Construct the likelihood function**

Assuming spikes are independent across time, the joint likelihood is:

$$\mathcal{L}(\boldsymbol{\theta}) = \prod_{t=1}^{T} P(y_t \mid \mathbf{x}_t, \boldsymbol{\theta}) = \prod_{t=1}^{T} \frac{\lambda_t^{y_t} \cdot e^{-\lambda_t}}{y_t!}$$

</v-click>
---

<v-click>

**Step 5: Take the log to simplify**

$$\log \mathcal{L}(\boldsymbol{\theta}) = \sum_{t=1}^{T} \left[ y_t \log \lambda_t - \lambda_t - \log(y_t!) \right]$$

Drop the constant term $\log(y_t!)$ that does not depend on $\boldsymbol{\theta}$:

$$\log \mathcal{L}(\boldsymbol{\theta}) = \sum_{t=1}^{T} \left[ y_t \log \lambda_t - \lambda_t \right]$$

</v-click>

---

### LNP Model: Matrix Form and Optimization

<v-click>

**Step 6: Matrix form**

Substituting $\lambda_t = \exp(\mathbf{x}_t^\top \boldsymbol{\theta})$, express in matrix notation:

$$\log \mathcal{L}(\boldsymbol{\theta}) = \mathbf{y}^\top \log(\boldsymbol{\lambda}) - \mathbf{1}^\top \boldsymbol{\lambda}$$

where $\boldsymbol{\lambda} = \exp(\mathbf{X}\boldsymbol{\theta})$.

</v-click>

<v-click>

**Step 7: Negative log-likelihood (objective function)**

Minimize the negative log-likelihood:

$$-\log \mathcal{L}(\boldsymbol{\theta}) = -\left( \mathbf{y}^\top \log(\boldsymbol{\lambda}) - \mathbf{1}^\top \boldsymbol{\lambda} \right) = \mathbf{1}^\top \boldsymbol{\lambda} - \mathbf{y}^\top \log(\boldsymbol{\lambda})$$

</v-click>
---

<v-click>

**Step 8: Numerical optimization**

No closed-form solution — use `scipy.optimize.minimize`:

```python
def fit_lnp(stim, spikes, d=25):
    y = spikes
    constant = np.ones_like(y)
    X = np.column_stack([constant, make_design_matrix(stim)])
    x0 = np.random.normal(0, .2, d + 1)  # random initialization
    res = minimize(neg_log_lik_lnp, x0, args=(X, y))
    return res.x
```

</v-click>

---

### LNP Model: End-to-End Pipeline

```
Raw Data
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  Design matrix X[t] = [stim[t-24], ..., stim[t]]        │  ← past 25 bins of stimulus
│  Observation   y[t] = spikes[t]                         │  ← current spike count
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  Linear:      z_t = X[t] · θ + b                        │  ← weighted sum
│  Nonlinear:   λ_t = exp(z_t)                            │  ← map to positive
│  Poisson:     y_t ~ Poisson(λ_t)                        │  ← probabilistic model
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  Minimize  −[y^T log(λ) − 1^T λ]                       │  ← objective
│  Solve θ via scipy.optimize.minimize                     │  ← numerical optimization
└─────────────────────────────────────────────────────────┘
    │
    ▼
  θ → temporal receptive field (TRF)
```

---

### LG GLM vs LNP GLM: Comparison

| | LG (Linear-Gaussian) | LNP (Poisson) |
|---|---|---|
| **Prediction** | $\hat{y} = X\theta$ | $\hat{y} = \exp(X\theta)$ |
| **Output range** | $(-\infty, +\infty)$ | $(0, +\infty)$ |
| **Noise assumption** | Gaussian $\epsilon \sim \mathcal{N}(0, \sigma^2)$ | Poisson $y \sim \text{Poisson}(\lambda)$ |
| **Fitting** | Closed-form $\hat{\theta} = (X^TX)^{-1}X^Ty$ | Numerical optimization (no closed form) |
| **Use case** | Continuous prediction | Non-negative integer counts |
| **Neuroscience** | Stimulus filter estimation | Firing rate modeling |

<v-click>

**Key differences**:

- LG can predict **negative spikes** (unreasonable) ❌
- LNP guarantees predictions are **always positive** via $\exp$ ✅
- LNP better matches the **statistical properties** of spike data (counts, sparse, variance ≈ mean)

</v-click>

---

### Linear-Gaussian GLM: Spike Filtering

Model: spike count $y_t$ depends on the stimulus over the past $d$ time steps.

<v-click>

**Design matrix**: each row contains the $d$ preceding stimulus values

```python
def make_design_matrix(stim, d=25):
    nt = len(stim)
    X = np.zeros((nt, d))
    for t in range(nt):
        for j in range(d):
            if t - j >= 0:
                X[t, j] = stim[t - j]
    return X
```

</v-click>

<v-click>

**Fitting**: same OLS formula $\hat{\boldsymbol{\theta}} = (X^T X)^{-1} X^T \mathbf{y}$

**Interpretation**: $\hat{\boldsymbol{\theta}}$ is the **stimulus filter** — how the neuron integrates stimulus over time. Similar to the Spike-Triggered Average (STA).

</v-click>

---

### Poisson GLM: Spike Rate Modeling

Spike counts are non-negative integers → use Poisson distribution.

<v-click>

**Model**: $y_t \sim \text{Poisson}(\lambda_t)$ where $\lambda_t = \exp(\mathbf{x}_t^T \boldsymbol{\theta})$

The $\exp$ ensures $\lambda_t > 0$.

</v-click>

<v-click>

**Log-likelihood**:

$$\log L(\boldsymbol{\theta}) = \sum_t \left[ y_t \log \lambda_t - \lambda_t \right] = \sum_t \left[ y_t (\mathbf{x}_t^T \boldsymbol{\theta}) - \exp(\mathbf{x}_t^T \boldsymbol{\theta}) \right]$$

</v-click>

<v-click>

**No closed-form solution** — use numerical optimization:

```python
from scipy.optimize import minimize

def neg_log_lik_lnp(theta, X, y):
    lam = np.exp(X @ theta)
    return -np.sum(y * np.log(lam) - lam)

def fit_lnp(stim, spikes, d=25):
    X = make_design_matrix(stim, d)
    X = np.column_stack([X, np.ones(len(spikes))])  # add bias
    result = minimize(neg_log_lik_lnp, x0=np.zeros(d+1), args=(X, spikes))
    return result.x
```

</v-click>

---

### Why Logistic Regression?

The previous models (LG, LNP) predict **continuous** or **count** outputs. But many neuroscience questions are **binary**:

<v-click>

| Question | Output type | Model |
|----------|-------------|-------|
| Stimulus filtering | Spike count (0, 1, 2, ...) | Poisson GLM |
| Firing rate | Positive real number | LNP GLM |
| **Decision decoding** | **Binary (0 or 1)** | **Logistic regression** |

</v-click>

<v-click>

**The problem**: Linear regression can predict probabilities outside $[0, 1]$. Poisson regression predicts counts, not probabilities.

**The solution**: Logistic regression — a GLM with a **sigmoid link function** and **Bernoulli noise model**.

</v-click>

---

### Logistic Regression: The Sigmoid Link Function

The core idea: map a linear output to a probability using the **sigmoid** (logistic) function.

<v-click>

**The sigmoid function**:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

| $z$ | $\sigma(z)$ | Interpretation |
|-----|-------------|----------------|
| $z \to -\infty$ | $\to 0$ | Strong evidence for class 0 |
| $z = 0$ | $= 0.5$ | No evidence either way |
| $z \to +\infty$ | $\to 1$ | Strong evidence for class 1 |

</v-click>
---

<v-click>

**Key properties**:
- Output is always in $(0, 1)$ — interpretable as probability
- Monotonically increasing — larger $z$ → higher probability of class 1
- Symmetric: $\sigma(-z) = 1 - \sigma(z)$
- Derivative has a nice form: $\sigma'(z) = \sigma(z)(1 - \sigma(z))$

</v-click>

<v-click>

**In GLM terms**: the sigmoid is the **inverse link function** $g^{-1}$:

$$\underbrace{\sigma^{-1}(\hat{y})}_{\text{log-odds}} = \mathbf{x}^\top \boldsymbol{\theta} \quad \Leftrightarrow \quad \hat{y} = \sigma(\mathbf{x}^\top \boldsymbol{\theta})$$

The link function $g = \sigma^{-1}$ is the **logit** (log-odds): $g(p) = \log \frac{p}{1-p}$.

</v-click>

---

### Logistic Regression: Bernoulli Likelihood

The output $y$ is binary (0 or 1), so we use the **Bernoulli distribution**.

<v-click>

**Model**:

$$P(y = 1 \mid \mathbf{x}, \boldsymbol{\theta}) = \hat{y} = \sigma(\mathbf{x}^\top \boldsymbol{\theta})$$

</v-click>

<v-click>

**Bernoulli likelihood for one observation**:

$$P(y_i \mid \hat{y}_i) = \hat{y}_i^{\,y_i} (1 - \hat{y}_i)^{1 - y_i}$$

This is a compact way to write:
- If $y_i = 1$: probability = $\hat{y}_i$
- If $y_i = 0$: probability = $1 - \hat{y}_i$

</v-click>

<v-click>

**Log-likelihood for all data** (assuming independence):

$$\log \mathcal{L}(\boldsymbol{\theta}) = \sum_{i=1}^N \left[ y_i \log \hat{y}_i + (1 - y_i) \log(1 - \hat{y}_i) \right]$$

This is the **cross-entropy loss** (negated). It penalizes confident wrong predictions heavily.

</v-click>

<v-click>

**Negative log-likelihood** (what we minimize):

$$-\log \mathcal{L} = -\sum_{i=1}^N \left[ y_i \log \sigma(\mathbf{x}_i^\top \boldsymbol{\theta}) + (1 - y_i) \log(1 - \sigma(\mathbf{x}_i^\top \boldsymbol{\theta})) \right]$$

No closed-form solution — use numerical optimization (e.g., gradient descent, Newton's method).

</v-click>

---

### What is Overfitting?

A model that performs perfectly on training data but poorly on new data has **overfit**.

<v-click>

**Definition**: Overfitting occurs when a model learns the **noise** in the training data instead of the underlying pattern.

**Symptoms**:
- Training accuracy ≈ 100%
- Test accuracy << 100%
- Model weights are very large (to fit noise)

</v-click>

<v-click>

**When does it happen?**

When the model has too much **capacity** relative to the amount of data. In neuroscience, this is extremely common:

| Data type | Features ($d$) | Samples ($N$) | Ratio $d/N$ |
|-----------|---------------|---------------|-------------|
| Electrophysiology | ~100 neurons | ~50 trials | ~2 |
| fMRI | ~10,000 voxels | ~200 trials | ~50 |
| Calcium imaging | ~500 cells | ~100 time points | ~5 |

When $d > N$, overfitting is almost guaranteed.

</v-click>

<v-click>

**Geometric intuition**: In 2D, a single data point can be fit by infinitely many lines. With more features than samples, there are infinitely many weight vectors that achieve zero training error — most of them are meaningless.

</v-click>

---

### Overfitting: Visual Illustration

```
Training data:  ●  ●      ●  ●
                |  |      |  |
                ▼  ▼      ▼  ▼

Underfitting:   ─────────────────     (too simple, high bias)
                A straight line through noisy data

Good fit:       ──╱╲──╱╲──             (captures the pattern)
                Smooth curve through data

Overfitting:    ╱╲╱╲╱╲╱╲╱╲╱╲          (memorizes noise)
                Wiggly curve passing through every point
```

<v-click>

**The bias-variance tradeoff**:

| | Underfitting | Good fit | Overfitting |
|---|---|---|---|
| **Bias** | High | Low | Low |
| **Variance** | Low | Low | High |
| **Training error** | High | Low | ≈ 0 |
| **Test error** | High | Low | High |

</v-click>

<v-click>

**How to detect overfitting**: Cross-validation.

If training accuracy >> cross-validated accuracy, the model is overfitting.

```python
from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(model, X, y, cv=8)
# Compare: model.score(X_train, y_train) vs cv_scores.mean()
```

</v-click>

---

### Regularization: The Core Idea

Regularization reduces overfitting by **constraining the model's freedom**.

<v-click>

**Intuition**: Instead of asking "find the best $\boldsymbol{\theta}$", we ask "find the best $\boldsymbol{\theta}$ **that is small**".

This adds a **penalty** for large weights to the objective function:

$$\text{Objective} = \underbrace{-\log \mathcal{L}(\boldsymbol{\theta})}_{\text{fit the data}} + \underbrace{\Omega(\boldsymbol{\theta})}_{\text{penalty for complexity}}$$

</v-click>

<v-click>

**Why does this help?**

Large weights allow the model to fit noise. By penalizing large weights:
- The model is **smoother** — small changes in input → small changes in output
- The model is **simpler** — fewer effective parameters
- The model **generalizes** better to unseen data

</v-click>
---

<v-click>

**Frequency perspective**:

Think of the model as fitting a sum of sine waves. High-frequency components capture noise; low-frequency components capture the true signal.

- **No regularization**: fits all frequencies (including noise)
- **Regularization**: suppresses high-frequency components (smooths the model)

This is analogous to low-pass filtering in signal processing.

</v-click>

<v-click>

**Bias-variance tradeoff**:

Regularization **increases bias** (the model can't fit the true function perfectly) but **decreases variance** (the model is more stable across different training sets). The sweet spot is found via cross-validation.

</v-click>

---

### $L_2$ Regularization (Ridge)

Penalizes the **sum of squared** weights.

<v-click>

**Objective**:

$$-\log \mathcal{L}'(\boldsymbol{\theta}) = -\log \mathcal{L}(\boldsymbol{\theta}) + \frac{\beta}{2} \sum_j \theta_j^2 = -\log \mathcal{L}(\boldsymbol{\theta}) + \frac{\beta}{2} \|\boldsymbol{\theta}\|_2^2$$

where $\beta > 0$ is the **regularization strength**.

</v-click>

<v-click>

**Effect**: All weights are **shrunk toward zero**, but none are exactly zero.

```
No regularization:    θ = [18.3, -15.7, 12.1, -8.4, ...]   (large values)
L2 regularization:    θ = [ 0.3,  -0.2,  0.1, -0.1, ...]   (small values)
```

</v-click>

<v-click>

**Geometric intuition**: The $L_2$ constraint region is a **circle** (in 2D) or **hypersphere** (in higher dimensions). The solution is where the likelihood contour first touches the circle — typically at a point where all coordinates are nonzero but small.

</v-click>

---

### $L_1$ Regularization (Lasso)

Penalizes the **sum of absolute** weights.

<v-click>

**Objective**:

$$-\log \mathcal{L}'(\boldsymbol{\theta}) = -\log \mathcal{L}(\boldsymbol{\theta}) + \frac{\beta}{2} \sum_j |\theta_j| = -\log \mathcal{L}(\boldsymbol{\theta}) + \frac{\beta}{2} \|\boldsymbol{\theta}\|_1$$

</v-click>

---
layout: center
---

## W1D4: Dimensionality Reduction

---

### Covariance Matrix

The **covariance matrix** captures how pairs of variables co-vary.

<v-click>

**Definition**: For data matrix $\mathbf{X}$ (each row = one sample, each column = one feature):

$$\hat{\Sigma}_{ij} = \frac{1}{N} \sum_{n=1}^N (x_i^{(n)} - \bar{x}_i)(x_j^{(n)} - \bar{x}_j)$$

In matrix form (after mean-centering): $\hat{\Sigma} = \frac{1}{N}\mathbf{X}^\top\mathbf{X}$

</v-click>

<v-click>

**Properties**:
- Symmetric: $\hat{\Sigma}_{ij} = \hat{\Sigma}_{ji}$
- Diagonal = variances of each feature
- Off-diagonal = covariances between feature pairs
- Positive semi-definite (eigenvalues ≥ 0)

</v-click>


---

### Eigenvalues and Eigenvectors

The **eigenvectors** of the covariance matrix define the directions of maximum variance. The **eigenvalues** tell us how much variance each direction captures.

<v-click>

**Definition**: For matrix $\Sigma$:

$$\Sigma \mathbf{w} = \lambda \mathbf{w}$$

where $\mathbf{w}$ is an eigenvector and $\lambda$ is the corresponding eigenvalue.

</v-click>

<v-click>

**Geometric meaning**:
- Eigenvectors point in the directions where the data spreads most
- Eigenvalues measure the amount of spread along each direction
- Eigenvectors are orthogonal (perpendicular) to each other

</v-click>


---

### Principal Component Analysis (PCA)

PCA finds a new coordinate system aligned with the directions of maximum variance.

<v-click>

**Algorithm**:
1. Mean-center the data: $\mathbf{X} \leftarrow \mathbf{X} - \bar{\mathbf{X}}$
2. Compute covariance matrix: $\hat{\Sigma} = \frac{1}{N}\mathbf{X}^\top\mathbf{X}$
3. Find eigenvectors and eigenvalues of $\hat{\Sigma}$
4. Sort by eigenvalue (descending)
5. Project data onto top $K$ eigenvectors

</v-click>

<v-click>

**Projection** (scores):

$$\mathbf{S} = \mathbf{X} \mathbf{W}_{1:K}$$

where $\mathbf{W}_{1:K}$ contains the top $K$ eigenvectors as columns.

</v-click>
---

<v-click>

**Key insight**: PCA is equivalent to finding the **best rank-$K$ approximation** to the data in the least-squares sense (Eckart-Young theorem).

</v-click>

<p align="center">
  <img src="./assets/latent_space_plots_pca.png" alt="PCA Latent Space" width="400" style="display:inline; margin:0 10px;" />
  <img src="./assets/pca-components.png" alt="PCA Components" width="300" style="display:inline; margin:0 10px;" />
</p>

---

### Variance Explained: How Many Components?

Each eigenvalue $\lambda_i$ represents the variance captured by the $i$-th principal component.

<v-click>

**Cumulative variance explained**:

$$\text{Variance explained}(K) = \frac{\sum_{i=1}^K \lambda_i}{\sum_{i=1}^N \lambda_i}$$

</v-click>

<v-click>

**Intrinsic vs Extrinsic dimensionality**:

| | Extrinsic | Intrinsic |
|---|---|---|
| **Definition** | Number of measured features ($N$) | Number of components needed ($K$) |
| **Example (MNIST)** | 784 pixels | ~50–100 components for 90% variance |

</v-click>

<v-click>

**Scree plot**: Plot eigenvalues in descending order. The "elbow" indicates where additional components contribute little variance — this suggests the intrinsic dimensionality.

</v-click>


---

### PCA Reconstruction: Compressing Data

PCA enables **lossy compression**: store only the top $K$ components instead of all $N$ dimensions.

<v-click>

**Forward (projection)**: $\mathbf{S} = \mathbf{X} \mathbf{W}$

**Inverse (reconstruction)**: $\hat{\mathbf{X}} = \mathbf{S}_{1:K} \mathbf{W}_{1:K}^\top + \bar{\mathbf{X}}$

</v-click>

<v-click>

**Reconstruction quality** depends on $K$:

| $K$ | Variance captured | Quality |
|-----|-------------------|---------|
| 1 | Very low | Blob of average intensity |
| ~20 | Moderate | Blurry digits recognizable |
| ~100 | High | Nearly indistinguishable from original |
| 784 | 100% | Perfect reconstruction |

</v-click>


---

### PCA for Denoising

PCA can remove noise by projecting onto the low-dimensional subspace that captures the signal.

<v-click>

**Idea**: Noise is spread across all dimensions equally, while signal is concentrated in the top components. By keeping only the top $K$ components, we discard most of the noise.

</v-click>

<v-click>

**Algorithm**:
1. Find PCA basis from clean data (or noisy data)
2. Project noisy data onto the PCA basis
3. Keep only the top $K$ components
4. Reconstruct: $\hat{\mathbf{X}}_{\text{clean}} = \mathbf{S}_{1:K} \mathbf{W}_{1:K}^\top + \bar{\mathbf{X}}$

</v-click>

<v-click>

**Key insight**: The optimal $K$ depends on the noise level:
- More noise → fewer components (more aggressive denoising)
- Less noise → more components (preserve detail)

</v-click>

<v-click>

**Limitation**: PCA denoising assumes signal lies in a linear subspace. Nonlinear methods (e.g., autoencoders) can be more effective for complex signals.

</v-click>

---

### PCA vs t-SNE: Linear vs Nonlinear Visualization

Both methods reduce high-dimensional data to 2D for visualization, but they have fundamentally different goals.

<v-click>

**PCA (linear)**:
- Preserves **global** structure (large distances)
- Maximizes variance along orthogonal axes
- Deterministic, fast, interpretable
- Result: overlapping clusters for complex data

</v-click>

<v-click>

**t-SNE (nonlinear)**:
- Preserves **local** structure (neighborhood relationships)
- Maps similar points to nearby positions
- Stochastic, slower, not easily interpretable
- Result: well-separated clusters

</v-click>


---

### t-SNE: The Perplexity Parameter

t-SNE has one key hyperparameter: **perplexity**, which roughly controls the effective number of neighbors.

<v-click>

**What perplexity does**:
- **Low perplexity** (e.g., 2–5): focuses on very local structure, creates many small clusters
- **Medium perplexity** (e.g., 30): balances local and global structure
- **High perplexity** (e.g., 50+): emphasizes global structure, may merge distinct clusters

</v-click>

<v-click>

**Guidelines**:
- There is no "correct" perplexity — explore multiple values
- Typical range: 5–50
- Results can change significantly with different perplexity
- Always report perplexity when presenting t-SNE results

</v-click>


---
layout: center
---

## W1D5: Deep Learning

<div class="text-center mt-4 mb-4">

🎬 **Video**: [3Blue1Brown Convolution](https://www.bilibili.com/video/BV1Vd4y1e7pj/?spm_id_from=333.337.search-card.all.click&vd_source=4a0efd9e69f1eda05dec65b957c7e492)

</div>

---

### Feedforward Neural Networks

A feedforward network transforms input $\mathbf{r}$ through a series of **layers** to produce output $y$.

<p align="center">
  <img src="./assets/one-layer-network.png" alt="One-layer network" width="350" />
</p>

<v-click>

**Single hidden layer**:

$$\mathbf{h} = \phi(\mathbf{W}^{in} \mathbf{r} + \mathbf{b}^{in}), \quad y = \mathbf{W}^{out} \mathbf{h} + \mathbf{b}^{out}$$

where $\phi$ is a **nonlinear activation function**.

</v-click>
---

<v-click>

**Why nonlinearity matters**: Without it, stacking linear layers is equivalent to a single linear transformation:

$$y = \mathbf{W}^{out}(\mathbf{W}^{in} \mathbf{r} + \mathbf{b}^{in}) + \mathbf{b}^{out} = (\mathbf{W}^{out}\mathbf{W}^{in})\mathbf{r} + \text{bias}$$

Nonlinear activations allow the network to compute **arbitrary functions** (universal approximation theorem).

</v-click>

<v-click>

**Key concepts**:
- **Width**: number of units per layer ($M$)
- **Depth**: number of hidden layers
- **Parameters**: all weights $\mathbf{W}$ and biases $\mathbf{b}$

</v-click>

---

### Activation Functions

Activation functions introduce nonlinearity into neural networks.

<p align="center">
  <img src="./assets/relu.png" alt="ReLU" width="200" style="display:inline; margin:0 10px;" />
  <img src="./assets/sigmoid.png" alt="Sigmoid" width="200" style="display:inline; margin:0 10px;" />
  <img src="./assets/prelu.png" alt="PReLU" width="200" style="display:inline; margin:0 10px;" />
</p>

<v-click>

| Function | Formula | Range | Use case |
|----------|---------|-------|----------|
| **ReLU** | $\max(0, x)$ | $[0, \infty)$ | Hidden layers (default) |
| **Sigmoid** | $\frac{1}{1+e^{-x}}$ | $(0, 1)$ | Output for binary probability |
| **Softmax** | $\frac{e^{x_i}}{\sum_j e^{x_j}}$ | $(0, 1)$, sums to 1 | Output for classification |
| **PReLU** | $\max(0, x) + \alpha \min(0, x)$ | $(-\infty, \infty)$ | When ReLU "dies" |

</v-click>
---

<v-click>

**Why ReLU works well**:
- Gradient is 1 for positive inputs → no vanishing gradient
- Computationally efficient
- Sparse activations (many zeros)

</v-click>

<v-click>

**"Dying ReLU" problem**: If a neuron's input is always negative, its gradient is always 0 — it never learns. Solutions: PReLU, Leaky ReLU, better initialization.

</v-click>

---

### Loss Functions

The loss function measures how bad the network's predictions are.

<v-click>

| Loss | Formula | Use case |
|------|---------|----------|
| **MSE** | $\frac{1}{N}\sum(y - \tilde{y})^2$ | Regression (continuous output) |
| **BCE** | $-\sum[\tilde{y}\log y + (1-\tilde{y})\log(1-y)]$ | Binary classification |
| **NLL** | $-\sum \log p_{\tilde{y}}$ | Multi-class classification |

</v-click>

<v-click>

**Choosing the right loss**:
- Predicting a continuous value → MSE
- Predicting a probability (0–1) → BCE
- Predicting a class label → NLL (with softmax)

</v-click>

<v-click>

**BCE vs MSE for pixel reconstruction**: BCE penalizes confident wrong predictions more heavily (gradient $\sim 1/\hat{y}$), making it better for binary-ish data like images.

</v-click>

<p align="center">
  <img src="./assets/bce-mse.png" alt="BCE vs MSE" width="500" />
</p>

---

### Gradient Descent and Backpropagation

Training a network = finding parameters that minimize the loss.

<v-click>

**Gradient descent** iterates three steps:

1. **Forward pass**: compute output and loss
2. **Backward pass**: compute gradients $\frac{\partial L}{\partial \theta}$ via **backpropagation**
3. **Update**: $\theta \leftarrow \theta - \alpha \frac{\partial L}{\partial \theta}$

</v-click>

<v-click>

**Backpropagation** applies the chain rule layer by layer:

$$\frac{\partial L}{\partial \mathbf{W}^{in}} = \frac{\partial L}{\partial \mathbf{h}} \cdot \frac{\partial \mathbf{h}}{\partial \mathbf{W}^{in}}$$

PyTorch computes this automatically with `loss.backward()`.

</v-click>

<v-click>

**SGD vs GD**:
- **GD**: compute gradient over ALL training data (accurate but slow)
- **SGD**: compute gradient over a mini-batch (noisy but fast)
- **Adam**: SGD with adaptive learning rate and momentum (default choice)

</v-click>

---

### Convolution: From Moving Average to Neural Networks

Convolution is a fundamental operation that slides a small **kernel** across data, computing a weighted sum at each position.

<v-click>

**Start simple: 1D moving average**

Given a noisy signal $[3, 5, 2, 8, 1, 4]$, a moving average with window size 3 smooths it:

$$\text{output}[i] = \frac{1}{3}(x[i-1] + x[i] + x[i+1])$$

This is a convolution with kernel $f = [\frac{1}{3}, \frac{1}{3}, \frac{1}{3}]$.

</v-click>

<v-click>

**General 1D convolution**:

$$\text{output}[i] = \sum_{k=-K/2}^{K/2} f[k] \cdot x[i+k]$$

The kernel $f$ is a small array of **learnable weights** that slides across the input.

</v-click>


---

### Different Kernels, Different Effects

The **kernel** determines what feature the convolution detects. Same input, different kernels → completely different outputs.

<v-click>

| Kernel | Name | Effect |
|--------|------|--------|
| $[1/3, 1/3, 1/3]$ | Moving average | Smooths / low-pass filter |
| $[-1, 0, 1]$ | Central difference | Detects **edges** (changes) |
| $[1, -2, 1]$ | Second derivative | Detects **curvature** |
| $[0, 1, 0]$ | Identity | Returns original signal |

</v-click>

<v-click>

**Example: edge detection**

```
Input:     [0, 0, 0, 5, 5, 5, 0, 0, 0]    (step function)
Kernel:    [-1, 0, 1]
Output:    [0, 0, 5, 5, 0, -5, -5, 0, 0]   (edges at transitions!)
```

</v-click>

---

### From 1D to 2D: Image Convolutions

The same idea extends to 2D images — the kernel becomes a small matrix.

<v-click>

**2D convolution**:

$$\text{output}(x, y) = \sum_{k_x, k_y} f(k_x, k_y) \cdot I(x+k_x, y+k_y)$$

where $f$ is a $K \times K$ kernel and $I$ is the input image.

</v-click>
---

<v-click>

**Common 2D kernels**:

| Kernel | Size | Effect |
|--------|------|--------|
| $\frac{1}{9}\begin{bmatrix}1&1&1\\1&1&1\\1&1&1\end{bmatrix}$ | 3×3 | Box blur (smooth) |
| $\begin{bmatrix}0&-1&0\\-1&4&-1\\0&-1&0\end{bmatrix}$ | 3×3 | Edge detection (Laplacian) |
| $\begin{bmatrix}-1&0&1\\-2&0&2\\-1&0&1\end{bmatrix}$ | 3×3 | Sobel (vertical edges) |
| $\begin{bmatrix}-1&-2&-1\\0&0&0\\1&2&1\end{bmatrix}$ | 3×3 | Sobel (horizontal edges) |
| $\begin{bmatrix}0&0&0\\0&1&0\\0&0&0\end{bmatrix}$ | 3×3 | Identity |

</v-click>

---

### Convolution Kernels: Visual Demo

Different kernels produce dramatically different outputs on the same image:

<img src="./assets/conv_demo_result.png" alt="Convolution Demo" style="max-width: 80%; margin: 10px auto; display: block;" />


---

### Convolutional Neural Networks (CNNs)

CNNs extend convolution to **multiple learned kernels** across **multiple layers**.

<v-click>

**Key ideas**:
- **Local receptive fields**: each unit looks at a small $K \times K$ patch
- **Multiple channels**: each channel applies a different kernel → detects different features

</v-click>

<v-click>

**Parameters**:

| Parameter | Meaning | Effect |
|-----------|---------|--------|
| $K$ (kernel size) | Size of the sliding window | Larger → bigger receptive field |
| $C_{out}$ (channels) | Number of different kernels | More → richer features |
| Stride | Step size when sliding | Larger → smaller output |
| Padding | Zeros around input | Same padding → output = input size |

</v-click>

---

### CNN Architecture

A typical CNN alternates convolution and pooling layers, then ends with fully connected layers.

<p align="center">
  <img src="./assets/conv-network.png" alt="CNN Architecture" width="400" style="display:inline; margin:0 10px;" />
  <img src="./assets/convnet.png" alt="ConvNet" width="300" style="display:inline; margin:0 10px;" />
</p>

<v-click>

```
Input image
    │
    ▼
Conv + ReLU → Pool → Conv + ReLU → Pool → ... → Flatten → FC → Output
```

</v-click>

<v-click>

**Parameter count comparison** (for 48×64 image, 6 channels, K=7):

| Layer type | Parameters |
|------------|------------|
| Fully connected | $48 \times 64 \times 48 \times 64 \approx 9.4M$ |
| Convolutional | $6 \times 1 \times 7 \times 7 = 294$ |

Weight sharing dramatically reduces parameters!

</v-click>

<p align="center">
  <img src="./assets/weight-sharing.png" alt="Weight Sharing" width="600" />
</p>

<v-click>

**CNNs in neuroscience**: The visual cortex has a similar hierarchical structure:
- V1: edge detectors (like conv filters)
- V4: texture/shape detectors
- IT: object detectors

</v-click>

---

### Encoding vs Decoding Models

Two directions of modeling neural data:

<v-click>

| | Encoding | Decoding |
|---|---|---|
| **Direction** | Stimulus → Neural response | Neural response → Stimulus |
| **Question** | How does the brain represent stimuli? | How much information does the brain contain? |
| **Input** | External stimulus | Neural activity |
| **Output** | Predicted neural response | Predicted stimulus/behavior |
| **Example** | CNN predicting V1 responses | Logistic regression decoding choice |

</v-click>

<v-click>

**Normative encoding model**: Train a deep network on a **behavioral task** (not neural data), then compare its internal representations to neural recordings. If they match, the brain may be solving the same task.

</v-click>

---

### Representational Similarity Analysis (RSA)

RSA compares representations across different systems (brain vs model).

<v-click>

**Step 1: Compute RDM** (Representational Dissimilarity Matrix)

$$\mathbf{M} = 1 - \frac{1}{N}\mathbf{Z}\mathbf{Z}^\top$$

where $\mathbf{Z}$ is the z-scored response matrix. $M_{ss'}$ = dissimilarity between stimuli $s$ and $s'$.

</v-click>

<v-click>

**Step 2: Correlate RDMs**

Compare the RDM from brain data with the RDM from each model layer. Higher correlation → more similar representation.

</v-click>

<v-click>

**Why RSA?**
- Works across different dimensionalities (brain: ~20,000 neurons; model: 10 units)
- Doesn't require one-to-one mapping between neurons and units
- Captures the **structure** of representations, not just single-neuron tuning

</v-click>

---

### Autoencoders

Autoencoders learn **compressed representations** by reconstructing their own input.

<p align="center">
  <img src="./assets/ae-ann-1h.png" alt="Autoencoder Architecture" width="450" />
</p>

<v-click>

**Architecture**:

```
Input (784) → Encoder → Bottleneck (K) → Decoder → Output (784)
```

- **Encoder**: compresses high-dimensional input to low-dimensional latent space
- **Decoder**: reconstructs input from latent representation
- **Loss**: MSE or BCE between input and reconstruction

</v-click>
---

<v-click>

**Bottleneck effect**: Forcing information through a small bottleneck ($K \ll N$) makes the network learn the most important features — a form of **nonlinear PCA**.

</v-click>

<v-click>

**Latent space visualization**: When $K=2$, we can plot each input as a point $(z_1, z_2)$ colored by its class. Well-separated clusters indicate good representations.

</v-click>

<v-click>

**Improving autoencoders**:
- Add more hidden layers (deeper network)
- Use spherical latent space (NormalizeLayer)
- Better weight initialization (Kaiming)
- Choose appropriate activation functions (PReLU for small bottlenecks)

</v-click>

---

### Summary

<div class="grid grid-cols-2 gap-6">
<div class="p-4 bg-gray-800/50 rounded-lg">

### W1D1–W1D2: Foundations

- **Model types**: descriptive, mechanistic, teleological
- **Model fitting**: MSE, MLE, bootstrapping
- **Polynomial regression**: design matrix + OLS

</div>
<div class="p-4 bg-gray-800/50 rounded-lg">

### W1D3: Generalized Linear Models

- **LG GLM**: stimulus filtering
- **Poisson GLM**: spike rate modeling
- **Logistic**: binary classification
- **L1/L2**: regularization

</div>
</div>

<div class="grid grid-cols-2 gap-6 mt-4">
<div class="p-4 bg-gray-800/50 rounded-lg">

### W1D4: Dimensionality Reduction

- **PCA**: eigenvectors of covariance matrix
- **Variance explained**: intrinsic dimensionality
- **Reconstruction**: low-rank approximation
- **Denoising**: signal subspace projection
- **t-SNE**: nonlinear visualization

</div>
<div class="p-4 bg-gray-800/50 rounded-lg">

### W1D5: Deep Learning

- **Feedforward networks**: layers, activations, loss
- **CNNs**: convolution, pooling, weight sharing
- **Encoding/Decoding**: stimulus↔response
- **RSA**: comparing representations
- **Autoencoders**: compression, latent space

</div>
</div>
---

<div class="grid grid-cols-1 gap-6 mt-4">
<div class="p-4 bg-gray-800/50 rounded-lg">

### Key Formulas

$$\hat{\boldsymbol{\theta}} = (X^T X)^{-1} X^T \mathbf{y} \quad \text{(OLS)}$$
$$\mathbf{S} = \mathbf{X}\mathbf{W}, \quad \hat{\mathbf{X}} = \mathbf{S}_{1:K}\mathbf{W}_{1:K}^\top \quad \text{(PCA)}$$
$$\mathbf{h} = \phi(\mathbf{W}^{in}\mathbf{r} + \mathbf{b}^{in}), \quad y = \mathbf{W}^{out}\mathbf{h} + \mathbf{b}^{out} \quad \text{(Neural net)}$$
$$\mathbf{M} = 1 - \frac{1}{N}\mathbf{Z}\mathbf{Z}^\top \quad \text{(RSA)}$$

</div>
</div>
