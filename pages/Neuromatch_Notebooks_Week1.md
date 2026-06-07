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

### Logistic Regression: Binary Classification

Decode binary decisions (left/right) from neural activity.

<v-click>

**Sigmoid function**:
$\sigma(z) = \frac{1}{1 + e^{-z}}$

Maps any real number to $(0, 1)$ — interpretable as probability.

</v-click>

<v-click>

**Model**: $P(y=1 \mid \mathbf{x}) = \sigma(\mathbf{x}^T \boldsymbol{\theta})$

**Loss**: negative log-likelihood (cross-entropy)

$$\mathcal{L} = -\sum_i \left[ y_i \log \hat{p}_i + (1-y_i) \log(1-\hat{p}_i) \right]$$

</v-click>

<v-click>

**Code** (using scikit-learn):

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(penalty=None)
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
```

</v-click>

---

### Regularization: Preventing Overfitting

When features $\gg$ samples, the model memorizes noise (**overfitting**).

<v-click>

**L2 (Ridge)**: adds $\frac{\beta}{2}\|\boldsymbol{\theta}\|^2$ to the loss. Shrinks all weights toward zero.

$$\mathcal{L}_{\text{L2}} = \mathcal{L} + \frac{\beta}{2}\sum_j \theta_j^2$$

</v-click>

<v-click>

**L1 (Lasso)**: adds $\frac{\beta}{2}\sum_j |\theta_j|$. Drives some weights to **exactly zero** (sparsity).

$$\mathcal{L}_{\text{L1}} = \mathcal{L} + \frac{\beta}{2}\sum_j |\theta_j|$$

</v-click>

<v-click>

**Hyperparameter $C = 1/\beta$**: controls regularization strength.

- Small $C$ → strong regularization → simple model
- Large $C$ → weak regularization → complex model

**Cross-validation** to choose $C$:

```python
from sklearn.model_selection import cross_val_score

for C in C_values:
    model = LogisticRegression(C=C, penalty='l2')
    scores = cross_val_score(model, X, y, cv=8)
    print(f"C={C:.4f}, accuracy={np.mean(scores):.3f}")
```

</v-click>

---

### Summary

<div class="grid grid-cols-3 gap-6">
<div class="p-4 bg-gray-800/50 rounded-lg">

### W1D1: Model Types

- **What**: describe data (histograms, ISI)
- **How**: simulate mechanisms (LIF)
- **Why**: explain purpose (entropy)

</div>
<div class="p-4 bg-gray-800/50 rounded-lg">

### W1D2: Model Fitting

- **MSE**: minimize squared error
- **MLE**: maximize likelihood
- **Bootstrap**: quantify uncertainty
- **Polynomial**: design matrix + OLS

</div>
<div class="p-4 bg-gray-800/50 rounded-lg">

### W1D3: GLMs

- **LG GLM**: stimulus filtering
- **Poisson GLM**: spike rate modeling
- **Logistic**: binary classification
- **L1/L2**: regularization

</div>
</div>

<v-click>

$$\hat{\boldsymbol{\theta}} = (X^T X)^{-1} X^T \mathbf{y}$$

This single formula underlies linear regression, LG GLM, and polynomial regression. The GLM extends it with link functions and different noise models.

</v-click>

---
layout: center
---

## Challenge Problem: Optimal Integration

---

### The Problem

Given the ODE and initial condition:

$$\frac{dp}{dt} = -\frac{1}{5}\,p(t) + \sin(t), \qquad p(0) = 1$$

**Goal**: compute $p(10)$ as accurately as possible.

**Constraint**: you have a budget of **exactly 40 function evaluations** — each call to $f(t, p) = -\frac{1}{5}p + \sin(t)$ costs 1 evaluation. You must use all 40.

<v-click>

**Scoring**: your score is $-\log_{10}(\text{error})$. Higher is better.

| Score | Error       | Grade     |
| ----- | ----------- | --------- |
| 1     | 0.1         | Poor      |
| 3     | 0.001       | Good      |
| 5     | 0.00001     | Excellent |
| 7+    | $< 10^{-7}$ | Perfect   |

</v-click>

---

### Part A: Exact Solution (Analytical)

First, derive the exact value of $p(10)$ so you can measure your error.

<v-click>

**Method**: this is a first-order linear ODE. Use the integrating factor $e^{t/5}$:

$$\frac{d}{dt}\!\left[e^{t/5}\,p\right] = e^{t/5}\sin(t)$$

</v-click>

<v-click>

Integrate both sides and apply $p(0)=1$:

$$p(t) = \frac{1}{17}\!\left(\cos t + 4\sin t\right) + \frac{16}{17}\,e^{-t/5}$$

</v-click>

<v-click>

**Verify**: $p(0) = \frac{1}{17}(1 + 0) + \frac{16}{17} = 1$ ✓

$$\boxed{\;p(10) = \frac{1}{17}\!\left(\cos 10 + 4\sin 10\right) + \frac{16}{17}\,e^{-2}\;}$$

</v-click>

<v-click>

**Numerically**: $p(10) \approx -0.05264 + 0.12618 = 0.07355$

</v-click>

---

**Code to compute it**:

```python
import numpy as np
def p_exact(t):
    return (np.cos(t) + 4*np.sin(t)) / 17 + 16/17 * np.exp(-t/5)
print(f"p(10) = {p_exact(10):.10f}")  # 0.073546...
```

---

### Part B: Uniform Euler (Baseline)

Use standard Euler with uniform step size. With 40 evaluations budget and Euler's 1 evaluation per step:

$$\Delta t = \frac{10}{40} = 0.25$$

<v-click>

**Step-by-step trace** (first 3 steps):

| Step | $t$ | $p(t)$ | $f(t,p) = -p/5 + \sin t$ | $p + \Delta t \cdot f$ |
| ---- | --- | ------ | ------------------------ | ---------------------- |
| 0    | 0.00| 1.0000 | $-0.200 + 0.000 = -0.200$ | $1.000 + 0.25(-0.200) = 0.950$ |
| 1    | 0.25| 0.9500 | $-0.190 + 0.247 = +0.057$ | $0.950 + 0.25(0.057) = 0.964$ |
| 2    | 0.50| 0.9643 | $-0.193 + 0.479 = +0.286$ | $0.964 + 0.25(0.286) = 1.036$ |

Each step: compute slope at current point → extend linearly by $\Delta t$ → that's the new value.

</v-click>

<v-click>

With uniform Euler ($\Delta t = 0.25$, 40 evaluations), error $\approx 0.0039$. Score $\approx 2.4$. **Can you do better?**

</v-click>

---

### Part C: Your Strategy (40 Evaluations)

You have 40 evaluations. Design your own integration strategy. Here are ideas to explore:

<div class="grid grid-cols-2 gap-8">
<div>

**Idea 1: Non-uniform steps**

The solution changes rapidly near $t=0$ (exponential decay dominates) and slowly near $t=10$ (transient died out). Use smaller $\Delta t$ early, larger later.

<v-click>

```python
# Geometric step sequence: dt_i = dt_0 * r^i
# Choose dt_0 and r so that sum(dt_i) = 10
# and sum(evaluations) = 40
```

</v-click>

</div>
<div>

**Idea 2: Richardson extrapolation**

Run Euler twice at different step sizes, then combine:

$$p_{\text{better}} = \frac{2^p \cdot p_{\Delta t/2} - p_{\Delta t}}{2^p - 1}$$

where $p$ is the order of the method ($p = 1$ for Euler, so $2^1 = 2$).

This cancels the $O(\Delta t)$ error term, giving $O(\Delta t^2)$ accuracy — for free!

<v-click>

Cost: 40 evaluations split into 20 coarse + 20 fine, but you only get the final value.

</v-click>

</div>
</div>

---

**Idea 3: RK2 with non-uniform steps**

RK2 uses 2 evaluations per step → 20 steps with non-uniform spacing. Combine with adaptive spacing for maximum effect.

**Idea 4: Multistep (Adams-Bashforth)**

After a few startup steps, use past values to predict the next — 1 evaluation per step but higher-order accuracy:

$$p_{n+1} = p_n + \Delta t\!\left(\frac{3}{2}f_n - \frac{1}{2}f_{n-1}\right) \quad \text{(AB2, order 2, 1 eval/step)}$$

---

### Hints & Extensions

**Hints**:

1. The solution has two timescales: fast exponential decay ($\tau = 5$) and slow oscillation ($T = 2\pi$). Put steps where the action is.
2. For Richardson extrapolation: run Euler with 20 steps ($\Delta t = 0.5$) and 20 steps ($\Delta t$ halved to 0.25 gives 40 steps — but you only have 40 total. Instead, run 20 coarse + 20 fine, but the fine uses the coarse as a starting point after $t=5$.)
3. For Adams-Bashforth: you need a "startup" method (Euler or RK2) for the first step, then AB2 for the rest — 1 eval/step after startup.

**Extensions** (if you have time):

- **Adaptive step size**: after each step, estimate the local error (e.g., take one full step vs two half steps). If error is large, shrink $\Delta t$; if small, grow it. Budget 40 evaluations but choose step sizes *on the fly*.
- **Compare to RK4**: with 40 evaluations, RK4 gets only 10 steps ($\Delta t = 1$). Is it still more accurate than 40 Euler steps with optimal spacing?
- **Neural application**: the same ODE structure $\dot{p} = -p/\tau + I(t)$ is a synaptic current model with time constant $\tau = 5$ms and sinusoidal input. What does $p(t)$ represent biologically?
