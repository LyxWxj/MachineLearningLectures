---
title: Neuromatch Notebooks - Week 3
---

# Neuromatch Notebooks — Week 3

Bayesian Decision · Latent Dynamics · Reinforcement Learning · Network Causality

---

## Overview

Week 3 focuses on **inference, learning, and causality** — from Bayesian decision-making to reinforcement learning to causal network estimation:

| Day      | Topic              | Core Skills                                        |
| -------- | ------------------ | -------------------------------------------------- |
| **W3D2** | Bayesian Decision  | Bayes' rule, Gaussian posterior, utility functions |
| **W3D3** | Latent Dynamics    | SPRT / Drift-diffusion model, EM algorithm         |
| **W3D4** | Reinforcement Learning | Temporal difference learning, Q-learning, Dyna-Q |
| **W3D5** | Network Causality  | Interventions, regression, instrumental variable estimation |

**Overarching Theme**: How do agents infer latent states, learn from experience, and estimate causal structure in complex environments?

---

## W3D2: Bayesian Decision

---

### Section 1: Bayes' Rule — Why Do We Need It?

In the real world, we can never directly observe the state of interest — we can only see **indirect data**. For example:

- A doctor sees test results (data) and wants to infer whether a patient has a disease (latent state)
- A fisher sees ripples on the water (data) and wants to judge whether there are fish below (latent state)

**Core question**: How can we **rationally update** our beliefs about a latent state given observed data?

**Joint probability factorization** — the starting point for all inference:

$$
p(x, y) = p(x|y) \cdot p(y) = p(y|x) \cdot p(x)
$$

This equation tells us: the joint relationship between two variables can be decomposed as "conditional x marginal." This is the bridge connecting **observations** and **latent states**.

**Marginalization** — when some variables are "nuisance variables," we need to "integrate them out":

$$
p(x) = \sum_y p(x, y) = \sum_y p(x|y) \cdot p(y)
$$

**Why marginalization?** Because ultimately we care about the total probability of the data $p(m)$, not which intermediate state $s$ it passed through.

---

### Bayes' Rule

**Bayes' rule** combines the above tools to perform the inversion from "data to belief":

$$
\boxed{p(s|m) = \frac{p(m|s) \cdot p(s)}{p(m)} = \frac{p(m|s) \cdot p(s)}{\sum_{s'} p(m|s') \cdot p(s')}}
$$

| Term         | Name         | Why we need it                                       |
| ------------ | ------------ | ---------------------------------------------------- |
| $p(s)$       | **Prior**    | Our existing knowledge about the state before seeing data (e.g., prevalence from epidemiological data) |
| $p(m\mid s)$ | **Likelihood** | How plausible is data $m$ if the state is $s$? (e.g., sensitivity of a detection instrument) |
| $p(s\mid m)$ | **Posterior** | Our **updated belief** about the state after seeing data — this is what we actually want |
| $p(m)$       | **Evidence** | Normalization constant ensuring the posterior is a valid probability distribution |

**Core significance**: Bayes' rule provides a **principled framework** for fusing prior knowledge with new observations into an updated belief. It is not an approximation or heuristic — it is a direct consequence of probability theory.

---

### Section 2: Why Compare Posterior Odds?

Bayes' rule gives us the value of $p(s=+1|m)$ — but this absolute number alone is not sufficient.

**Problem**: Suppose you computed $p(s=+1|m) = 0.03$. Does this mean $s=+1$ is likely?

**Answer: not necessarily.** It depends on how large the posterior of the alternative $s=-1$ is. If $p(s=-1|m) = 0.01$, then $s=+1$ is actually more likely! Looking at $p(s=+1|m) = 0.03$ alone does not allow a judgment.

**Root cause**: The normalization constant $p(m)$ in Bayes' rule changes with the data. $p(s=+1|m) = 0.03$ could mean "s=+1 is very unlikely" (if $p(s=-1|m) = 0.97$) or "s=+1 is relatively more likely" (if $p(s=-1|m) = 0.01$).

**Solution**: Directly compare the posteriors of the two hypotheses — look at the **ratio** (odds) rather than absolute probabilities.

$$
\text{Posterior odds} = \frac{p(s=+1|m)}{p(s=-1|m)}
$$

- Odds > 1 → $s=+1$ is more likely
- Odds < 1 → $s=-1$ is more likely
- Odds = 1 → both are equally likely

---

**A key mathematical simplification**: When taking the ratio, the normalization constant $p(m)$ **cancels out** exactly:

$$
\frac{p(s=+1|m)}{p(s=-1|m)} = \frac{p(m|s=+1) \cdot p(s=+1) / \cancel{p(m)}}{p(m|s=-1) \cdot p(s=-1) / \cancel{p(m)}} = \frac{p(m|s=+1)}{p(m|s=-1)} \cdot \frac{p(s=+1)}{p(s=-1)}
$$

This means we **do not need to compute** the troublesome marginal probability $p(m) = \sum_{s'} p(m|s')p(s')$ — the likelihood ratio and prior odds suffice.

---

### Why Take the Logarithm Further?

Using Bayes' rule directly with multiple observations encounters two problems; the **log-odds** form solves both simultaneously.

**Problem 1: numerical underflow.** Multiplying many small probabilities causes floating-point underflow. Taking the logarithm converts multiplication to addition, avoiding this issue.

**Problem 2: interpretability.** In the raw odds space, the effect of each piece of evidence is **multiplicative** (multiplying by the likelihood ratio). After taking the logarithm, it becomes **additive** — each piece of evidence contributes an independent additive term, and we can directly see "how far this piece of evidence pushed things."

**Decomposition of log-odds**:

$$
\underbrace{\log \frac{p(s=+1|m)}{p(s=-1|m)}}_{\text{log posterior odds}} = \underbrace{\log \frac{p(m|s=+1)}{p(m|s=-1)}}_{\text{log likelihood ratio}} + \underbrace{\log \frac{p(s=+1)}{p(s=-1)}}_{\text{log prior odds}}
$$

**Core insight**: In log-odds space, evidence accumulates in an **additive fashion**. Each new observation shifts the log posterior by one log-likelihood-ratio amount. This is why "more data → more certainty" is mathematically natural.

---

### Section 3: From Beliefs to Action — Utility Functions

Bayesian inference tells us what we should **believe**, but knowing probabilities alone is not enough — we also need to **make decisions**.

**Problem**: The posterior probability $P(\text{good location}|\text{data}) = 0.7$ — what does it mean? Should we go fish at that location?

**The answer depends on consequences**: If going to the good location earns 100 yuan and going to the bad location loses 50 yuan, then 0.7 probability may be worth the risk; but if going to the bad location endangers your life, 0.7 may not be safe enough.

**Expected utility**: Quantifying the "expected benefit" of each action:

$$
\mathbb{E}[U(a)] = \sum_s U(s, a) \cdot P(s)
$$

where $U(s, a)$ is the **utility** of taking action $a$ in state $s$ (a numerical quantification of benefit/cost).

**Optimal decision**: Choose the action that maximizes expected utility:

$$
a^* = \arg\max_a \sum_s U(s, a) \cdot P(s)
$$

**Core significance**: Bayesian decision theory unifies **inference** (what to believe) and **decision-making** (what to do) in a single framework. First use Bayes' rule to obtain the posterior, then use the utility function to make the optimal choice.

---

### Section 4: Why Are Gaussian Distributions So Important?

Real-world continuous variables (such as neural firing rates, measurement errors, market prices) are typically not binary but continuous. We need a distribution that can describe continuous variables while being mathematically tractable.

**Gaussian (normal) distributions** are widely used for three reasons:

1. **Central limit theorem**: The sum of many independent random variables tends toward a Gaussian — it is the "natural default choice"
2. **Maximum entropy principle**: Given constraints on mean and variance, the Gaussian distribution has maximum entropy — it is the most "conservative" assumption
3. **Conjugacy**: Gaussian likelihood x Gaussian prior = Gaussian posterior — mathematically closed, no approximation needed

**Probability density function**:

$$
\mathcal{N}(x; \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)
$$

**Precision** (inverse of variance): $\kappa = \frac{1}{\sigma^2}$ — measures the "sharpness" of the distribution. Higher precision means more confidence in our estimate.

---

### Gaussian Product: How Does Information Combine?

When we have two pieces of evidence about the same variable (such as prior knowledge and a new observation), how do we **fuse** them?

**Product of two Gaussians** (likelihood x prior):

$$
\mathcal{N}(x; \mu_1, \sigma_1^2) \cdot \mathcal{N}(x; \mu_2, \sigma_2^2) \propto \mathcal{N}(x; \mu_3, \sigma_3^2)
$$

where:

$$
\mu_3 = \frac{\sigma_2^{-2} \mu_1 + \sigma_1^{-2} \mu_2}{\sigma_1^{-2} + \sigma_2^{-2}}, \qquad \sigma_3^{-2} = \sigma_1^{-2} + \sigma_2^{-2}
$$

**Why should we care?**

- The **posterior mean** is a **precision-weighted average** of the prior mean and likelihood mean — whoever is more certain (higher precision) has more influence
- The **posterior precision** is the sum of the two precisions — information always **accumulates**, and combining two estimates is never worse than either alone
- This gives an **exact analytical solution** for "evidence fusion" without numerical computation

**Intuition**: Imagine two people separately estimating the length of an object. One says "about 10 cm, not very sure" (large variance), the other says "about 12 cm, quite confident" (small variance). The optimal estimate will be closer to 12 cm, and more certain than either individual estimate.

---

### Section 5: 2D Gaussian — Capturing Relationships Between Variables

When we care about two variables simultaneously (e.g., the activity of neuron $i$ and neuron $j$), they may not be independent. The **covariance matrix** is the mathematical tool for describing such relationships.

**Joint distribution**:

$$
p(\mathbf{x}) = \mathcal{N}(\mathbf{x}; \boldsymbol{\mu}, \boldsymbol{\Sigma})
$$

$$
\boldsymbol{\Sigma} = \begin{bmatrix} \sigma_1^2 & \rho \sigma_1 \sigma_2 \\ \rho \sigma_1 \sigma_2 & \sigma_2^2 \end{bmatrix}
$$

where $\rho = \frac{\text{Cov}(X_1, X_2)}{\sigma_1 \sigma_2}$ is the **correlation coefficient**, measuring the degree to which the two variables "co-vary."

**Marginalization** — when we care about only one variable, "integrate out" the other:

$$
p(x_1) = \int p(x_1, x_2) \, dx_2 = \mathcal{N}(x_1; \mu_1, \sigma_1^2)
$$

**Conditional distribution** — when we **observe** the value of one variable, our belief about the other:

$$
p(x_1 | x_2) = \mathcal{N}\left(x_1; \mu_1 + \rho\frac{\sigma_1}{\sigma_2}(x_2 - \mu_2), \, \sigma_1^2(1-\rho^2)\right)
$$

---

### Section 6: Loss Functions — "Best" Depends on What You Care About

Given a posterior distribution, we still need a **point estimate** (a single numerical answer). But the posterior is a distribution — how do we extract a single number from it?

**Key question**: Different "costs of error" lead to different optimal estimators.

**Mean Squared Error (MSE)**:

$$
L(\hat{\mu}, \mu) = (\hat{\mu} - \mu)^2 \quad \Rightarrow \quad \hat{\mu}^* = \text{posterior mean } \mathbb{E}[\mu \mid m]
$$

MSE penalizes large errors heavily (quadratic); the posterior mean balances all possible values.

**Absolute Error**:

$$
L(\hat{\mu}, \mu) = |\hat{\mu} - \mu| \quad \Rightarrow \quad \hat{\mu}^* = \text{posterior median}
$$

Absolute error treats all error sizes equally; the median is the most robust measure of central tendency.

**0-1 Loss**:

$$
L(\hat{\mu}, \mu) = \mathbb{1}[\hat{\mu} \neq \mu] \quad \Rightarrow \quad \hat{\mu}^* = \text{posterior mode (MAP)}
$$

Only cares about "correct or not," not "how far off," so it picks the most probable value.

---

**Expected loss** (risk):

$$
R(\hat{\mu}) = \int L(\hat{\mu}, \mu) \, p(\mu|m) \, d\mu
$$

**Bayesian estimator**: $\hat{\mu}^* = \arg\min_{\hat{\mu}} R(\hat{\mu})$

**Core insight**: The "best" estimate is not an absolute concept — it depends on what you care about (the loss function). In neuroscience, different experimental objectives (e.g., maximizing accuracy vs. minimizing reaction time) lead to different optimal strategies.

---

## W3D3: Latent Dynamics

---

### Section 1: Sequential Probability Ratio Test (SPRT) — When to Stop Observing?

In many scenarios, data does not all arrive at once but **accumulates gradually**. A key practical question is: **when should I stop collecting data and make a decision?**

**Problem scenarios**:

- A doctor performs a series of tests, updating diagnostic confidence after each one — when can they conclude?
- An animal continuously collects clues about food locations during foraging — when should it act?

**Core idea of SPRT** (Wald, 1945): Maintain a **log-likelihood ratio** statistic, and stop and decide when it crosses a threshold.

**Why likelihood ratio?** Facing two hypotheses $s=+1$ and $s=-1$, **all** information in the data relevant to distinguishing them is captured by the likelihood ratio — it is a sufficient statistic. No other statistic can distinguish the two hypotheses more efficiently.

**Why take the logarithm?** The raw likelihood ratio is in product form; taking the log converts it to a sum — each step only requires adding the latest contribution, which is an $O(1)$ incremental update, and avoids numerical underflow.

**Log-likelihood ratio** (LLR): After $T$ observations

$$
L_T = \log \frac{p(m_{1:T}|s=+1)}{p(m_{1:T}|s=-1)} = \sum_{t=1}^{T} \underbrace{\log \frac{p(m_t|s=+1)}{p(m_t|s=-1)}}_{\Delta_t}
$$

**Recursive form**: $L_T = L_{T-1} + \Delta_t$ — each step only adds the latest evidence contribution, without recomputing the entire sequence.

---

### A Concrete Example: Step-by-Step Diagnosis

Suppose a doctor is diagnosing a disease, obtaining an observation $m_t$ after each test.

**Known information**:

- $s=+1$: diseased; $s=-1$: healthy
- Test A is positive with 80% probability when diseased, and 10% when healthy
- Test B is positive with 90% probability when diseased, and 30% when healthy

**First test A result is positive** ($m_1 = +$):

$$
\Delta_1 = \log\frac{p(m_1=+|s=+1)}{p(m_1=+|s=-1)} = \log\frac{0.8}{0.1} = \log 8 \approx 2.08
$$

$\Delta_1 > 0$: this observation **supports disease**. The log-likelihood ratio jumps from $L_0 = 0$ to $L_1 \approx 2.08$.

---

**Second test B result is also positive** ($m_2 = +$):

$$
\Delta_2 = \log\frac{0.9}{0.3} = \log 3 \approx 1.10
$$

$L_2 = L_1 + \Delta_2 \approx 3.18$ — evidence accumulates further, leaning more toward disease.

**If test B result is negative** ($m_2 = -$):

$$
\Delta_2 = \log\frac{p(m_2=-|s=+1)}{p(m_2=-|s=-1)} = \log\frac{0.1}{0.7} \approx -1.95
$$

$L_2 \approx 2.08 + (-1.95) = 0.13$ — this test **supports health**, pulling the evidence back close to zero.

**Core significance**: The sign of $L_T$ tells us "which side the evidence currently favors," and the magnitude tells us "how strongly it favors that side." Each $\Delta_t$ is the **push of this piece of evidence** — positive pushes toward $s=+1$, negative pushes toward $s=-1$, and the magnitude reflects how "persuasive" this evidence is.

---

### SPRT: Likelihood Ratio vs. Posterior Odds

**Note: SPRT accumulates the likelihood ratio, not posterior odds.** Recall the decomposition of Bayes' rule from W3D2:

$$
\underbrace{\frac{p(s=+1|m)}{p(s=-1|m)}}_{\text{posterior odds}} = \underbrace{\frac{p(m|s=+1)}{p(m|s=-1)}}_{\text{likelihood ratio}} \times \underbrace{\frac{p(s=+1)}{p(s=-1)}}_{\text{prior odds}}
$$

The prior odds are determined and fixed before the experiment begins, and do not need to be recalculated at each step. SPRT only needs to accumulate the data evidence (likelihood ratio), and then uses thresholds to absorb the effects of the prior and target error rates.

- $m_{1:T} = (m_1, m_2, \ldots, m_T)$: **all observed data** from time 1 to $T$ (observation sequence)
- $p(m_{1:T}\mid s)$: given state $s$, the probability of the **entire observation sequence** (joint likelihood)
- $p(m_t\mid s)$: given state $s$, the probability of a **single observation** $m_t$ (single-step likelihood)

**Why can the joint likelihood decompose into a product of single-step likelihoods?** Key assumption: **given the latent state $s$, observations are conditionally independent** — i.e., $m_t$ depends only on the current state $s$, not on previous observations.

$$
p(m_{1:T}|s) = \prod_{t=1}^{T} p(m_t|s)
$$

Taking the log converts the product to a sum — this is where the summation comes from. If there is temporal dependence between observations, this decomposition no longer holds.

---

### SPRT Decision Rule

**Decision rule**: Stop when $L_T$ crosses a threshold:

$$
\begin{cases} L_T \geq \theta_A & \Rightarrow \text{accept } s = +1 \\ L_T \leq \theta_B & \Rightarrow \text{accept } s = -1 \\ \theta_B < L_T < \theta_A & \Rightarrow \text{continue sampling} \end{cases}
$$

**Relationship between thresholds and target error rate $\alpha$**:

$$
\theta_A = \log\frac{1-\alpha}{\alpha}, \qquad \theta_B = \log\frac{\alpha}{1-\alpha}
$$

**Where did the prior go?** The above thresholds implicitly assume **prior odds of 1** ($p(s=+1) = p(s=-1) = 0.5$). If the prior is unequal, the threshold needs to be adjusted by adding $\log\frac{p(s=+1)}{p(s=-1)}$.

Essentially, SPRT's stopping rule is equivalent to checking whether the **posterior odds** have crossed a threshold determined by the target error rate — the prior is "absorbed" into the threshold.

---

**Why is SPRT optimal?**

Wald and Wolfowitz proved that under given error rate constraints, SPRT has the **smallest average sample size**. In other words, it achieves a predetermined accuracy with the least data — this is optimal in an information-theoretic sense.

**Comparison with fixed-sample tests**:

| Method         | Approach                                          | Disadvantage                                      |
| -------------- | ------------------------------------------------- | ------------------------------------------------- |
| Fixed sample   | Predetermine sample size $N$, collect all, then decide | May collect too much data, or not enough          |
| SPRT           | Judge while collecting, stop when sufficient      | Need to maintain cumulative statistic              |

---

### Section 2: Drift-Diffusion Model (DDM) — Evidence Accumulation in Continuous Time

SPRT is in discrete time, but the brain's decision process is continuous. The **drift-diffusion model** is the natural extension of SPRT to continuous time, and is one of the most successful models in neuroscience for describing decision behavior.

**Decomposition of each evidence step**:

$$
\Delta_t = b + c \cdot \varepsilon_t
$$

where:

- $b = \frac{2\mu^2}{\sigma^2}$ — **drift rate** (signal, deterministic component): represents the "average push" of evidence
- $c = \frac{2\mu}{\sigma}$ — **diffusion coefficient** (noise scaling factor): represents the uncertainty of evidence
- $\varepsilon_t \sim \mathcal{N}(0, 1)$ — standard normal noise

**Cumulative evidence** follows a random walk with drift:

$$
L_T \sim \mathcal{N}(b \cdot T, \, c^2 \cdot T)
$$

Mean grows linearly (signal accumulation), variance also grows linearly (noise accumulation). The **signal-to-noise ratio** improves as $\sqrt{T}$ — this is the mathematical essence of "more data → better decisions."

---

### Speed-Accuracy Tradeoff in DDM

**Speed-accuracy tradeoff**: The choice of threshold determines whether decisions are "fast" or "accurate."

- **High threshold**: Need more evidence before deciding → more accurate but slower
- **Low threshold**: Decide with less evidence → faster but more error-prone

**Analytical accuracy**:

$$
P(\text{correct}) = \frac{1}{2} + \frac{1}{2} \operatorname{erf}\left(\frac{\mu}{\sqrt{2} \cdot \sigma / \sqrt{T}}\right)
$$

**Neuroscience connection**:

- Neural activity in brain decision areas (e.g., LIP) indeed shows DDM-like evidence accumulation patterns
- Under different task conditions (emphasizing speed vs. accuracy), subject behavior can be well fit by adjusting DDM threshold parameters
- This provides strong evidence that "the brain performs optimal sequential testing"

---

### Section 3: Expectation-Maximization (EM) Algorithm — What to Do with Latent Variables?

Many real-world models contain **variables that cannot be directly observed** (latent variables). For example:

- Which neural population is active at a given moment? (latent state)
- Which mixture of Gaussians generated the data? (mixture components)

**Problem**: If the latent variables were known, maximum likelihood estimation would be easy; but when latent variables are unknown, direct optimization of the likelihood function is usually intractable.

**Core idea of the EM algorithm** (Dempster et al., 1977): Since latent variables are unknown, **guess** their values, then optimize parameters based on the guess; then use the optimized parameters to **update the guess**; and alternate.

**Two alternating steps**:

1. **E-step** (Expectation step): Under current parameters $\theta^{(t)}$, compute the **posterior expectation** of latent variables — i.e., "given current parameters, what are the latent variables most likely to be?"
2. **M-step** (Maximization step): Fix the expectations of latent variables, update parameters to maximize expected log-likelihood — i.e., "if the latent variables are as the E-step guessed, what are the optimal parameters?"

**Monotonicity guarantee**:

$$
\mathcal{L}(\theta^{(t+1)}) \geq \mathcal{L}(\theta^{(t)})
$$

Each step never makes the likelihood worse — this guarantees the algorithm converges (at least to a local optimum).

---

### EM: Forward-Backward Algorithm

For Hidden Markov Models, the E-step needs to compute "the probability of being in state $i$ at time $t$, given all observed data." This seemingly requires enumerating all possible state sequences — exponential complexity. The **forward-backward algorithm** uses dynamic programming to reduce this to linear complexity.

**Forward probability** $\alpha_i(t) = p(y_{1:t}, x_t = i)$ — "the sum of probabilities of all paths from the start to time $t$ and being in state $i$":

$$
\alpha_i(t) = p(y_t | x_t = i) \sum_j A_{ji} \cdot \alpha_j(t-1)
$$

Meaning: to reach state $i$, the previous step must have been in some state $j$, then transitioned to $i$, while observing $y_t$.

**Backward probability** $\beta_i(t) = p(y_{t+1:T} | x_t = i)$ — "the sum of probabilities of all paths from state $i$ at time $t$ to generate the remaining observations":

$$
\beta_i(t) = \sum_j p(y_{t+1} | x_{t+1} = j) \cdot \beta_j(t+1) \cdot A_{ij}
$$

---

### EM: Posterior Marginal Probabilities

The forward probability $\alpha_i(t)$ and backward probability $\beta_i(t)$ each converge on time $t$ from "past" and "future" directions respectively. Multiplying them gives the **posterior marginal probability**:

$$
\gamma_i(t) = p(x_t = i | y_{1:T}) = \frac{\alpha_i(t) \cdot \beta_i(t)}{p(Y_{1:T})}
$$

- $\alpha_i(t)$: accounts for "how we got here from the past"
- $\beta_i(t)$: accounts for "how we proceed into the future"
- Their product: the **total weight** of this path

**Pairwise marginal probability** (joint posterior at adjacent times):

$$
\xi_{ij}(t) = p(x_t = i, x_{t+1} = j | y_{1:T})
$$

$\gamma_i(t)$ and $\xi_{ij}(t)$ are the outputs of the E-step — they are all the "expected counts" needed by the M-step.

---

### EM: M-Step Updates — Why Are They So Simple?

Given the expectations computed in the E-step, the M-step has closed-form solutions, and the form is surprisingly intuitive.

**Transition matrix update**:

$$
\hat{A}_{ij} = \frac{\sum_t \xi_{ij}(t)}{\sum_t \gamma_i(t)}
$$

Numerator = "expected number of transitions from $i$ to $j$", denominator = "expected number of times in state $i$."

**Observation parameter update** (e.g., Poisson firing rate):

$$
\hat{\lambda}_i = \frac{\sum_t \gamma_i(t) \cdot y_t}{\sum_t \gamma_i(t) \cdot \Delta t}
$$

Numerator = "expected total firing in state $i$", denominator = "expected total time in state $i$."

**Initial state probability**:

$$
\hat{\psi}_i = \frac{1}{N_{\text{trials}}} \sum_{\text{trials}} \gamma_i(1)
$$

---

## W3D4: Reinforcement Learning

---

### Section 1: Temporal Difference (TD) Learning — How to Predict the Future?

Animals and humans can learn to predict future rewards. **TD learning** provides a mathematical description of this learning process.

**Core question**: Given the current state $s_t$, what is the cumulative future reward (**return**)?

**Return** (discounted cumulative reward):

$$
G_t = \sum_{k=0}^{\infty} \gamma^k \cdot r_{t+k+1} = r_{t+1} + \gamma \cdot G_{t+1}
$$

where $\gamma \in [0,1)$ is the **discount factor** — future rewards are weighted less than immediate rewards. The closer $\gamma$ is to 1, the more "patient" the agent is.

**State value function** — expected return:

$$
V^\pi(s) = \mathbb{E}_\pi[G_t | s_t = s]
$$

This is the total reward one can expect when in state $s$ following policy $\pi$.

---

### TD Error: The Driving Force of Learning

Given the value function $V$, we can measure how good our predictions are — the **TD error**:

$$
\delta_t = r_{t+1} + \gamma \cdot V(s_{t+1}) - V(s_t)
$$

- $r_{t+1} + \gamma V(s_{t+1})$: **what was actually obtained** (immediate reward + estimate for next state)
- $V(s_t)$: **what was expected** (current estimate of state value)
- $\delta_t$: the difference between the two — "how much better/worse the outcome was than I expected"

**Why is it called "temporal difference"?** Because the error is computed between **adjacent time steps**, rather than waiting for the entire return to be realized. This allows learning to proceed **online, step by step**.

---

### A Concrete Example

Consider a simple path: $A \to B \to C$ (reaching $C$ gives reward $r=+4$). Discount factor $\gamma = 0.9$, learning rate $\alpha = 0.5$.

**Initial value estimates** (all zero): $V(A)=0, \; V(B)=0, \; V(C)=0$

**Step 1**: Walk from $A$ to $B$, receive reward $r=0$ (no reward along the way)

$$
\delta_1 = r + \gamma V(B) - V(A) = 0 + 0.9 \times 0 - 0 = 0
$$

$$
V(A) \leftarrow 0 + 0.5 \times 0 = 0 \quad \text{（no change, since no useful information seen yet）}
$$

**Step 2**: Walk from $B$ to $C$, receive reward $r=+4$

$$
\delta_2 = r + \gamma V(C) - V(B) = 4 + 0.9 \times 0 - 0 = 4
$$

$$
V(B) \leftarrow 0 + 0.5 \times 4 = 2
$$

$V(B)$ updated from 0 to 2 — because starting from $B$ did yield a reward.

---

**Here is the key**: Now go back to $A$ and walk $A \to B$ again ($r=0$):

$$
\delta = r + \gamma V(B) - V(A) = 0 + 0.9 \times 2 - 0 = 1.8
$$

$$
V(A) \leftarrow 0 + 0.5 \times 1.8 = 0.9
$$

$V(A)$ updated from 0 to 0.9 — even though no reward was ever directly received from $A$, the value information from $B$ **propagated backward** to $A$. This is the core mechanism of "temporal difference."

---

### TD Learning Update Rule

**Update rule**:

$$
V(s_t) \leftarrow V(s_t) + \alpha \cdot \delta_t
$$

$$
V(s_t) \leftarrow V(s_t) + \alpha \left[ r_{t+1} + \gamma V(s_{t+1}) - V(s_t) \right]
$$

where $\alpha \in (0, 1]$ is the **learning rate** — controlling the step size of each update.

**Interpretation**:

- $\delta_t > 0$: outcome better than expected → increase $V(s_t)$ ("this state is better than I thought")
- $\delta_t < 0$: outcome worse than expected → decrease $V(s_t)$ ("this state is worse than I thought")
- $\delta_t = 0$: outcome matches expectations → no change ("everything went as expected")

---

**Core insight — bootstrapping**: TD learning uses the current estimate $V(s_{t+1})$ as part of the target, rather than waiting for the actual return $G_t$. This means:

- No need to wait until "the game is over" to learn
- Can learn from incomplete experiences
- The cost is that the target itself is an estimate and may be biased

**Neuroscience connection**: The firing patterns of dopamine neurons closely match TD error — firing increases when outcomes are better than expected, decreases when worse than expected, and stays the same when matching expectations. This provides neural-level evidence that "the brain does TD learning."

---

### Section 2: Multi-Armed Bandits — The Exploration-Exploitation Dilemma

The simplest action selection problem: $K$ "slot machines" (actions), each with an unknown reward distribution. You must trade off between **trying new actions** (exploration) and **repeating the best action** (exploitation).

**Why explore?** If you have never tried a particular action, you will never know if it is better than your current best. But exploration has a cost — you may lose rewards in the process.

**Action value function**:

$$
q(a) = \mathbb{E}[r_t | a_t = a]
$$

This is the "true quality" of action $a$ — but we do not know it; we can only estimate it through trial.

**$\varepsilon$-greedy policy** — a simple but effective balance:

$$
a_t = \begin{cases} \arg\max_a q_t(a) & \text{with probability } 1 - \varepsilon \\ \text{random action} & \text{with probability } \varepsilon \end{cases}
$$

- $\varepsilon = 0$: pure greedy, never explore — may get stuck on a suboptimal action
- $\varepsilon = 1$: pure explore, never exploit — wastes what has already been learned
- $\varepsilon = 0.1$: most of the time exploit the known best, occasionally try new actions

---

### Multi-Armed Bandit Learning

**Running mean update** (sample average method):

$$
q_{t+1}(a) = q_t(a) + \frac{1}{n_t}\left[r_t - q_t(a)\right]
$$

**Why use this form instead of directly averaging?** Because it allows **incremental updates** — no need to store all historical data, just maintain the current estimate and trial count.

**Generalized update** (fixed learning rate):

$$
q_{t+1}(a) = q_t(a) + \alpha\left[r_t - q_t(a)\right]
$$

- Sample average ($\alpha = 1/n_t$): optimal for stationary environments, equal weight to all historical data
- Fixed $\alpha$ (e.g., $\alpha = 0.1$): better for non-stationary environments, more weight on recent data

**Effects of $\varepsilon$ and $\alpha$**:

- Large $\varepsilon$: more exploration, fast initial learning but unstable later
- Small $\varepsilon$: less exploration, may miss good actions initially but more stable later
- Large $\alpha$: fast learning but sensitive to noise
- Small $\alpha$: slow learning but smoother

---

### Section 3: Q-Learning — From Prediction to Control

TD learning solves the **prediction** problem (estimate value given a policy). But a true agent also needs to **choose actions**. Q-learning extends TD learning to the **control** problem.

**Action value function** (Q-function) — simultaneously describes "how good a state is" and "how good an action is":

$$
Q^\pi(s, a) = \mathbb{E}_\pi[G_t | s_t = s, a_t = a]
$$

If we know the optimal $Q^*$, the optimal policy is: in each state, choose the action with the highest $Q$ value.

**Q-learning update** (Watkins & Dayan, 1992):

$$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_{t+1} + \gamma \max_a Q(s_{t+1}, a) - Q(s_t, a_t) \right]
$$

**Key property — off-policy**: The update target uses $\max_a Q(s_{t+1}, a)$ (greedy action), **not** the action actually taken. This means:

- The **target policy** is greedy (we want to learn the optimal policy)
- The **behavior policy** can be exploratory ($\varepsilon$-greedy)
- Learning and exploration can be **decoupled** — explore while learning the optimal

**Why is this important?** Off-policy learning allows an agent to learn from **others' experiences**, or from **historical data**, without needing to explore itself.

---

### SARSA: On-Policy TD Control — A More Cautious Learner

**SARSA update**:

$$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_{t+1} + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t) \right]
$$

Note: $a_{t+1}$ is the **action actually to be taken**, not the optimal action.

**Comparison**:

| Aspect               | Q-learning                               | SARSA                                    |
| -------------------- | ---------------------------------------- | ---------------------------------------- |
| Target               | $r + \gamma \max_a Q(s', a)$            | $r + \gamma Q(s', a')$                  |
| Policy               | Off-policy (learns optimal policy)       | On-policy (learns current policy)        |
| Effect of exploration| Ignored (only cares about optimal)       | Considered (accounts for exploration cost)|

---

**Intuition — cliff walking example**:

Imagine a grid world with a cliff. The optimal path is along the cliff edge (shortest).

- **Q-learning**: Learns the optimal path (along the cliff), because the penalty of falling off during exploration does not affect the learned policy
- **SARSA**: Learns a safe path (away from the cliff), because during exploration there is a probability of randomly stepping into the cliff, and this cost is factored into the value estimate

Both are "correct" — they just answer different questions.

---

### Section 4: Model-Based Reinforcement Learning — Dyna-Q

So far, all methods have been **model-free**: learning value functions directly from experience without building an internal representation of the environment. But humans clearly have "mental models" — we **simulate** possible outcomes in our minds. Dyna-Q formalizes this "model-based planning."

**World model**: A learned representation of the environment, caching observed transitions:

$$
\text{Model}(s, a) \rightarrow (r, s')
$$

"In state $s$, taking action $a$ yields reward $r$ and reaches state $s'$."

**Core idea**: Model-free methods only learn from real experience; if we can build an internal model of the environment, we can **simulate** experience in our minds to accelerate learning — this is what "model-based" means.

---

### Dyna-Q Algorithm

**Dyna-Q** (Sutton, 1990) — each real interaction simultaneously does three things:

1. **(a)** Take action $a$, observe reward $r$ and next state $s'$
2. **(b)** **Direct RL**: update $Q$ with real experience
3. **(c)** **Model update**: $\text{Model}(s, a) \leftarrow (r, s')$
4. **(d)** **Planning loop** (repeat $k$ times):
   - Randomly recall a previously experienced $(s, a)$
   - Use the model to "imagine" the result $(r, s')$
   - Update $Q$ with imagined experience

**Why is planning useful?** Each real step executes $k$ planning steps, so learning is about $k+1$ times faster. More critically, planning can propagate information **without actual interaction** — extremely important when real interactions are costly.

---

### Dyna-Q: Adapting to Change

**Scenario**: A shortcut appears in the environment (e.g., a wall suddenly disappears).

| Agent Type                        | Adaptation Speed | Reason                                                                         |
| --------------------------------- | ---------------- | ------------------------------------------------------------------------------ |
| **Model-free** (Q-learning only)  | Slow             | Can only update values by actually walking the new path                        |
| **Model-based** (Dyna-Q)          | Fast             | After discovering the shortcut, planning can immediately propagate information to all relevant states |

**Intuition**: The model acts as a "mental simulator." When you discover the world has changed, you can mentally replan all affected paths without actually traversing them.

**Tradeoffs**:

- **Advantage**: high sample efficiency, fast adaptation to change
- **Disadvantage**: requires more computation ($k$ planning steps per real step), and if the model has errors, planning may propagate incorrect information

**Neuroscience connection**: The hippocampal "replay" phenomenon (reactivation of experienced sequences during sleep and rest) may be the neural mechanism by which the brain implements Dyna-Q-style planning.

---

## W3D5: Network Causality

---

### Section 1: Correlation != Causation — Why Does This Distinction Matter?

"Ice cream sales are highly correlated with drowning incidents" — does this mean ice cream causes drowning? Obviously not. Both are driven by a **third variable** (summer heat). This simple example reveals a fundamental problem: **correlations in observational data cannot directly tell us about causal relationships**.

**Why does causal inference matter?**

- In neuroscience: Two brain regions show correlated activity — is it because one drives the other, or because both are driven by a third region?
- In medicine: Patients who take medication have higher recovery rates — is it because the drug works, or because healthier patients are more likely to seek treatment?
- In policy: After implementing a policy, indicators improve — is it because the policy worked, or because other factors changed simultaneously?

**Interventionist definition** (Pearl, 2000):

$$
A \text{ causes } B \iff \text{if we forcibly change } A, \text{ then } B \text{ will change accordingly}
$$

The key phrase is "**forcibly change**" — not "observed $A$ change" but "actively intervened to change $A$."

---

### Intervention vs. Conditioning — the do-operator

**Core distinction**:

$$
P(B | A = a) \neq P(B | \text{do}(A = a))
$$

- $P(B \mid A = a)$: **conditional probability** — observed $A = a$ (may have confounders)
- $P(B \mid \text{do}(A = a))$: **interventional probability** — force $A = a$ (cuts off confounders)

**Average causal effect**:

$$
\delta_{A \rightarrow B} = \mathbb{E}[B \mid \text{do}(A = 1)] - \mathbb{E}[B \mid \text{do}(A = 0)]
$$

**Randomized controlled trial (RCT)** — the gold standard for implementing interventions: randomly assign treatment, break all confounders, difference in group means = causal effect.

**Problem**: We usually cannot conduct RCTs. How can we estimate causal relationships from **observational data**?

**Answer**: The next three sections introduce three approaches —

1. **Using natural perturbations** (Section 2): spontaneous activity fluctuations as natural experiments
2. **Using temporal structure** (Section 3): causes precede effects; use regression to extract temporal relationships
3. **Using instrumental variables** (Section 4): find variables that bypass confounders

Common strategy: **break the influence of confounders** so that observations approximate interventions.

---

### Section 2: Estimating Causal Connections via Perturbations

For **biological neural networks** (neuron populations in the brain), we have a special advantage: **dynamical structure**. Causal influences between neurons are transmitted through time — $x_t^i$ influences $x_{t+1}^j$. Here $x_t^i$ denotes the activity of neuron $i$ at time $t$, and $\mathbf{A}$ denotes synaptic connection strengths.

**Neural dynamics model**:

$$
\mathbf{x}_{t+1} = \sigma(\mathbf{A} \mathbf{x}_t + \boldsymbol{\varepsilon}_t)
$$

- $\mathbf{x}_t$ — neural activity vector at time $t$
- $\mathbf{A}$ — **connectivity matrix** (the causal structure we want to estimate)
- $\sigma$ — sigmoid activation function
- $\boldsymbol{\varepsilon}_t$ — noise

Goal: Recover $\mathbf{A}$ from the observed time series $\{\mathbf{x}_t\}$.

---

### Perturbation-Based Estimator

Using the **spontaneous activity fluctuations** of neurons as "natural perturbations":

$$
\hat{\delta}_{x^i \rightarrow x^j} = \frac{1}{N} \sum_{t: x_t^i=1} x_{t+1}^j - \frac{1}{N} \sum_{t: x_t^i=0} x_{t+1}^j
$$

**Intuitive explanation**: When $i$ happens to be active, what is the average activity of $j$ the next moment? When $i$ is inactive? The difference reflects $i$'s causal influence on $j$.

**Limitations**: This assumes $i$'s activity is "random" — but in reality, $i$'s activity may be correlated with other factors that influence $j$ (confounders), leading to biased estimates. The regression and instrumental variable methods that follow are designed to overcome this limitation.

---

### Section 3: When Does Correlation Approximate Causation?

Correlation is a simple, cheap statistic. Under what conditions can it give us the correct causal answer?

**Lagged correlation matrix**:

$$
\mathbf{R} = \text{corrcoef}\left(\begin{bmatrix} \mathbf{X}_{t+1} \\ \mathbf{X}_t \end{bmatrix}\right)
$$

The off-diagonal blocks capture temporal dependence — the correlation between $x_t^i$ and $x_{t+1}^j$.

| Network Size          | Correlation ≈ Causation? | Reason                                                    |
| --------------------- | ------------------------ | --------------------------------------------------------- |
| Small (~6 neurons)    | Approximately valid      | Few confounders, limited influence of indirect paths      |
| Large (~100 neurons)  | Severely fails           | Many indirect paths produce spurious correlations         |

---

**Why does it fail in large-scale networks?**

Imagine $A \rightarrow B \rightarrow C$:

- $A$ and $C$ are strongly correlated (because $A$ influences $C$ through $B$)
- But there is no direct $A \rightarrow C$ connection
- Correlation cannot distinguish "direct connection" from "indirect path"

In a network, each additional node causes the number of indirect paths to grow exponentially, causing correlation to increasingly deviate from causation.

---

### Section 4: Regression and Causal Estimation — The Magic of Logit Transform

Regression is the most basic tool in statistics. Through a clever transformation, we can use it to estimate causal connections.

**Problem**: The neural dynamics model $\mathbf{x}_{t+1} = \sigma(\mathbf{A} \mathbf{x}_t + \boldsymbol{\varepsilon}_t)$ is nonlinear (due to sigmoid), so linear regression cannot be applied directly.

**Solution**: Use the **Logit transform** (inverse of sigmoid) to convert the nonlinear model into a linear one:

$$
\sigma^{-1}(x) = \log\frac{x}{1-x}
$$

Applying to both sides:

$$
\sigma^{-1}(\mathbf{x}_{t+1}) = \mathbf{A} \mathbf{x}_t + \boldsymbol{\varepsilon}_t
$$

This becomes a standard **linear regression** problem!

**Why is Lasso (L1 regularization) needed?**

Neural connections are typically **sparse** — each neuron is directly connected to only a few others. Ordinary regression would estimate many small nonzero connections (spurious connections caused by noise). Lasso uses an L1 penalty to drive unimportant coefficients to **exactly zero**:

$$
\hat{\mathbf{A}} = \arg\min_{\mathbf{A}} \|\sigma^{-1}(\mathbf{Y}) - \mathbf{A}\mathbf{X}\|_2^2 + \lambda \|\mathbf{A}\|_1
$$

The larger $\lambda$ is, the more coefficients are shrunk to zero — exactly the sparse structure we expect.

---

### Omitted Variable Bias — The Fundamental Limitation of Regression

**Core problem**: Regression coefficients are only causal when **all confounders are included**.

If neuron $k$ simultaneously influences $x^i$ and $x^j$ but is not included in the regression:

$$
\hat{A}_{ij} \neq A_{ij} \quad \text{（biased estimate）}
$$

**Intuitive explanation**: Regression sees that $x^i$ and $x^j$ are related and attributes it to $A_{ij}$. But if this relationship is actually caused by $k$ driving both ($k \rightarrow i$ and $k \rightarrow j$), regression will overestimate $A_{ij}$.

**Condition for unbiased regression**:

$$
\hat{A}_{ij} = A_{ij} \iff \text{all parents of } x^j \text{ are included as regression variables}
$$

**Consequences of partial observation**: When only a subset of neurons is observed:

$$
\text{performance} \propto \frac{N_{\text{observed}}}{N_{\text{total}}}
$$

The more neurons observed → the better the causal estimate. This is the theoretical basis for "record as many neurons as possible" in experimental design.

---

### Section 5: Instrumental Variables — When Confounders Cannot Be Observed

In many cases, we cannot observe all confounders. **Instrumental variables (IV)** provide a method for estimating causal effects in the presence of unobserved confounders.

**Core idea of instrumental variables**: Find a variable $Z$ that satisfies three conditions:

1. **Relevance**: $Z$ affects the cause variable $X$
2. **Exclusion**: $Z$ affects the outcome $Y$ **only through** $X$
3. **Independence**: $Z$ is uncorrelated with confounders

If such a $Z$ exists, we can "bypass" the confounders.

---

### Two-Stage Least Squares (2SLS)

**2SLS** — the standard method for instrumental variable estimation:

**First stage**: Strip away the part of $X$ that is correlated with confounders, keeping only the part explainable by $Z$:

$$
\hat{X} = \alpha \cdot Z
$$

**Second stage**: Use the "clean" $\hat{X}$ to estimate the causal effect on $Y$:

$$
Y = \beta \cdot \hat{X}
$$

The coefficient $\beta$ is the **causal effect** of $X$ on $Y$ — unaffected by confounders.

**Why does it work?** $\hat{X}$ contains only the variation in $X$ driven by $Z$. Since $Z$ is uncorrelated with confounders, $\hat{X}$ is also uncorrelated with confounders, so the regression coefficient is causal.

---

### Natural Instrumental Variables in Neural Systems

In neural dynamics, we can exploit **temporal structure** to construct instrumental variables.

**Setup**:

$$
x_{t+1}^j = \sigma\left(A_{ij} x_t^i + \underbrace{\sum_{k \neq i} A_{kj} x_t^k}_{\text{confounders}} + \varepsilon_t\right)
$$

The term $\sum_{k \neq i} A_{kj} x_t^k$ represents **unobserved common inputs** — it simultaneously affects $x_t^i$ and $x_{t+1}^j$, a classic confounder.

**Natural instrumental variable**: $x_{t-1}^i$ (past activity of neuron $i$)

- Relevant: $x_{t-1}^i$ affects $x_t^i$ through $A_{ii}$ (self-connection exists)
- Exclusive: The effect of $x_{t-1}^i$ on $x_{t+1}^j$ is entirely mediated through $x_t^i$
- Independent: $x_{t-1}^i$ is independent of the current noise $\varepsilon_t$

---

### IV Estimator

**IV estimation**:

$$
\hat{A}_{ij}^{\text{IV}} = \frac{\text{Cov}(x_{t-1}^i, x_{t+1}^j)}{\text{Cov}(x_{t-1}^i, x_t^i)}
$$

Even in the presence of confounders, this is **unbiased**!

- Numerator: total association between $i$'s past activity and $j$'s future activity
- Denominator: $i$'s autocorrelation (normalization factor)
- Ratio = causal effect of $i$ on $j$

---

### Granger Causality — Predictive Power as a Causal Proxy

When instrumental variable conditions are not met, there is a weaker but more general method: **Granger causality**.

**Core idea**: If $x$ truly causally influences $y$, then knowing past values of $x$ should help us better **predict** future values of $y$.

**Test method**:

**Null hypothesis** ($H_0$): $x$ does not Granger-cause $y$ — past $x$ does not help predict $y$:

$$
y_t = a_0 + a_1 y_{t-1} + \varepsilon_t
$$

**Alternative hypothesis** ($H_a$): $x$ Granger-causes $y$ — adding past $x$ improves prediction:

$$
y_t = a_0 + a_1 y_{t-1} + b_1 x_{t-1} + \varepsilon_t
$$

**Test**: Perform an F-test on $b_1 = 0$. If rejected, then $x$ Granger-causes $y$.

**Important limitations**:

- Granger causality is about **predictive improvement**, not true causal relationships
- If unobserved common drivers exist, Granger causality can produce spurious conclusions
- It is a **necessary condition** for causation, not a **sufficient condition**: if $x$ does not Granger-cause $y$, then $x$ cannot truly cause $y$; but the converse does not hold

---

## Summary

---

### Week 3: Core Concepts

### W3D2: Bayesian Decision

- Bayes' rule: fuse prior and data into posterior
- Gaussian product: analytical information combination
- Loss functions: optimal estimation under different costs
- Marginalization and conditioning: handling partial observations

### W3D3: Latent Dynamics

- SPRT: optimal sequential testing
- DDM: continuous-time evidence accumulation
- EM algorithm: iterative optimization with latent variables
- Forward-backward: efficient posterior computation

### W3D4: Reinforcement Learning

- TD learning: learning value from prediction errors
- Multi-armed bandits: exploration-exploitation tradeoff
- Q-learning vs SARSA: off-policy vs on-policy
- Dyna-Q: model-based planning accelerates learning

### W3D5: Network Causality

- Intervention defines causation: do-operator and RCT
- Regression + Lasso: sparse causal structure learning
- Omitted variable bias: fundamental limitation of regression
- Instrumental variables: bypassing unobserved confounders

---

### Key Formulas and Their Significance

$$
p(s|m) = \frac{p(m|s) \cdot p(s)}{\sum_{s'} p(m|s') \cdot p(s')} \quad \text{（Bayes' rule: mathematical foundation of belief updating）}
$$

$$
\mu_3 = \frac{\sigma_2^{-2}\mu_1 + \sigma_1^{-2}\mu_2}{\sigma_1^{-2} + \sigma_2^{-2}}, \quad \sigma_3^{-2} = \sigma_1^{-2} + \sigma_2^{-2} \quad \text{（Gaussian product: analytical solution for information fusion）}
$$

$$
\delta_t = r_{t+1} + \gamma V(s_{t+1}) - V(s_t) \quad \text{（TD error: driving signal for learning）}
$$

$$
Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma \max_{a'} Q(s',a') - Q(s,a)] \quad \text{（Q-learning: off-policy learning of optimal action values）}
$$

$$
\hat{A}_{ij}^{\text{IV}} = \frac{\text{Cov}(x_{t-1}^i, x_{t+1}^j)}{\text{Cov}(x_{t-1}^i, x_t^i)} \quad \text{（Instrumental variables: estimating causal effects in the presence of confounders）}
$$
