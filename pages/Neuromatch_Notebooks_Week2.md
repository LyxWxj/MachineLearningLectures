---
theme: seriph
themeConfig:
  primary: "#5d8392"
title: "Neuromatch Notebooks - Week 2"
transition: slide-left
math: true
---

# Neuromatch Notebooks — Week 2

<div class="text-gray-400 mt-2">
Linear Systems · Biological Neuron Models · Dynamical Systems
</div>

<Toc minDepth="2" maxDepth="2" />

---
layout: center
---

## Overview

<v-click>

Week 2 focuses on **dynamical systems and neural models** — from linear systems to biological neuron models to network dynamics:

| Day      | Topic                      | Core Skill                               |
| -------- | -------------------------- | ---------------------------------------- |
| **W2D3** | Linear Systems             | Euler integration, Oscillations, AR models |
| **W2D4** | Biological Neuron Models   | LIF neuron, Synapses, STDP               |
| **W2D5** | Dynamical Systems          | Firing rate models, Wilson-Cowan, Phase plane |

</v-click>

<v-click>

<div class="mt-4 p-4 bg-blue-900/20 rounded-lg text-center">

**The unifying theme**: how do neurons and networks evolve in time, and how can we model their dynamics mathematically?

</div>

</v-click>

---
layout: center
---

## W2D3: Linear Systems

---

### Section 1: One-Dimensional Differential Equations

The simplest dynamical system: $\dot{x} = ax$

<v-click>

**Analytical solution**: $x(t) = x_0 e^{at}$

| $a$ | Behavior |
|-----|----------|
| $a < 0$ | Exponential decay → 0 |
| $a > 0$ | Exponential growth → ∞ |
| $a = \text{complex}$ | Oscillation (with growth/decay) |

</v-click>

<v-click>

**Forward Euler integration** (numerical solution):

$$x(t_i) = x(t_{i-1}) + \dot{x}(t_{i-1}) \cdot dt$$

</v-click>

---

### Section 2: Oscillatory Dynamics

When $a$ is complex ($a = \text{real} + i \cdot \text{imag}$), the system oscillates:

<v-click>

**Key insight**: 
- Real part → growth/decay rate
- Imaginary part → oscillation frequency

**Euler's formula**: $e^{i\theta} = \cos\theta + i\sin\theta$

This is why complex eigenvalues produce oscillation: $e^{i \cdot \text{imag} \cdot t}$ traces a circle in the complex plane.

$$x(t) = x_0 e^{(\text{real} + i \cdot \text{imag})t} = x_0 e^{\text{real} \cdot t} \cdot [\cos(\text{imag} \cdot t) + i \sin(\text{imag} \cdot t)]$$

</v-click>

<v-click>

**For stable oscillation at frequency $f$**: set real part = 0, imag = $2\pi f$

</v-click>

---

### Section 3: Two-Dimensional Linear Systems

Extending to 2D: $\dot{\mathbf{x}} = \mathbf{A}\mathbf{x}$

<v-click>

$$\begin{bmatrix} \dot{x}_1 \\ \dot{x}_2 \end{bmatrix} = \begin{bmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \end{bmatrix}$$

**System function**:

```python
def system(t, x, a00, a01, a10, a11):
    x1dot = a00 * x[0] + a01 * x[1]
    x2dot = a10 * x[0] + a11 * x[1]
    return np.array([x1dot, x2dot])
```

</v-click>

<v-click>

**Eigenvalues determine behavior**:
- Both negative → stable node (converge to origin)
- Both positive → unstable node (diverge)
- Mixed signs → saddle point
- Complex → oscillation (spiral)

</v-click>

---

### Markov Process: Telegraph Process

A two-state Markov process — the **telegraph process** — models a neuron ion channel that switches between **Closed (0)** and **Open (1)** states:

<v-click>

**State transition matrix**:

$$\begin{bmatrix} P(\text{Closed}) \\ P(\text{Open}) \end{bmatrix}_{k+1} = \underbrace{\begin{bmatrix} 1-\mu_{c2o} & \mu_{o2c} \\ \mu_{c2o} & 1-\mu_{o2c} \end{bmatrix}}_{\mathbf{A}} \begin{bmatrix} P(\text{Closed}) \\ P(\text{Open}) \end{bmatrix}_k$$

| Entry | Meaning |
|-------|---------|
| $1-\mu_{c2o}$ | Stay closed |
| $\mu_{c2o}$ | Closed → Open |
| $\mu_{o2c}$ | Open → Closed |
| $1-\mu_{o2c}$ | Stay open |

</v-click>
---

<v-click>

**Simulation via Poisson process**:

```python
def ion_channel_opening(c2o, o2c, T, dt):
  t = np.arange(0, T, dt)
  x = np.zeros_like(t)
  switch_times = []
  x[0] = 0
  myrand = np.random.random_sample(size=len(t))
  for k in range(len(t)-1):
    if x[k] == 0 and myrand[k] < c2o*dt:
      x[k+1:] = 1
      switch_times.append(k*dt)
    elif x[k] == 1 and myrand[k] < o2c*dt:
      x[k+1:] = 0
      switch_times.append(k*dt)
  return t, x, switch_times
```

</v-click>

---

### Probability Propagation

Instead of running many stochastic simulations, we can propagate probabilities **directly** in one pass:

<v-click>

```python
def simulate_prob_prop(A, x0, dt, T):
  t = np.arange(0, T, dt)
  x = x0
  for k in range(len(t)-1):
    x_kp1 = np.dot(A, x[-1])      # matrix × state vector
    x = np.vstack([x, x_kp1])     # append new state
  return x, t
```

**Key advantage**: one simulation run gives the full probability trajectory, no repeated sampling needed.

</v-click>

---

### Equilibrium via Eigendecomposition

At equilibrium, probabilities stop changing: $\mathbf{A}\mathbf{x}^* = \mathbf{x}^*$

<v-click>

This is exactly the **eigenvalue equation** with $\lambda = 1$.

```python
lam, v = np.linalg.eig(A)
print(f"Eigenvalues: {lam}")
print(f"Eigenvector 1: {v[:, 0]}")
print(f"Eigenvector 2: {v[:, 1]}")
```

</v-click>

<v-click>

**Two eigenvalues**:

| Eigenvalue | Meaning |
|------------|---------|
| $\lambda_1 = 1$ | **Stable** — the stationary distribution (eigenvector rescaled to sum to 1) |
| $\lambda_2 < 1$ | **Decaying** — transient component that vanishes over time |

</v-click>
---

<v-click>

**Stationary distribution** (rescale eigenvector so probabilities sum to 1):

$$P(\text{Open}) = \frac{\mu_{c2o}}{\mu_{c2o} + \mu_{o2c}} \qquad P(\text{Closed}) = \frac{\mu_{o2c}}{\mu_{c2o} + \mu_{o2c}}$$

For $\mu_{c2o}=0.02,\; \mu_{o2c}=0.1$: $P(\text{Open}) \approx 0.167$ — the channel is open ~16.7% of the time.

</v-click>

<v-click>

**Intuition**: the transition matrix acts like a "mixing" operator. Regardless of the initial state, repeated application always converges to the same ratio — this is the **memoryless** property of Markov processes.

</v-click>

---

### Random Walks and Diffusion

A random walk: at each step, move $\Delta x = \pm 1$ with equal probability.

<v-click>

```python
def random_walk_simulator(N, T, mu=0, sigma=1):
    steps = np.random.normal(mu, sigma, size=(N, T))
    sim = np.cumsum(steps, axis=1)
    return sim
```

</v-click>

<v-click>

**Properties of random walks**:
- Mean stays near 0 (independent of time)
- Variance grows linearly with time: $\text{Var} \propto t$
- This is a **diffusive process**

</v-click>

---

### Ornstein-Uhlenbeck (OU) Process

Combines deterministic drift with random diffusion:

$$x_{k+1} = x_\infty + \lambda(x_k - x_\infty) + \sigma \eta$$

<v-click>

```python
def simulate_ddm(lam, sig, x0, xinfty, T):
    t = np.arange(0, T, 1.)
    x = np.zeros_like(t)
    x[0] = x0
    for k in range(len(t)-1):
        x[k+1] = xinfty + lam * (x[k] - xinfty) + sig * np.random.standard_normal()
    return t, x
```

</v-click>

<v-click>

**Equilibrium variance** (when $\lambda < 1$):

$$\text{Var} = \frac{\sigma^2}{1 - \lambda^2}$$

Unlike random walk, variance **saturates** due to the restoring drift toward $x_\infty$.

</v-click>

---

### Autoregressive (AR) Models

Flip the perspective: given data, learn the dynamics.

<v-click>

**First-order AR**: $x_{k+1} = \lambda x_k + \eta$

**Higher-order AR**: $x_{k+1} = \alpha_0 + \alpha_1 x_k + \alpha_2 x_{k-1} + \dots + \alpha_r x_{k-r}$

</v-click>

<v-click>

**Residual** = data − prediction:

```python
res = x2 - (p[0] + lam_hat * x1[:, 1])
```

</v-click>

<v-click>

**Key finding**: humans are terrible at generating random sequences! An AR model can predict "random" human input better than chance (error < 0.5).

The 6th-order AR model finds the sweet spot between underfitting and overfitting.

</v-click>

---
layout: center
---

## W2D4: Biological Neuron Models

---

### The Leaky Integrate-and-Fire (LIF) Model

The simplest mathematical model of a neuron:

<v-click>

$$\tau_m \frac{dV}{dt} = -(V - E_L) + \frac{I}{g_L}$$

**Parameters**:
| Symbol | Meaning | Typical value |
|--------|---------|---------------|
| $\tau_m$ | Membrane time constant | 10 ms |
| $g_L$ | Leak conductance | 10 nS |
| $E_L$ | Resting potential | −75 mV |
| $V_{th}$ | Spike threshold | −55 mV |
| $V_{reset}$ | Reset potential | −75 mV |
| $t_{ref}$ | Refractory time | 2 ms |

</v-click>

---

### LIF Neuron: Euler Integration

```python
def run_LIF(pars, Iinj, stop=False):
    # ... parameter setup ...
    for it in range(Lt - 1):
        if tr > 0:                          # refractory period
            v[it] = V_reset
            tr = tr - 1
        elif v[it] >= V_th:                 # spike!
            rec_spikes.append(it)
            v[it] = V_reset
            tr = tref / dt

        # Calculate the increment of the membrane potential
        dv = (dt / tau_m) * (-(v[it] - E_L) + Iinj[it] / g_L)

        # Update the membrane potential
        v[it + 1] = v[it] + dv
```

<v-click>

**Key outputs**: membrane potential trajectory `v` and spike times `sp`

</v-click>

---

### Firing Rate and Spike Irregularity

**F-I curve**: output firing rate as a function of input current.

<v-click>

**CV of ISI** (Coefficient of Variation of Inter-Spike Intervals):

$$\text{CV}_{\text{ISI}} = \frac{\text{std}(\text{ISI})}{\text{mean}(\text{ISI})}$$

| CV value | Meaning |
|----------|---------|
| 0 | Perfectly regular (clock-like) |
| 1 | Poisson process (maximum irregularity) |

```python
def isi_cv_LIF(spike_times):
    if len(spike_times) >= 2:
        isi = np.diff(spike_times)
        cv = np.std(isi) / np.mean(isi)
    return isi, cv
```

</v-click>

---

### Input Correlations and Output Correlations

How do correlated inputs affect output correlations?

<v-click>

**Correlated input** to two neurons:

$$\frac{I_i}{g_L} = \mu + \sigma(\sqrt{1-c}\xi_i + \sqrt{c}\xi_c)$$

where $c \in [0,1]$ controls the fraction of common input.

</v-click>

<v-click>

**Sample correlation coefficient**:

```python
def my_CC(i, j):
    cov = np.sum((i - np.mean(i)) * (j - np.mean(j)))
    var_i = np.sum((i - np.mean(i))**2)
    var_j = np.sum((j - np.mean(j))**2)
    return cov / np.sqrt(var_i * var_j)
```

</v-click>

<v-click>

**Key finding**: output correlation < input correlation. The neuron acts as a "correlation filter."

</v-click>

---

### Conductance-Based Synapses

Real neurons receive synaptic inputs modeled as conductance changes:

<v-click>

$$\tau_m \frac{dV}{dt} = -(V-E_L) - \frac{g_E}{g_L}(V-E_E) - \frac{g_I}{g_L}(V-E_I) + \frac{I_{\text{inj}}}{g_L}$$

</v-click>

<v-click>

**Free Membrane Potential (FMP)**: membrane potential with spike threshold removed (artificially set $V_{th} = \infty$).

- Mean FMP above threshold → regular firing
- Mean FMP below threshold → irregular, noise-driven firing
- **Balance of excitation/inhibition** determines firing regime

</v-click>

---

### Short-Term Plasticity (STP)

Synapses can change strength based on recent spike history:

<v-click>

**Short-Term Depression (STD)**: synapse weakens with repeated use
- Parameters: $U_0 = 0.5$, $\tau_d = 100$ ms, $\tau_f = 50$ ms

**Short-Term Facilitation (STF)**: synapse strengthens with repeated use
- Parameters: $U_0 = 0.2$, $\tau_d = 100$ ms, $\tau_f = 750$ ms

</v-click>

<v-click>

**STP dynamics**:

$$\frac{du}{dt} = -\frac{u}{\tau_f} + U_0(1-u^-)\delta(t-t_{sp})$$
$$\frac{dR}{dt} = \frac{1-R}{\tau_d} - u^+ R^- \delta(t-t_{sp})$$

</v-click>

---

### Spike-Timing Dependent Plasticity (STDP)

Synaptic weight changes based on the **timing** of pre- and postsynaptic spikes:

<v-click>

$$\Delta W = \begin{cases} A_+ e^{(t_{pre}-t_{post})/\tau_+} & \text{if } t_{post} > t_{pre} \text{ (LTP)} \\ -A_- e^{-(t_{pre}-t_{post})/\tau_-} & \text{if } t_{post} < t_{pre} \text{ (LTD)} \end{cases}$$

</v-click>

<v-click>

**Tracking variables**:

```python
def generate_P(pars, pre_spike_train_ex):
    A_plus, tau_stdp = pars['A_plus'], pars['tau_stdp']
    dt = pars['dt']
    P = np.zeros(pre_spike_train_ex.shape)
    for it in range(Lt - 1):
        dP = -(dt / tau_stdp) * P[:, it] + A_plus * pre_spike_train_ex[:, it + 1]
        P[:, it + 1] = P[:, it] + dP
    return P
```

</v-click>

<v-click>

**Key insight**: STDP causes synapses from **correlated** presynaptic neurons to be strengthened, while uncorrelated synapses weaken — a form of unsupervised learning.

</v-click>

---
layout: center
---

## W2D5: Dynamical Systems

---

### Single Population Firing Rate Model

Instead of modeling individual neurons, model the **average firing rate** of a population:

<v-click>

$$\tau \frac{dr}{dt} = -r + F(w \cdot r + I_{\text{ext}})$$

**Sigmoid transfer function**:

$$F(x; a, \theta) = \frac{1}{1 + e^{-a(x-\theta)}} - \frac{1}{1 + e^{a\theta}}$$

```python
def F(x, a, theta):
    f = (1 + np.exp(-a * (x - theta)))**-1 - (1 + np.exp(a * theta))**-1
    return f
```

</v-click>

---

### Fixed Points and Stability

**Fixed point**: value of $r$ where $\frac{dr}{dt} = 0$

<v-click>

$$-r^* + F(w \cdot r^* + I_{\text{ext}}) = 0$$

**Eigenvalue** (stability):

$$\lambda = \frac{-1 + w \cdot F'(w \cdot r^* + I_{\text{ext}})}{\tau}$$

| $\lambda$ | Stability |
|-----------|-----------|
| $\lambda < 0$ | Stable (attracting) |
| $\lambda > 0$ | Unstable (repelling) |

</v-click>

<v-click>

```python
def eig_single(fp, tau, a, theta, w, I_ext, **other_pars):
    eig = (-1 + w * dF(w * fp + I_ext, a, theta)) / tau
    return eig
```

</v-click>

---

### Wilson-Cowan Model: E/I Populations

Two coupled populations (Excitatory + Inhibitory):

<v-click>

$$\tau_E \frac{dr_E}{dt} = -r_E + F_E(w_{EE}r_E - w_{EI}r_I + I_E^{\text{ext}})$$
$$\tau_I \frac{dr_I}{dt} = -r_I + F_I(w_{IE}r_E - w_{II}r_I + I_I^{\text{ext}})$$

```python
def simulate_wc(tau_E, a_E, theta_E, tau_I, a_I, theta_I,
                wEE, wEI, wIE, wII, I_ext_E, I_ext_I,
                rE_init, rI_init, dt, range_t, **other_pars):
    for k in range(Lt - 1):
        drE = (dt/tau_E) * (-rE[k] + F(wEE*rE[k] - wEI*rI[k] + I_ext_E[k], a_E, theta_E))
        drI = (dt/tau_I) * (-rI[k] + F(wIE*rE[k] - wII*rI[k] + I_ext_I[k], a_I, theta_I))
        rE[k+1] = rE[k] + drE
        rI[k+1] = rI[k] + drI
    return rE, rI
```

</v-click>

---

### Phase Plane Analysis

Plot $r_E$ vs $r_I$ to visualize system dynamics:

<v-click>

**Nullclines**: curves where $\frac{dr_E}{dt} = 0$ or $\frac{dr_I}{dt} = 0$

```python
def get_E_nullcline(rE, a_E, theta_E, wEE, wEI, I_ext_E, **other_pars):
    rI = 1/wEI * (wEE * rE - F_inv(rE, a_E, theta_E) + I_ext_E)
    return rI

def get_I_nullcline(rI, a_I, theta_I, wIE, wII, I_ext_I, **other_pars):
    rE = 1/wIE * (wII * rI + F_inv(rI, a_I, theta_I) - I_ext_I)
    return rE
```

</v-click>

<v-click>

**Vector field**: arrows showing $(\frac{dr_E}{dt}, \frac{dr_I}{dt})$ at each point

```python
def EIderivs(rE, rI, tau_E, a_E, theta_E, wEE, wEI, I_ext_E,
             tau_I, a_I, theta_I, wIE, wII, I_ext_I, **other_pars):
    drEdt = (-rE + F(wEE*rE - wEI*rI + I_ext_E, a_E, theta_E)) / tau_E
    drIdt = (-rI + F(wIE*rE - wII*rI + I_ext_I, a_I, theta_I)) / tau_I
    return drEdt, drIdt
```

</v-click>

---

### Jacobian Matrix and Stability

For the 2D Wilson-Cowan system, stability is determined by the **Jacobian**:

<v-click>

$$J = \begin{bmatrix} \frac{\partial G_E}{\partial r_E} & \frac{\partial G_E}{\partial r_I} \\ \frac{\partial G_I}{\partial r_E} & \frac{\partial G_I}{\partial r_I} \end{bmatrix}$$

```python
def get_eig_Jacobian(fp, tau_E, a_E, theta_E, wEE, wEI, I_ext_E,
                     tau_I, a_I, theta_I, wIE, wII, I_ext_I, **other_pars):
    rE, rI = fp
    J = np.zeros((2, 2))
    J[0, 0] = (-1 + wEE * dF(wEE*rE - wEI*rI + I_ext_E, a_E, theta_E)) / tau_E
    J[0, 1] = (-wEI * dF(wEE*rE - wEI*rI + I_ext_E, a_E, theta_E)) / tau_E
    J[1, 0] = (wIE * dF(wIE*rE - wII*rI + I_ext_I, a_I, theta_I)) / tau_I
    J[1, 1] = (-1 - wII * dF(wIE*rE - wII*rI + I_ext_I, a_I, theta_I)) / tau_I
    evals = np.linalg.eig(J)[0]
    return evals
```

</v-click>

---

### Limit Cycles and Oscillations

When eigenvalues become **complex**, the system oscillates:

<v-click>

**Oscillatory parameters**: $w_{EE}=6.4$, $w_{EI}=4.8$, $w_{IE}=6.0$, $w_{II}=1.2$, $I_E^{\text{ext}}=0.8$

- Trajectories form a **limit cycle** in the phase plane
- E and I populations alternate in activity
- Frequency determined by the imaginary part of eigenvalues

</v-click>

<v-click>

**Bifurcation**: dramatic change in system behavior as parameters change
- Changing $\tau_I$ can switch between steady state and oscillations
- Nullclines stay the same, but vector field changes

</v-click>

---

### Inhibition-Stabilized Network (ISN)

Two regimes based on $\frac{\partial G_E}{\partial r_E}$:

<v-click>

| Regime | Condition | Behavior |
|--------|-----------|----------|
| **non-ISN** | $\frac{\partial G_E}{\partial r_E} < 0$ | Adding inhibition to I → E decreases |
| **ISN** | $\frac{\partial G_E}{\partial r_E} > 0$ | Adding inhibition to I → E also decreases (paradoxically) |

</v-click>

<v-click>

**ISN is common in cortex**: strong recurrent excitation ($w_{EE}$ large) creates a regime where inhibition is needed for stability.

</v-click>

---

### Working Memory: Persistent Activity

Brief input can trigger **sustained activity** that outlasts the stimulus:

<v-click>

**Mechanism**: multiple fixed points + noise

1. System starts at low-activity fixed point
2. Brief pulse pushes state past the unstable fixed point
3. System settles at high-activity fixed point
4. This represents "memory" of the stimulus

</v-click>

<v-click>

**Wilson-Cowan model demonstrates this**:
- Without pulse: system stays at resting state
- With sufficient pulse: system switches to persistent activity
- Critical pulse amplitude determines the transition

</v-click>

---
layout: center
---

## Summary

---

### Week 2: Key Concepts

<div class="grid grid-cols-3 gap-4">
<div class="p-3 bg-gray-800/50 rounded-lg text-center">

### W2D3: Linear Systems

- Euler integration
- Eigenvalue analysis
- Random walks & OU process
- Autoregressive models

</div>
<div class="p-3 bg-gray-800/50 rounded-lg text-center">

### W2D4: Neuron Models

- LIF neuron dynamics
- Conductance-based synapses
- Short-term plasticity
- STDP learning rule

</div>
<div class="p-3 bg-gray-800/50 rounded-lg text-center">

### W2D5: Network Dynamics

- Firing rate models
- Wilson-Cowan model
- Phase plane analysis
- Fixed points & stability

</div>
</div>

---

### Key Formulas

$$\tau_m \frac{dV}{dt} = -(V-E_L) + \frac{I}{g_L} \quad \text{(LIF neuron)}$$

$$\tau \frac{dr}{dt} = -r + F(w \cdot r + I_{\text{ext}}) \quad \text{(Firing rate model)}$$

$$\tau_E \frac{dr_E}{dt} = -r_E + F_E(w_{EE}r_E - w_{EI}r_I + I_E^{\text{ext}}) \quad \text{(Wilson-Cowan)}$$

$$\lambda = \frac{-1 + w \cdot F'(w \cdot r^* + I_{\text{ext}})}{\tau} \quad \text{(Eigenvalue/stability)}$$

$$\text{Var} = \frac{\sigma^2}{1-\lambda^2} \quad \text{(OU equilibrium variance)}$$