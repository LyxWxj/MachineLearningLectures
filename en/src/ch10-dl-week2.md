# Deep Learning Advanced — Week 2

Regularization · Convolutional Neural Networks · Generative Models · Diffusion Models · Sequences and Language

---

## Overview

Week 2 extends the view from fully connected networks to **richer architectures and tasks** — convolutional networks for images, generative models for creating new samples, and sequence models for language:

| Day      | Topic                | Core Skills                                                        |
| -------- | -------------------- | ------------------------------------------------------------------ |
| **W2D1** | Regularization       | Overfitting, Frobenius norm, early stopping, data augmentation, memorization |
| **W2D2** | Convolutional Neural Networks (CNN) | Convolution operation, pooling, classic architectures, feature visualization |
| **W2D3** | Generative Models    | Autoencoders, PCA, Variational Autoencoders (VAE), GAN basics      |
| **W2D4** | Diffusion Generative Models | Forward diffusion, Score function, reverse SDE, denoising score matching |
| **W2D5** | Sequences and Language | Word embeddings (Word2Vec/FastText), text classification, RNN basics |

**Recurring themes**: How to make models **generalize** rather than memorize? How to **generate** new data? How to handle **sequential** data?

---

## W2D1: Regularization

---

### 1. The Overfitting Problem

**Overfitting**: The model performs well on the training set but poorly on the test set.

**Root cause**: The model capacity (number of parameters) is too large relative to the amount of training data, causing the model to "memorize" noise in the training data.

**Classic behavior**: Training loss continues to decrease, but validation loss first decreases then increases.

---

### 2. Frobenius Norm and Weight Decay

**Frobenius norm** measures the "size" of a weight matrix:

$$
\|A\|_F = \sqrt{\sum_{i,j} |a_{ij}|^2}
$$

**L2 regularization** (weight decay): Adds a penalty based on the Frobenius norm of weights to the loss function:

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + \lambda \sum_l \|\mathbf{W}^{(l)}\|_F^2
$$

**Effect**: Encourages smaller weights, reduces model complexity, and improves generalization.

```python
# L2 regularization in PyTorch
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)
```

---

### 3. Early Stopping

**Idea**: Stop training when validation set performance no longer improves.

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

**Hyperparameter**: `patience` (how many epochs of no improvement to tolerate)

---

### 4. Memorization Experiment

**Surprising finding**: Even with completely random labels, a sufficiently large network can still achieve 100% training accuracy!

This shows that training accuracy is **not** a good metric for evaluating models — you must look at the validation/test set.

---

### 5. Data Augmentation

"Create" more samples by applying transformations to the training data:

| Transformation    | Applicable Scenarios           |
| ----------------- | ------------------------------ |
| Horizontal flip   | Natural images                 |
| Random crop       | Image classification           |
| Color jitter      | Image classification           |
| Random rotation   | Rotation-invariant tasks       |
| Mixup             | Linear interpolation of two images |
| CutMix            | Pasting a region from one image onto another |

```python
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
])
```

---

## W2D2: Convolutional Neural Networks (CNN)

---

### 1. Why Do We Need CNNs?

Problems with fully connected networks for images:
- Too many parameters: $28 \times 28 = 784$ pixels, hidden layer 1000 → $784 \times 1000 = 784,000$ parameters
- No translation invariance

**Core ideas of CNNs**:
- **Local connectivity**: Each neuron only looks at a small region of the input
- **Parameter sharing**: The same convolutional kernel slides across the entire input
- **Translation equivariance**: When an object translates in the image, the feature map also translates

---

### 2. Convolution Operation

#### 2.1 1D Convolution

$$
y[i] = \sum_{k=0}^{K-1} w[k] \cdot x[i+k]
$$

#### 2.2 2D Convolution

$$
y[i,j] = \sum_{m=0}^{M-1}\sum_{n=0}^{N-1} w[m,n] \cdot x[i+m, j+n]
$$

**Convolution operation process:**

![Convolution Operation](../../assets/conv_operation.png)

- **Input**: 5×5 matrix
- **Kernel**: 3×3 Laplacian kernel (edge detection)
- **Process**: Kernel slides over input, element-wise multiplication then sum at each position
- **Output**: 3×3 feature map (smaller than input due to no padding)

**Effects of different convolution kernels:**

![Different Kernels](../../assets/conv_kernels.png)

**Common kernel values:**

![Kernel Values](../../assets/conv_kernel_values.png)

| Kernel | Purpose | Characteristics |
|--------|---------|-----------------|
| **Identity** | No change | Output = Input |
| **Gaussian** | Blur/smoothing | Positive weights, center-heavy, for denoising |
| **Sobel X** | Vertical edge detection | Horizontal differencing |
| **Sobel Y** | Horizontal edge detection | Vertical differencing |
| **Laplacian** | Edge detection | Second derivative, sensitive to noise |
| **Sharpen** | Sharpening | Enhances center pixel, suppresses neighbors |

```python
# PyTorch convolution layer
conv = nn.Conv2d(
    in_channels=3,      # Number of input channels (RGB = 3)
    out_channels=16,    # Number of output channels (number of kernels)
    kernel_size=3,      # Kernel size
    stride=1,           # Stride
    padding=1           # Padding (preserves spatial dimensions)
)

x = torch.randn(1, 3, 32, 32)  # (batch, channels, height, width)
y = conv(x)  # (1, 16, 32, 32)
```

#### 2.3 Output Size Calculation

$$
H_{\text{out}} = \left\lfloor\frac{H_{\text{in}} + 2 \cdot \text{padding} - \text{kernel\_size}}{\text{stride}}\right\rfloor + 1
$$

---

### 3. Pooling

**Max pooling**: Takes the maximum value in a local region

$$
y[i,j] = \max_{(m,n) \in \mathcal{R}_{i,j}} x[m,n]
$$

**Average pooling**: Takes the mean of a local region

$$
y[i,j] = \frac{1}{|\mathcal{R}_{i,j}|} \sum_{(m,n) \in \mathcal{R}_{i,j}} x[m,n]
$$

**Max pooling vs Average pooling:**

![Pooling Comparison](../../assets/pooling_comparison.png)

- **Max pooling**: Retains the maximum value in each window (red highlight), preserves strongest features
- **Average pooling**: Computes the mean of each window, preserves overall information

**Pooling stride effect:**

![Pooling Stride](../../assets/pooling_stride.png)

- 2×2 pooling + stride=2: spatial size halved (6×6 → 3×3)
- Windows don't overlap, each element participates in only one pooling operation

```python
# Max pooling
max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
x = torch.randn(1, 16, 32, 32)
y = max_pool(x)
print(y.shape)
# torch.Size([1, 16, 16, 16])  — spatial size halved

# Average pooling
avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
y = avg_pool(x)
print(y.shape)
# torch.Size([1, 16, 16, 16])

# Global average pooling (often used as the last layer in classification networks)
gap = nn.AdaptiveAvgPool2d(1)  # output size fixed to 1×1
y = gap(x)
print(y.shape)
# torch.Size([1, 16, 1, 1])  — each channel becomes a scalar
```

**Effect**: Reduces spatial resolution, increases the receptive field, and provides some translation invariance.

---

### 4. Classic CNN Architectures

| Architecture   | Year  | Key Innovation                          |
| -------------- | ----- | --------------------------------------- |
| **LeNet**      | 1998  | Classic structure of convolution + pooling + fully connected |
| **AlexNet**    | 2012  | ReLU, Dropout, GPU training             |
| **VGG**        | 2014  | Small kernels (3×3) stacked             |
| **GoogLeNet**  | 2014  | Inception module (multi-scale)          |
| **ResNet**     | 2015  | Residual connections (skip connections) |
| **DenseNet**   | 2017  | Dense connections                        |

---

### 5. Residual Connections

The core innovation of ResNet:

$$
\mathbf{y} = F(\mathbf{x}) + \mathbf{x}
$$

**Why it works**:
- Gradients can flow directly through skip connections, alleviating the vanishing gradient problem
- The network only needs to learn the "residual" $F(\mathbf{x}) = \mathbf{y} - \mathbf{x}$
- Identity mapping is easy to learn, so additional layers do not harm performance

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
        return F.relu(x + residual)  # Skip connection
```

---

## W2D3: Generative Models

---

### 1. Generative Models vs. Discriminative Models

|              | Discriminative Models | Generative Models    |
| ------------ | --------------------- | -------------------- |
| **Goal**     | $p(y \mid x)$         | $p(x)$ or $p(x, y)$ |
| **Meaning**  | Given input $x$, what is the probability it belongs to class $y$? | Learn the distribution of data itself, so you can sample and generate new data |
| **Tasks**    | Classification, regression | Generating new samples, density estimation |
| **Examples** | MLP, CNN              | VAE, GAN, Diffusion  |

**Intuition**:
- **Discriminative model**: Given a photo of a cat, ask "Is this a cat or a dog?" → outputs $p(\text{cat} \mid \text{image})$
- **Generative model**: Learns "what cat photos look like" → can draw a new cat image from scratch

---

### 2. Autoencoder

#### 2.1 Core Idea

An autoencoder is an **unsupervised** neural network that learns a **compressed representation** of data.

**Structure**: Encoder + bottleneck + decoder

$$
\mathbf{x} \xrightarrow{\text{Encoder}} \mathbf{h} \xrightarrow{\text{Decoder}} \hat{\mathbf{x}}
$$

- **Encoder** $f_\phi$: compresses high-dimensional input $\mathbf{x} \in \mathbb{R}^D$ to low-dimensional latent $\mathbf{h} \in \mathbb{R}^d$ ($d \ll D$)
- **Decoder** $g_\theta$: reconstructs original input from latent $\mathbf{h}$
- **Bottleneck**: forces the network to learn **essential features** (cannot simply copy input)

#### 2.2 Simplest Autoencoder

```python
import torch
import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=32):
        super().__init__()
        # Encoder: 784 → 128 → 32
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, hidden_dim),
            nn.ReLU()
        )
        # Decoder: 32 → 128 → 784
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()  # output in [0,1] (suitable for MNIST)
        )

    def forward(self, x):
        h = self.encoder(x)           # encode: (B, 784) → (B, 32)
        x_hat = self.decoder(h)       # decode: (B, 32) → (B, 784)
        return x_hat

# Create model
model = AutoEncoder(input_dim=784, hidden_dim=32)

# Check parameter counts
print(f"Encoder params: {sum(p.numel() for p in model.encoder.parameters())}")
# Encoder params: 101152 = 784*128 + 128 + 128*32 + 32

print(f"Decoder params: {sum(p.numel() for p in model.decoder.parameters())}")
# Decoder params: 101536 = 32*128 + 128 + 128*784 + 784

print(f"Total: {sum(p.numel() for p in model.parameters())}")
# Total: 202688
```

#### 2.3 Training an Autoencoder

```python
# Training loop
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

for epoch in range(num_epochs):
    for batch_x, _ in train_loader:  # Note: no labels needed!
        batch_x = batch_x.view(-1, 784)  # Flatten

        x_hat = model(batch_x)            # Forward pass
        loss = loss_fn(x_hat, batch_x)    # Reconstruction error

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

**Key points**:
- Loss function is **reconstruction error** (MSE or cross-entropy)
- **No labels needed** — this is unsupervised learning
- Smaller bottleneck dimension $d$ → more compression → blurrier reconstruction

#### 2.4 Linear Autoencoder ≈ PCA

When both encoder and decoder are **linear mappings** (no activation functions), the subspace learned by the autoencoder is the same as PCA.

$$
\hat{\mathbf{x}} = W_2 W_1 \mathbf{x}
$$

where $W_1 \in \mathbb{R}^{d \times D}$ is the encoding matrix and $W_2 \in \mathbb{R}^{D \times d}$ is the decoding matrix. The optimal solution's column space matches the first $d$ principal components of PCA.

#### 2.5 Limitations of Autoencoders

Autoencoders learn to **compress and reconstruct**, but their latent space has **no structure**:
- Latent space is discontinuous — two adjacent $\mathbf{h}$ values may decode to completely different outputs
- Cannot **sample** from the latent space to generate new data — because we don't know which $\mathbf{h}$ values are "valid"

> This is exactly the problem VAE solves.

---

### 3. Variational Autoencoder (VAE)

#### 3.1 Core Idea

> **Goal of generative models**: Learn the data distribution $p(x)$ so we can sample and generate new data.

**VAE's strategy**: Assume data is generated by a low-dimensional latent variable $\mathbf{z}$ (sample $\mathbf{z}$ first, then map through Decoder to get $\mathbf{x}$):

$$
p(\mathbf{x}) = \int p(\mathbf{x}|\mathbf{z})\,p(\mathbf{z})\,d\mathbf{z}
$$

> **Manifold hypothesis**: The actual degrees of freedom in high-dimensional data are far lower than the dimension. For example, 28×28 handwritten digits (784 dimensions) have only about 10-20 factors of variation (stroke thickness, tilt angle, etc.). VAE uses a low-dimensional vector $\mathbf{z}$ (e.g., 20 dimensions) to represent these latent factors, and assumes $\mathbf{z}$ follows a standard normal distribution (prior assumption).

**Two core difficulties**:

1. $p(\mathbf{x}) = \int p(\mathbf{x}|\mathbf{z})p(\mathbf{z})d\mathbf{z}$ requires integrating over all possible $\mathbf{z}$ — infeasible in high dimensions
2. Training also requires the posterior $p(\mathbf{z}|\mathbf{x}) = \frac{p(\mathbf{x}|\mathbf{z})p(\mathbf{z})}{p(\mathbf{x})}$, but the denominator $p(\mathbf{x})$ is the intractable integral

> **VAE's solution — variational inference**: Since the true posterior $p(\mathbf{z}|\mathbf{x})$ is intractable, train an Encoder network $q_\phi(\mathbf{z}|\mathbf{x})$ (outputting Gaussian distribution parameters $\mu, \sigma$) to approximate it. Then optimize a computable lower bound (ELBO) as a substitute for the intractable integral.
>
> "Variational" comes from the Calculus of Variations — finding the optimal one among a family of functions.

#### 3.2 Architecture

```
x  →  [Encoder q_φ(z|x)]  →  μ, log σ²
                                ↓
                         z = μ + σ·ε    (ε ~ N(0,I), reparameterization)
                                ↓
z  →  [Decoder p_θ(x|z)]  →  x̂ (reconstruction)
```

- **Encoder** $q_\phi(\mathbf{z}|\mathbf{x})$: data → latent distribution parameters (mean $\mu$, variance $\sigma^2$)
- **Decoder** $p_\theta(\mathbf{x}|\mathbf{z})$: latent → reconstructed data

#### 3.3 ELBO Derivation

**Starting point**: Maximize log-likelihood $\log p_\theta(\mathbf{x})$, but $p_\theta(\mathbf{x}) = \int p_\theta(\mathbf{x}|\mathbf{z})p(\mathbf{z})d\mathbf{z}$ is intractable.

> Why use $\log$ instead of $p(\mathbf{x})$?
> 1. Numerical stability — $p(\mathbf{x})$ is extremely small in high dimensions (e.g., $10^{-300}$), normal after log
> 2. Multiplication becomes addition — joint likelihood $\prod p(\mathbf{x}_i)$ becomes $\sum \log p(\mathbf{x}_i)$

**Introduce variational distribution** $q_\phi(\mathbf{z}|\mathbf{x})$, apply Jensen's inequality:

$$
\log p_\theta(\mathbf{x}) = \log \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}\left[\frac{p_\theta(\mathbf{x}, \mathbf{z})}{q_\phi(\mathbf{z}|\mathbf{x})}\right] \geq \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}\left[\log \frac{p_\theta(\mathbf{x}, \mathbf{z})}{q_\phi(\mathbf{z}|\mathbf{x})}\right]
$$

This lower bound is the **ELBO** (Evidence Lower Bound).

**ELBO decomposition**:

$$
\text{ELBO} = \underbrace{\mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}[\log p_\theta(\mathbf{x}|\mathbf{z})]}_{\text{Reconstruction term}} - \underbrace{D_{\text{KL}}(q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))}_{\text{KL regularization term}}
$$

**Relationship between ELBO and true likelihood**:

$$
\log p_\theta(\mathbf{x}) = \text{ELBO} + D_{\text{KL}}(q_\phi(\mathbf{z}|\mathbf{x}) \| p_\theta(\mathbf{z}|\mathbf{x}))
$$

Since $D_{\text{KL}} \geq 0$, we have $\log p_\theta(\mathbf{x}) \geq \text{ELBO}$. Maximizing ELBO simultaneously achieves:
1. Maximize data likelihood (improve generative model)
2. Minimize approximation gap (improve Encoder approximation)

#### 3.4 Final Loss Function

$$
\mathcal{L}_{\text{VAE}} = -\mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}[\log p_\theta(\mathbf{x}|\mathbf{z})] + D_{\text{KL}}(q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))
$$

| Term | Meaning | Intuition |
|---|---|---|
| $-\mathbb{E}[\log p_\theta(\mathbf{x}\|\mathbf{z})]$ | Negative log-likelihood of Decoder reconstructing $\mathbf{x}$ from $\mathbf{z}$ | If Decoder outputs Gaussian, this is MSE |
| $D_{\text{KL}}(q \| p)$ | How far the Encoder's distribution deviates from prior $\mathcal{N}(0,I)$ | Makes latent space smooth and sampleable |

> **Why is the reconstruction loss MSE?** Assuming $p_\theta(\mathbf{x}|\mathbf{z}) = \mathcal{N}(\mathbf{x}; \boldsymbol{\mu}_\theta(\mathbf{z}), \sigma^2 I)$, then $-\log p_\theta(\mathbf{x}|\mathbf{z}) = \frac{\|\mathbf{x} - \boldsymbol{\mu}_\theta(\mathbf{z})\|^2}{2\sigma^2} + \text{const}$, which is MSE after dropping constants.

#### 3.5 Reparameterization Trick

> **Problem**: Sampling $\mathbf{z}$ from $q_\phi(\mathbf{z}|\mathbf{x}) = \mathcal{N}(\boldsymbol{\mu}_\phi(\mathbf{x}), \boldsymbol{\sigma}^2_\phi(\mathbf{x}))$ is non-differentiable — gradients cannot flow from Decoder back to Encoder.

> **Solution**: "Outsource" the randomness to external noise
>
> $$\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(0, I)$$
>
> - $\boldsymbol{\epsilon}$ is sampled from a fixed distribution (independent of parameters $\phi$)
> - $\mathbf{z}$ is a differentiable deterministic function of $\boldsymbol{\mu}$ and $\boldsymbol{\sigma}$: $\frac{\partial \mathbf{z}}{\partial \boldsymbol{\mu}} = 1, \frac{\partial \mathbf{z}}{\partial \boldsymbol{\sigma}} = \boldsymbol{\epsilon}$

```python
def reparameterize(mu, log_var):
    """Reparameterization: z = μ + σ * ε, ε ~ N(0,I)"""
    std = torch.exp(0.5 * log_var)  # σ = exp(0.5 * log σ²)
    eps = torch.randn_like(std)      # ε ~ N(0,I)
    return mu + eps * std            # z = μ + σ * ε
```

> **Generality of reparameterization**: Reparameterization is not unique to VAE — it's a general method for stochastic computational graphs. **The forward process in Diffusion $\mathbf{x}_t = \sqrt{\bar\alpha_t} \mathbf{x}_0 + \sqrt{1-\bar\alpha_t}\boldsymbol{\epsilon}$ is essentially reparameterization.**

#### 3.6 Closed-Form KL Divergence

When $q(\mathbf{z}|\mathbf{x}) = \mathcal{N}(\boldsymbol{\mu}, \text{diag}(\boldsymbol{\sigma}^2))$ and $p(\mathbf{z}) = \mathcal{N}(0, I)$, the KL divergence has a closed-form solution:

$$
D_{\text{KL}} = -\frac{1}{2}\sum_{j=1}^{d}\left(1 + \log\sigma_j^2 - \mu_j^2 - \sigma_j^2\right)
$$

> This avoids the variance introduced by Monte Carlo sampling to estimate the KL divergence.

#### 3.7 Complete VAE Implementation

```python
class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, latent_dim=20):
        super().__init__()
        # Encoder: outputs μ and log σ²
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)      # μ
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)   # log σ²

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar


def vae_loss(x_hat, x, mu, logvar):
    """VAE loss = reconstruction loss + KL divergence"""
    # Reconstruction loss (MSE or BCE)
    recon_loss = nn.functional.mse_loss(x_hat, x, reduction='sum')

    # Closed-form KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + kl_loss
```

#### 3.8 The Essential Dilemma of VAE

**Reconstruction vs regularization tension**:

- **Reconstruction term** wants the Encoder to pack as much information as possible into $\mathbf{z}$ → smaller $\sigma$ is better
- **KL term** wants $q(\mathbf{z}|\mathbf{x})$ to be close to $\mathcal{N}(0,I)$ → $\mu \to 0$, $\sigma \to 1$

This tension causes VAE-generated images to tend to be blurry — it's the "average" of all possible reconstructions.

**Posterior collapse**: When the Decoder is too powerful (e.g., autoregressive Decoder), it can ignore $\mathbf{z}$ and generate $\mathbf{x}$ directly from context. Then $q(\mathbf{z}|\mathbf{x}) \approx p(\mathbf{z})$, and $\mathbf{z}$ no longer carries any information.

Mitigation: KL annealing, $\beta$-VAE (adjusting KL weight), Free bits (setting minimum KL per dimension).

#### 3.9 VAE vs Autoencoder

| | Autoencoder | VAE |
|---|---|---|
| **Encoder output** | A deterministic point $\mathbf{h}$ | A distribution $q(\mathbf{z}\|\mathbf{x})$ |
| **Latent space** | No structure, not sampleable | Structured (close to $\mathcal{N}(0,I)$), sampleable |
| **Can generate?** | ❌ No | ✅ Yes, sample from $p(\mathbf{z})$ |
| **Loss function** | Reconstruction error | Reconstruction loss + KL divergence |
| **Training** | Deterministic | Stochastic (reparameterization) |

#### 3.10 Applications of VAE

- **Stable Diffusion's image codec** (KL-VAE): compresses 512×512 images to 64×64 latent space
- Anomaly detection, data augmentation, semi-supervised learning
- Latent space interpolation, attribute editing

---

### 4. GAN Basics

**Generative Adversarial Network**: Two networks trained adversarially

- **Generator** $G$: Generates fake samples from noise $\mathbf{z}$
- **Discriminator** $D$: Distinguishes real from fake samples

$$
\min_G \max_D \; \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}}[\log D(\mathbf{x})] + \mathbb{E}_{\mathbf{z} \sim p(\mathbf{z})}[\log(1 - D(G(\mathbf{z})))]
$$

---

## W2D4: Diffusion Generative Models

---

### 1. Core Idea

> If you know how to gradually "corrode" an image into noise, then learning to "reverse the corrosion" is equivalent to learning to generate.

**Two-step process**:

1. **Forward process (fixed, no learning)**: Gradually add Gaussian noise to data, becoming pure noise after $T$ steps
2. **Reverse process (needs learning)**: Train a neural network to gradually denoise, turning pure noise back into data

```
Forward: x_0 → x_1 → x_2 → ... → x_T    (gradual noising, signal decays)
Reverse: x_T → x_{T-1} → ... → x_0      (gradual denoising, network predicts noise)
```

**Signal-to-noise ratio**: $\mathrm{SNR}(t) = \frac{\bar\alpha_t}{1-\bar\alpha_t}$, monotonically decreasing with $t$.

---

### 2. Forward Process (Noising)

#### 2.1 Single-step noising

Define noise schedule $\beta_1, \beta_2, \ldots, \beta_T$ (typically $\beta_t \in [0.0001, 0.02]$, $T=1000$):

$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}\, x_{t-1}, \beta_t I)$$

Reparameterized form:

$$x_t = \sqrt{1-\beta_t}\, x_{t-1} + \sqrt{\beta_t}\, \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, I)$$

Define $\alpha_t = 1 - \beta_t$, equivalent form:

$$x_t = \sqrt{\alpha_t}\, x_{t-1} + \sqrt{1-\alpha_t}\, \epsilon_t$$

> **Variance preservation**: The coefficient design ensures $(\sqrt{\alpha_t})^2 + (\sqrt{1-\alpha_t})^2 = 1$, so variance neither explodes nor collapses after each noising step.

#### 2.2 Closed-form solution (key derivation)

Define $\bar{\alpha}_t = \prod_{s=1}^{t}\alpha_s$. Using the additivity of independent Gaussians ($a\epsilon_1 + b\epsilon_2 \sim \mathcal{N}(0, (a^2+b^2)I)$), we can directly compute $x_t$ at any time step in one shot:

$$x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1-\bar{\alpha}_t}\, \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

Or in distribution form:

$$q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}\, x_0, (1-\bar{\alpha}_t)\, I)$$

When $t = T$, $\bar{\alpha}_T \approx 0$, so $x_T \approx \mathcal{N}(0, I)$.

> **This is one of the most important derivations in Diffusion**: The closed-form solution allows randomly sampling any $t$ during training and constructing $x_t$ in one step, without iteratively adding noise.

---

### 3. Reverse Process (Core Difficulty)

#### 3.1 Problem: Why is $q(x_{t-1}|x_t)$ intractable?

Expanding with Bayes' theorem:

$$q(x_{t-1}|x_t) = \frac{q(x_t|x_{t-1}) \cdot q(x_{t-1})}{q(x_t)}$$

- $q(x_t|x_{t-1})$ is a known Gaussian ✓
- $q(x_{t-1})$ and $q(x_t)$ are **marginal distributions**, requiring integration over the entire data distribution:

$$q(x_{t-1}) = \int q(x_{t-1}|x_0) \cdot q_{\text{data}}(x_0) \, dx_0$$

This is a **mixture** (not linear combination) of $N$ Gaussians — an extremely complex multimodal distribution with no closed-form expression.

> **Key distinction — Gaussian "linear combination" vs "mixture"**:
> - **Linear combination** $Z = aX + bY$: arithmetic on sampled values, result is still Gaussian (as in noise merging during forward process)
> - **Mixture** $p(x) = \sum_i w_i \cdot \mathcal{N}(x; \mu_i, \sigma_i^2)$: weighted average of probability densities, result is generally not Gaussian (multimodal)

#### 3.2 Breakthrough: Given $x_0$, everything becomes known Gaussians

$$q(x_{t-1}|x_t, x_0) = \frac{q(x_t|x_{t-1}) \cdot q(x_{t-1}|x_0)}{q(x_t|x_0)}$$

All three terms on the right are known Gaussian distributions! Three known Gaussians doing Bayesian operations, result is still Gaussian.

> **Essence**: Given $x_0$, the "$N$-Gaussian mixture" collapses to "one definite Gaussian".

#### 3.3 Completing the square

Deriving the posterior mean and variance through completing the square:

$$\tilde\beta_t = \frac{\beta_t(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}$$

$$\tilde\mu_t = \frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\,x_t + \frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{1-\bar\alpha_t}\,x_0$$

$$q(x_{t-1}|x_t, x_0) = \mathcal{N}(x_{t-1}; \tilde{\mu}_t(x_t, x_0), \tilde{\beta}_t I)$$

> **Intuition for $\tilde\mu_t$**: It's a "compromise" between $x_t$ (current noisy image) and $x_0$ (original data) — when noise is high ($t$ large), it relies more on $x_0$; when noise is low, it relies more on $x_t$.

#### 3.4 Replacing $x_0$ with noise $\epsilon$ (connecting to neural network)

During inference, we don't have $x_0$. Using the forward closed-form solution to invert: $x_0 = \frac{x_t - \sqrt{1-\bar\alpha_t}\,\epsilon}{\sqrt{\bar\alpha_t}}$, substituting into $\tilde\mu_t$:

$$\tilde\mu_t = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon\right)$$

> **Core conclusion**: $\tilde\mu_t$ depends only on $x_t$ (known) and $\epsilon$ (the only unknown). Training a neural network $\epsilon_\theta(x_t, t)$ to predict $\epsilon$ completes the reverse denoising.

#### 3.5 Complete logic chain

1. Want $q(x_{t-1}|x_t)$ → **intractable** (needs entire data distribution)
2. Fall back to $q(x_{t-1}|x_t, x_0)$ → **tractable** (three known Gaussians, completing the square)
3. Posterior mean $\tilde\mu_t(x_t, x_0)$ → depends on $x_0$, not available at inference
4. Replace $x_0$ with $\epsilon$ → $\tilde\mu_t(x_t, \epsilon)$
5. Train network to predict $\epsilon$ → $\epsilon_\theta(x_t, t) \approx \epsilon$

---

### 4. Loss Function Derivation

#### 4.1 Variational Lower Bound (VLB)

Objective: Maximize $\log p_\theta(x_0)$. Same approach as VAE — introduce the forward process $q(x_{1:T}|x_0)$ as a "bridge", derive lower bound via Jensen's inequality:

$$\log p_\theta(x_0) \geq \mathbb{E}_{q}\left[\log \frac{p_\theta(x_{0:T})}{q(x_{1:T}|x_0)}\right]$$

**Comparison: VAE vs Diffusion**

| | VAE | Diffusion |
|---|---|---|
| Latent variable | $\mathbf{z}$ (single vector) | $x_1, x_2, \ldots, x_T$ (entire Markov chain) |
| Approximate posterior | $q_\phi(\mathbf{z}\|x_0)$ (learnable Encoder) | $q(x_{1:T}\|x_0)$ (fixed forward noising, no learning) |
| Generative model | $p_\theta(x_0, \mathbf{z}) = p_\theta(x_0\|\mathbf{z}) \cdot p(\mathbf{z})$ | $p_\theta(x_{0:T}) = p(x_T) \cdot \prod p_\theta(x_{t-1}\|x_t)$ |

#### 4.2 Decomposition into per-step KL

Through Bayesian flipping and telescoping, VLB decomposes into:

$$-\text{ELBO} = \underbrace{D_{\text{KL}}(q(x_T|x_0) \| p(x_T))}_{L_T} + \sum_{t=2}^{T} \underbrace{D_{\text{KL}}(q(x_{t-1}|x_t, x_0) \| p_\theta(x_{t-1}|x_t))}_{L_{t-1}} - \underbrace{\mathbb{E}_{q}[\log p_\theta(x_0|x_1)]}_{L_0}$$

- $L_T$: Forward endpoint matching prior, constant ($\approx 0$)
- $L_{t-1}$: **Core term**, model's denoising step matching true reverse posterior
- $L_0$: Final reconstruction loss

#### 4.3 Computing the core term $L_{t-1}$

$$L_{t-1} = D_{\text{KL}}(q(x_{t-1}|x_t, x_0) \| p_\theta(x_{t-1}|x_t))$$

- True posterior: Gaussian with mean $\tilde\mu_t(x_t, x_0)$ and variance $\tilde\beta_t$
- Model distribution: Also set to Gaussian, variance fixed to $\tilde\beta_t$, only learn mean $\mu_\theta(x_t, t)$

KL divergence between two Gaussians with equal variance = squared difference of means:

$$L_{t-1} = \frac{1}{2\tilde\beta_t}\|\tilde\mu_t(x_t, x_0) - \mu_\theta(x_t, t)\|^2$$

#### 4.4 Parameterization as noise prediction

Replace the $\epsilon$ in the true posterior mean with network prediction $\epsilon_\theta$:

$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon_\theta(x_t, t)\right)$$

After substitution, $x_t$ terms cancel:

$$L_{t-1} = \frac{\beta_t^2}{2\tilde\beta_t \alpha_t (1-\bar\alpha_t)}\|\epsilon - \epsilon_\theta(x_t, t)\|^2$$

#### 4.5 Simplified loss (DDPM)

> **Ho et al. 2020 simplified loss**: Removing the time-step weighting coefficient and using the simplified loss directly works better in practice:
>
> $$\mathcal{L}_{\text{simple}} = \mathbb{E}_{x_0, \epsilon \sim \mathcal{N}(0,I), t \sim U\{1,T\}}\left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]$$
>
> where $x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\epsilon$.

> **Why does removing weights work better?** The simplified loss gives uniform weight to all time steps, giving more attention to high noise levels (responsible for global structure), producing better generation quality in practice.

---

### 5. Three Equivalent Prediction Targets

Everything stems from the forward closed-form solution: $x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon$

Given $x_t$ (known), $\epsilon$, $x_0$, $v$ are interconvertible:

$$\epsilon = \frac{x_t - \sqrt{\bar\alpha_t}\,x_0}{\sqrt{1-\bar\alpha_t}}, \quad x_0 = \frac{x_t - \sqrt{1-\bar\alpha_t}\,\epsilon}{\sqrt{\bar\alpha_t}}$$

Velocity $v$ is a linear combination of $\epsilon$ and $x_0$:

$$v = \sqrt{\bar\alpha_t}\,\epsilon - \sqrt{1-\bar\alpha_t}\,x_0$$

**Comparison of three prediction targets**:

| | $\epsilon$ prediction | $x_0$ prediction | $v$ prediction |
|---|---|---|---|
| Network output | Predicts noise | Predicts denoised original image | "Rotation" of signal and noise |
| Small $t$ (less noise) | Weak noise signal | Close to $x_t$, easy | Stable |
| Large $t$ (more noise) | Close to $x_t$, easy | Far from $x_t$, difficult | Stable |
| Numerical stability | Unstable as $t \to 0$ | Unstable as $t \to T$ | **Stable at both ends** |
| Representative work | DDPM (Ho 2020) | DALL·E (Ramesh et al.) | Stable Diffusion v2+ |

---

### 6. Sampling: DDPM and DDIM

#### 6.1 DDPM sampling (stochastic, T steps)

$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\,\epsilon_\theta(x_t, t)\right) + \sigma_t\, z, \quad z \sim \mathcal{N}(0, I)$$

This is a discretization of an SDE, adding new random noise at each step. Must complete all $T$ steps.

```python
# DDPM sampling
x_T = torch.randn(...)  # Start from pure noise
for t in reversed(range(1, T+1)):
    z = torch.randn_like(x_T) if t > 1 else 0
    eps_pred = model(x_T, t)
    x_T = (1/sqrt(alpha_t)) * (x_T - (beta_t/sqrt(1-alpha_bar_t)) * eps_pred) + sigma_t * z
```

#### 6.2 DDIM sampling (deterministic, accelerated)

> **DDIM core insight**: The reverse process doesn't have to be a stochastic Markov chain. The only constraint is that the forward marginal distribution remains unchanged.

**Two-step strategy**:

1. Estimate $\hat{x}_0 = \frac{x_t - \sqrt{1-\bar{\alpha}_t}\,\epsilon_\theta(x_t, t)}{\sqrt{\bar\alpha_t}}$ with the network
2. "Re-noise" to target time: $x_{t-1} = \sqrt{\bar{\alpha}_{t-1}}\,\hat{x}_0 + \sqrt{1-\bar{\alpha}_{t-1}}\,\epsilon_\theta(x_t, t)$

> **Why can we skip steps?** The formula only uses cumulative coefficients $\bar\alpha$, not single-step decay $\alpha_t$. Define subsequence $\tau = [T, T-k, T-2k, \ldots, 0]$, each step directly jumps $k$ time steps.

```python
# DDIM sampling (deterministic, σ=0)
x_T = torch.randn(...)
tau = list(range(0, T, T // S))  # Subsequence, e.g., S=50 steps
for t in reversed(tau):
    eps_pred = model(x_T, t)
    x0_pred = (x_T - sqrt(1 - alpha_bar_t) * eps_pred) / sqrt(alpha_bar_t)
    x_T = sqrt(alpha_bar_prev) * x0_pred + sqrt(1 - alpha_bar_prev) * eps_pred
```

**DDPM vs DDIM comparison**:

| | DDPM | DDIM |
|---|---|---|
| Stochasticity | Yes (new noise each step) | No (deterministic, reproducible) |
| Steps | Must be T steps | Can skip steps (e.g., 50 steps) |
| Latent interpolation | Not supported | Supported ($x_T \leftrightarrow x_0$ deterministic bijection) |

> **Essence**: DDIM removes stochastic noise, converting SDE to ODE.

---

### 7. Score Matching Perspective

#### 7.1 Score Function

$$s(x) = \nabla_x \log p(x)$$

> **Intuition**: Imagine the data distribution $p(x)$ as a topographic map, where "elevation" represents probability density. The score function is the uphill direction at each point — pointing toward the fastest increase in probability density.
> - At peaks, score $\approx 0$ (already at the summit)
> - At valleys, score points toward the nearest peak

> **Why learn score instead of directly learning $p(x)$?** Directly learning $p(x)$ requires ensuring it integrates to 1 (the normalization constant $Z$ is extremely hard to compute in high dimensions), while $\nabla_x \log p(x) = \nabla_x \log \frac{p^*(x)}{Z} = \nabla_x \log p^*(x)$ — the $Z$ disappears when taking the gradient w.r.t. $x$.

#### 7.2 Predicting noise = learning Score

> **Core conclusion**:
> $$\epsilon_\theta(x_t, t) \approx -\sqrt{1-\bar\alpha_t}\,\nabla_{x_t}\log q(x_t)$$

Intuition: Noise pushes you away from data, score pulls you back — they differ by a negative sign and a scaling factor.

#### 7.3 Probability Flow ODE

Song et al. 2021 proved: For any SDE, there exists a deterministic ODE such that both have exactly the same probability distribution at every time step:

$$\frac{dx}{dt} = f(x,t) - \frac{1}{2}g(t)^2 \nabla_x \log p_t(x)$$

> **Connection between DDIM and PF-ODE**: DDIM's deterministic sampling is essentially solving the Probability Flow ODE.

---

### 8. Complete Training and Sampling Code

**Training**:

```python
for x0 in dataloader:
    # Randomly sample time steps
    t = torch.randint(1, T+1, (batch_size,))
    # Sample noise
    eps = torch.randn_like(x0)
    # Construct x_t (closed-form, one shot)
    x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * eps
    # Predict noise
    eps_pred = model(x_t, t)
    # Simplified loss
    loss = F.mse_loss(eps_pred, eps)
    loss.backward()
    optimizer.step()
```

**DDPM sampling**:

```python
x = torch.randn(batch_size, C, H, W)
for t in reversed(range(1, T+1)):
    eps_pred = model(x, t)
    mu = (1/sqrt(alpha_t)) * (x - (beta_t/sqrt(1-alpha_bar_t)) * eps_pred)
    if t > 1:
        x = mu + sqrt(beta_t) * torch.randn_like(x)
    else:
        x = mu
```

**DDIM sampling**:

```python
x = torch.randn(batch_size, C, H, W)
tau = list(range(0, T, T // 50))  # 50 steps
for i, t in enumerate(reversed(tau)):
    eps_pred = model(x, t)
    x0_pred = (x - sqrt(1-alpha_bar_t) * eps_pred) / sqrt(alpha_bar_t)
    t_prev = tau[len(tau)-2-i] if i < len(tau)-1 else 0
    alpha_bar_prev = alpha_bars[t_prev] if t_prev > 0 else 1.0
    x = sqrt(alpha_bar_prev) * x0_pred + sqrt(1-alpha_bar_prev) * eps_pred
```

---

## W2D5: Sequences and Language

---

### 1. Word Embeddings

Mapping discrete words to a continuous vector space.

#### 1.1 Word2Vec

**Core idea**: Words that appear in similar contexts have similar meanings.

**Skip-gram model**: Given a center word, predict the context words

$$
P(w_{\text{context}} | w_{\text{center}}) = \frac{\exp(\mathbf{v}_{\text{context}} \cdot \mathbf{v}_{\text{center}})}{\sum_{w} \exp(\mathbf{v}_w \cdot \mathbf{v}_{\text{center}})}
$$

#### 1.2 Interesting Properties of Word Embeddings

**Analogy relationships**:

$$
\mathbf{v}_{\text{king}} - \mathbf{v}_{\text{man}} + \mathbf{v}_{\text{woman}} \approx \mathbf{v}_{\text{queen}}
$$

**Cosine similarity**:

$$
\text{sim}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|}
$$

---

### 2. Text Classification Pipeline

```
Text → Tokenization → Embedding → Aggregation (Pooling/Attention) → Classifier
```

```python
class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, pretrained_embeddings):
        super().__init__()
        self.embedding = nn.EmbeddingBag.from_pretrained(pretrained_embeddings)
        self.fc1 = nn.Linear(embed_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)  # Average word embeddings
        x = F.relu(self.fc1(embedded))
        return self.fc2(x)
```

---

### 3. RNN Basics

A natural choice for processing sequential data:

$$
\mathbf{h}_t = \sigma(\mathbf{W}_{hh} \mathbf{h}_{t-1} + \mathbf{W}_{xh} \mathbf{x}_t + \mathbf{b})
$$

**Problems**:
- Vanishing/exploding gradients: Gradients decay or grow exponentially in long sequences
- Sequential computation: Cannot be parallelized

**LSTM solution**: Introduces gating mechanisms (forget gate, input gate, output gate) and cell state

**Simplified GRU**: Only two gates (reset gate, update gate), fewer parameters

---

### 4. Context-Free vs. Contextualized Embeddings

| Type             | Representative    | Characteristics                                   |
| ---------------- | ----------------- | ------------------------------------------------- |
| **Context-free** | Word2Vec, FastText | Each word has a fixed vector                      |
| **Contextualized** | ELMo, BERT      | The same word has different vectors in different contexts |

**Advantage of contextualized embeddings**: Can distinguish polysemous words (e.g., the different meanings of "bank" in "river bank" and "bank account").

---

## References

- [Deep Learning Book, Chapter 7 (Regularization)](https://www.deeplearningbook.org/contents/regularization.html)
- [Deep Learning Book, Chapter 9 (CNNs)](https://www.deeplearningbook.org/contents/convnets.html)
- [Deep Learning Book, Chapter 20 (Generative Models)](https://www.deeplearningbook.org/contents/generative.html)
- [What are Diffusion Models? (Lil'Log)](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
- [The Illustrated Word2Vec](https://jalammar.github.io/illustrated-word2vec/)
