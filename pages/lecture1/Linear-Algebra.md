---
title: "Part 2: Linear Algebra"
transition: slide-left
math: true
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

| Operation           | Definition                     | Example                                          |
| ------------------- | ------------------------------ | ------------------------------------------------ |
| **Transpose**       | $(A^T)_{ij} = A_{ji}$          | $\begin{bmatrix} 1 & 3 \\ 2 & 4 \end{bmatrix}$   |
| **Matrix Addition** | $(A+B)_{ij} = A_{ij} + B_{ij}$ | $\begin{bmatrix} 6 & 8 \\ 10 & 12 \end{bmatrix}$ |
| **Scalar Addition** | $(A+c)_{ij} = A_{ij} + c$      | $\begin{bmatrix} 3 & 4 \\ 5 & 6 \end{bmatrix}$   |

---

### Matrix Operations (cont.)

| Operation                   | Definition                               | Example                                            |
| --------------------------- | ---------------------------------------- | -------------------------------------------------- |
| **Element-wise (Hadamard)** | $(A \odot B)_{ij} = A_{ij} \cdot B_{ij}$ | $\begin{bmatrix} 5 & 12 \\ 21 & 32 \end{bmatrix}$  |
| **Matrix Multiplication**   | $(AB)_{ij} = \sum_k A_{ik}B_{kj}$        | $\begin{bmatrix} 19 & 22 \\ 43 & 50 \end{bmatrix}$ |
