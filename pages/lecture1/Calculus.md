## Part 1: Calculus

### Limit: Definition

The **limit** describes the value that $f(x)$ approaches as $x$ gets arbitrarily close to a point $a$:

$$\lim_{x \to a} f(x) = L$$

<v-click>

Formally: for every $\varepsilon > 0$, there exists $\delta > 0$ such that

$$0 < |x - a| < \delta \implies |f(x) - L| < \varepsilon$$

</v-click>

<v-click>

<div class="mt-4 p-3 bg-blue-900/20 rounded-lg">

**Intuition**: We can make $f(x)$ as close to $L$ as we want by choosing $x$ sufficiently close to $a$ (but $x \neq a$).

</div>

</v-click>

---

### Limit: Examples

<div class="grid grid-cols-2 gap-8">
<div>

<v-click>

**Direct substitution** (when continuous):

$$\lim_{x \to 2} (x^2 + 1) = 2^2 + 1 = 5$$

</v-click>

<v-click>

**Indeterminate form** $\frac{0}{0}$ — factor and cancel:

$$\lim_{x \to 1} \frac{x^2 - 1}{x - 1} = \lim_{x \to 1} \frac{(x-1)(x+1)}{x-1} = \lim_{x \to 1}(x+1) = 2$$

</v-click>

</div>
<div>

<v-click>

**Important limits**:

$$\lim_{x \to 0} \frac{\sin x}{x} = 1$$

$$\lim_{x \to 0} \frac{e^x - 1}{x} = 1$$

$$\lim_{x \to \infty}\left(1 + \frac{1}{x}\right)^x = e$$

</v-click>

</div>
</div>

---

### Limit: Properties

<v-click>

**1. Linearity**:

$$\lim_{x \to a} [f(x) + g(x)] = \lim_{x \to a} f(x) + \lim_{x \to a} g(x)$$

$$\lim_{x \to a} [c \cdot f(x)] = c \cdot \lim_{x \to a} f(x)$$

</v-click>

<v-click>

**2. Product and Quotient**:

$$\lim_{x \to a} [f(x) \cdot g(x)] = \lim_{x \to a} f(x) \cdot \lim_{x \to a} g(x)$$

$$\lim_{x \to a} \frac{f(x)}{g(x)} = \frac{\displaystyle\lim_{x \to a} f(x)}{\displaystyle\lim_{x \to a} g(x)} \quad \text{(if } \lim_{x \to a} g(x) \neq 0\text{)}$$

</v-click>

<v-click>

**3. Squeeze Theorem**: If $g(x) \leq f(x) \leq h(x)$ near $a$, and $\lim_{x \to a} g(x) = \lim_{x \to a} h(x) = L$, then $\lim_{x \to a} f(x) = L$.

</v-click>

---

### Limit → Derivative

The derivative is **defined** as a limit:

$$f'(x) = \lim_{\Delta x \to 0} \frac{f(x + \Delta x) - f(x)}{\Delta x}$$

<v-click>

<div class="mt-4 p-3 bg-blue-900/20 rounded-lg">

**Connection**: Without limits, we cannot rigorously define derivatives. The limit captures the idea of "instantaneous rate of change" — the slope of the tangent line as the two points on the secant line merge into one.

</div>

</v-click>

<v-click>

<div class="mt-4 p-3 bg-green-900/20 rounded-lg">

**Example**: Derivative of $f(x) = x^2$ from first principles:

$$f'(x) = \lim_{\Delta x \to 0} \frac{(x+\Delta x)^2 - x^2}{\Delta x} = \lim_{\Delta x \to 0} \frac{2x\Delta x + (\Delta x)^2}{\Delta x} = \lim_{\Delta x \to 0}(2x + \Delta x) = 2x$$

</div>

</v-click>

---

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
<v-click>
<div class="mt-4 p-4 bg-blue-900/20 rounded-lg">

Question: we have some $x_0$ and $x_1 = x_0 - \beta f'(x_0)$ ($\beta$ is small enough), then $f(x_0) \mathbf{O} f(x_1)$?

</div>
</v-click>

---

<div class="grid grid-cols-2 gap-8">
<div>

### Intuition

<v-click>

$$x_1 = x_0 - \beta f'(x_0)$$

- If $f'(x_0) > 0$: $f$ is increasing, $x_1 < x_0$ (move left) $\Rightarrow$ $f$ decreases
- If $f'(x_0) < 0$: $f$ is decreasing, $x_1 > x_0$ (move right) $\Rightarrow$ $f$ decreases
- Anyway $x_1$ moves in the direction that reduces $f$

$$\boxed{f(x_0) \geq f(x_1)}$$

</v-click>

</div>
<div>

### Derivation (Taylor Expansion)

<v-click>

$$x_1 = x_0 - \beta f'(x_0)$$

$$f(x_1) \approx f(x_0) + f'(x_0)(x_1 - x_0)$$

Substitute $x_1 - x_0 = -\beta f'(x_0)$:

$$f(x_1) \approx f(x_0) - \beta\left(f'(x_0)\right)^2$$

Since $\beta > 0$ and $\left(f'(x_0)\right)^2 \geq 0$:

$$\boxed{f(x_1) \leq f(x_0)}$$

</v-click>

</div>
</div>

<v-click>

<div class="mt-2 p-2 bg-blue-900/20 rounded-lg text-center">

**Taylor Expansion**: For a smooth function $f$ near point $a$:

$$f(x) = f(a) + f'(a)(x-a) + \frac{f''(a)}{2!}(x-a)^2 + \cdots+\frac{f^{(n)}(a)}{n!}(x-a)^n+o((x-a)^n)$$

For $a=0$,$f(x) = f(0) + f'(0)(x) + \frac{f''(0)}{2!}(x)^2 + \cdots+\frac{f^{(n)}(0)}{n!}(x)^n+o((x)^n)$

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

**Key point**: points which are extrema $\rightarrow f'(x)=0$.

**The Basis of Optimization**

</v-click>
<v-click>
<div class="mt-4 p-3 bg-blue-900/20 rounded-lg">
Points where f'(x) =0,then f(x)?
</div>
</v-click>

</div>
</div>

---

### Taylor Expansion Visualization

<div class="grid grid-cols-2 gap-6" style="height: 380px;">
<div>

<TaylorExpansion type="poly6" />

</div>
<div>

<TaylorExpansion type="ln" />

</div>
</div>

<div class="bg-blue-900/20 rounded-lg text-center">

$$f(x) = f(a) + f'(a)(x-a) + \frac{f''(a)}{2!}(x-a)^2 + \cdots+\frac{f^{(n)}(a)}{n!}(x-a)^n+o((x-a)^n)$$

</div>

---

### Taylor Expansion Visualization-2

<div class="grid grid-cols-2 gap-6" style="height: 380px;">
<div>

<TaylorExpansion type="exp" />

</div>
<div>

<TaylorExpansion type="sin" />

</div>
</div>
<div class="bg-blue-900/20 rounded-lg text-center">

$$f(x) = f(a) + f'(a)(x-a) + \frac{f''(a)}{2!}(x-a)^2 + \cdots+\frac{f^{(n)}(a)}{n!}(x-a)^n+o((x-a)^n)$$

</div>

---

### Questions

<v-click>

<div class="p-2 bg-blue-900/20 rounded-lg">

**Q1**: For $f(x) = x^6 - 3x^4 + 2x^3 + x$ at $a=0$: why does $T_2(x)$ equal $T_1(x)$, and $T_5(x)$ equal $T_4(x)$?

</div>

</v-click>

<v-click>

<div class="mt-2 p-2 bg-blue-900/20 rounded-lg">

**Q2**: For $\sin x$ at $a=0$: why does every even-order expansion equal the previous odd-order one? (i.e. $T_{2k} = T_{2k-1}$)

</div>

</v-click>

<v-click>

<div class="mt-2 text-sm text-gray-400 text-center">

Hint: Look at which coefficients are zero and why.

</div>

</v-click>

<v-click>

<div class="mt-2 p-2 bg-green-900/20 rounded-lg">

**A1**: $f(x) = x^6 - 3x^4 + 2x^3 + x$ has **no** $x^2$ or $x^5$ terms. At $a=0$, the Taylor coefficients $c_2 = \frac{f''(0)}{2!} = 0$ and $c_5 = \frac{f^{(5)}(0)}{5!} = 0$, so adding these terms changes nothing.

</div>

</v-click>

<v-click>

<div class="mt-2 p-2 bg-green-900/20 rounded-lg">

**A2**: $\sin x$ is an **odd function** ($\sin(-x) = -\sin x$), so its Taylor series at $a=0$ contains only **odd** powers: $\sin x = x - \frac{x^3}{3!} + \frac{x^5}{5!} - \cdots$. All even-order coefficients are zero, hence $T_{2k} = T_{2k-1}$.

</div>

</v-click>

---

### Common Differentiation Rules

| Function $y$  | Derivative $\dfrac{dy}{dx}$           | Differential $dy$               | Rule              |
| ------------- | ------------------------------------- | ------------------------------- | ----------------- |
| $y = cf(x)$   | $c\dfrac{df}{dx}$                     | $dy=c\, df=cf'(x)dx$            | Constant multiple |
| $y = f + g$   | $\dfrac{df}{dx} + \dfrac{dg}{dx}$     | $dy=df + dg=(f'(x)+g'(x))dx$    | Sum               |
| $y = f(g(x))$ | $\dfrac{df}{dg} \cdot \dfrac{dg}{dx}$ | $dy=f'(g)\, dg=f'(g(x))g'(x)dx$ | **Chain rule**    |

<v-click>

<div class="mt-4 p-3 bg-blue-900/20 rounded-lg">

The **chain rule** is critical for backpropagation in neural networks. It computes derivatives of composite functions: propagating gradients backward layer by layer.

$$\frac{\partial L}{\partial w_1} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial z} \cdot \frac{\partial z}{\partial w_1}$$

</div>

</v-click>

---

### Gradient Descent Visualization

<div class="grid grid-cols-2 gap-8">
<div style="height: 320px;">

<GradientDescent />
<v-click>

**Two starting points**:

- **Red path**: start from $x_0 = -0.5$
- **Green path**: start from $x_0 = 4.5$
- Both converge to $x = 2$

</v-click>
</div>
<div>

**Goal**: find the minimum of $f(x) = (x-2)^2 + 1$

<v-click>

**Recall**: we proved $f(x_0) \geq f(x_1)$ for each step $x_1 = x_0 - \eta f'(x_0)$.

This guarantees the function value **never increases** at each iteration.

We call this **Gradient Descent**.
</v-click>

<v-click>

**Update rule**: $x_{n+1} = x_n - \eta \cdot f'(x_n)$, with $\eta = 0.3$

No matter which side we start, gradient descent always descends toward the minimum.

</v-click>

</div>
</div>

---

### Learning Rate: Too Small vs Too Large

<div class="grid grid-cols-2 gap-8">
<div style="height: 340px;">

<GradientDescentLR />

</div>
<div>

**Function**:

$$f(x) = (x - 2)^2 + 1$$

**Gradient**:

$$f'(x) = 2(x - 2)$$

**Update rule**:

$$x_{n+1} = x_n - \eta \cdot f'(x_n)$$

<v-click>

**Try adjusting $\eta$** with the slider and observe:

<div class="mt-4 p-3 bg-yellow-900/20 rounded-lg">

**Tradeoff**: Small $\eta$ is safe but slow. Large $\eta$ is fast but risky. Choosing the right learning rate is critical in ML training.

</div>

</v-click>

</div>
</div>

---

### Multiple Local Minima

<div class="grid grid-cols-2 gap-8">
<div style="height: 340px;">

<GradientDescentMulti />

</div>
<div>

**Function**:

$$f(x) = x^4 - 4x^3 + 2x^2 + 4x$$

**Gradient**:

$$f'(x) = 4x^3 - 12x^2 + 4x + 4$$

<v-click>

**Critical points** (where $f'(x) = 0$):

| Point             | Type      |
| ----------------- | --------- |
| $x \approx -0.41$ | Local min |
| $x = 1$           | Local max |
| $x \approx 2.41$  | Local min |

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

### Second-Order Derivative

The **second-order derivative** is the derivative of the derivative, measuring how the rate of change itself changes:

$$f''(x) = \frac{d^2 f}{dx^2} = \frac{d}{dx}\left(\frac{df}{dx}\right) = \lim_{\Delta x \to 0} \frac{f'(x + \Delta x) - f'(x)}{\Delta x}$$

<v-click>

**Geometric meaning**: $f''(x)$ describes the **concavity** of the function.

- $f''(x) > 0$: the function is **concave up** (curves upward, like a cup) → local minimum at critical point
- $f''(x) < 0$: the function is **concave down** (curves downward, like a cap) → local maximum at critical point
- $f''(x) = 0$: possible **inflection point** (concavity changes)

</v-click>

<v-click>

<div class="mt-4 p-3 bg-blue-900/20 rounded-lg">

**Example**: $f(x) = \frac{x^3}{3} - x$
$f'(x) = x^2 - 1, \quad f''(x) = 2x$

At $x = 1$: $f'(1) = 0$ (critical point), $f''(1) = 2 > 0$ → **local minimum**

At $x = -1$: $f'(-1) = 0$ (critical point), $f''(-1) = -2 < 0$ → **local maximum**

</div>

</v-click>

---

### Second-Order Derivative & Optimization

<div class="grid grid-cols-2 gap-8">
<div>

<v-click>

**Second-Order Taylor Expansion** near point $a$:

$$f(x) \approx f(a) + f'(a)(x-a) + \frac{f''(a)}{2}(x-a)^2$$

</v-click>

<v-click>

At a critical point where $f'(a) = 0$:

$$f(x) \approx f(a) + \frac{f''(a)}{2}(x-a)^2$$

- If $f''(a) > 0$: $f(x) \geq f(a)$ → **local minimum**
- If $f''(a) < 0$: $f(x) \leq f(a)$ → **local maximum**

</v-click>

</div>
<div>

<v-click>

<div class="p-3 bg-blue-900/20 rounded-lg">

**In machine learning**: The second derivative tells us about the **curvature** of the loss surface.

- Large $f''$ → steep curvature → small steps are safer
- Small $f''$ → flat region → can take larger steps

This motivates **second-order optimization methods** like Newton's method:

$$x_{n+1} = x_n - \frac{f'(x_n)}{f''(x_n)}$$

</div>

</v-click>

</div>
</div>

---

### Second-Order Partial Derivative

For a multivariable function $f(x_1, x_2, \ldots, x_n)$, we can take partial derivatives with respect to the same variable twice:

$$\frac{\partial^2 f}{\partial x_i^2} = \frac{\partial}{\partial x_i}\left(\frac{\partial f}{\partial x_i}\right)$$

<v-click>

**Example**: $f(x, y) = x^2 y + 3xy^2$

$$\frac{\partial f}{\partial x} = 2xy + 3y^2, \quad \frac{\partial^2 f}{\partial x^2} = 2y$$

$$\frac{\partial f}{\partial y} = x^2 + 6xy, \quad \frac{\partial^2 f}{\partial y^2} = 6x$$

</v-click>

<v-click>

<div class="mt-4 p-3 bg-blue-900/20 rounded-lg">

**Interpretation**: $\frac{\partial^2 f}{\partial x_i^2}$ measures the concavity of $f$ along the $x_i$ direction, holding all other variables constant.

</div>

</v-click>

---

### Second-Order Mixed Partial Derivative

The **mixed partial derivative** involves taking partial derivatives with respect to **different variables**:

$$\frac{\partial^2 f}{\partial x_j \partial x_i} = \frac{\partial}{\partial x_j}\left(\frac{\partial f}{\partial x_i}\right)$$

<v-click>

**Example**: $f(x, y) = x^2 y + 3xy^2$

$$\frac{\partial f}{\partial x} = 2xy + 3y^2 \quad \Rightarrow \quad \frac{\partial^2 f}{\partial y \partial x} = \frac{\partial}{\partial y}(2xy + 3y^2) = 2x + 6y$$

$$\frac{\partial f}{\partial y} = x^2 + 6xy \quad \Rightarrow \quad \frac{\partial^2 f}{\partial x \partial y} = \frac{\partial}{\partial x}(x^2 + 6xy) = 2x + 6y$$

</v-click>

<v-click>

<div class="p-1 bg-green-900/20 rounded-lg">

**Clairaut's Theorem (Symmetry of Mixed Partials)**:

If $f$ has continuous second-order partial derivatives, then:

$$\boxed{\frac{\partial^2 f}{\partial x_j \partial x_i} = \frac{\partial^2 f}{\partial x_i \partial x_j}}$$

The order of differentiation does not matter!

</div>

</v-click>

---

### Hessian Matrix

The **Hessian matrix** collects all second-order partial derivatives of a multivariable function:

$$H(f) = \begin{bmatrix} \dfrac{\partial^2 f}{\partial x_1^2} & \dfrac{\partial^2 f}{\partial x_1 \partial x_2} & \cdots & \dfrac{\partial^2 f}{\partial x_1 \partial x_n} \\[6pt] \dfrac{\partial^2 f}{\partial x_2 \partial x_1} & \dfrac{\partial^2 f}{\partial x_2^2} & \cdots & \dfrac{\partial^2 f}{\partial x_2 \partial x_n} \\[6pt] \vdots & \vdots & \ddots & \vdots \\[6pt] \dfrac{\partial^2 f}{\partial x_n \partial x_1} & \dfrac{\partial^2 f}{\partial x_n \partial x_2} & \cdots & \dfrac{\partial^2 f}{\partial x_n^2} \end{bmatrix}$$

<v-click>

**Example**: $f(x, y) = x^2 y + 3xy^2$

$$H = \begin{bmatrix} \dfrac{\partial^2 f}{\partial x^2} & \dfrac{\partial^2 f}{\partial x \partial y} \\[6pt] \dfrac{\partial^2 f}{\partial y \partial x} & \dfrac{\partial^2 f}{\partial y^2} \end{bmatrix} = \begin{bmatrix} 2y & 2x + 6y \\ 2x + 6y & 6x \end{bmatrix}$$

</v-click>

---

### Hessian & Optimization

<div class="grid grid-cols-2 gap-8">
<div>

<v-click>

**Multivariable second-order Taylor expansion** near point $\mathbf{a}$:

$$f(\mathbf{x}) \approx f(\mathbf{a}) + \nabla f(\mathbf{a})^T (\mathbf{x} - \mathbf{a}) + \frac{1}{2}(\mathbf{x} - \mathbf{a})^T H(\mathbf{a}) (\mathbf{x} - \mathbf{a})$$

</v-click>

<v-click>

At a critical point where $\nabla f(\mathbf{a}) = \mathbf{0}$:

$$f(\mathbf{x}) \approx f(\mathbf{a}) + \frac{1}{2}(\mathbf{x} - \mathbf{a})^T H(\mathbf{a}) (\mathbf{x} - \mathbf{a})$$

</v-click>

</div>
<div>

<v-click>

<div class="p-3 bg-blue-900/20 rounded-lg">

**Classifying critical points** using the Hessian:

- $H$ **positive definite** → all eigenvalues $> 0$ → **local minimum**
- $H$ **negative definite** → all eigenvalues $< 0$ → **local maximum**
- $H$ **indefinite** → mixed signs → **saddle point**

</div>

</v-click>

<v-click>

<div class="mt-4 p-3 bg-yellow-900/20 rounded-lg">

**In ML**: The Hessian informs optimization algorithms. Newton's method uses $H^{-1}$ to adapt step size per direction, converging faster than gradient descent on smooth loss surfaces.

$$\mathbf{w} \leftarrow \mathbf{w} - H^{-1} \nabla L(\mathbf{w})$$

</div>

</v-click>

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

| $f(x)$              | $\int f(x)\, dx$           |
| ------------------- | -------------------------- |
| $x^n$ ($n \neq -1$) | $\dfrac{x^{n+1}}{n+1} + C$ |
| $\dfrac{1}{x}$      | $\ln\|x\| + C$             |
| $e^x$               | $e^x + C$                  |

</v-click>

---

### Properties of Indefinite Integrals

<v-click>

**1. Linearity (Sum Rule)**:

$$\int \left(f(x) + g(x)\right) dx = \int f(x)\, dx + \int g(x)\, dx$$

</v-click>

<v-click>

**2. Constant Multiple Rule**:

$$\int C f(x)\, dx = C \int f(x)\, dx$$

Equivalently: $\int C f(x)\, dx = \int f(x)\, d(Cx) = C \int f(x)\, dx$

</v-click>

<v-click>

<div class="mt-4 p-3 bg-blue-900/20 rounded-lg">

**Example**: $\int (3x^2 + 2x)\, dx = 3\int x^2\, dx + 2\int x\, dx = 3 \cdot \frac{x^3}{3} + 2 \cdot \frac{x^2}{2} + C = x^3 + x^2 + C$

</div>
</v-click>

<v-click>
<div class="mt-4 p-3 bg-blue-900/20 rounded-lg">

**Exerciese**: $\int x + e^x \, dx,\int x + \frac{1}{x}\,dx$

</div>
</v-click>

---

### Integration by Substitution (凑微分法)

$$\int f'(x)dx=\int df(x)=f(x)+C$$
<v-click>

If $\int f(x)\, dx = F(x) + C$, then:

$$\boxed{\int f(g(x))\, g'(x)\, dx =\int f(g(x))dg(x)=\int f(u)du=F(u)+C= F(g(x)) + C}$$
</v-click>

<v-click>

**Example**: $\int 2x \cos(x^2)\, dx$

Let $u = x^2$, then $du = 2x\, dx$.

$$\int 2x \cos(x^2)\, dx = \int \cos u\, du = \sin u + C = \sin(x^2) + C$$

</v-click>

<v-click>

**Example**: $\int e^{3x}\, dx$

Let $u = 3x$, then $du = 3\, dx$, so $dx = \frac{1}{3} du\r
\int e^{3x}\, dx = \frac{1}{3}\int e^u\, du = \frac{1}{3}e^{3x} + C$

</v-click>

<div class="bg-blue-900/20 rounded-lg">

**Exerciese**: $\int e^{-x} \,dx$

</div>

---

### Integration by Parts

From the product rule $(uv)' = u'v + uv'$, we get:

$$\boxed{\int u\, dv = uv - \int v\, du} or \boxed{\int u\, dv + \int v\, du=uv}$$

<v-click>

**Example**: $\int x e^x\, dx$

Let $u = x$, $dv = e^x dx$. Then $du = dx$, $v = e^x$.

</v-click>
<v-click>

$$\int x e^x\, dx = x e^x - \int e^x\, dx = x e^x - e^x + C = (x-1)e^x + C$$

</v-click>

<v-click>
<div class="mt-4 p-3 bg-blue-900/20 rounded-lg">

**Exerciese**: $\int \ln\,x \, dx,\int xe^{-x}\,dx$

</div>
</v-click>

---

### Integral

<script setup>
import { ref } from 'vue'
const n = ref(8)
</script>

An **integral** is the inverse of the derivative, representing the accumulated quantity over $[a, b]$:

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

### What is a Differential Equation?

A **differential equation** is an equation that relates a function to its derivatives.

<v-click>

**Simple example**:

$$\frac{dy}{dt} = ky$$

This says: _the rate of change of $y$ is proportional to $y$ itself._

</v-click>

<v-click>

**Solutions** are functions, not numbers:

$$y(t) = Ce^{kt}$$

where $C$ is an arbitrary constant determined by initial conditions.

</v-click>

---

<v-click>

<div class="mt-3 p-3 bg-blue-900/20 rounded-lg">

**Everyday examples**:

- **Population growth**: $\frac{dP}{dt} = rP$ — growth rate proportional to current population
- **Radioactive decay**: $\frac{dN}{dt} = -\lambda N$ — decay rate proportional to remaining atoms
- **Cooling**: $\frac{dT}{dt} = -k(T - T_{\text{env}})$ — Newton's law of cooling

</div>

</v-click>

<v-click>

<div class="bg-green-900/20 rounded-lg">

**Key idea**: Instead of specifying a function directly, we specify a **rule about how it changes**. Solving the ODE means finding the function that satisfies this rule.

</div>

</v-click>

---

### Ordinary Differential Equations (ODE)

An **ODE** relates a function $y(t)$ to its derivatives. The **order** is the highest derivative present.

<v-click>

**Examples**:

| ODE                  | Order | Type                    |
| -------------------- | ----- | ----------------------- |
| $y' + 2y = 0$        | 1st   | Linear, homogeneous     |
| $y' + 2y = e^t$      | 1st   | Linear, non-homogeneous |
| $y'' + 3y' + 2y = 0$ | 2nd   | Linear, homogeneous     |

</v-click>

<v-click>

<div class="mt-3 p-3 bg-blue-900/20 rounded-lg">

**Why ODEs matter in ML**: Many dynamical systems (e.g. neural ODEs, diffusion models) are formulated as ODEs. Training them requires solving ODEs and differentiating through the solutions.

</div>

</v-click>

---

### First-Order Linear ODE

A **first-order linear ODE** has the form: $y' + p(t)\,y = g(t)$

<v-click>

**Solution method** — Integrating Factor:

1. Compute $\mu(t) = e^{\int p(t)\,dt}$
2. Multiply both sides: $\mu(t)\,y' + \mu(t)\,p(t)\,y = \mu(t)\,g(t)$
3. Left side becomes $\frac{d}{dt}[\mu(t)\,y]$
4. Integrate: $\mu(t)\,y = \int \mu(t)\,g(t)\,dt + C$

</v-click>

<v-click>

**Example**: $y' + 2y = e^t$

$$\mu(t) = e^{\int 2\,dt} = e^{2t}$$

$$\frac{d}{dt}\!\left[e^{2t}y\right] = e^{2t} \cdot e^t = e^{3t}$$

$$e^{2t}y = \frac{1}{3}e^{3t} + C \quad\Rightarrow\quad \boxed{y = \frac{1}{3}e^{t} + Ce^{-2t}}$$

</v-click>

---

### Second-Order Linear ODE

A **second-order linear ODE** has the form:
$y'' + a\,y' + b\,y = g(t)$

<v-click>

**Homogeneous case** ($g(t) = 0$): solve the **characteristic equation**
$r^2 + ar + b = 0$

| Roots                            | General Solution                                      |
| -------------------------------- | ----------------------------------------------------- |
| Two real $r_1 \neq r_2$          | $y = C_1 e^{r_1 t} + C_2 e^{r_2 t}$                   |
| Repeated $r_1 = r_2 = r$         | $y = (C_1 + C_2 t)\,e^{rt}$                           |
| Complex $r = \alpha \pm \beta i$ | $y = e^{\alpha t}(C_1 \cos\beta t + C_2 \sin\beta t)$ |

</v-click>

<v-click>

**Example**: $y'' + 3y' + 2y = 0$

$$r^2 + 3r + 2 = 0 \;\Rightarrow\; (r+1)(r+2) = 0 \;\Rightarrow\; r_1 = -1,\; r_2 = -2$$

$$\boxed{y = C_1 e^{-t} + C_2 e^{-2t}}$$

</v-click>
