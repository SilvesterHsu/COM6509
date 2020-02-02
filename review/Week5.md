# Generalization: Model Validation
## Lab
The true distribution of the data samples, $y$, $x$ is $\mathbb{P}(y, x)$. Then we can define our expected loss, or risk, to be,
$$
R(\mathbf{w}) = \int L(y, x, \mathbf{w}) \mathbb{P}(y, x) \text{d}y \text{d}x.
$$
**Sample Based Approximations:**
$$
\int f(z) p(z) \text{d}z\approx \frac{1}{s}\sum_{i=1}^s f(z_i).
$$
Thus, we can approximate our true integral with the sum,
$$
R(\mathbf{w}) \approx \frac{1}{n}\sum_{i=1}^n L(y_i, x_i, \mathbf{w}),
$$
Minimizing this sum directly is known as **empirical risk minimization**.
