[[toc]]
# Non-Linear Response from A Linear Model
Introduce **basis functions** for non-linear regression models.
## I. Nonlinear Regression
* Problem with Linear Regression: $\mathbf{x}$ may not be linearly related to $\mathbf{y}$

* Solution: create a feature space: define $\phi(\mathbf{x})$ where $\phi(\cdot)$ is a nonlinear function of $\mathbf{x}$.

* Model for target is a linear combination of these nonlinear functions (polynomial basis, RBF basis,  Fourier basis and Relu basis).

For example,
$$f(\mathbf{x}) = \sum_{j=0}^m w_j \phi_j(\mathbf{x})$$
### 0. Defermine Linear Functions (input & parameters)
(a) $f(x) = w_1x_1 + w_2$

(b) $f(x) = w_1\exp(x_1) + w_2x_2 + w_3$

(c\) $f(x) = \log(x_1^{w_1}) + w_2x_2^2 + w_3$

(d) $f(x) = \exp(-\sum_i(x_i - w_i)^2)$

(e) $f(x) = \exp(-\mathbf{w}^\top \mathbf{x})$

>
>(a) The model is linear in both the inputs and the parameters.
>
>(b) The model is non-linear in the inputs, but linear in the parameters.
>
>(c\) The model is non-linear in the inputs, but linear in the parameters.
>
>(d) The model is non-linear in both the inputs and the parameters.
>
>(e) The model is non-linear in both the inputs and the parameters.

### 1. Polynomial Basis
Here, the feature space is
$$\boldsymbol{\phi} = [1, x, x^2]$$
Thus, the model becomes,
$$
\begin{align}
    f(x) &= {\color{\redColor}w_0}\phi_0(x) + {\color{\magentaColor}w_1 \phi_1(x)} + {\color{\blueColor}w_2
                                                                                 \phi_2(x)}\\
         &= {\color{\redColor}w_0} \quad\;\;\;\,+ {\color{\magentaColor}w_1x} \quad\;\,+ {\color{\blueColor}w_2 x^2}
\end{align}
$$
And the result could be,

![](img/Jietu20200117-013438.jpg)
### 2. Radial Basis Functions
Here, the feature space is
$$\phi_j(x) = \exp\left(-\frac{(x-\mu_j)^2}{2\ell^2}\right)$$
And the model becomes,
$$f(x) = {\color{\redColor}w_0 e^{-2(x+1)^2}}  + {\color{\magentaColor}w_1e^{-2x^2}} + {\color{\blueColor}w_2 e^{-2(x-1)^2}}$$
And the result could be,

![](img/Jietu20200117-021524.jpg)

### 3. Polynomial Discussion
Increase the **Polynomial Order** will result in a model closer to the **training data**.
* The training loss decreases as the polynomial order increase.
* The validation loss increase as the polynomial order increase.

Hence, this leads to the hyper parameters optimization problem. It can be solve by **cross-validation**.

Specially, we discuss the **$K$-fold cross-validation**.
1. Seplit the data into $K$ equally
2. Each block as a validation set test the model which is trained with the other $K-1$ blocks.
3. Average over the resulting $K$ loss.

![](img/Jietu20200118-194929.jpg)

and the average loss for LOOCV is:
$$
\mathfrak{L}^{CV} = \frac{1}{N}\sum_{n=1}^N(y_n - \widehat{w}_{-n}^Tx_n)^2
$$
where $\widehat{w}_{-n}$ is the parameters without $n$th training.
> Tips:
>
> The data can be seplit into folds in different ways.

![](img/Jietu20200118-215215.jpg)
![](img/Jietu20200118-222405.jpg)
Another way is known as **regularisation**.

In general, the higher the sum of the absolute values in $\mathbf{w}$, the more complex the model
$$
\sum_iw_i^2
$$
or after vectorizing,
$$
\mathbf{w}^T\mathbf{w}
$$

In order to keep it low, we add it into the average squared loss $\mathfrak{L}$:
$$
\mathfrak{L}' = \mathfrak{L}+\lambda\mathbf{w}^T\mathbf{w}
$$

![](img/Jietu20200118-222451.jpg)
Then the solution is given by:
$$
\widehat{\mathbf{w}} = (\mathbf{X}^T\mathbf{X}+N\lambda\mathbf{I})^{-1}\mathbf{X}^T\mathbf{y}
$$
## II. Vectorize Nonlinear Regression
The basic definition of functions are

$$f(\mathbf{x}_i) = \mathbf{w}^\top \boldsymbol{\phi}_i$$
$$
\begin{align*}
    \boldsymbol{\phi}_i =
                \begin{bmatrix}
                    \phi_0(\mathbf{x}_i)\\
                    \phi_1(\mathbf{x}_i)\\
                    \vdots\\
                    \phi_m(\mathbf{x}_i)
                \end{bmatrix}.
\end{align*}
$$
We maximize the $\log$ likelihood to get the parameters $\mathbf{w}$ and $\sigma^2$.

And the $\log$ likelihood is
$$
L(\mathbf{w},\sigma^2)= -\frac{n}{2}\log \sigma^2
          -\frac{n}{2}\log 2\pi -\frac{\sum
            _{i=1}^{n}\left(y_i-\mathbf{w}^{\top}\boldsymbol{\phi}_i\right)^{2}}{2\sigma^2}
$$
$$
E(\mathbf{w},\sigma^2)= \frac{n}{2}\log
          \sigma^2 + \frac{\sum
            _{i=1}^{n}\left(y_i-\mathbf{w}^{\top}\boldsymbol{\phi}_i\right)^{2}}{2\sigma^2}
$$
> 误差函数等效于负对数似然，此处忽略常数 $-\frac{n}{2}\log 2\pi$.

After expand the Brackets
$$
\begin{align}
E(\mathbf{w},\sigma^2) &=  \frac{n}{2}\log \sigma^2 + \frac{1}{2\sigma^2}\sum _{i=1}^{n}y_i^{2}-\frac{1}
{\sigma^2}\sum _{i=1}^{n}y_i\mathbf{w}^{\top}\boldsymbol{\phi}_i
+\frac{1}{2\sigma^2}\sum_{i=1}^{n}
\mathbf{w}^{\top}\boldsymbol{\phi}_i\boldsymbol{\phi}_i^{\top}\mathbf{w}\\

&= \frac{n}{2}\log \sigma^2 + \frac{1}{2\sigma^2}\sum _{i=1}^{n}y_i^{2}-\frac{1}{\sigma^2}\mathbf{w}^\top\sum_{i=1}^{n}\boldsymbol{\phi}_i y_i
+\frac{1}{2\sigma^2}\mathbf{w}^{\top}\left[\sum_{i=1}^{n}\boldsymbol{\phi}_i\boldsymbol{\phi}_i^{\top}\right]\mathbf{w}
\end{align}
$$

We calculate the differentiating, and it leads to
$$\mathbf{w}^{*}=\left[\sum _{i=1}^{n}\boldsymbol{\phi}_i\boldsymbol{\phi}_i^{\top}\right]^{-1}\sum _{i=1}^{n}\boldsymbol{\phi}_iy_i = \left(\boldsymbol{\Phi}^\top \boldsymbol{\Phi}\right)^{-1} \boldsymbol{\Phi}^\top \mathbf{y}\\
\left.\sigma^2\right.^{{*}}=\frac{\sum _{i=1}^{n}\left(y_i-\left.\mathbf{w}^{*}\right.^{\top}\boldsymbol{\phi}_i\right)^{2}}{n}
$$
