# Objective Function and Supervised Learning
<!-- TOC -->
- [Classification](#classification)
  - [Premise](#premise)
  - [Prediction Function](#prediction-function)
  - [Decision Boundary](#decision-boundary)
  - [Perception Algorithm](#perception-algorithm)
    - [Process](#process)
    - [Why it works?](#why-it-works)
- [Regression](#regression)
  - [Premise](#premise-1)
  - [Steepest Descent](#steepest-descent)
  - [Stochastic Gradient Descent Algorithm](#stochastic-gradient-descent-algorithm)
    - [Process](#process-1)
    - [The Meaning of Stochastic](#the-meaning-of-stochastic)
- [Loss Function](#loss-function)
  - [Definition](#definition)
  - [Aim](#aim)
  - [Turning Points](#turning-points)
<!-- /TOC -->
## Classification
### Premise
* Features: $\mathbf{x}_i$ for the $i$th data point
* Labels: $y_i$ is either $-1$ or $+1$
### Prediction Function
For classification,
* Prediction Function:
$$
f(\mathbf{x}_i) = \text{sign}(\mathbf{w}^T\mathbf{x}_i+b)
$$
* Activative Function:
$$
g(z)=\text{sign}(x)=
\left\{\begin{matrix}
+1 & z>0\\
-1 &  z<0
\end{matrix}\right.
$$
### Decision Boundary
It is a **plane** that distinguishes between different classes, such as -1 and +1.
* Equation:
$$
\mathbf{w}^T\mathbf{x}+b=0
$$
> In the current situation,
> $$
> w_1x_{i,1}+w_2x_{i,2}+b=0
> $$

> 二维空间中，决策边界是一条线；三维空间中，它是一个平面，称为超平面（Hyperplane）.

### Perception Algorithm
#### Process
1. If the vector $\mathbf{x}_i$ is **correctly** classified, **change nothing**.

2. If it is **incorrectly** classified,
   * $\mathbf{x}_i$ is **positive**, and incorrectly classified as **negative**,
   $$
   \mathbf{w}_\text{new} = \mathbf{w}_\text{old} + \eta\mathbf{x}_i\\
   b_\text{new} = b_\text{old} + \eta
   $$
   * $\mathbf{x}_i$ is **negative**, and incorrectly classified as **positive**,
   $$
   \mathbf{w}_\text{new} = \mathbf{w}_\text{old} - \eta\mathbf{x}_i\\
   b_\text{new} = b_\text{old} - \eta
   $$
#### Why it works?
Point $\mathbf{x}_i$ has label $y_i=+1$, but incorrectly classified as $-1$.
$$
f(\mathbf{x}_i) = \text{sign}(\mathbf{w}^T\mathbf{x}_i+b) = -1
$$
We have
$$
\mathbf{w}^T\mathbf{x}_i+b < 0
$$

Then according to the process,
$$
\mathbf{w}_\text{new} = \mathbf{w} + \eta\mathbf{x}_i\\
b_\text{new} = b + \eta
$$
After the process,
$$
\begin{align*}
f(\mathbf{x}_i) &= \text{sign}(\mathbf{w}_\text{new}^T\mathbf{x}_i+b_\text{new})\\
&=\text{sign}((\mathbf{w} + \eta\mathbf{x}_i)^T\mathbf{x}_i+(b+\eta))\\
&=\text{sign}(\mathbf{w}^T\mathbf{x}_i+b+\eta\mathbf{x}_i^T\mathbf{x}_i+\eta)
\end{align*}
$$
where $\eta\mathbf{x}_i^T\mathbf{x}_i+\eta$ (positive) pushes $\text{sign}(\cdot)$ to be posotive.
## Regression
### Premise
* Features: $\mathbf{x}_i$ for the $i$th data point
* Labels: $y_i$ match $y_i=m\mathbf{x}_i+c+\varepsilon$
### Steepest Descent
1. Initialize with a guess for $m$ and $c$.
2. Offset Gradient:
$$
\frac{\text{d}E(m,c)}{\text{d}c} = -2 \sum_{i=1}^{n}(y_i-m\mathbf{x}_i-c)
$$
3. Slope Gradient:
$$
\frac{\text{d}E(m,c)}{\text{d}m} = -2 \sum_{i=1}^{n}\mathbf{x}_i(y_i-m\mathbf{x}_i-c)
$$
### Stochastic Gradient Descent Algorithm
#### Process
Update the guess parameters by subtracting the gradient from the guess.
$$
c_{\text{new}} = c_{\text{old}} - \eta\frac{\text{d}E(m,c)}{\text{d}c}\\
m_{\text{new}} = m_{\text{old}} - \eta\frac{\text{d}E(m,c)}{\text{d}m}\\
$$
Each time, only take a *small* step by using learning rate ($\eta$), otherwise we might overshoot the minimum.
#### The Meaning of Stochastic
We present each data point in a random order, and process one point a time.
> 打乱数据集，然后顺序抽取。
>
> 但实际上应该，在SGD在每次迭代时，从整个数据集中随机选择一个数据点，这样极大的减少了计算量。

Since the data is normally presented in a random order,
$$
m_{\text{new}} = m_{\text{old}} +2\eta[x_i(y_i-m_{\text{old}}x_i-c_{\text{old}})]
$$
## Loss Function
### Definition
Loss function measures how the gap between the model and the real true.

|         名字          |                          公式                          |
| :-------------------: | :----------------------------------------------------: |
| Squared loss function |  $L_n(t_n,f(x_n;w_o,w_1)) = (t_n - f(x_n;w_o,w_1))^2$  |
| Average loss function | $L = \frac{1}{N}\sum_{n=1}^{N}L_n(t_n,f(x_n;w_o,w_1))$ |

More generally,
$$
L = \frac{1}{N}\sum_{n=1}^{N}(t_n - \mathbf{w}^T\mathbf{x}_n)^2
$$
### Aim
$$
\underset{w_0,w_1} {\text{argmin}}\frac{1}{N}\sum_{n=1}^{N}L_n(t_n,f(x_n;w_o,w_1))
$$
or
$$
\underset{w_0,w_1} {\text{argmin}}\frac{1}{N}\sum_{n=1}^{N}(t_n - \mathbf{w}^T\mathbf{x}_n)^2
$$
### Turning Points
The point touch the minima.
* Set the gradient of the loss function to 0.
$$
\frac{\delta f(w)}{\delta w} = 0
$$
For example,
$$
\frac{\partial L}{\partial \mathbf{w}} = \frac{2}{N}\mathbf{x}^T\mathbf{x}\mathbf{w} - \frac{2}{N}\mathbf{x}^T\mathbf{t} = 0 \\
\widehat{\mathbf{w}} = (\mathbf{x}^T\mathbf{x})^{-1}\mathbf{x}^T\mathbf{t}
$$
