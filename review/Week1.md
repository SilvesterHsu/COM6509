# Introduction to Machine Learning and Probability
## Course Book
1. Simon Rogers and Mark Girolami, **A First Course in Machine Learning**, Chapman and Hall/CRC Press, 2nd Edition, 2016
2. Christopher Bishop, **Pattern Recognition and Machine Learning**, Springer-Verlag, 2006

## Basic Definition
### Introduction
|                 类别                  |      种类      |           例子           |
| :-----------------------------------: | :------------: | :----------------------: |
|              Supervised               | Classification | Tumor: malignant? begin? |
|              Supervised               |   Regression   | House Price: \$100~\$200 |
| Unsupervised (offer $X$, without $y$) |   Clustering   |         K-means          |
### Model
* Linaer Model: $y = w_1x+w_0$
* Objective Function: Calculate the error between every point.
$$
\begin{align*}
E(w_0,w_1) &= \sum_{i=1}^{N}(y_i - f(x_i))^2\\
           &= \sum_{i=1}^{N}(y_i - (w_1x+w_0))^2
\end{align*}
$$
* Aim: Estimate the parameters $w_0$ and $w_1$, which <u>best fit the data</u> (lowest error).
$$
\theta = \text{argmin } E(w_0,w_1)
$$
## Probability
### Important Formula
|       名字       |       解释        |                   公式                    |
| :--------------: | :---------------: | :---------------------------------------: |
|    Joint Prob    |    $X$ and $Y$    |    $P(X=x_i,Y = y_i) = \frac{n_ij}{N}$    |
|  Marginal Prob   | Regardless of $Y$ | $P(X=x_i) = \sum_{i=1}^{L}P(X=x_i,Y=y_i)$ |
| Conditional Prob |  Under $X = x_i$  |   $P(Y = y_i|X=x_i) = \frac{n_ij}{c_i}$   |

* Marginal Probability is the calulation of Joint Probability (视作投影)
![Important Formula](img/Jietu20200106-172942.jpg)
