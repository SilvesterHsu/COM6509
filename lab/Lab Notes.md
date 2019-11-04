# COM6509 Machine Learning and Adaptive Intelligence

## Lab 1 Questions

### Question 1

Who invented python and why? What was the language designed to do? What is the origin of the name "python"? Is the language a compiled language? Is it an object orientated language?

#### My Answer

**Creator:** Guido van Rossum

**Designed to do:** Python's design philosophy emphasizes code readability with its notable use of significant whitespace. Its language constructs and object-oriented approach aim to help programmers write clear, logical code for small and large-scale projects

**Origin name:** Monty Python's Flying Circus

#### Answer

Python was created by Guido van Rossum. It was designed to be highly *extensible*. The name comes from the Monty Python comedy group. Python is "only" [compiled to bytecode](https://stackoverflow.com/questions/6889747/is-python-interpreted-or-compiled-or-both). It is an object-oriented language.



### Question 2

Read on the internet about the following python libraries: `numpy`, `matplotlib`, `scipy` and `pandas`. What functionality does each provide in python. What is the `pylab` library and how does it relate to the other libraries?

#### My Answer

**numpy:** NumPy is the fundamental package for scientific computing with Python.

**matplotlib:** Matplotlib is a Python 2D plotting library which produces publication quality figures in a variety of hardcopy formats and interactive environments across platforms.

**scipy:** SciPy is a Python-based ecosystem of open-source software for mathematics, science, and engineering.

**pandas:** pandas is an open source, BSD-licensed library providing high-performance, easy-to-use data structures and data analysis tools for the Python programming language.

#### Answer

`numpy`: it adds support to work with arrays and multidimensional matrices and a collection of mathematical functions to operate over those arrays and matrices.

`matplotlib`: it is a plotting library that allows embedding plots in different applications.

`scipy`: it is a library for scientific computing including numerical and symbolic maths.

`pylab`: it is part of matplotlib that looks to resemble the plotting functionalities of [MATLAB](https://en.wikipedia.org/wiki/MATLAB)

### Question 3

What is jupyter and why was it invented? Give some examples of functionality it gives over standard python. What is the jupyter project? Name two languages involved in the Jupyter project other than python.

#### My Answer

**Other language:** Scala, R, Spark, Mesos Stack

#### Answer

Jupyter was invented to support interactive data science and scientific computing across programming languages. Additional functionalities to python include the possibility to include live code, equations, visualisations and narrative text. The [Jupyter Project](https://en.wikipedia.org/wiki/Project_Jupyter) is a non-profit organisation created to "develop open-source software, open-standards, and services for interactive computing across dozens of programming languages". Other languages involved in the Jupyter Project are Julia and R.

### Question 4

We now have an estimate of the probability a film has greater than 40 deaths. The estimate seems quite high. What could be wrong with the estimate? Do you think any film you go to in the cinema has this probability of having greater than 40 deaths?

Why did we have to use `float` around our counts of deaths and total films? What would the answer have been if we hadn't used the `float` command? If we were using Python 3 would we have this problem?

#### My Answer

**Q1:** The collection of film data needs to be spread throughout the movie catalog, not a horror type movie.

**Q2:** In *python2*, if we don't use `float`, the result would be integer which is wrong. However, this won't matter in *python3*.

#### Answer

There are two observations to make here: 1) Few films have a very large number of deaths, e.g. well beyond 200, that bias the computation of the probability 2) the probability that we computed was obtained from a dataset that cares precisely about films with a significant number of deaths. It was not calculated using a larger dataset that perhaps could contain many more films with fewer deaths.

As long as we use Python 3, we don't need to worry about using `float`.

### Question 5

Compute the probability for the number of deaths being over 40 for each year we have in our `film_deaths` data frame. Store the result in a `numpy` array and plot the probabilities against the years using the `plot` command from `matplotlib`. Do you think the estimate we have created of 𝑃(𝑦|𝑡)P(y|t) is a good estimate? Write your code and your written answers in the box below.

#### My Answer

```python
prob_death = np.array([])
for year in range(film_deaths.Year.min(),film_deaths.Year.max()+1):
    deaths = (film_deaths.Body_Count[film_deaths.Year==year]>40).sum()
    total_films = (film_deaths.Year==year).sum()
    prob_death_year = float(deaths)/float(total_films) if total_films != 0 else 0
    prob_death = np.append(prob_death, prob_death_year)
plt.plot(range(film_deaths.Year.min(),film_deaths.Year.max()+1), prob_death, 'b.')
plt.title('Probability of deaths being over 40 for each year')
plt.ylabel('Probability')
plt.xlabel('year')
```

![](https://tva1.sinaimg.cn/large/006y8mN6gy1g7reyj0zooj30as07qdg9.jpg)

#### Answer

```python
# We import numpy as np
import numpy as np
# We start by selecting the unique years that appear in the dataset
unique_years = film_deaths.Year.sort_values().unique()
number_years = unique_years.size
pyear = np.empty(number_years)
for i in range(number_years):   
    deaths = (film_deaths.Body_Count[film_deaths.Year==unique_years[i]]>40).sum()
    total_films = (film_deaths.Year==unique_years[i]).sum()
    pyear[i] = deaths/total_films
    
# Now we plot
plt.plot(unique_years, pyear, 'rx')
plt.title('P(deaths>40|year)')
plt.ylabel('probability')
plt.xlabel('year')
```

![](https://tva1.sinaimg.cn/large/006y8mN6gy1g7rez42dadj30at07q3yw.jpg)

### Question 6

Write code that computes 𝑃(𝑦) by adding 𝑃(𝑦,𝑡) for all values of 𝑡.

#### My Answer

```python
p_y = (film_deaths.Body_Count>40).sum()/film_deaths.Body_Count.count()
p_y_add = 0
for t in np.unique(film_deaths.Year): 
    p_y_and_t = float((film_deaths.Body_Count[film_deaths.Year==t]>40).sum())/float(film_deaths.Body_Count.count())
    p_y_add += p_y_and_t
print("P(y): {}\nSum of P(y,t): {}".format(p_y,p_y_add))
```

```
P(y): 0.37767220902612825
Sum of P(y,t): 0.37767220902612836
```

#### Answer

```python
# For this question, we can reuse some of the code we already wrote for Question 5.  
# In this question, t corresponds to the different years and y refers to the number of deaths 
# that are greater than 40. We already have a vector that contains the unique years (unique_years)
p_y_and_t_all = np.empty(number_years)
for i in range(number_years):
    p_y_and_t_all[i] = (film_deaths.Body_Count[film_deaths.Year==unique_years[i]]>40).sum()/film_deaths.Body_Count.count()

Py = np.sum(p_y_and_t_all)
Py
# Notice that this value should be exactly the same obtained from the code appearing before Question 4
```

```
0.37767220902612825
```

### Question 7

Now we see we have several additional features including the quality rating (`IMDB_Rating`). Let's assume we want to predict the rating given the other information in the data base. How would we go about doing it?

Using what you've learnt about joint, conditional and marginal probabilities, as well as the sum and product rule, how would you formulate the question you want to answer in terms of probabilities? Should you be using a joint or a conditional distribution? If it's conditional, what should the distribution be over, and what should it be conditioned on?

#### My Answer

It should use conditional distribution.

#### Answer

We can use a conditional probability as a predictive model of the probability of quality rating values given other variables in the dataset. We can ask questions like *what is the probability that the rating for a particular film is equal to 7 given other features for the film?* In this case the distribution will be over `IMDB_Rating` and it would be conditioned on all the other variables `Film`, `Year`, `Body_Count`, `MPAA_Rating`, `Genre`, ...

## Lab 2 Questions

### Question 1

Data ethics. If you find data available on the internet, can you simply use it without consequence? If you are given data by a fellow researcher can you publish that data on line?

#### My Answer

No, the reliability of the data should be considered before using the data.

No, the researcher has copyright to the data and cannot publish the data at will.

#### Answer

It depends on the **licensing** terms of the data. However, if the data has no licensing terms but it is related to a **person** in someway (e.g. health data)then there may be **consequences**, depending on local laws.

In general no, you would certainly have to get their **permission**. Be very careful about data containing any **personal** information. Even data that is anonymized can be "de-anonymized".

### Question 2

Have a look at the matrix `Y_with_NaNs`. The movies data is now in a data frame which contains one column for each user rating the movie. There are some entries that contain 'NaN'. What does the 'NaN' mean in this context?

#### My Answer

NaN indicates data vacancies.

#### Answer

In this context, NaN represents a movie that has not been given a rating by the user of that column.

### Question 3

The dataframes `Y_with_NaNs` and `Y` contain the same information but organised in a different way. Explain what is the difference. We have also included two columns for ratings in dataframe `Y`, `ratingsorig` and `ratings`. Explain the difference.

#### My Answer

`Y_with_NaNs` is a matrix of user and movie; `Y` is a list, and each column is a feature.

The main difference between `ratings` and `ratingsorig` is whether to subtract the mean, and subtracting the mean is more conducive to calculation.

#### Answer

The dataframe `Y_with_NaNs` is organised as a **matrix**, <u>with rows being movies and columns being users</u>. The dataframe `Y` takes this information and makes each entry in `Y_with_NaNs` it's own <u>row with the first column being the user, the next the movie, the next the original rating and the last the transformed rating</u>. In mathematical terms, `Y_with_NaNs` is a matrix with the ijth entry representing the rating $X$ of the ith movie for the jth user and the `Y` is a dataframe with columns, j i $X$. Thus `Y_with_NaNs` is movie-orientation while `Y` is user-orientation.

The ratingsorig is the raw rating given by a user, ratings is the original rating with the mean rating subtracted.

### Question 4

What is the gradient of the objective function with respect to $v_{k, \ell}$? Write your answer in the box below, and explain which differentiation techniques you used to get there. 

#### My Answer

$$
\frac{\text{d}E(\mathbf{U}, \mathbf{V})}{\text{d}v_{k,\ell}} = -2 \sum_i s_{i,k}u_{i,\ell}(y_{i, k} - \mathbf{u}_i^\top\mathbf{v}_{k})
$$

![lab2-01.jpg](https://tva1.sinaimg.cn/large/006y8mN6gy1g8lewmu3tdj30sg0r9juu.jpg)

#### Answer

Initially we have 
$$
E(\mathbf{U}, \mathbf{V}) = \sum_{i,j} s_{i,j} (y_{i,j} - \mathbf{u}_i^\top \mathbf{v}_j)^2
$$
In order to find $ \frac{\text{d}E(\mathbf{U}, \mathbf{V})}{\text{d}v_{k,\ell}}$ we first have to use the product rule as there are 2 distinct functions however as $s_{i,j}$ is a constant, the first one equals zero and so we are left with 
$$
\frac{\text{d}E(\mathbf{U}, \mathbf{V})}{\text{d}v_{k,\ell}} = \sum_{i,j} s_{i,j} \times 
\frac{\text{d}(y_{i, j} - \mathbf{u}_i^\top\mathbf{v}_{j})^2}{\text{d}v_{k,\ell}}
$$
Next we have to perform $\frac{\text{d}(y_{i, j} - \mathbf{u}_i^\top\mathbf{v}_{j})^2}{\text{d}v_{k,\ell}}$ which is done using the chain rule. This gives 
$$
\frac{\text{d}(y_{i, j} - \mathbf{u}_i^\top\mathbf{v}_{j})^2}{\text{d}v_{k,\ell}} = 
2(y_{i, j} - \mathbf{u}_i^\top\mathbf{v}_{j}) \times \frac{-\text{d}\mathbf{u}_i^\top\mathbf{v}_{j}}{\text{d}v_{k,j}}
$$
Now, $\frac{\text{d}\mathbf{u}_i^\top\mathbf{v}_{j}}{\text{d}v_{k,\ell}}$ needs to be differentated. $\mathbf{u}_i^\top\mathbf{v}_{j}$ is the inner product of the vector $\mathbf{u}_i$ and  $\mathbf{v}_{j}$ and so is a sum of products. The only time this will not differentiate to zero is when one of the products in the sum contains $\mathbf{v}_{k,\ell}$, all other terms differentiate to zero. This means that the only inner products that do not differentiate to zero are those that involve $\mathbf{v}_k$. These products differentiate to $u_{i,\ell}$. This gives that:
$$
\frac{\text{d}\mathbf{u}_i^\top\mathbf{v}_{k}}{\text{d}v_{k,\ell}} = u_{i,\ell}
$$
The implications in the full equation is that $j=k$ and so the sum over $j$ now disappears as these other sums are all zero.
And so, putting all this working together gives the final result as:
$$
\frac{\text{d}E(\mathbf{U}, \mathbf{V})}{\text{d}v_{k,\ell}} = -2 \sum_i s_{i,k}u_{i,\ell}(y_{i, k} - \mathbf{u}_i^\top\mathbf{v}_{k}).
$$

### Question 5

What happens as you increase the number of iterations? What happens if you increase the learning rate?

#### My Answer

After increasing the number of iterations, the MSE drops to a lower point (there may be an overfitting problem).

After increasing the learning rate, the gradient drops faster, but the MSE may not converge.

#### Answer

As you increase the number of iterations, the objective function starts **decreasing**.

As you increase the learning rate, initially it **learns** **quicker**, but then later on it takes **steps** that are **too big** and you end up shooting the gradient to places where the **error** is **bigger**.

### Question 6

Create a function that provides the prediction of the ratings for the users in the dataset. Is the quality of the predictions affected by the number of iterations or the learning rate? The function should receive `Y`, `U` and `V` and return the predictions and the absolute error between the predictions and the actual rating given by the users. The predictions and <u>the absolute error should be added as additional columns to the dataframe `Y</u>`.

#### My Answer

```python
def Prediction(U, V, Y):
    Y_without_NaNs = pd.DataFrame(data=np.dot(
        U, V.T).T + Y['ratingsorig'].mean(), columns=Y_with_NaNs.columns, index=Y_with_NaNs.index)
    diff = Y_without_NaNs - Y_with_NaNs
    total = diff.fillna(value=0).apply(abs).sum().sum()
    return Y_without_NaNs, total

prediction, loss = Prediction(U, V, Y)
```

#### Answer

```python
def prediction(Y,U,V):
    pred_df = pd.DataFrame(index = Y.index, columns = ['prediction'])
    abs_error_df = pd.DataFrame(index = Y.index, columns = ['absolute error'])
    for i in Y.index:
        row = Y.iloc[i]
        user = row['users']
        film = row['movies']
        rating = row['ratings']
        pred_df.loc[i] = np.dot(U.loc[user], V.loc[film]) # vTu
        abs_error_df.loc[i] = abs(pred_df.iloc[i, 0]-rating)      
    return pred_df, abs_error_df

pred_df, abs_error_df = prediction(Y, U, V)
Y['prediction'] = pred_df
Y['absolute error'] = abs_error_df
```

### Question 7

Create a stochastic gradient descent version of the algorithm. Monitor the objective function after every 1000 updates to ensure that it is decreasing. When you have finished, plot the movie map and the user map in two dimensions (you can use the columns of the matrices $\mathbf{U}$ for the user map and the columns of $\mathbf{V}$ for the movie map). Provide three observations about these maps.

#### My Answer

None

#### Answer

```python
# Question 7 Code Answer
import matplotlib.pylab as plt
%matplotlib inline

def compute_obj(Y, U, V):
    obj = 0
    for i in Y.index:
        row = Y.iloc[i]
        user = row['users']
        film = row['movies']
        rating = row['ratings']
        prediction = np.dot(U.loc[user], V.loc[film]) # vTu
        diff = prediction - rating # vTu - y
        obj += diff*diff
    return obj

def obj_gradient(rating, u, v):
    prediction = np.dot(u, v)
    diff = prediction - rating
    gU = 2* diff * v
    gV = 2* diff * u

    return gU, gV


def SGD(Y, U, V, learn_rate = 0.01, max_iter = 100, check_obj = 1000):
    update_counter = 0
    obj_prev = None
    converge = False
    
    num_updates = []
    objectives = []
    idx_list = Y.index.values
    for iteration in range(max_iter):
        if converge:
            break
        np.random.shuffle(idx_list)
        for i in idx_list:
            if converge:
                break
            row = Y.iloc[i]
            user = row['users']
            film = row['movies']
            rating = row['ratings']            
            gU, gV = obj_gradient(rating, U.loc[user], V.loc[film])
            U.loc[user] -= learn_rate * gU
            V.loc[film] -= learn_rate * gV 
            update_counter += 1
            if update_counter %check_obj == 0:
                obj = compute_obj(Y, U, V)
                num_updates.append(update_counter)
                objectives.append(obj)
                print('Update %s times, objective %s'%(update_counter, obj))
                if obj_prev == None or obj_prev > obj:
                    obj_prev = obj
                else:
                    converge = True
                    
    plt.plot(num_updates, objectives, 'rx-')
    plt.title('Objectives over updates')
    plt.show()
                    
    return U, V                
            
    
q = 2
U = pd.DataFrame(np.random.normal(size=(nUsersInExample, q))*0.001, index=my_batch_users)
V = pd.DataFrame(np.random.normal(size=(n_movies, q))*0.001, index=indexes_unique_movies)

U, V = SGD(Y, U, V)
```

```python
plt.plot(V[0],V[1],'rx')
plt.title('Movie map')
plt.show()

plt.plot(U[0],U[1],'bx')
plt.title('User map')
plt.show()
```



### Question 8

Use stochastic gradient descent to make a movie map for the MovieLens 100k data. Plot the map of the movies when you are finished.

#### My Answer

```python
# %% Prepare datasets
import numpy as np
import pandas as pd
import pods
import zipfile
import os

# Download datasets
if 'ml-latest-small.zip' not in os.listdir():
    pods.util.download_url(
        "http://files.grouplens.org/datasets/movielens/ml-latest-small.zip")
if 'ml-latest-small' not in os.listdir():
    zip_console = zipfile.ZipFile('ml-latest-small.zip', 'r')
    for name in zip_console.namelist():
        zip_console.extract(name, './')
# %% Format datasets


def ReadData(data_dir: str) -> pd.core.frame.DataFrame:
    return pd.read_csv(data_dir)


def GetRandomSample(data, columns='userId', sample_number=10, seed=410) -> np.ndarray:
    np.random.seed(seed)
    item_uni_list = len(data[columns].unique())
    disorder = np.random.permutation(item_uni_list)
    return disorder[0:sample_number]


# basic parameters
studentId = 410
sample_number = 50

# load data & select user
data_dir = "./ml-latest-small/ratings.csv"
data = ReadData(data_dir).drop('timestamp', axis=1)
sample_user = GetRandomSample(data, 'userId', sample_number, studentId)
data_sample = data[data['userId'].isin(sample_user)]

# Put data into a metrix (movieId x userId)
index_movieId = data_sample.movieId.unique()
index_movieId.sort()
index_userId = sample_user
temp = np.full((len(index_movieId), len(index_userId)), np.nan)
Y_with_NaNs = pd.DataFrame(temp, index=index_movieId, columns=index_userId)
# Copy data
for user, movie, rating in iter(data_sample.values):
    Y_with_NaNs.loc[movie, user] = rating

# Put data into DaraFrame (mean has been subtracted)
Y = data_sample.copy()
Y['rating_nor'] = Y['rating'] - Y['rating'].mean()

# %%


def Grad(U, V, Y):
    dU = pd.DataFrame(np.zeros((U.shape)), index=U.index)
    dV = pd.DataFrame(np.zeros((V.shape)), index=V.index)
    loss = 0
    for user, movie, _, rating in iter(Y.values):
        predict = np.dot(U.loc[user], V.loc[movie])
        diff = np.squeeze(predict) - rating
        loss += diff * diff
        dU.loc[user] += 2 * diff * V.loc[movie]
        dV.loc[movie] += 2 * diff * U.loc[user]
    return loss, dU, dV


# Initialize & set parameters
latent_dimension = 2
lr = 0.01
U = pd.DataFrame(np.random.normal(size=(len(Y_with_NaNs.columns), latent_dimension)),
                 index=Y_with_NaNs.columns) * 0.001  # user
V = pd.DataFrame(np.random.normal(size=(len(Y_with_NaNs.index), latent_dimension)),
                 index=Y_with_NaNs.index) * 0.001  # movies
# %% Iteration
iterations = 40
learn_rate = 0.01
for i in range(iterations):
    loss, dU, dV = Grad(U, V, Y)
    #loss, dU, dV = objective_gradient(Y, U, V)
    print("Iteration", i, "MSE: ", loss)
    U -= learn_rate * dU
    V -= learn_rate * dV
# %%


def Prediction(U, V, Y):
    Y_without_NaNs = pd.DataFrame(data=np.dot(
        U, V.T).T + Y['rating'].mean(), columns=Y_with_NaNs.columns, index=Y_with_NaNs.index)
    diff = Y_without_NaNs - Y_with_NaNs
    total = diff.fillna(value=0).apply(abs).sum().sum()
    return Y_without_NaNs, total


prediction, loss = Prediction(U, V, Y)

```

#### Answer

```python
# Code for question 8 here.
ratings = pd.read_csv("./ml-latest-small/ratings.csv") 
Y_full = pd.DataFrame({'users': ratings['userId'], 'movies': ratings['movieId'], 'ratingsorig': ratings['rating']})
Y_full['ratings'] = Y_full['ratingsorig'] - np.mean(Y_full['ratingsorig'])
indexes_unique_users = ratings['userId'].unique()
n_users = indexes_unique_users.shape[0]
indexes_unique_movies = ratings['movieId'].unique()
n_movies = indexes_unique_movies.shape[0]
q = 2
U = pd.DataFrame(np.random.normal(size=(n_users, q))*0.001, index=indexes_unique_users)
V = pd.DataFrame(np.random.normal(size=(n_movies, q))*0.001, index=indexes_unique_movies)

# max_iter should be larger, we just put 10 here for making it faster.
U, V = SGD(Y_full, U, V, max_iter = 10, check_obj = Y_full.shape[0]) # more iterations are needed for optimization
```

```python
plt.plot(V[0],V[1],'rx')
plt.title('Movie map')
plt.show()
```

