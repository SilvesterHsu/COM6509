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