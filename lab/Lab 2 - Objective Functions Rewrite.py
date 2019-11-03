#%% Prepare datasets
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
#%% Format datasets

def ReadData(data_dir: str) -> pd.core.frame.DataFrame:
    return pd.read_csv(data_dir)


def GetRandomSample(data, columns='userId', sample_number=10, seed=410) -> np.ndarray:
    np.random.seed(seed)
    item_uni_list = len(data[columns].unique())
    disorder = np.random.permutation(item_uni_list)
    return disorder[0:sample_number]


# basic parameters
studentId = 410
sample_number = 10

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

#%%
def Grad(U, V, Y):
    dU = pd.DataFrame(np.zeros((U.shape)), index=U.index)
    dV = pd.DataFrame(np.zeros((V.shape)), index=V.index)
    loss = 0
    for user, movie, _, rating in iter(Y.values):
        predict = np.dot(U.loc[user], V.loc[movie])
        diff = np.squeeze(predict) - rating
        loss += diff**2
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
#%% Iteration
iterations = 20
learn_rate = 0.01
for i in range(iterations):
    loss, dU, dV = Grad(U, V, Y)
    #loss, dU, dV = objective_gradient(Y, U, V)
    print("Iteration", i, "MSE: ", loss)
    U -= learn_rate*dU
    V -= learn_rate*dV
