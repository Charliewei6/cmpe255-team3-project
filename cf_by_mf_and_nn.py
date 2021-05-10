import time
import numpy as np
import pandas as pd
import keras

from keras.layers import Dropout, Flatten,Activation,Input,Embedding
from keras.models import Model
from keras.layers.merge import dot
from keras.optimizers import Adam
from keras.layers import Dense , merge
from sklearn.metrics import mean_squared_error
from math import sqrt


np.random.seed(123)
# load data
def loadData():
    ratings = pd.read_csv('./data/rating.csv', parse_dates=['timestamp'])
    return ratings

# for test convenience, only use num% of data
def cutData(num, ratings):
    rand_userIds = np.random.choice(ratings['userId'].unique(), size=int(len(ratings['userId'].unique())*num), replace=False)
    ratings = ratings.loc[ratings['userId'].isin(rand_userIds)]
    return ratings

def splitData(ratings):
    users = ratings.userId.unique()
    movies = ratings.movieId.unique()

    userid2idx = {o:i for i,o in enumerate(users)}
    movieid2idx = {o:i for i,o in enumerate(movies)}

    ratings['userId'] = ratings['userId'].apply(lambda x: userid2idx[x])
    ratings['movieId'] = ratings['movieId'].apply(lambda x: movieid2idx[x])
    split = np.random.rand(len(ratings)) < 0.8
    train_ratings = ratings[split]
    test_ratings = ratings[~split]
    print(train_ratings.shape , test_ratings.shape)
    return train_ratings, test_ratings
 
def embeddingModel(ratings):
    n_movies=len(ratings['movieId'].unique())
    n_users=len(ratings['userId'].unique())
    n_latent_factors=64  # hyperparamter to deal with. 

    user_input=Input(shape=(1,),name='user_input',dtype='int64')
    user_embedding=Embedding(n_users,n_latent_factors,name='user_embedding')(user_input)
    user_vec =Flatten(name='FlattenUsers')(user_embedding)

    movie_input=Input(shape=(1,),name='movie_input',dtype='int64')
    movie_embedding=Embedding(n_movies,n_latent_factors,name='movie_embedding')(movie_input)
    movie_vec=Flatten(name='FlattenMovies')(movie_embedding)

    sim=dot([user_vec,movie_vec],name='Simalarity-Dot-Product',axes=1)
    model =keras.models.Model([user_input, movie_input],sim)
    return model

def embeddingNNModel(ratings):
    n_movies=len(ratings['movieId'].unique())
    n_users=len(ratings['userId'].unique())
    n_latent_factors=50  # hyperparamter to deal with. 

    user_input=Input(shape=(1,),name='user_input',dtype='int64')
    user_embedding=Embedding(n_users,n_latent_factors,name='user_embedding')(user_input)
    user_vec =Flatten(name='FlattenUsers')(user_embedding)
    user_vec=Dropout(0.40)(user_vec)

    movie_input=Input(shape=(1,),name='movie_input',dtype='int64')
    movie_embedding=Embedding(n_movies,n_latent_factors,name='movie_embedding')(movie_input)
    movie_vec=Flatten(name='FlattenMovies')(movie_embedding)
    movie_vec=Dropout(0.40)(movie_vec)

    sim=dot([user_vec,movie_vec],name='Simalarity-Dot-Product',axes=1)
    nn_inp=Dense(96,activation='relu')(sim)
    nn_inp=Dropout(0.4)(nn_inp)
    nn_inp=Dense(1,activation='relu')(nn_inp)
    nn_model =keras.models.Model([user_input, movie_input],nn_inp)
    return nn_model

def fit(model, train_ratings, epochs, batch_size):
    model.compile(optimizer=Adam(lr=1e-4),loss='mse')
    model.fit([train_ratings.userId,train_ratings.movieId], train_ratings.rating, epochs=epochs, batch_size=batch_size, verbose=1)
    return model

def predict(model, test_ratings):
    pre_ratings = model.predict([test_ratings.userId,test_ratings.movieId])
    return pre_ratings

def rmse(prediction, ground_truth):
    return sqrt(mean_squared_error(prediction, ground_truth))

start = time.time()
ratings = loadData()
#ratings_cut = cutData(0.3, ratings)
train_ratings, test_ratings = splitData(ratings)
nn_model = embeddingNNModel(ratings)
batch_size = 512
epochs = 10
new_model = fit(nn_model, train_ratings, epochs, batch_size)
pre_ratings = predict(new_model, test_ratings)
RMSE = rmse(pre_ratings, test_ratings.rating)
print (f'Batch_size is: {batch_size}, epochs is: {epochs}, Neural Network RMSE is: {RMSE}')
end = time.time()
print(f"Runtime of the program is {end - start}")


