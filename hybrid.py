import time
import numpy as np
import pandas as pd
import keras
import random

from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


from keras.layers import Dropout, Flatten,Activation,Input,Embedding
from keras.models import Model
from keras.layers.merge import dot
from keras.optimizers import Adam
from keras.layers import Dense , merge
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.model_selection import train_test_split


np.random.seed(123)
# load data
def loadData():
    ratings = pd.read_csv('./data/rating.csv', parse_dates=['timestamp'])
    movies = pd.read_csv('./data/movie.csv')
    merge_ratings_movies = pd.merge(movies, ratings, on='movieId', how='inner')
    del merge_ratings_movies["userId"]
    movies_avg_ratings = merge_ratings_movies.groupby('movieId').mean()
    movies_ratings=pd.merge(movies, movies_avg_ratings, on='movieId', how='inner')
    return ratings, movies_ratings

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

    train_ratings, test_ratings= train_test_split(ratings, test_size=0.2, random_state=42)
    train_ratings_ori = train_ratings.copy()
    test_ratings_ori = test_ratings.copy()
    
    train_ratings['userId'] = train_ratings['userId'].apply(lambda x: userid2idx[x])
    train_ratings['movieId'] = train_ratings['movieId'].apply(lambda x: movieid2idx[x])
    test_ratings['userId'] = test_ratings['userId'].apply(lambda x: userid2idx[x])
    test_ratings['movieId'] = test_ratings['movieId'].apply(lambda x: movieid2idx[x])

    return train_ratings, test_ratings, train_ratings_ori, test_ratings_ori
 
def findTfidfMatrix(movies_ratings): 
    tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3),max_features=10000, min_df=0, stop_words='english')
    tfidf_matrix = tf.fit_transform(movies_ratings['genres'])
    return tfidf_matrix

def findSimilar(tfidf_matrix, movies_ratings):
    cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)
    results = {}

    for idx, row in movies_ratings.iterrows():
        similar_indices = cosine_similarities[idx].argsort()[:-100:-1]
        similar_items = [(cosine_similarities[idx][i], movies_ratings['movieId'][i]) for i in similar_indices]

        results[row['movieId']] = similar_items[1:]
    return results

def cbSuggest(item_id, amount, results, movies_ratings):
    rating_sum = 0
    recs = results[item_id]
    for rec in recs:
        if rec[1]!=item_id:
            index = movies_ratings[movies_ratings['movieId'] == rec[1]].index
            rating = movies_ratings.iloc[index]['rating']
            rating_sum += float(rating)
            amount -= 1
        if amount <= 0:
            break
    return rating_sum/5

def cbPredict(test_ratings_ori, results, movies_ratings):
    cb_pre_ratings = []
    for index, row in test_ratings_ori.iterrows():
        pre_rateing = cbSuggest(row['movieId'], 5, results, movies_ratings)
        cb_pre_ratings.append(pre_rateing)
    cb_pre_ratings = np.asarray(cb_pre_ratings)
    cb_pre_ratings_reshape = np.reshape(cb_pre_ratings, (cb_pre_ratings.shape[0], 1))
    return cb_pre_ratings_reshape

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

def cfFit(model, train_ratings, epochs, batch_size):
    model.compile(optimizer=Adam(lr=1e-4),loss='mse')
    model.fit([train_ratings.userId,train_ratings.movieId], train_ratings.rating, epochs=epochs, batch_size=batch_size, verbose=1)
    return model

def cfPredict(model, test_ratings):
    pre_ratings = model.predict([test_ratings.userId,test_ratings.movieId])
    return pre_ratings

def finalPredict1(cbPre, cfPre):
    finalPre = 0.25 * cbPre + 0.75 * cfPre
    return finalPre

def finalPredict2(cbPre, cfPre):
    finalPre = 0.1 * cbPre + 0.9 * cfPre
    return finalPre

def rmse(prediction, ground_truth):
    return sqrt(mean_squared_error(prediction, ground_truth))

start = time.time()
ratings, movies_ratings = loadData()
#ratings_cut = cutData(0.3, ratings)
train_ratings, test_ratings, train_ratings_ori, test_ratings_ori = splitData(ratings)
tfidf_matrix = findTfidfMatrix(movies_ratings)
results = findSimilar(tfidf_matrix, movies_ratings)
cb_pre_ratings = cbPredict(test_ratings_ori, results, movies_ratings)
nn_model = embeddingNNModel(ratings)
batch_size = 512
epochs = 10
new_model = cfFit(nn_model, train_ratings, epochs, batch_size)
cf_pre_ratings = cfPredict(new_model, test_ratings)
final_pre_ratings1 = finalPredict1(cb_pre_ratings, cf_pre_ratings)
final_pre_ratings2 = finalPredict2(cb_pre_ratings, cf_pre_ratings)
RMSE = rmse(cf_pre_ratings, test_ratings.rating)
RMSE1 = rmse(final_pre_ratings1, test_ratings.rating)
RMSE2 = rmse(final_pre_ratings2, test_ratings.rating)
print (f'Batch_size is: {batch_size}, epochs is: {epochs}, Neural Network RMSE is: {RMSE}')
print (f'cbPre is 0.25, cfPre is 0.75,  Hybrid RMSE is: {RMSE1}')
print (f'cbPre is 0.1, cfPre is 0.9,  Hybrid RMSE is: {RMSE2}')
end = time.time()
print(f"Runtime of the program is {end - start}")