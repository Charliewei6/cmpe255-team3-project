# cmpe255-team3-project

## Members
Yiru Sun 015262897 yiru.sun@sjsu.edu<br/>
Huiying Li 012289069 huiying.li@sjsu.edu<br/>
Xichao Wei 015324764 xichao.wei@sjsu.edu<br/>

## Dataset
MovieLens 20M Dataset<br/>
Over 20 Million Movie Ratings and Tagging Activities Since 1995<br/>
https://www.kaggle.com/grouplens/movielens-20m-dataset/download

## Goal
Our group aims at doing research for using deep learning method to make a recommender system by traing MovieLens datasets. Each of our team members focus on digging into a particular deep learning recommender system from Kaggle, optimizing it, testing it, and comparing the new result to the original one.

## Methods
### Method 1: Building Recommender System using Implicit Feedback (Yiru Sun)
*Artical:* 
* Recommender System Deep Learning

*Link:* 
* https://www.kaggle.com/taruntiwarihp/recommender-system-deep-learning/notebook

*Description:* 
* This method treats users’ rating as an interaction to the movie. The recommendation system transfers the dataset to an implicit feedback dataset. Then, it recommends movies to a customer based on his own rating habits.

*Pros:* 
* This method is customized to each user.

*Cons:* 
* This method is dependent on the user's interaction with the movie, but not the rating value of the movie. This means the recommendation system will recommend similar movies to a user, because the user will more likely to give a rating feedback to these movies, no matter the rating value will be high or low.

*Steps:* 
1. load data with columns: userId, movieId, rating, timestamp.
2. use leave-one-out test method to split the training and testing data. use a user's latest reviews for testing.
3. convert the dataset into an implicit feedback dataset.
4. use Neural Collaborative Filtering model to train.
5. test result using Hit Ratio @ 10 method.

*Train method:* 
* Neural Collaborative Filtering model, by using pytorchligntning package

*Test method:* 
* Hit Ratio @ 10, leave-one-out test method

*Optimization*
* In the original method, if a user has rated a movie, the label is equals to 1. In this method, I calculate the quantile value of a user's all rating data. Only if a user's rating value is above his own 75% quantile value, the label is set to 1. Otherwise, it is set to 0.
* In this method, If a user's rating value is above his own 50% quantile value, the label is set to 1. Otherwise, it is 0. Besides, similiar to the original method from the artical, I randamly generate 2 negetive samples for eachrow of data, to make sure the ratio of negative to positive samples is 4:1

*Results*
1. Original 0 4 The Hit Ratio @ 10 is: 0.82
2. Optimization 1 0.5 2 The Hit Ratio @ 10 is: 0.80
3. Optimization 2 0.25 3 The Hit Ratio @ 10 is: 0.85
4. Optimization 2 0.25 4 The Hit Ratio @ 10 is: 0.85
5. Optimization 2 0.25 5 The Hit Ratio @ 10 is: 0.85
6. Optimization 2 0.1 6 The Hit Ratio @ 10 is: 0.86

*Time*
* Total test time:


### Method 2: Building Recommender System using Matrix Factorizaiton with Neural Network
*Artical:* 
* [CF Based RecSys by Low Rank Matrix Factorization](https://www.kaggle.com/rajmehra03/cf-based-recsys-by-low-rank-matrix-factorization#Collaborative-Filtering-Based-Recommender-Systems-using-Low-Rank-Matrix-Factorization(User-&-Movie-Embeddings)-&-Neural-Network-in-Keras.)
* [Creating a Hybrid Content-Collaborative Movie Recommender Using Deep Learning](https://towardsdatascience.com/creating-a-hybrid-content-collaborative-movie-recommender-using-deep-learning-cc8b431618af)
* [Simple Collaberative Recommendation using LightFM](https://www.kaggle.com/panks03/simple-collaberative-recommendation-using-lightfm)


*Description:* 
* This method use maxtrix factorization method to break down the rating matirx to user embedding and movie emdedding. And then use densely-connect NN layer to train the model.

*Steps:* 
1. load data
2. split training and testing data. training data is 0.8, testing data is 0.2
3. Create embedding for user and movie, add densely-connected NN layer to the model if using Neural Network.
4. Train the model with traning data set.
5. Predic the rating using the model.
6. Calculate rmse.

*Problem:*
* Collaborative filtering has cold start problem.

*Solution:*
* Create hybrid recommender system with content-based filtering to Collaborative filtering

*To Do List:*
1. Collaborative filtering: turning parameters with 30% data, metric is RMSE.
2. Collaborative filtering: use the best parameter set from step1 to train the model, and then test the model with the whole dataset.
3. Create hybrid recommender system
4. Hybrid recommender system: turning parameters
5. Compare collaborative filtering and hybrid recommender system

*Experiemnts*
1. Collaborative filtering: turning parameters with 30% data

| Number | Batch_size  | Epochs | RMSE |  Runtime | Output
|--------|-------------|--------|------|----------|--|
|1|32|5|-|More than 4 hours||
|2|128|5|0.8376757057764654|02:12:45|[output_2](https://github.com/Charliewei6/cmpe255-team3-project/blob/main/output/output_2.png)
|3|512|10|0.8173020977949738|01:15:39|[output_3](https://github.com/Charliewei6/cmpe255-team3-project/blob/main/output/output_3.png)
|4|128|10|0.8146115762344519|02:45:02|[output_4](https://github.com/Charliewei6/cmpe255-team3-project/blob/main/output/output_4.png)

2. Collaborative filtering: all data

 Batch_size  | Epochs | RMSE |  Runtime | Output
-------------|--------|------|----------|--|
512|10|0.9760667964050797|07:13:56|[output_5](https://github.com/Charliewei6/cmpe255-team3-project/blob/main/output/output_5.png)

3. Hybrid Recommender System: all data

 Batch_size  | Epochs | Num of movies | Percentage of CB | Percentage of CF | RMSE |  Runtime | Output
-------------|--------|------|----------|--|--|--|--|
(Base Line)512|10|-|0|100%|0.8098864027483295|11:49:06|[hybrid_2](https://github.com/Charliewei6/cmpe255-team3-project/blob/main/output/hybrid_2.png)
512|10|5|10%|90%|0.817477271184282|
512|10|5|25%|75%|0.8423216447352478|

4. LightFM: all data

* epochs= 50, num_threads= 4

* Train AUC Score: 0.9694958
* Test AUC Score: 0.9507404

### How to run program in local?
* python fileName

### How to run program using HPC?
* Upload necessary files to HPC, in our case, upload movie.csv, rating.csv, all python file, submit script([submit_hybrid.sh](https://github.com/Charliewei6/cmpe255-team3-project/blob/main/submit_hybrid.sh))
* In submit script, change job-name, output, error, time, load module, and file path if necessary.
* run: sbatch submit.sh
* Use output file to retrive result.
