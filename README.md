# cmpe255-team3-project

## Members
Yiru Sun 015262897 yiru.sun@sjsu.edu<br/>
Huiying Li 012289069 huiying.li@sjsu.edu

## Dataset
MovieLens 20M Dataset<br/>
Over 20 Million Movie Ratings and Tagging Activities Since 1995<br/>
https://www.kaggle.com/grouplens/movielens-20m-dataset/download

## Goal
Our group aims at doing research for using deep learning method to make a recommender system by traing MovieLens datasets. Each of our team members focus on digging into a particular deep learning recommender system from Kaggle, optimizing it, testing it, and comparing the new result to the original one.

## Methods
### Method 1: Building Recommender System using Implicit Feedback (Yiru Sun)
Artical: Recommender System Deep Learning<br/>
Link: https://www.kaggle.com/taruntiwarihp/recommender-system-deep-learning/notebook<br/>
Description: This method treats users’ rating as an interaction to the movie. The recommendation system transfers the dataset to an implicit feedback dataset. Then, it recommends movies to a customer based on his own rating habits.<br/>
Pros: This method is customized to each user.<br/>
Cons: This method is dependent on the user's interaction with the movie, but not the rating value of the movie. This means the recommendation system will recommend similar movies to a user, because the user will more likely to give a rating feedback to these movies, no matter the rating value will be high or low.<br/>
Steps: #1 load data with columns: userId, movieId, rating, timestamp.<br/>
       #2 use leave-one-out test method to split the training and testing data. use a user's latest reviews for testing.<br/>
       #3 convert the dataset into an implicit feedback dataset.
       #4 use Neural Collaborative Filtering model to train.<br/>
       #5 test result using Hit Ratio @ 10 method.<br/>
Test method: Hit Ratio @ 10, leave-one-out test method<br/>

