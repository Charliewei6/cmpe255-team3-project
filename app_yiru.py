import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl

np.random.seed(123)

# load data
def loadData():
    ratings = pd.read_csv('archive/rating.csv', parse_dates=['timestamp'])
    return ratings

# for test convenience, only use num% of data
def cutData(num, ratings):
    rand_userIds = np.random.choice(ratings['userId'].unique(), size=int(len(ratings['userId'].unique())*num), replace=False)
    ratings = ratings.loc[ratings['userId'].isin(rand_userIds)]
    return ratings

# leave the latest one out for test
def trainTestSplit(ratings):
    ratings['rank_latest'] = ratings.groupby(['userId'])['timestamp'].rank(method = 'first', ascending=False)
    train_ratings = ratings[ratings['rank_latest'] != 1]
    test_ratings = ratings[ratings['rank_latest'] == 1]
    train_ratings = train_ratings[['userId', 'movieId', 'rating']]
    test_ratings = test_ratings[['userId', 'movieId', 'rating']]
    return train_ratings, test_ratings

class MovieLensTrainDataset(Dataset):
    """MovieLens Pytorch Dataset for Training
    Args:
        ratings(pd.DataFrame): Dataframe containing the movie ratings
        all_movieIds (list): List containing all movieIds
    """
    def __init__(self, ratings, all_movieIds, algorithm):
        self.users, self.items, self.labels = self.get_dataset(ratings, all_movieIds, algorithm)
        
    def __len__(self):
        return len(self.users)
    
    def __getitem__ (self, idx):
        return self.users[idx], self.items[idx], self.labels[idx]
    
    def get_dataset(self, ratings, all_movieIds, algorithm):
        users, items, labels = [], [], []
        if algorithm == 1:
            user_item_set = set(zip(ratings['userId'], ratings['movieId']))
            num_negatives = 4
            for u, i in user_item_set:
                users.append(u)
                items.append(i)
                labels.append(1)
                for _ in range(num_negatives):
                    negative_item = np.random.choice(all_movieIds)
                    while (u, negative_item) in user_item_set:
                        negative_item = np.random.choice(all_movieIds)
                    users.append(u)
                    items.append(negative_item)
                    labels.append(0)
        elif algorithm == 2:
            user_item_set = set(zip(ratings['userId'], ratings['movieId'], ratings['rating']))
            user_rating_set = ratings[['userId', 'rating']]
            user_rating = user_rating_set.groupby('userId')['rating'].quantile([0.85]).unstack().reset_index()
            rating_q = {}
            for i in range(len(user_rating)):
                rating_q[user_rating.iloc[i][0]] = user_rating.iloc[i][1]
            for u, i, r in user_item_set:
                users.append(u)
                items.append(i)
                if r >= rating_q[u]:
                    labels.append(1)
                else:
                    labels.append(0)
        elif algorithm == 3:
            user_item_set = set(zip(ratings['userId'], ratings['movieId'], ratings['rating']))
            user_rating_set = ratings[['userId', 'rating']]
            user_rating = user_rating_set.groupby('userId')['rating'].quantile([0.25]).unstack().reset_index()
            rating_q = {}
            num_negatives = 3
            for i in range(len(user_rating)):
                rating_q[user_rating.iloc[i][0]] = user_rating.iloc[i][1]
            for u, i, r in user_item_set:
                users.append(u)
                items.append(i)
                if r >= rating_q[u]:
                    labels.append(1)
                else:
                    labels.append(0)
                for _ in range(num_negatives):
                    negative_item = np.random.choice(all_movieIds)
                    while (u, negative_item, r) in user_item_set:
                        negative_item = np.random.choice(all_movieIds)
                    users.append(u)
                    items.append(negative_item)
                    labels.append(0)
        return torch.tensor(users), torch.tensor(items), torch.tensor(labels)

class NCF(pl.LightningModule):
    """ Neural Collaborative Filtering (NCF)
    
        Args:
            num_users (int): Number of unique users
            num_items (int): Number of unique items
            ratings (pd.DataFrame): Dataframe containing the movie ratings for training
            all_movieIds (list): List containing all movieIds (train + test)
    """
    
    def __init__(self, num_users, num_items, ratings, all_movieIds, algorithm):
        super().__init__()
        self.user_embedding = nn.Embedding(num_embeddings=num_users, embedding_dim=8)
        self.item_embedding = nn.Embedding(num_embeddings=num_items, embedding_dim=8)
        self.fc1 = nn.Linear(in_features=16, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=32)
        self.output = nn.Linear(in_features=32, out_features=1)
        self.ratings = ratings
        self.all_movieIds = all_movieIds
        self.algorithm = algorithm
        
    def forward(self, user_input, item_input):
        # Pass through embedding layers
        user_embedded = self.user_embedding(user_input)
        item_embedded = self.item_embedding(item_input)

        # Concat the two embedding layers
        vector = torch.cat([user_embedded, item_embedded], dim=-1)

        # Pass through dense layer
        vector = nn.ReLU()(self.fc1(vector))
        vector = nn.ReLU()(self.fc2(vector))

        # Output layer
        pred = nn.Sigmoid()(self.output(vector))

        return pred
    
    def training_step(self, batch, batch_idx):
        user_input, item_input, labels = batch
        predicted_labels = self(user_input, item_input)
        loss = nn.BCELoss()(predicted_labels, labels.view(-1, 1).float())
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def train_dataloader(self):
        return DataLoader(MovieLensTrainDataset(self.ratings, self.all_movieIds, self.algorithm), batch_size=512, num_workers=4)

def testing(ratings, test_ratings, all_movieIds, model):
    # User-item pairs for testing
    test_user_item_set = set(zip(test_ratings['userId'], test_ratings['movieId']))

    # Dict of all items that are interacted with by each user
    user_interacted_items = ratings.groupby('userId')['movieId'].apply(list).to_dict()

    hits = []
    for (u,i) in test_user_item_set:
        interacted_items = user_interacted_items[u]
        not_interacted_items = set(all_movieIds) - set(interacted_items)
        selected_not_interacted = list(np.random.choice(list(not_interacted_items), 99))
        test_items = selected_not_interacted + [i]
        
        predicted_labels = np.squeeze(model(torch.tensor([u]*100), 
                                            torch.tensor(test_items)).detach().numpy())
        
        top10_items = [test_items[i] for i in np.argsort(predicted_labels)[::-1][0:10].tolist()]
        
        if i in top10_items:
            hits.append(1)
        else:
            hits.append(0)
            
    print("The Hit Ratio @ 10 is {:.2f}".format(np.average(hits)))

if __name__ == "__main__":
    ratings = loadData()
    cutDataPercent = 0.3
    ratings = cutData(cutDataPercent, ratings)
    train_ratings, test_ratings = trainTestSplit(ratings)
    num_users = ratings['userId'].max()+1
    num_items = ratings['movieId'].max()+1
    all_movieIds = ratings['movieId'].unique()

    print("Method1:")
    model = NCF(num_users, num_items, train_ratings, all_movieIds, algorithm=1)
    trainer = pl.Trainer(max_epochs=5, gpus=None, reload_dataloaders_every_epoch=True,
                     progress_bar_refresh_rate=50, logger=False, checkpoint_callback=False)
    trainer.fit(model)
    testing(ratings, test_ratings, all_movieIds, model)

    print("Method2:")
    model = NCF(num_users, num_items, train_ratings, all_movieIds, algorithm=2)
    trainer = pl.Trainer(max_epochs=5, gpus=None, reload_dataloaders_every_epoch=True,
                     progress_bar_refresh_rate=50, logger=False, checkpoint_callback=False)
    trainer.fit(model)
    testing(ratings, test_ratings, all_movieIds, model)

    print("Method3:")
    model = NCF(num_users, num_items, train_ratings, all_movieIds, algorithm=3)
    trainer = pl.Trainer(max_epochs=5, gpus=None, reload_dataloaders_every_epoch=True,
                     progress_bar_refresh_rate=50, logger=False, checkpoint_callback=False)
    trainer.fit(model)
    testing(ratings, test_ratings, all_movieIds, model)

