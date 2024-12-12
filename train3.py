import torch
import torch.nn as nn
from tensorflow.python.ops.ragged.ragged_dispatch import ragged_binary_elementwise_op

import dataset as ds
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class InteractionDataset(Dataset):
    def __init__(self, users, items, labels, item_mapping):
        self.users = users - users.min()
        self.items = items.map(item_mapping)
        self.labels = labels / 5

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.labels[idx]


ratings = ds.readRatings()
movies = ds.readMovies()
ratings = ratings.join(movies.set_index('movieId'), on='movieId')

item_mapping = {item_id: idx for idx, item_id in enumerate(ratings['movieId'].unique())}

# Exemple de dataset
train_dataset = InteractionDataset(ratings["userId"], ratings["movieId"], ratings["rating"], item_mapping)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

class NeuCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim):
        super(NeuCF, self).__init__()
        # Embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        # GMF layer
        self.gmf_layer = nn.Linear(embedding_dim, embedding_dim)

        # MLP layers
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, embedding_dim)
        )

        # Output layer
        self.output = nn.Linear(embedding_dim * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, user_ids, item_ids):
        user_embed = self.user_embedding(user_ids)
        item_embed = self.item_embedding(item_ids)

        # GMF interaction
        gmf = user_embed * item_embed

        # MLP interaction
        mlp_input = torch.cat([user_embed, item_embed], dim=-1)
        mlp = self.mlp(mlp_input)

        # Combine GMF and MLP
        combined = torch.cat([gmf, mlp], dim=-1)

        # Output layer
        pred = self.sigmoid(self.output(combined))
        return pred

def train(model : NeuCF, criterion, optimizer, device):
    num_epochs = 50  # Pots ajustar-ho segons el dataset

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for user_ids, item_ids, labels in train_loader:
            # Mou dades a la GPU si tens CUDA disponible
            user_ids, item_ids, labels = user_ids.to(device), item_ids.to(device), labels.to(device)

            # Reseta els gradients
            optimizer.zero_grad()

            # Prediccions
            predictions = model(user_ids, item_ids).squeeze()

            # Calcula la pèrdua
            loss = criterion(predictions, labels.float())

            # Backpropagation i optimització
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader):.4f}")


def setTraining():
    global ratings, movies
    num_users = len(ratings['userId'].unique())  # Comptar usuaris únics
    num_items = len(ratings['movieId'].unique())  # Comptar pel·lícules úniques


    embedding_dim = 60

    model = NeuCF(num_users, num_items, embedding_dim)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.015)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train(model, criterion, optimizer, device)

def getUserEmbbeding():
    movies = ds.readMovies()
    userRanking = ds.readRankings()

    userPreferences = {}

    moviesGenres = movies.set_index('movieId')['genres'].str.get_dummies(sep='|')
    moviesGenres = ds.alignGenresWithRatings(userRanking, moviesGenres)

    for user_id, user_ratings in userRanking.iterrows():
        rated_movies = user_ratings[user_ratings > 0]

        genres_of_rated_movies = moviesGenres.loc[rated_movies.index]

        user_embedding = genres_of_rated_movies.T.dot(rated_movies)
        if user_embedding.max() > 0:  # Evita dividir per zero
            user_embedding = user_embedding / user_embedding.max()
        user_embedding *= 0.01

        userPreferences[user_id] = user_embedding.to_numpy()

    # Convertir a DataFrame per visualitzar
    userPreferencesDF = pd.DataFrame(userPreferences).T
    return userPreferencesDF

setTraining()