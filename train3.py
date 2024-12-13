import torch
import torch.nn as nn
from tensorflow.python.ops.ragged.ragged_dispatch import ragged_binary_elementwise_op

import dataset as ds
import pandas as pd
import torch.nn.functional as F
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

model_path = "model.pth"
ratings = ds.readRatings()
movies = ds.readMovies()
ratings = ratings.join(movies.set_index('movieId'), on='movieId')

item_mapping = {item_id: idx for idx, item_id in enumerate(ratings['movieId'].unique())}

# Exemple de dataset
train_dataset = InteractionDataset(ratings["userId"], ratings["movieId"], ratings["rating"], item_mapping)
train_loader = DataLoader(train_dataset, batch_size=200, shuffle=True)

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
            nn.Linear(embedding_dim * 2, 256),  # Més unitats a la primera capa
            nn.BatchNorm1d(256),  # Normalització de lot
            nn.ReLU(),
            nn.Linear(256, 128),  # Segona capa augmentada
            nn.BatchNorm1d(128),  # Normalització de lot
            nn.ReLU(),
            nn.Linear(128, 64),  # Tercera capa augmentada
            nn.BatchNorm1d(64),  # Normalització de lot
            nn.ReLU(),
            nn.Linear(64, embedding_dim),  # Capa de sortida
            nn.Dropout(0.3)  # Dropout per evitar sobreajustament
        )

        # Output layer
        self.output = nn.Linear(embedding_dim * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, user_ids, item_ids):
        user_embed = self.user_embedding(user_ids)
        item_embed = self.item_embedding(item_ids)

        gmf = user_embed * item_embed
        mlp_input = torch.cat([user_embed, item_embed], dim=-1)
        mlp = self.mlp(mlp_input)

        combined = torch.cat([gmf, mlp], dim=-1)

        pred = self.sigmoid(self.output(combined))
        return pred

    def forwardAdapt(self, user_embed, item_embed):

        gmf = user_embed * item_embed

        mlp_input = torch.cat([user_embed, item_embed], dim=-1)
        mlp = self.mlp(mlp_input)

        combined = torch.cat([gmf, mlp], dim=-1)

        pred = self.sigmoid(self.output(combined))
        return pred

    def find_similar_movies(model, movie_id, top_k=5):

        movie_emb = model.item_embedding(torch.tensor(movie_id))

        all_movie_embs = model.item_embedding.weight

        similarity = F.cosine_similarity(movie_emb, all_movie_embs, dim=1)

        top_similar = torch.argsort(similarity, descending=True)[:top_k]

        return top_similar, similarity[top_similar]
    def adapt_user_embedding(self, user_id, new_movie_ids, new_ratings, lr=0.2, steps=200):

        user_emb = self.user_embedding(user_id).clone().detach().requires_grad_(True)
        optimizer = torch.optim.SGD([user_emb], lr=lr)

        new_ratings = torch.tensor(new_ratings, dtype=torch.float32)

        # Entrena només sobre les noves valoracions
        for _ in range(steps):
            optimizer.zero_grad()
            movie_embs = self.item_embedding(new_movie_ids).clone().detach().requires_grad_(True)

            predictions = self.forwardAdapt(user_emb, movie_embs)
            loss = nn.MSELoss()(predictions/5, new_ratings/5)
            loss.backward()
            optimizer.step()

        self.user_embedding.weight.data[user_id] = user_emb.data

def train(model : NeuCF, criterion, optimizer, device, scheduler):
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for user_ids, item_ids, labels in train_loader:
            # Mou dades a la GPU si tens CUDA disponible
            user_ids, item_ids, labels = user_ids.to(device), item_ids.to(device), labels.to(device)
            optimizer.zero_grad()
            predictions = model(user_ids, item_ids).squeeze()
            loss = criterion(predictions, labels.float())

            # Backpropagation i optimització
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        val_loss = total_loss / len(train_loader)
        scheduler.step(val_loss)

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader):.4f}")



def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def loadModel(num_users, num_items, embedding_dim):
    model = NeuCF(num_users, num_items, embedding_dim)
    # Carrega l'estat guardat
    model.load_state_dict(torch.load(model_path))

    # Passa el model a eval per fer prediccions
    model.eval()
    return model

def setTraining():
    global ratings, movies
    num_users = len(ratings['userId'].unique())  # Comptar usuaris únics
    num_items = len(ratings['movieId'].unique())  # Comptar pel·lícules úniques
    embedding_dim = 128

    model = NeuCF(num_users, num_items, embedding_dim)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.apply(init_weights)

    train(model, criterion, optimizer, device, scheduler)
    torch.save(model.state_dict(), model_path)

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

#setTraining()