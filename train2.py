import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import dataset as ds

# 1. Configuració de dimensions
embedding_dim = 20
hidden_dim1 = 20
hidden_dim2 = 10
output_dim = 1

num_epochs = 2
learning_rate = 0.001
train_losses = []

# 2. Usant GPU si està disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 3. Definim el model amb PyTorch
class MovieRecommendationMLP(nn.Module):
    def __init__(self, num_users, num_movies, embedding_dim, hidden_dim1, hidden_dim2, output_dim):
        super(MovieRecommendationMLP, self).__init__()
        # Embeddings per usuaris i pel·lícules
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.movie_embeddings = nn.Embedding(num_movies, embedding_dim)

        # Capes ocultes
        self.fc1 = nn.Linear(2 * embedding_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, output_dim)

        # Activacions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, user_idx, movie_idx):
        user_emb = self.user_embeddings(user_idx)
        movie_emb = self.movie_embeddings(movie_idx)
        x = torch.cat((user_emb, movie_emb), dim=-1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return self.sigmoid(x)

# 4. Inicialitzar el model, optimitzador i funció de pèrdua
num_users = 610  # Nombre d'usuaris
num_movies = 9742  # Nombre de pel·lícules

model = MovieRecommendationMLP(num_users, num_movies, embedding_dim, hidden_dim1, hidden_dim2, output_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
mse_loss = nn.MSELoss()

# 5. Entrenament
def train():
    global model, optimizer, train_losses

    # Llegir dades
    movies = ds.readMovies()
    userRanking = ds.readRankings()

    moviesGenres = movies.set_index('movieId')['genres'].str.get_dummies(sep='|')
    moviesGenres = ds.alignGenresWithRatings(userRanking, moviesGenres)

    userPreferences = {}

    for user_id, user_ratings in userRanking.iterrows():
        rated_movies = user_ratings[user_ratings > 0]

        genres_of_rated_movies = moviesGenres.loc[rated_movies.index]

        user_embedding = genres_of_rated_movies.T.dot(rated_movies)
        if user_embedding.max() > 0:
            user_embedding = user_embedding / user_embedding.max()
        user_embedding *= 0.01

        userPreferences[user_id] = user_embedding.to_numpy()

    userPreferencesDF = pd.DataFrame(userPreferences).T
    print(userPreferencesDF)

    np.random.seed(42)

    # Inicialització dels embeddings
    user_embeddings = torch.tensor(np.array(list(userPreferences.values()))).to(device)
    movie_embeddings = torch.rand(len(movies), embedding_dim).to(device) * 0.01

    ratings = ds.readRatings()
    ratings["rating"] = ratings["rating"] / 5  # Normalització

    for epoch in range(num_epochs):
        epoch_loss = 0
        print(f"Epoch {epoch + 1}:")

        for i in range(len(ratings)):
            user_idx = ratings.iloc[i]['userId']
            movie_idx = movies[movies['movieId'] == ratings.iloc[i]['movieId']].index[0]
            rating = ratings.iloc[i]['rating']

            user_emb = self.user_embeddings(user_idx.long())
            movie_emb = movie_embeddings[int(movie_idx)].reshape(1, -1)

            # Forward pass
            y_pred = model(user_emb, movie_emb)

            # Càlcul de la pèrdua
            loss = mse_loss(y_pred, torch.tensor([[rating]], device=device))
            epoch_loss += loss.item()

            # Backward pass i actualització de pesos
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_losses.append(epoch_loss / len(ratings))
        print(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(ratings):.4f}")

    print("Entrenament completat!")

def setupEnvironment():
    dataset = ds.readRankings()
    split_index = int(len(dataset) * 0.8)  # 80% for training
    train_data = dataset.iloc[:split_index]  # train  80%
    test_data = dataset.iloc[split_index:]  # test 20%

    pivotRatingTest = test_data.pivot_table(index='userId', columns='movieId', values='rating', fill_value=0)
    pivotRatingTrain = train_data.pivot_table(index='userId', columns='movieId', values='rating', fill_value=0)

train()
