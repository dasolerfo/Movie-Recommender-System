import dataset as ds
import numpy as np
import pandas as pd
import random

# 1. Configuració de dimensions
num_genres = 10
embedding_dim = 20
hidden_dim1 = 20
hidden_dim2 = 10
output_dim = 1


user_embeddings = []
movie_embeddings = []

num_epochs = 50
learning_rate = 0.01
train_losses = []

# Pesos del MLP
W1 = np.random.rand(2 * embedding_dim, hidden_dim1) * 0.01
b1 = np.zeros((1, hidden_dim1))
W2 = np.random.rand(hidden_dim1, hidden_dim2) * 0.01
b2 = np.zeros((1, hidden_dim2))
W3 = np.random.rand(hidden_dim2, output_dim) * 0.01
b3 = np.zeros((1, output_dim))



def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)


def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)


def leaky_relu_derivative(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    sig = sigmoid(x)
    return sig * (1 - sig)

# Funció de pèrdua
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


# Forward pass
def forward(user_emb, movie_emb):
    x = np.hstack((user_emb, movie_emb))
    z1 = np.dot(x, W1) + b1
    a1 = leaky_relu(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = leaky_relu(z2)
    z3 = np.dot(a2, W3) + b3
    y_pred = sigmoid(z3)
    return x, z1, a1, z2, a2, z3, y_pred


# Backward pass
# Adam: Inicialització de moments
adam_params = {
    "W1": {"m": np.zeros_like(W1), "v": np.zeros_like(W1)},
    "b1": {"m": np.zeros_like(b1), "v": np.zeros_like(b1)},
    "W2": {"m": np.zeros_like(W2), "v": np.zeros_like(W2)},
    "b2": {"m": np.zeros_like(b2), "v": np.zeros_like(b2)},
    "W3": {"m": np.zeros_like(W3), "v": np.zeros_like(W3)},
    "b3": {"m": np.zeros_like(b3), "v": np.zeros_like(b3)},
    "user_embeddings": {"m": np.zeros_like(user_embeddings), "v": np.zeros_like(user_embeddings)},
    "movie_embeddings": {"m": np.zeros_like(movie_embeddings), "v": np.zeros_like(movie_embeddings)},
}


# Funció per aplicar Adam
def adam_update(param, grad, config, lr, beta1=0.9, beta2=0.999, epsilon=1e-8, t=1):
    # Comprovar si 'm' i 'v' estan inicialitzats; si no, inicialitzar-los amb zeros
    if len(config["m"]) == 0:
        config["m"] = np.zeros_like(grad)
    if len(config["v"]) == 0:
        config["v"] = np.zeros_like(grad)

    config["m"] = beta1 * config["m"] + (1 - beta1) * grad
    config["v"] = beta2 * config["v"] + (1 - beta2) * (grad ** 2)

    m_hat = config["m"] / (1 - beta1 ** t)
    v_hat = config["v"] / (1 - beta2 ** t)


    param -= lr * m_hat / (np.sqrt(v_hat) + epsilon)
    return param


# Backward pass amb Adam
def backward_adam(x, z1, a1, z2, a2, z3, y_pred, y_true, user_idx, movie_idx, lr=0.01, t=1):
    global W1, b1, W2, b2, W3, b3, adam_params, user_embeddings, movie_embeddings

    m = 1 #len(y_true)

    # Gradients de la pèrdua
    dz3 = (y_pred - y_true) * sigmoid_derivative(z3)
    dW3 = np.dot(a2.T, dz3) / m
    db3 = np.sum(dz3, axis=0, keepdims=True) / m

    da2 = np.dot(dz3, W3.T)
    dz2 = da2 * leaky_relu_derivative(z2)
    dW2 = np.dot(a1.T, dz2) / m
    db2 = np.sum(dz2, axis=0, keepdims=True) / m

    da1 = np.dot(dz2, W2.T)
    dz1 = da1 * leaky_relu_derivative(z1)
    dW1 = np.dot(x.reshape(1, -1).T, dz1) / m
    db1 = np.sum(dz1, axis=0, keepdims=True) / m

    # Gradients per als embeddings
    dx = np.dot(dz1, W1.T)
    user_grad = dx[:, :embedding_dim]
    movie_grad = dx[:, embedding_dim:]

    # Actualització amb Adam
    W3 = adam_update(W3, dW3, adam_params["W3"], lr, t=t)
    b3 = adam_update(b3, db3, adam_params["b3"], lr, t=t)
    W2 = adam_update(W2, dW2, adam_params["W2"], lr, t=t)
    b2 = adam_update(b2, db2, adam_params["b2"], lr, t=t)
    W1 = adam_update(W1, dW1, adam_params["W1"], lr, t=t)
    b1 = adam_update(b1, db1, adam_params["b1"], lr, t=t)

    #user_embeddings[user_idx] = adam_update(
    #    user_embeddings[user_idx], user_grad, adam_params["user_embeddings"], lr, t=t
    #)
    movie_embeddings[movie_idx] = adam_update(
        movie_embeddings[movie_idx].reshape(1, -1), movie_grad, adam_params["movie_embeddings"], lr, t=t
    )


def train():
    global user_embeddings, movie_embeddings
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
    print(userPreferencesDF)

    np.random.seed(42)

    #input()

    # Embeddings ajustables //TODO: ARREGLAR AIXO
    user_embeddings = userPreferences
    movie_embeddings = np.random.rand(len(movies), embedding_dim) * 0.01

    user_indices =  range(610)
    ratings = ds.readRatings()
    ratings["rating"] = ratings["rating"] / 5

    for epoch in range(num_epochs):
        epoch_loss = 0
        print("Iteracio numero: " + epoch.__str__())

        for i in range(len(ratings)):

            user_idx = ratings.iloc[i]['userId']
            movie_idx = movies[movies['movieId'] == ratings.iloc[i]['movieId']].index[0]
            rating = ratings.iloc[i]['rating']

            user_emb = user_embeddings[user_idx].reshape(1, -1)
            movie_emb = movie_embeddings[movie_idx].reshape(1, -1)

            # Forward pass
            x, z1, a1, z2, a2, z3, y_pred = forward(user_emb, movie_emb)

            # Pèrdua
            print("Rating: ", rating, "  Predicció: ", y_pred)
            loss = mse_loss(rating, y_pred)
            epoch_loss += loss

            # Backward pass amb Adam
            backward_adam(x, z1, a1, z2, a2, z3, y_pred, rating, user_idx, movie_idx, lr=learning_rate, t=epoch + 1)

        train_losses.append(epoch_loss / len(user_indices))
        if (epoch + 1) % 50 == 0:
            print(f"Època {epoch + 1}, Pèrdua d'entrenament: {epoch_loss / len(user_indices):.4f}")

    print("Entrenament finalitzat!")

def setupEnvironment():
    dataset = ds.readRankings()
    split_index = int(len(dataset) * 0.8)  # 80% for training
    train_data = dataset.iloc[:split_index]  # train  80%
    test_data = dataset.iloc[split_index:]  # test 20%

    pivotRatingTest = test_data.pivot_table(index='userId', columns='movieId', values='rating', fill_value=0)
    pivotRatingTrain  = test_data.pivot_table(index='userId', columns='movieId', values='rating', fill_value=0)

train()


