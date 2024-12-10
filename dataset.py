from tokenize import String

import torch
import pandas as pd
from pandas import DataFrame

class Movie:
    id: int
    name: String
    topics: list

def readRatings() -> DataFrame:
    dataset = pd.read_csv("datasets/ml-latest-small/ratings.csv")
    dataset.sort_values("timestamp", ascending=True, inplace=True)
    return dataset

def readRankings() -> DataFrame:
    dataset  = pd.read_csv("datasets/ml-latest-small/ratings.csv")
    dataset.sort_values("timestamp", ascending=True, inplace=True)
    print(dataset["userId"].max())
    pivotRating = dataset.pivot_table(index='userId', columns='movieId', values='rating', fill_value=0)
    print(pivotRating.head())
    return pivotRating

def filterMoviesByRatings(moviesGenres: pd.DataFrame, pivotRating: pd.DataFrame) -> pd.DataFrame:
    # Filtrar les pel·lícules presents a pivotRating.columns
    common_movies = moviesGenres.index.intersection(pivotRating.columns)
    filteredGenres = moviesGenres.loc[common_movies]
    return filteredGenres


def alignGenresWithRatings(pivotRating: pd.DataFrame, moviesGenres: pd.DataFrame) -> pd.DataFrame:
    rated_movie_ids = pivotRating.columns
    alignedGenres = moviesGenres.reindex(rated_movie_ids).fillna(0)
    print(f"Gèneres alineats: {alignedGenres.shape[0]} pel·lícules (de {moviesGenres.shape[0]} originals).")
    return alignedGenres


def generateGenresFromRatings(pivotRating: pd.DataFrame, movies: pd.DataFrame) -> pd.DataFrame:
    # Obtenim les pel·lícules presents a pivotRating
    rated_movie_ids = pivotRating.columns

    # Filtrar el DataFrame de pel·lícules
    filteredMovies = movies.set_index('movieId').loc[rated_movie_ids]

    # Generar la matriu de gèneres
    moviesGenres = filteredMovies['genres'].str.get_dummies(sep='|')

    print(f"Generada matriu de gèneres per {moviesGenres.shape[0]} pel·lícules.")

    return moviesGenres

def readMovies() -> DataFrame:
    dataset = pd.read_csv("datasets/ml-latest-small/movies.csv")
    return dataset


readRankings()
