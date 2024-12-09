from tokenize import String

import torch
import pandas as pd
from pandas import DataFrame

class Movie:
    id: int
    name: String
    topics: list

def readRankings() -> DataFrame:
    dataset  = pd.read_csv("datasets/ml-latest-small/ratings.csv")
    dataset.sort_values("timestamp", ascending=True, inplace=True)
    print(dataset["userId"].max())
    pivotRating = dataset.pivot_table(index='userId', columns='movieId', values='rating', fill_value=0)
    print(pivotRating.head())
    return pivotRating

def readMovies() -> DataFrame:
    dataset = pd.read_csv("datasets/ml-latest-small/movies.csv")
    return dataset


readRankings()
