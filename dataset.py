from tokenize import String

import torch
import pandas as pd

class Movie:
    id: int
    name: String
    topics: list

def readRankings():
    dataset = pd.read_csv("rankings.csv")
    print(dataset.head())

def readMovies():
    dataset = pd.read_csv("movies.csv")
    print(dataset.head())