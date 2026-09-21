import pandas as pd


# =========================================================
# LOAD MOVIE DATA
# =========================================================

def load_movies():
    movies = pd.read_csv("data/movies.csv")
    return movies


# =========================================================
# LOAD RATING DATA
# =========================================================

def load_ratings():
    ratings = pd.read_csv("data/ratings.csv")
    return ratings


# =========================================================
# LOAD TAG DATA
# =========================================================

def load_tags():
    tags = pd.read_csv("data/tags.csv")
    return tags


# =========================================================
# CLEAN MOVIE DATA
# =========================================================

def clean_movies(movies):

    movies = movies.drop_duplicates()

    movies = movies.dropna(subset=["movieId", "title"])

    movies["genres"] = movies["genres"].fillna("Unknown")

    return movies


# =========================================================
# CLEAN RATING DATA
# =========================================================

def clean_ratings(ratings):

    ratings = ratings.drop_duplicates()

    ratings = ratings.dropna(
        subset=["userId", "movieId", "rating"]
    )

    return ratings


# =========================================================
# GET DATA
# =========================================================

def get_data():

    movies = load_movies()
    ratings = load_ratings()
    tags = load_tags()

    movies = clean_movies(movies)
    ratings = clean_ratings(ratings)

    return movies, ratings, tags