import pandas as pd


# =========================================================
# GET USER PROFILE
# =========================================================

def get_user_profile(user_id, movies, ratings):

    # Get ratings of selected user
    user_ratings = ratings[ratings["userId"] == user_id].copy()

    # Merge ratings with movie information
    user_movies = user_ratings.merge(
        movies,
        on="movieId",
        how="left"
    )

    # -----------------------------------------------------
    # Basic Statistics
    # -----------------------------------------------------

    total_movies = len(user_movies)

    average_rating = (
        user_movies["rating"].mean()
        if total_movies > 0
        else 0
    )

    # -----------------------------------------------------
    # Highly Rated Movies
    # -----------------------------------------------------

    liked_movies = user_movies[
        user_movies["rating"] >= 4
    ].sort_values(
        by="rating",
        ascending=False
    )

    # -----------------------------------------------------
    # Genre Analysis
    # -----------------------------------------------------

    genre_list = []

    for genres in user_movies["genres"].dropna():

        if genres != "(no genres listed)":
            genre_list.extend(genres.split("|"))

    genre_counts = pd.Series(
        genre_list
    ).value_counts()

    # -----------------------------------------------------
    # Favorite Genres
    # -----------------------------------------------------

    favorite_genres = genre_counts.head(5)

    return {
        "user_ratings": user_movies,
        "total_movies": total_movies,
        "average_rating": average_rating,
        "liked_movies": liked_movies,
        "genre_counts": genre_counts,
        "favorite_genres": favorite_genres
    }