import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def create_user_item_matrix(ratings):
    """
    Create a User-Item rating matrix.
    Rows = Users
    Columns = Movies
    Values = Ratings
    """

    user_item_matrix = ratings.pivot_table(
        index="userId",
        columns="movieId",
        values="rating"
    ).fillna(0)

    return user_item_matrix


def get_collaborative_recommendations(
    user_id,
    movies,
    ratings,
    number_of_recommendations=10
):
    """
    Generate recommendations using
    user-based collaborative filtering.
    """

    # ---------------------------------------------------------
    # Create User-Item Matrix
    # ---------------------------------------------------------

    user_item_matrix = create_user_item_matrix(ratings)

    # Check whether user exists
    if user_id not in user_item_matrix.index:
        return pd.DataFrame()

    # ---------------------------------------------------------
    # Calculate similarity between users
    # ---------------------------------------------------------

    user_similarity = cosine_similarity(
        user_item_matrix
    )

    user_similarity_df = pd.DataFrame(
        user_similarity,
        index=user_item_matrix.index,
        columns=user_item_matrix.index
    )

    # ---------------------------------------------------------
    # Find similar users
    # ---------------------------------------------------------

    similar_users = (
        user_similarity_df[user_id]
        .drop(user_id)
        .sort_values(ascending=False)
    )

    similar_users = similar_users[
        similar_users > 0
    ].head(10)

    if similar_users.empty:
        return pd.DataFrame()

    # ---------------------------------------------------------
    # Movies already watched by selected user
    # ---------------------------------------------------------

    watched_movies = set(
        ratings[
            ratings["userId"] == user_id
        ]["movieId"]
    )

    # ---------------------------------------------------------
    # Calculate recommendation scores
    # ---------------------------------------------------------

    recommendation_scores = {}

    for similar_user_id, similarity_score in similar_users.items():

        similar_user_ratings = ratings[
            ratings["userId"] == similar_user_id
        ]

        for _, row in similar_user_ratings.iterrows():

            movie_id = row["movieId"]
            rating = row["rating"]

            # Skip movies already watched
            if movie_id in watched_movies:
                continue

            weighted_score = (
                similarity_score * rating
            )

            if movie_id not in recommendation_scores:
                recommendation_scores[movie_id] = {
                    "score": 0,
                    "weight": 0
                }

            recommendation_scores[movie_id]["score"] += (
                weighted_score
            )

            recommendation_scores[movie_id]["weight"] += (
                similarity_score
            )

    if not recommendation_scores:
        return pd.DataFrame()

    # ---------------------------------------------------------
    # Convert scores into final prediction
    # ---------------------------------------------------------

    final_scores = []

    for movie_id, values in recommendation_scores.items():

        if values["weight"] > 0:

            predicted_rating = (
                values["score"] /
                values["weight"]
            )

            final_scores.append(
                {
                    "movieId": movie_id,
                    "predicted_rating": predicted_rating
                }
            )

    recommendations = pd.DataFrame(
        final_scores
    )

    # ---------------------------------------------------------
    # Sort recommendations
    # ---------------------------------------------------------

    recommendations = recommendations.sort_values(
        by="predicted_rating",
        ascending=False
    ).head(
        number_of_recommendations
    )

    # ---------------------------------------------------------
    # Add movie information
    # ---------------------------------------------------------

    recommendations = recommendations.merge(
        movies[
            [
                "movieId",
                "title",
                "genres"
            ]
        ],
        on="movieId",
        how="left"
    )

    recommendations["predicted_rating"] = (
        recommendations["predicted_rating"]
        .round(2)
    )

    # ---------------------------------------------------------
    # Return final result
    # ---------------------------------------------------------

    return recommendations[
        [
            "movieId",
            "title",
            "genres",
            "predicted_rating"
        ]
    ].reset_index(drop=True)