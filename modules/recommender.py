import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def get_personalized_recommendations(
    user_id,
    movies,
    ratings,
    tfidf_matrix,
    number_of_recommendations=10
):
    """
    Generate personalized movie recommendations
    based on movies liked by the selected user.
    """

    # ---------------------------------------------------------
    # Get selected user's ratings
    # ---------------------------------------------------------

    user_ratings = ratings[
        ratings["userId"] == user_id
    ].copy()

    if user_ratings.empty:
        return pd.DataFrame()


    # ---------------------------------------------------------
    # Find movies liked by user
    # ---------------------------------------------------------

    liked_movies = user_ratings[
        user_ratings["rating"] >= 4
    ]

    # If user has no 4+ ratings, use highest rated movies
    if liked_movies.empty:

        liked_movies = (
            user_ratings
            .sort_values(
                by="rating",
                ascending=False
            )
            .head(5)
        )


    # ---------------------------------------------------------
    # Find TF-IDF indexes of liked movies
    # ---------------------------------------------------------

    movie_indexes = []

    for movie_id in liked_movies["movieId"]:

        matches = movies.index[
            movies["movieId"] == movie_id
        ].tolist()

        if matches:
            movie_indexes.append(matches[0])


    if not movie_indexes:
        return pd.DataFrame()


    # ---------------------------------------------------------
    # Create user profile vector
    # ---------------------------------------------------------

    user_profile_vector = (
        tfidf_matrix[movie_indexes]
        .mean(axis=0)
    )

    # Convert np.matrix to normal NumPy array
    user_profile_vector = np.asarray(
        user_profile_vector
    )

    # Make sure it is 2D
    user_profile_vector = user_profile_vector.reshape(
        1,
        -1
    )


    # ---------------------------------------------------------
    # Calculate cosine similarity
    # ---------------------------------------------------------

    similarity_scores = cosine_similarity(
        user_profile_vector,
        tfidf_matrix
    ).flatten()


    # ---------------------------------------------------------
    # Remove movies already watched by user
    # ---------------------------------------------------------

    watched_movies = set(
        user_ratings["movieId"]
    )

    recommendations = movies[
        ~movies["movieId"].isin(watched_movies)
    ].copy()


    # ---------------------------------------------------------
    # Add similarity scores
    # ---------------------------------------------------------

    recommendation_indexes = recommendations.index

    recommendations["similarity"] = [
        similarity_scores[index]
        for index in recommendation_indexes
    ]


    # ---------------------------------------------------------
    # Sort recommendations
    # ---------------------------------------------------------

    recommendations = recommendations.sort_values(
        by="similarity",
        ascending=False
    )


    # ---------------------------------------------------------
    # Select top recommendations
    # ---------------------------------------------------------

    recommendations = recommendations.head(
        number_of_recommendations
    )


    # ---------------------------------------------------------
    # Convert similarity to percentage
    # ---------------------------------------------------------

    recommendations["similarity"] = (
        recommendations["similarity"] * 100
    ).round(2)


    # ---------------------------------------------------------
    # Return final result
    # ---------------------------------------------------------

    return recommendations[
        [
            "movieId",
            "title",
            "genres",
            "similarity"
        ]
    ].reset_index(drop=True)