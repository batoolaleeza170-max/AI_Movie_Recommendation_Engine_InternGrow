import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# CREATE CONTENT MODEL
# =========================================================

def create_content_model(movies):

    movie_data = movies.copy()

    movie_data["genres"] = movie_data["genres"].fillna("")

    # Convert genres into TF-IDF
    vectorizer = TfidfVectorizer(
        stop_words="english"
    )

    tfidf_matrix = vectorizer.fit_transform(
        movie_data["genres"]
    )

    return movie_data, tfidf_matrix


# =========================================================
# GET SIMILAR MOVIES
# =========================================================

def get_similar_movies(
    movie_title,
    movie_data,
    tfidf_matrix,
    number_of_movies=10
):

    movie_matches = movie_data[
        movie_data["title"].str.lower()
        == movie_title.lower()
    ]

    if movie_matches.empty:
        return pd.DataFrame()

    movie_index = movie_matches.index[0]

    # Calculate similarity ONLY for selected movie
    similarity_scores = cosine_similarity(
        tfidf_matrix[movie_index],
        tfidf_matrix
    ).flatten()

    # Get top indexes
    similar_indices = similarity_scores.argsort()[
        ::-1
    ]

    # Remove selected movie
    similar_indices = [
        index
        for index in similar_indices
        if index != movie_index
    ]

    similar_indices = similar_indices[
        :number_of_movies
    ]

    recommendations = movie_data.iloc[
        similar_indices
    ][
        ["movieId", "title", "genres"]
    ].copy()

    recommendations["similarity"] = [
        similarity_scores[index] * 100
        for index in similar_indices
    ]

    recommendations["similarity"] = (
        recommendations["similarity"].round(2)
    )

    return recommendations.reset_index(
        drop=True
    )