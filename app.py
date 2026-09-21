import streamlit as st

from modules.data_loader import get_data
from modules.profile import get_user_profile
from modules.hybrid import get_hybrid_recommendations
from modules.content_based import (
    create_content_model,
    get_similar_movies
)

from modules.recommender import (
    get_personalized_recommendations
)

from modules.history import (
    save_recommendations,
    load_history,
    clear_history
)

from modules.collaborative import (
    get_collaborative_recommendations
)


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Movie Recommendation Engine",
    page_icon="🎬",
    layout="wide"
)


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_project_data():

    movies, ratings, tags = get_data()

    return movies, ratings, tags


movies, ratings, tags = load_project_data()


# =========================================================
# CREATE CONTENT MODEL
# =========================================================

@st.cache_resource
def load_content_model(movies):

    movie_data, tfidf_matrix = create_content_model(
        movies
    )

    return movie_data, tfidf_matrix


movie_data, tfidf_matrix = load_content_model(
    movies
)


# =========================================================
# TITLE
# =========================================================

st.title("🎬 AI Movie Recommendation Engine")

st.write(
    "An AI-powered movie recommendation system "
    "using machine learning and recommendation techniques."
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🎯 Recommendation System")

st.sidebar.markdown(
    "Select a user and explore personalized movie recommendations."
)


# =========================================================
# USER SELECTION
# =========================================================

user_ids = sorted(
    ratings["userId"].unique()
)

selected_user = st.sidebar.selectbox(
    "👤 Select User",
    user_ids
)


# =========================================================
# USER PROFILE
# =========================================================

profile = get_user_profile(
    selected_user,
    movies,
    ratings
)


# =========================================================
# USER PROFILE SECTION
# =========================================================

st.header("👤 User Profile Analysis")

col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "Movies Rated",
        profile["total_movies"]
    )


with col2:

    st.metric(
        "Average Rating",
        f"{profile['average_rating']:.2f} ⭐"
    )


with col3:

    if len(profile["favorite_genres"]) > 0:

        favorite_genre = (
            profile["favorite_genres"].index[0]
        )

    else:

        favorite_genre = "N/A"

    st.metric(
        "Favorite Genre",
        favorite_genre
    )


# =========================================================
# FAVORITE GENRES
# =========================================================

st.header("❤️ Favorite Genres")

if len(profile["favorite_genres"]) > 0:

    st.bar_chart(
        profile["favorite_genres"]
    )

else:

    st.info(
        "No genre information available."
    )


# =========================================================
# CONTENT-BASED RECOMMENDATION
# =========================================================

st.header("🎯 Similar Movie Recommendation")

st.write(
    "Select a movie and the system will find "
    "movies with similar genres."
)


movie_titles = sorted(
    movie_data["title"].tolist()
)


selected_movie = st.selectbox(
    "🎬 Select a Movie",
    movie_titles
)


if st.button(
    "🔍 Find Similar Movies",
    type="primary"
):

    with st.spinner(
        "Finding similar movies..."
    ):

        recommendations = get_similar_movies(
            selected_movie,
            movie_data,
            tfidf_matrix,
            number_of_movies=10
        )


    if recommendations.empty:

        st.error(
            "Movie not found."
        )

    else:

        st.success(
            f"Movies similar to **{selected_movie}**"
        )

        st.dataframe(
            recommendations,
            use_container_width=True,
            hide_index=True
        )


# =========================================================
# PERSONALIZED RECOMMENDATIONS
# =========================================================

st.header("🤖 Personalized Recommendations")

st.write(
    "Recommendations are generated based on "
    "the movies liked by the selected user."
)


if st.button(
    "🎯 Generate Personalized Recommendations",
    type="primary"
):

    with st.spinner(
        "Analyzing user preferences..."
    ):

        personalized = get_personalized_recommendations(
            selected_user,
            movies,
            ratings,
            tfidf_matrix,
            number_of_recommendations=10
        )


    if personalized.empty:

        st.warning(
            "Not enough rating data available "
            "for this user."
        )

    else:

        # Save recommendations to history
        save_recommendations(
            selected_user,
            personalized
        )

        st.success(
            f"Personalized recommendations for User "
            f"{selected_user}"
        )

        st.dataframe(
            personalized,
            use_container_width=True,
            hide_index=True
        )


# =========================================================
# RECOMMENDATION HISTORY
# =========================================================

st.header("📜 Recommendation History")

history = load_history()


if not history.empty:

    user_history = history[
        history["userId"] == selected_user
    ].copy()


    if not user_history.empty:

        user_history = user_history.sort_values(
            by="date",
            ascending=False
        )


        st.dataframe(
            user_history[
                [
                    "movieId",
                    "title",
                    "genres",
                    "similarity",
                    "date"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )


        if st.button(
            "🗑️ Clear Recommendation History"
        ):

            clear_history()

            st.success(
                "Recommendation history cleared."
            )

            st.rerun()


    else:

        st.info(
            "No recommendation history available "
            "for this user."
        )


else:

    st.info(
        "No recommendation history available yet. "
        "Generate personalized recommendations first."
    )


# =========================================================
# COLLABORATIVE FILTERING
# =========================================================

st.header("🤝 Collaborative Filtering Recommendations")

st.write(
    "Recommendations are generated by analyzing "
    "rating patterns of users with similar interests."
)


if st.button(
    "🤝 Generate Collaborative Recommendations",
    type="primary"
):

    with st.spinner(
        "Finding users with similar movie preferences..."
    ):

        collaborative = get_collaborative_recommendations(
            selected_user,
            movies,
            ratings,
            number_of_recommendations=10
        )


    if collaborative.empty:

        st.warning(
            "Not enough user rating data available "
            "to generate collaborative recommendations."
        )

    else:

        st.success(
            f"Collaborative recommendations for User "
            f"{selected_user}"
        )

        st.dataframe(
            collaborative,
            use_container_width=True,
            hide_index=True
        )
# =========================================================
# HYBRID RECOMMENDATIONS
# =========================================================

st.header("🔀 Hybrid Movie Recommendations")

st.write(
    "Combines Content-Based Filtering and "
    "Collaborative Filtering for personalized recommendations."
)

if st.button(
    "🔀 Generate Hybrid Recommendations",
    type="primary"
):

    with st.spinner(
        "Combining recommendation models..."
    ):

        content_recommendations = get_similar_movies(
            selected_movie,
            movie_data,
            tfidf_matrix,
            number_of_movies=10
        )

        collaborative_recommendations = (
            get_collaborative_recommendations(
                selected_user,
                movies,
                ratings,
                number_of_recommendations=10
            )
        )

        hybrid = get_hybrid_recommendations(
            content_recommendations,
            collaborative_recommendations,
            number_of_recommendations=10
        )

    if hybrid.empty:

        st.warning(
            "Unable to generate hybrid recommendations."
        )

    else:

        st.success(
            "Hybrid recommendations generated successfully."
        )

        st.dataframe(
            hybrid,
            use_container_width=True,
            hide_index=True
        )

# =========================================================
# HIGHLY RATED MOVIES
# =========================================================

st.header("⭐ Highly Rated Movies")

liked_movies = profile["liked_movies"]


if len(liked_movies) > 0:

    display_movies = liked_movies[
        ["title", "genres", "rating"]
    ].head(10)


    st.dataframe(
        display_movies,
        use_container_width=True,
        hide_index=True
    )

else:

    st.info(
        "This user has not given any rating "
        "of 4 or higher."
    )


# =========================================================
# USER RATING HISTORY
# =========================================================

st.header("📊 User Rating History")

user_history = profile["user_ratings"][
    ["title", "genres", "rating"]
].sort_values(
    by="rating",
    ascending=False
)


st.dataframe(
    user_history,
    use_container_width=True,
    hide_index=True
)


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "AI Movie Recommendation Engine | "
    "Content-Based Filtering using TF-IDF and Cosine Similarity"
)