import pandas as pd


def get_hybrid_recommendations(
    content_recommendations,
    collaborative_recommendations,
    number_of_recommendations=10
):
    """
    Combine content-based and collaborative recommendations.
    """

    if content_recommendations.empty and collaborative_recommendations.empty:
        return pd.DataFrame()

    content = content_recommendations.copy()
    collaborative = collaborative_recommendations.copy()

    # Rename scores
    if not content.empty:
        content = content.rename(
            columns={"similarity": "content_score"}
        )

    if not collaborative.empty:
        collaborative = collaborative.rename(
            columns={"predicted_rating": "collaborative_score"}
        )

    # Merge both recommendation lists
    if not content.empty and not collaborative.empty:

        hybrid = pd.merge(
            content,
            collaborative,
            on=["movieId", "title", "genres"],
            how="outer"
        )

    elif not content.empty:

        hybrid = content.copy()
        hybrid["collaborative_score"] = 0

    else:

        hybrid = collaborative.copy()
        hybrid["content_score"] = 0

    # Fill missing values
    hybrid["content_score"] = hybrid[
        "content_score"
    ].fillna(0)

    hybrid["collaborative_score"] = hybrid[
        "collaborative_score"
    ].fillna(0)

    # Normalize collaborative score to percentage
    hybrid["collaborative_score"] = (
        hybrid["collaborative_score"] / 5
    ) * 100

    # Hybrid score: 50% content + 50% collaborative
    hybrid["hybrid_score"] = (
        hybrid["content_score"] * 0.5
        + hybrid["collaborative_score"] * 0.5
    )

    hybrid = hybrid.sort_values(
        by="hybrid_score",
        ascending=False
    ).head(number_of_recommendations)

    hybrid["hybrid_score"] = (
        hybrid["hybrid_score"].round(2)
    )

    return hybrid[
        [
            "movieId",
            "title",
            "genres",
            "hybrid_score"
        ]
    ].reset_index(drop=True)