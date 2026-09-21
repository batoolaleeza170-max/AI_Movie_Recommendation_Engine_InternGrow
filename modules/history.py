import pandas as pd
import os
from datetime import datetime

HISTORY_FILE = "data/recommendation_history.csv"


def save_recommendations(user_id, recommendations):
    """
    Save personalized recommendations to CSV history.
    """

    if recommendations.empty:
        return

    history_data = recommendations.copy()

    history_data["userId"] = user_id
    history_data["date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    columns = [
        "userId",
        "movieId",
        "title",
        "genres",
        "similarity",
        "date"
    ]

    history_data = history_data[columns]

    # Create data folder if it does not exist
    os.makedirs("data", exist_ok=True)

    # Append if file already exists
    if os.path.exists(HISTORY_FILE):
        history_data.to_csv(
            HISTORY_FILE,
            mode="a",
            header=False,
            index=False
        )
    else:
        history_data.to_csv(
            HISTORY_FILE,
            index=False
        )


def load_history():
    """
    Load recommendation history.
    """

    if not os.path.exists(HISTORY_FILE):
        return pd.DataFrame(
            columns=[
                "userId",
                "movieId",
                "title",
                "genres",
                "similarity",
                "date"
            ]
        )

    return pd.read_csv(HISTORY_FILE)


def clear_history():
    """
    Delete recommendation history.
    """

    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)