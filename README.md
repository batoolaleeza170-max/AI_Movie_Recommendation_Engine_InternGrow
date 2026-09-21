# 🎬 AI Movie Recommendation Engine

An AI-powered movie recommendation system built with Python, Pandas, Scikit-learn, and Streamlit.

The system analyzes user ratings and movie genres to provide personalized movie recommendations using Content-Based Filtering, Collaborative Filtering, and a Hybrid Recommendation Model.

---

## 🚀 Features

- 👤 User Profile Analysis
- 🎯 Content-Based Movie Recommendation
- 🤖 Personalized Recommendations
- 🤝 Collaborative Filtering
- 🔀 Hybrid Recommendation Model
- 📜 Recommendation History
- ⭐ Highly Rated Movies
- 📊 User Rating History
- 📈 Genre Preference Analysis
- 🖥️ Interactive Streamlit Web Interface

---

## 🧠 Recommendation Techniques

### 1. Content-Based Filtering

The system analyzes movie genres using:

- TF-IDF Vectorization
- Cosine Similarity

Movies with similar genre patterns are recommended based on the selected movie.

### 2. Collaborative Filtering

The system creates a user-movie rating matrix and compares users using cosine similarity.

Recommendations are generated from the rating patterns of users with similar preferences.

### 3. Hybrid Recommendation

The Hybrid model combines:

- Content-Based recommendations
- Collaborative Filtering recommendations

This produces a combined recommendation score for movies.

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Plotly
- SciPy

---

## 📂 Project Structure

```text
AI Recommendation Engine/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── movies.csv
│   ├── ratings.csv
│   ├── tags.csv
│   ├── links.csv
│   ├── recommendation_history.csv
│   └── README.txt
│
├── modules/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── profile.py
│   ├── content_based.py
│   ├── recommender.py
│   ├── history.py
│   ├── collaborative.py
│   └── hybrid.py
│
├── models/
│
└── assets/
    └── images/
