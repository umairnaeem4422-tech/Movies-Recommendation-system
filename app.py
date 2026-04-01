import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os
import requests
import gdown

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

@st.cache_resource
def load_data():
    sim_path = "/tmp/similarity.npy"
    if not os.path.exists(sim_path) or os.path.getsize(sim_path) < 1000000:
        gdown.download(
            "https://drive.google.com/uc?id=1qUpFEbbyF6JVMlFELYHbHY3_cvbXJXOa",
            sim_path,
            quiet=False,
            fuzzy=True
        )
    movies_dict = pickle.load(open(os.path.join(BASE_DIR, "movies_dict.pkl,"), "rb"))
    movies      = pd.DataFrame(movies_dict)
    similarity  = np.load(sim_path)
    return movies, similarity

movies, similarity = load_data()

TMDB_API_KEY    = "8265bd1679663a7ea12ac168da84d2e8"
PLACEHOLDER_IMG = "https://via.placeholder.com/500x750?text=No+Image"

def fetch_poster(movie_id: int) -> str:
    try:
        url      = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}&language=en-US"
        response = requests.get(url, timeout=5)
        data     = response.json()
        poster   = data.get("poster_path")
        return f"https://image.tmdb.org/t/p/w500/{poster}" if poster else PLACEHOLDER_IMG
    except Exception:
        return PLACEHOLDER_IMG

def recommend(movie: str):
    idx  = movies[movies["title"] == movie].index[0]
    top5 = sorted(enumerate(similarity[idx]), key=lambda x: x[1], reverse=True)[1:6]
    names, posters = [], []
    for i, _ in top5:
        names.append(movies.iloc[i].title)
        posters.append(fetch_poster(movies.iloc[i].movie_id))
    return names, posters

st.title("🎬 Movie Recommender System")

selected_movie = st.selectbox("Choose your favourite movie:", movies["title"].values)

if st.button("Recommend"):
    with st.spinner("Finding recommendations..."):
        names, posters = recommend(selected_movie)

    cols = st.columns(5)
    for col, name, poster in zip(cols, names, posters):
        with col:
            st.image(poster)
            st.caption(name)
