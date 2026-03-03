import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("netflix_titles.csv")
df.fillna('', inplace=True)
movies = df[df['type'] == 'Movie'].reset_index(drop=True)

movies['combined'] = movies['description'] + ' ' + movies['listed_in'] + ' ' + movies['cast'] + ' ' + movies['director']

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['combined'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

movie_index = pd.Series(movies.index, index=movies['title']).drop_duplicates()

def recommend(movie):
    if movie not in movie_index:
        return []
    idx = movie_index[movie]
    scores = list(enumerate(cosine_sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:6]
    return movies['title'].iloc[[i[0] for i in scores]]

st.title("Movie Recommendation System")

movie_name = st.text_input("Enter movie name")

if st.button("Recommend"):
    result = recommend(movie_name)
    for m in result:
        st.write(m)
