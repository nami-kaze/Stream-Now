from flask import Flask, render_template, request
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)

def clean_title(title):
    return re.sub("[^a-zA-Z0-9 ]", "", title)

def initialize_vectorizer():
    return TfidfVectorizer(ngram_range=(1, 2))

def get_tfidf_matrix(movies_df, vectorizer):
    return vectorizer.fit_transform(movies_df["clean_title"])

def search(title, vectorizer, tfidf_matrix, movies_df):
    title = clean_title(title)
    query_vec = vectorizer.transform([title])
    similarity = cosine_similarity(query_vec, tfidf_matrix).flatten()
    indices = np.argpartition(similarity, -5)[-5:]
    results = movies_df.loc[indices[::-1]]
    return results

def find_similar_movies(movie_id, ratings_df, movies_df):
    movie = movies_df[movies_df["movieId"] == movie_id]
    similar_users = ratings_df.query("movieId == @movie_id & rating > 4")["userId"].unique()
    similar_user_recs = (
        ratings_df.query("userId in @similar_users & rating > 4")["movieId"]
        .value_counts() / len(similar_users)
    )
    similar_user_recs = similar_user_recs[similar_user_recs > 0.10]
    
    all_users = (
        ratings_df.query("movieId in @similar_user_recs.index & rating > 4")
    )
    all_user_recs = all_users["movieId"].value_counts() / len(all_users["userId"].unique())
    
    rec_percentages = pd.concat([similar_user_recs, all_user_recs], axis=1)
    rec_percentages.columns = ["similar", "all"]
    rec_percentages["score"] = rec_percentages["similar"] / rec_percentages["all"]
    rec_percentages = rec_percentages.sort_values("score", ascending=False)
    
    return rec_percentages.head(10).merge(movies_df, left_index=True, right_on="movieId")[["score", "title", "genres"]]

# Load datasets
movies = pd.read_csv("./movies.csv")
ratings = pd.read_csv("./ratings.csv")

# Preprocess data
movies["clean_title"] = movies["title"].apply(clean_title)
vectorizer = initialize_vectorizer()
tfidf_matrix = get_tfidf_matrix(movies, vectorizer)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommendations', methods=['POST'])
def recommendations():
    if request.method == 'POST':
        movie_title = request.form['movie_title']
        results = search(movie_title, vectorizer, tfidf_matrix, movies)
        if not results.empty:
            movie_id = results.iloc[0]["movieId"]
            similar_movies = find_similar_movies(movie_id, ratings, movies)
            return render_template('recommendations.html', similar_movies=similar_movies)
    
    return render_template('no_results.html')

if __name__ == '__main__':
    app.run(debug=True)
