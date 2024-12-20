from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")

app = Flask(__name__)

def preprocess_data(ratings, min_movie_votes=10, min_user_votes=50):
    user_item_matrix = ratings.pivot(index='movieId', columns='userId', values='rating').fillna(0)
    movie_votes = ratings.groupby('movieId')['rating'].count()
    user_votes = ratings.groupby('userId')['rating'].count()
    filtered_matrix = user_item_matrix.loc[movie_votes[movie_votes > min_movie_votes].index, :]
    filtered_matrix = filtered_matrix.loc[:, user_votes[user_votes > min_user_votes].index]
    return filtered_matrix

def build_knn_model(matrix, metric='cosine', n_neighbors=10):
    csr_data = csr_matrix(matrix.values)
    knn = NearestNeighbors(metric=metric, algorithm='brute', n_neighbors=n_neighbors, n_jobs=-1)
    knn.fit(csr_data)
    return knn, csr_data

def get_movie_recommendation(movie_name, csr_data, movies, dataset, n_recommendations=10):
    movie_name = movie_name.lower().strip()
    matching_movies = movies[movies['title'].str.lower().str.contains(movie_name)]

    if len(matching_movies):        
        movie_idx= matching_movies.iloc[0]['movieId']
        movie_idx = dataset[dataset['movieId'] == movie_idx].index[0]
        scores , indices = knn.kneighbors(csr_data[movie_idx],n_neighbors=n_recommendations+1)    
        rec_movie_indices = sorted(list(zip(indices.squeeze().tolist(),scores.squeeze().tolist())),key=lambda x: x[1])[:0:-1]
        
        recs = []
        
        for val in rec_movie_indices:
            movie_idx = dataset.iloc[val[0]]['movieId']
            idx = movies[movies['movieId'] == movie_idx].index
            recs.append({'Title':movies.iloc[idx]['title'].values[0],'Score':val[1]})

    else:
        return f"No movies found matching '{movie_name}'. Please check your input."

    print(recs)

    return recs

final_dataset = preprocess_data(ratings)
final_dataset.reset_index(inplace=True)
knn, csr_data = build_knn_model(final_dataset)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/recommend", methods=["GET"])
def recommend():
    movie_name = request.args.get("movie_name", "")
    if not movie_name:
        return jsonify({"error": "Please provide a movie_name query parameter."}), 400
    recommendations = get_movie_recommendation(movie_name, csr_data, movies, final_dataset)
    if isinstance(recommendations, str):
        print("\n\n\nERROR\n\n\n")
        return jsonify({"error": recommendations}), 404
    elif isinstance(recommendations, dict) and "message" in recommendations:
        return jsonify(recommendations), 200
    else:
        return jsonify({"recommendations": recommendations}), 200

if __name__ == "__main__":
    app.run(debug=True, port=8000)
