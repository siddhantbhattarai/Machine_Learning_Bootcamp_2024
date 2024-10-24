from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load the dataset
url = 'https://raw.githubusercontent.com/siddhantbhattarai/Machine_Learning_Bootcamp_2024/refs/heads/main/Datasets/movies.csv'
movies_df = pd.read_csv(url)

# Data Preprocessing
movies_df['VOTES'] = movies_df['VOTES'].replace(',', '', regex=True).astype(float)
movies_df['YEAR'] = pd.to_numeric(movies_df['YEAR'].str.extract(r'(\d{4})')[0], errors='coerce')
movies_df['GENRE'] = movies_df['GENRE'].str.replace('\n', '').str.strip()
movies_df['ONE-LINE'] = movies_df['ONE-LINE'].str.strip()

# Handling Missing Data
imputer = SimpleImputer(strategy='mean')
movies_df[['RATING', 'VOTES', 'RunTime']] = imputer.fit_transform(movies_df[['RATING', 'VOTES', 'RunTime']])
movies_df['GENRE'] = movies_df['GENRE'].fillna('Unknown')

# Remove duplicate movies
movies_df = movies_df.drop_duplicates(subset=['MOVIES'])

# TF-IDF for text features
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
tfidf_one_line = tfidf_vectorizer.fit_transform(movies_df['ONE-LINE'])

# One-hot encode genres
onehot_encoder = OneHotEncoder()
genres_encoded = onehot_encoder.fit_transform(movies_df[['GENRE']]).toarray()

# Scaling numerical features
scaler = StandardScaler()
numeric_columns = ['RATING', 'VOTES', 'RunTime']
scaled_numeric = scaler.fit_transform(movies_df[numeric_columns])

# Combine features
features_combined = np.hstack([scaled_numeric, genres_encoded, tfidf_one_line.toarray()])

# KNN for collaborative filtering
knn = NearestNeighbors(metric='cosine', algorithm='brute')
knn.fit(features_combined)

# Content-based filtering using cosine similarity
cosine_sim = cosine_similarity(features_combined)

# Recommendation Functions
def knn_recommend(movie_index, n_recommendations=5):
    distances, indices = knn.kneighbors([features_combined[movie_index]], n_neighbors=n_recommendations+1)
    return movies_df.iloc[indices[0][1:]]

def content_based_recommend(movie_index, n_recommendations=5):
    sim_scores = list(enumerate(cosine_sim[movie_index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_indices = [i[0] for i in sim_scores[1:n_recommendations+1]]
    return movies_df.iloc[top_indices]

# Home route (Frontend)
@app.route('/')
def home():
    movie_list = movies_df['MOVIES'].tolist()  # Display all unique movies
    return render_template('index.html', movies=movie_list)

# API route to get recommendations
@app.route('/recommend', methods=['POST'])
def recommend():
    movie_name = request.form['movie'].strip().lower()  # Case-insensitive comparison
    matching_movies = movies_df[movies_df['MOVIES'].str.lower() == movie_name]
    
    if matching_movies.empty:
        return jsonify({'error': f"Movie '{movie_name}' not found"}), 404
    
    movie_index = matching_movies.index[0]
    
    # Get recommendations
    recommendations = content_based_recommend(movie_index)
    recommendations = recommendations[['MOVIES', 'YEAR', 'RATING']].to_dict(orient='records')
    
    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(debug=True)

