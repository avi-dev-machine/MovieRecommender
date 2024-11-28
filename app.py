from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss
import json
from pathlib import Path
import joblib
import gc
from tqdm import tqdm
import os




app = Flask(__name__)

class MovieRecommender:
    def __init__(self, cache_dir='recommender_cache'):
        """Initialize the movie recommender system"""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
    def load_and_prepare_data(self, movies_csv, ratings_csv=None):
        """
        Load and prepare movie data from CSV files
        
        Parameters:
        movies_csv: path to movies CSV file
        ratings_csv: (optional) path to ratings CSV file
        """
        print("Loading and preparing data...")
        
        # Load movies
        self.movies_df = pd.read_csv(movies_csv)
        
        # If ratings file is provided, calculate average ratings
        if ratings_csv:
            ratings_df = pd.read_csv(ratings_csv)
            # Calculate average ratings and number of ratings
            rating_stats = ratings_df.groupby('movieId').agg({
                'rating': ['mean', 'count']
            }).reset_index()
            rating_stats.columns = ['movieId', 'avg_rating', 'num_ratings']
            
            # Merge with movies
            self.movies_df = self.movies_df.merge(
                rating_stats, 
                on='movieId', 
                how='left'
            )
            
            # Fill NaN values
            self.movies_df['avg_rating'] = self.movies_df['avg_rating'].fillna(0)
            self.movies_df['num_ratings'] = self.movies_df['num_ratings'].fillna(0)
        
        # Clean and prepare text features
        self.prepare_text_features()
        
        return self.movies_df
    
    def prepare_text_features(self):
        """Prepare text features for recommendation"""
        # Clean genres
        self.movies_df['genres'] = self.movies_df['genres'].fillna('')
        self.movies_df['genres'] = self.movies_df['genres'].str.replace('|', ' ')
        
        # If there's a tags or keywords column, prepare it
        if 'tags' in self.movies_df.columns:
            self.movies_df['tags'] = self.movies_df['tags'].fillna('')
        else:
            self.movies_df['tags'] = ''
            
        # Create combined features for TF-IDF
        self.movies_df['combined_features'] = self.movies_df.apply(
            lambda x: f"{x['title']} {x['genres']} {x['tags']}", axis=1
        )
    
    def build_recommendation_system(self):
        """Build the recommendation system using TF-IDF and FAISS"""
        print("Building recommendation system...")
        
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english'
        )
        
        tfidf_matrix = self.vectorizer.fit_transform(
            self.movies_df['combined_features']
        )
        
        feature_matrix = tfidf_matrix.astype(np.float32)
        
        self.index = faiss.IndexFlatIP(feature_matrix.shape[1])
        self.index.add(feature_matrix.toarray())
        
        self.save_components()
        
        del tfidf_matrix, feature_matrix
        gc.collect()
    
    def save_components(self):
        """Save recommendation system components"""
        print("Saving components...")
        
        joblib.dump(
            self.vectorizer, 
            self.cache_dir / 'vectorizer.pkl'
        )
        
        faiss.write_index(
            self.index, 
            str(self.cache_dir / 'movie_index.faiss')
        )
        
        self.movies_df.to_pickle(self.cache_dir / 'movies.pkl')
    
    def load_components(self):
        """Load saved recommendation system components"""
        print("Loading recommendation system...")
        
        self.vectorizer = joblib.load(self.cache_dir / 'vectorizer.pkl')
        self.index = faiss.read_index(str(self.cache_dir / 'movie_index.faiss'))
        self.movies_df = pd.read_pickle(self.cache_dir / 'movies.pkl')
    
    def get_recommendations(self, query, n_recommendations=5):
        """Get movie recommendations based on query"""
        query_vector = self.vectorizer.transform([query]).toarray().astype(np.float32)
        
        D, I = self.index.search(query_vector, n_recommendations)
        
        recommendations = []
        for idx, score in zip(I[0], D[0]):
            movie = self.movies_df.iloc[idx]
            recommendations.append({
                'title': movie['title'],
                'genres': movie['genres'],
                'similarity_score': float(score),
                'avg_rating': float(movie['avg_rating']) if 'avg_rating' in movie else None,
                'num_ratings': int(movie['num_ratings']) if 'num_ratings' in movie else None
            })
        
        return recommendations

# Initialize recommender as a global variable
recommender = MovieRecommender()
def initialize_recommender():
    try:
        # data = request.get_json()
        movies_path = 'data/movies.csv'
        ratings_path = 'data/ratings.csv'
        print(ratings_path)
        if not os.path.exists(movies_path):
            # return jsonify({'success': False, 'error': 'Movies CSV file not found'})
            print('Movies CSV file not found')
            return
            
        recommender.load_and_prepare_data(
            movies_csv=movies_path,
            ratings_csv=ratings_path if os.path.exists(ratings_path) else None
        )
        recommender.build_recommendation_system()
        
        # return jsonify({'success': True, 'message': 'Recommender system initialized successfully'})
        print('Recommender system initialized successfully')
    except Exception as e:
        # return jsonify({'success': False, 'error': str(e)})
        print(str(e))

@app.route('/')
def home():
    initialize_recommender()
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    query = data.get('query', '')
    n_recommendations = data.get('n_recommendations', 5)
    
    try:
        recommendations = recommender.get_recommendations(query, n_recommendations)
        return jsonify({'success': True, 'recommendations': recommendations})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# @app.route('/initialize', methods=['GET'])
# def initialize_recommender():
#     try:
#         # data = request.get_json()
#         movies_path = 'data/movies.csv'
#         ratings_path = 'data/ratings.csv'
#         print(ratings_path)
#         if not os.path.exists(movies_path):
#             return jsonify({'success': False, 'error': 'Movies CSV file not found'})
            
#         recommender.load_and_prepare_data(
#             movies_csv=movies_path,
#             ratings_csv=ratings_path if os.path.exists(ratings_path) else None
#         )
#         recommender.build_recommendation_system()
        
#         return jsonify({'success': True, 'message': 'Recommender system initialized successfully'})
#     except Exception as e:
#         return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    # Check if recommendation system already exists
    if (Path(recommender.cache_dir) / 'movie_index.faiss').exists():
        recommender.load_components()
    
    app.run(debug=True, host='0.0.0.0', port=5000)