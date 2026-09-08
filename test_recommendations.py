#!/usr/bin/env python
"""Test script to verify recommendation functionality."""
from src import load_movies_data, load_similarity_matrix, MovieRecommender

# Load data
movies = load_movies_data()
similarity = load_similarity_matrix()

# Create recommender
recommender = MovieRecommender(movies, similarity)

# Test with a popular movie
test_movie = "Avatar"
titles, ids, scores = recommender.recommend(test_movie, num_recommendations=5)

print(f"\nRecommendations for '{test_movie}':\n")
for i, (title, movie_id, score) in enumerate(zip(titles, ids, scores), 1):
    print(f"{i}. {title}")
    print(f"   ID: {movie_id} | Similarity: {score*100:.2f}%\n")
