"""
Movie Recommendation System - Modular package structure.
"""
from .config import *
from .data_loader import load_movies_data, load_similarity_matrix
from .recommender import MovieRecommender
from .api import TMDBClient

__all__ = [
    'load_movies_data',
    'load_similarity_matrix',
    'MovieRecommender',
    'TMDBClient',
]
