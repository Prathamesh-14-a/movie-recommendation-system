"""
Data loading utilities for the Movie Recommendation System.
Handles loading of pickle files and data validation.
"""
import pickle
from typing import Tuple, Optional
import pandas as pd
import numpy as np
from pathlib import Path

from .config import MOVIES_PKL_PATH, SIMILARITY_MATRIX_PATH


def load_movies_data() -> Optional[pd.DataFrame]:
    """
    Load movies data from pickle file.
    
    Returns:
        DataFrame with columns: movie_id, title, overview, genres, keywords, cast, crew
        Returns None if file not found.
    """
    try:
        if not MOVIES_PKL_PATH.exists():
            raise FileNotFoundError(f"Movies pickle file not found: {MOVIES_PKL_PATH}")
        
        with open(MOVIES_PKL_PATH, "rb") as f:
            movies = pickle.load(f)
        
        if isinstance(movies, dict):
            movies = pd.DataFrame(movies)
        elif not isinstance(movies, pd.DataFrame):
            movies = pd.DataFrame(movies)
        
        return movies
    except Exception as e:
        print(f"Error loading movies data: {e}")
        return None


def load_similarity_matrix() -> Optional[np.ndarray]:
    """
    Load precomputed cosine similarity matrix from pickle file.
    
    Returns:
        2D numpy array of shape (n_movies, n_movies) with similarity scores
        Returns None if file not found.
    """
    try:
        if not SIMILARITY_MATRIX_PATH.exists():
            raise FileNotFoundError(f"Similarity matrix file not found: {SIMILARITY_MATRIX_PATH}")
        
        with open(SIMILARITY_MATRIX_PATH, "rb") as f:
            similarity = pickle.load(f)
        
        return similarity
    except Exception as e:
        print(f"Error loading similarity matrix: {e}")
        return None


def validate_movies_data(movies: pd.DataFrame) -> bool:
    """
    Validate that movies DataFrame has required columns.
    
    Args:
        movies: DataFrame to validate
        
    Returns:
        True if valid, False otherwise
    """
    required_columns = ['movie_id', 'title']
    return all(col in movies.columns for col in required_columns)


def get_movie_title_list(movies: pd.DataFrame) -> list:
    """
    Get sorted list of all movie titles for display.
    
    Args:
        movies: Movies DataFrame
        
    Returns:
        Sorted list of movie titles
    """
    return sorted(movies['title'].unique().tolist())
