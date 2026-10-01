"""
Data loading utilities for the Movie Recommendation System.
Handles loading of pickle files and data validation.
"""
import sys
import pickle
from typing import Tuple, Optional
import pandas as pd
import numpy as np
from pathlib import Path

from .config import MOVIES_PKL_PATH, SIMILARITY_NPZ_PATH, SIMILARITY_MATRIX_PATH

# Compatibility alias for environments where pickled objects reference numpy._core
if not hasattr(np, "_core"):
    try:
        sys.modules["numpy._core"] = np.core
        sys.modules["numpy._core.numeric"] = np.core.numeric
        sys.modules["numpy._core.multiarray"] = np.core.multiarray
    except Exception:
        pass


def is_lfs_pointer(file_path: Path) -> bool:
    """Check if a file is a Git LFS pointer text file instead of actual binary data."""
    try:
        with open(file_path, "rb") as f:
            header = f.read(50)
            return header.startswith(b"version https://git-lfs.github.com/spec/v1")
    except Exception:
        return False


def load_movies_data() -> Optional[pd.DataFrame]:
    """
    Load movies data from pickle file.
    
    Returns:
        DataFrame with columns: movie_id, title, tags
    """
    if not MOVIES_PKL_PATH.exists():
        raise FileNotFoundError(f"Movies file not found: {MOVIES_PKL_PATH}")
    
    if is_lfs_pointer(MOVIES_PKL_PATH):
        raise ValueError(
            f"{MOVIES_PKL_PATH.name} is a Git LFS pointer file, not actual data. "
            "Please commit the real file directly."
        )

    with open(MOVIES_PKL_PATH, "rb") as f:
        movies = pickle.load(f)
    
    if isinstance(movies, dict):
        movies = pd.DataFrame(movies)
    elif not isinstance(movies, pd.DataFrame):
        movies = pd.DataFrame(movies)
    
    return movies


def load_similarity_matrix() -> Optional[np.ndarray]:
    """
    Load precomputed cosine similarity matrix.
    Prefers compressed .npz format (29MB, fast & compatible with Git/Streamlit Cloud),
    and falls back to .pkl if present.
    
    Returns:
        2D numpy array of shape (n_movies, n_movies) with similarity scores
        Returns None if file not found.
    """
    # 1. Prefer compressed NPZ format (29MB, fast, direct git tracking)
    if SIMILARITY_NPZ_PATH.exists():
        if is_lfs_pointer(SIMILARITY_NPZ_PATH):
            raise ValueError(f"{SIMILARITY_NPZ_PATH.name} is a Git LFS pointer.")
        data = np.load(SIMILARITY_NPZ_PATH)
        return data["similarity"]
    
    # 2. Fall back to PKL format
    if SIMILARITY_MATRIX_PATH.exists():
        if is_lfs_pointer(SIMILARITY_MATRIX_PATH):
            raise ValueError(
                f"{SIMILARITY_MATRIX_PATH.name} is a Git LFS pointer. "
                "Use similarity_matrics.npz for cloud deployment."
            )
        with open(SIMILARITY_MATRIX_PATH, "rb") as f:
            similarity = pickle.load(f)
        return similarity
    
    raise FileNotFoundError(
        f"Neither {SIMILARITY_NPZ_PATH.name} nor {SIMILARITY_MATRIX_PATH.name} found in repository."
    )


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
