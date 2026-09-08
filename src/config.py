"""
Configuration and constants for the Movie Recommendation System.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# Paths (using Path for cross-platform compatibility)
DATA_DIR = PROJECT_ROOT
MODELS_DIR = PROJECT_ROOT

MOVIES_PKL_PATH = DATA_DIR / "movies.pkl"
SIMILARITY_MATRIX_PATH = DATA_DIR / "similarity_matrics.pkl"

# TMDB API Configuration
TMDB_API_KEY = os.getenv("TMDB_API_KEY", "")
TMDB_BASE_URL = "https://api.themoviedb.org/3/movie"
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"

# Fallback image URL
FALLBACK_IMAGE_URL = "https://via.placeholder.com/500x750?text=No+Poster"

# App Configuration
PAGE_CONFIG = {
    "page_title": "MovieMind - AI Movie Recommendations",
    "page_icon": "🎬",
    "layout": "wide",
    "initial_sidebar_state": "auto",
}

# Recommendation defaults
DEFAULT_NUM_RECOMMENDATIONS = 5
