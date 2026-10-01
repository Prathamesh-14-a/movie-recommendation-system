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
SIMILARITY_NPZ_PATH = DATA_DIR / "similarity_matrics.npz"
SIMILARITY_MATRIX_PATH = DATA_DIR / "similarity_matrics.pkl"

# TMDB API Configuration - Support both Streamlit Secrets (Cloud) and .env (Local)
def get_tmdb_api_key() -> str:
    """
    Retrieve TMDB API key from:
    1. Environment variables (e.g. from local .env via python-dotenv)
    2. Streamlit secrets (when running on Streamlit Cloud or if secrets.toml exists)
    """
    # 1. Check environment variable first (from .env or cloud environment)
    env_key = os.getenv("TMDB_API_KEY", "").strip()
    if env_key:
        return env_key

    # 2. Check Streamlit secrets ONLY if a secrets file exists on disk
    try:
        from pathlib import Path
        import streamlit as st
        secret_files = st.config.get_option("secrets.files") or []
        if any(Path(p).exists() for p in secret_files):
            if "TMDB_API_KEY" in st.secrets:
                key = st.secrets["TMDB_API_KEY"]
                if key and str(key).strip():
                    return str(key).strip()
    except Exception:
        pass

    return ""

TMDB_API_KEY = get_tmdb_api_key()
TMDB_BASE_URL = "https://api.themoviedb.org/3/movie"
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"

# Fallback image URL (offline SVG data URL so it never times out or fails on network issues)
FALLBACK_IMAGE_URL = (
    "data:image/svg+xml;charset=utf-8,"
    "%3Csvg xmlns='http://www.w3.org/2000/svg' width='500' height='750' viewBox='0 0 500 750'%3E"
    "%3Crect width='500' height='750' fill='%2311131f'/%3E"
    "%3Ccircle cx='250' cy='330' r='60' fill='%231e2338'/%3E"
    "%3Ctext x='250' y='345' font-size='48' text-anchor='middle'%3E%F0%9F%8E%AC%3C/text%3E"
    "%3Ctext x='250' y='430' font-family='sans-serif' font-size='22' font-weight='600' fill='%238b5cf6' text-anchor='middle'%3ENo Poster Available%3C/text%3E"
    "%3Ctext x='250' y='465' font-family='sans-serif' font-size='14' fill='%2364748b' text-anchor='middle'%3ETMDB Key Required%3C/text%3E"
    "%3C/svg%3E"
)

# App Configuration
PAGE_CONFIG = {
    "page_title": "MovieMind - AI Movie Recommendations",
    "page_icon": "🎬",
    "layout": "wide",
    "initial_sidebar_state": "auto",
}

# Recommendation defaults
DEFAULT_NUM_RECOMMENDATIONS = 5
