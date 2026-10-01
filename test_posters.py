#!/usr/bin/env python
"""Test TMDB API poster loading."""
import os
from dotenv import load_dotenv
from src.api import TMDBClient

# Load environment
load_dotenv()

api_key = os.getenv("TMDB_API_KEY")
print(f"API Key loaded: {'[OK]' if api_key else '[MISSING]'}")
print(f"API Key (first 10 chars): {api_key[:10]}..." if api_key else "No API key")

# Test poster loading
client = TMDBClient()

# Test with Avatar (movie_id: 19995)
print("\n--- Testing Avatar (ID: 19995) ---")
poster_url = client.fetch_poster(19995)
print(f"Poster URL: {poster_url}")

# Test with The Dark Knight (movie_id: 155)
print("\n--- Testing The Dark Knight (ID: 155) ---")
poster_url = client.fetch_poster(155)
print(f"Poster URL: {poster_url}")

# Test with an invalid movie_id
print("\n--- Testing Invalid ID (9999999) ---")
poster_url = client.fetch_poster(9999999)
print(f"Poster URL: {poster_url}")
