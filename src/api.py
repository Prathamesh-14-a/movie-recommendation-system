"""
TMDB API integration for fetching movie metadata and posters.
"""
from typing import Optional, Dict
import requests

from .config import TMDB_API_KEY, TMDB_BASE_URL, TMDB_IMAGE_BASE_URL, FALLBACK_IMAGE_URL, get_tmdb_api_key


class TMDBClient:
    """
    Client for TheMovieDatabase (TMDB) API.
    Handles fetching movie posters and additional metadata.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize TMDB client.
        
        Args:
            api_key: TMDB API key (uses get_tmdb_api_key from config if not provided)
        """
        self.api_key = api_key or get_tmdb_api_key()
        self.base_url = TMDB_BASE_URL
        self.image_base_url = TMDB_IMAGE_BASE_URL
        self.session = requests.Session()
        self.session.timeout = 5  # 5 second timeout
    
    def fetch_poster(self, movie_id: int) -> str:
        """
        Fetch poster URL for a movie.
        
        Args:
            movie_id: TMDB movie ID
            
        Returns:
            URL to movie poster, or fallback image if not available
        """
        if not self.api_key:
            return self.get_fallback_image("API key not configured")
        
        try:
            url = f"{self.base_url}/{movie_id}"
            params = {
                "api_key": self.api_key,
                "language": "en-US"
            }
            
            response = self.session.get(url, params=params, timeout=5)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('poster_path'):
                # Construct full image URL correctly
                poster_path = data['poster_path']
                return f"{self.image_base_url}{poster_path}"
            else:
                return self.get_fallback_image("No poster available")
                
        except requests.exceptions.RequestException as e:
            print(f"Error fetching poster for movie {movie_id}: {e}")
            return self.get_fallback_image(f"API Error: {str(e)[:30]}")
        except Exception as e:
            print(f"Unexpected error fetching poster: {e}")
            return self.get_fallback_image("Error loading poster")
    
    def fetch_movie_details(self, movie_id: int) -> Optional[Dict]:
        """
        Fetch detailed information for a movie.
        
        Args:
            movie_id: TMDB movie ID
            
        Returns:
            Dictionary with movie details or None if error
        """
        if not self.api_key:
            return None
        
        try:
            url = f"{self.base_url}/{movie_id}"
            params = {
                "api_key": self.api_key,
                "language": "en-US",
                "append_to_response": "credits"
            }
            
            response = self.session.get(url, params=params, timeout=5)
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching details for movie {movie_id}: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error fetching details: {e}")
            return None
    
    @staticmethod
    def get_fallback_image(reason: str = "No image") -> str:
        """
        Get fallback image URL.
        
        Args:
            reason: Reason for using fallback
            
        Returns:
            Fallback image URL
        """
        return FALLBACK_IMAGE_URL
