"""
Recommendation engine using content-based filtering with cosine similarity.
"""
from typing import Tuple, List, Optional
import pandas as pd
import numpy as np

from .config import DEFAULT_NUM_RECOMMENDATIONS


class MovieRecommender:
    """
    Content-based movie recommendation engine using cosine similarity.
    
    The system recommends movies based on similarity to a selected movie.
    Similarity is computed using cosine distance on feature vectors
    that include: genres, keywords, cast, director, and overview.
    """
    
    def __init__(self, movies: pd.DataFrame, similarity_matrix: np.ndarray):
        """
        Initialize the recommender with movies data and similarity matrix.
        
        Args:
            movies: DataFrame with movie information (must have 'title' and 'movie_id')
            similarity_matrix: 2D numpy array of cosine similarity scores
        """
        self.movies = movies
        self.similarity_matrix = similarity_matrix
        self._title_to_index = {title: idx for idx, title in enumerate(movies['title'])}
    
    def recommend(
        self,
        movie_title: str,
        num_recommendations: int = DEFAULT_NUM_RECOMMENDATIONS
    ) -> Tuple[List[str], List[int], List[float]]:
        """
        Recommend similar movies based on a given movie title.
        
        Args:
            movie_title: Title of the reference movie
            num_recommendations: Number of recommendations to return (default: 5)
        
        Returns:
            Tuple of:
            - List of recommended movie titles
            - List of corresponding movie IDs
            - List of similarity scores (0-1)
            
        Raises:
            ValueError: If movie title not found in database
        """
        # Find movie index
        if movie_title not in self._title_to_index:
            raise ValueError(f"Movie '{movie_title}' not found in database")
        
        movie_idx = self._title_to_index[movie_title]
        
        # Get similarity scores with all movies
        similarity_scores = self.similarity_matrix[movie_idx]
        
        # Create list of (index, similarity_score) tuples
        similar_movies = list(enumerate(similarity_scores))
        
        # Sort by similarity score (descending), exclude the movie itself (index 0)
        similar_movies = sorted(similar_movies, reverse=True, key=lambda x: x[1])[1:num_recommendations+1]
        
        # Extract results
        recommended_indices = [idx for idx, _ in similar_movies]
        scores = [score for _, score in similar_movies]
        
        # Get movie details
        recommended_movies = self.movies.iloc[recommended_indices]
        titles = recommended_movies['title'].tolist()
        movie_ids = recommended_movies['movie_id'].tolist()
        
        return titles, movie_ids, scores
    
    def get_available_movies(self) -> List[str]:
        """Get list of all available movies for selection."""
        return sorted(self.movies['title'].tolist())
    
    def get_movie_details(self, movie_title: str) -> Optional[dict]:
        """
        Get details for a specific movie.
        
        Args:
            movie_title: Movie title
            
        Returns:
            Dictionary with movie information or None if not found
        """
        movie_row = self.movies[self.movies['title'] == movie_title]
        if movie_row.empty:
            return None
        
        movie = movie_row.iloc[0]
        return {
            'title': movie['title'],
            'movie_id': movie['movie_id'],
            'overview': movie.get('overview', 'N/A'),
            'genres': movie.get('genres', []),
        }
