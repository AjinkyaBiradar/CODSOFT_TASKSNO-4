"""MovieLens 100K Recommendation System Package.
"""

from .data import load_data, get_popular_movies
from .content_based import ContentBasedRecommender, recommend_similar
from .collaborative import CollaborativeRecommender, recommend_for_user
from .evaluate import evaluate_model

__all__ = [
    "load_data",
    "get_popular_movies",
    "ContentBasedRecommender",
    "recommend_similar",
    "CollaborativeRecommender",
    "recommend_for_user",
    "evaluate_model",
]
