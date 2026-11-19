"""Collaborative Filtering Recommendation Module.

Implements Matrix Factorization via SVD (Singular Value Decomposition)
using scikit-learn TruncatedSVD, with support for Item-Item similarity,
and robust cold-start user handling.
"""

from typing import Optional, Dict, Any, List, Union
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

from src.data import load_data, get_popular_movies


class CollaborativeRecommender:
    """Collaborative filtering recommender using SVD matrix factorization and item similarity.
    """

    def __init__(
        self,
        movies_df: pd.DataFrame,
        ratings_df: pd.DataFrame,
        n_components: int = 20,
        random_state: int = 42
    ):
        self.movies_df = movies_df.copy().reset_index(drop=True)
        self.ratings_df = ratings_df.copy()
        self.n_components = n_components
        self.random_state = random_state

        self.user_item_matrix = None
        self.user_means = None
        self.global_mean = 3.0
        self.predicted_ratings_matrix = None
        self.all_users = set()
        self.all_movies = set()
        self.movie_info = {}

        self._fit()

    def _fit(self):
        """Fits SVD model on user-item ratings matrix.
        """
        # Map movie metadata for quick lookup
        self.movie_info = self.movies_df.set_index("movie_id")[["title", "genres"]].to_dict(orient="index")

        # Global statistics
        self.global_mean = float(self.ratings_df["rating"].mean())
        self.all_users = set(self.ratings_df["user_id"].unique())
        self.all_movies = set(self.movies_df["movie_id"].unique())

        # Construct User-Item Matrix
        pivot = self.ratings_df.pivot(index="user_id", columns="movie_id", values="rating")
        
        # Ensure all known movies and users are in index
        pivot = pivot.reindex(columns=sorted(list(self.all_movies)))
        self.user_item_matrix = pivot

        # User mean centering
        self.user_means = pivot.mean(axis=1).fillna(self.global_mean)
        pivot_centered = pivot.sub(self.user_means, axis=0).fillna(0.0)

        # Truncated SVD Matrix Factorization
        effective_components = min(self.n_components, pivot_centered.shape[1] - 1, pivot_centered.shape[0] - 1)
        self.svd = TruncatedSVD(n_components=effective_components, random_state=self.random_state)
        
        # Deconstruct and reconstruct predicted ratings
        user_factors = self.svd.fit_transform(pivot_centered)
        reconstructed = user_factors @ self.svd.components_

        # Add back user mean to restore rating scale [1.0, 5.0]
        pred_df = pd.DataFrame(
            reconstructed,
            index=pivot.index,
            columns=pivot.columns
        ).add(self.user_means, axis=0)

        # Clip predictions to valid rating bounds
        self.predicted_ratings_matrix = pred_df.clip(lower=1.0, upper=5.0)

        # Cache popular movies for cold-start users
        self.popular_movies = get_popular_movies(self.movies_df, self.ratings_df, min_ratings=30, top_n=50)

    def is_cold_start_user(self, user_id: int) -> bool:
        """Determines if a user is new or has no interaction history.
        """
        try:
            uid = int(user_id)
            return uid not in self.all_users
        except (ValueError, TypeError):
            return True

    def get_user_rated_movies(self, user_id: int) -> Dict[int, float]:
        """Returns dictionary of {movie_id: rating} for movies rated by user.
        """
        if self.is_cold_start_user(user_id):
            return {}
        user_ratings = self.ratings_df[self.ratings_df["user_id"] == int(user_id)]
        return dict(zip(user_ratings["movie_id"], user_ratings["rating"]))

    def predict_rating(self, user_id: int, movie_id: int) -> float:
        """Predicts rating for a specific user and movie.
        """
        if self.is_cold_start_user(user_id):
            if movie_id in self.movie_info:
                movie_ratings = self.ratings_df[self.ratings_df["movie_id"] == movie_id]["rating"]
                return float(movie_ratings.mean()) if not movie_ratings.empty else self.global_mean
            return self.global_mean

        uid = int(user_id)
        mid = int(movie_id)

        if uid in self.predicted_ratings_matrix.index and mid in self.predicted_ratings_matrix.columns:
            pred = self.predicted_ratings_matrix.loc[uid, mid]
            if not np.isnan(pred):
                return float(pred)

        return float(self.user_means.get(uid, self.global_mean))

    def recommend_for_user(
        self,
        user_id: int,
        n: int = 5,
        exclude_rated: bool = True
    ) -> pd.DataFrame:
        """Recommends top n movies for a given user.

        Handles cold-start users automatically by falling back to top-rated
        popular movies with a warning note.

        Args:
            user_id: Unique user identifier.
            n: Number of recommendations to return.
            exclude_rated: Whether to filter out movies already rated by the user.

        Returns:
            DataFrame with columns ['movie_id', 'title', 'genres', 'predicted_rating', 'is_cold_start'].
        """
        # Validate or handle cold-start
        try:
            uid = int(user_id)
        except (ValueError, TypeError):
            # Non-integer user ID -> treat as cold-start
            return self._cold_start_recommendations(n, reason="Invalid or non-numeric User ID")

        if self.is_cold_start_user(uid):
            return self._cold_start_recommendations(
                n,
                reason=f"User ID {uid} not found in training dataset (Cold-Start User)"
            )

        # Known user: Retrieve predicted ratings
        user_preds = self.predicted_ratings_matrix.loc[uid]

        # Exclude movies already rated if requested
        if exclude_rated:
            already_rated = set(self.ratings_df[self.ratings_df["user_id"] == uid]["movie_id"])
            user_preds = user_preds.drop(index=list(already_rated), errors="ignore")

        # Top N highest predicted ratings
        top_movie_ids = user_preds.sort_values(ascending=False).head(n)

        results = []
        for mid, score in top_movie_ids.items():
            meta = self.movie_info.get(mid, {"title": "Unknown Title", "genres": "Unknown"})
            results.append({
                "movie_id": int(mid),
                "title": meta["title"],
                "genres": meta["genres"],
                "predicted_rating": round(float(score), 2),
                "is_cold_start": False
            })

        return pd.DataFrame(results)

    def _cold_start_recommendations(self, n: int, reason: str = "") -> pd.DataFrame:
        """Helper to generate top-rated popular movie recommendations for cold-start users.
        """
        top_popular = self.popular_movies.head(n).copy()
        top_popular["predicted_rating"] = top_popular["score"].round(2)
        top_popular["is_cold_start"] = True
        top_popular["notes"] = reason

        cols = ["movie_id", "title", "genres", "predicted_rating", "is_cold_start"]
        return top_popular[cols].reset_index(drop=True)


# Default singleton instance
_DEFAULT_COLLAB_RECOMMENDER: Optional[CollaborativeRecommender] = None


def get_default_collab_recommender(
    movies_df: Optional[pd.DataFrame] = None,
    ratings_df: Optional[pd.DataFrame] = None
) -> CollaborativeRecommender:
    """Gets or initializes the cached CollaborativeRecommender instance.
    """
    global _DEFAULT_COLLAB_RECOMMENDER
    if _DEFAULT_COLLAB_RECOMMENDER is None:
        if movies_df is None or ratings_df is None:
            movies_df, ratings_df = load_data()
        _DEFAULT_COLLAB_RECOMMENDER = CollaborativeRecommender(movies_df, ratings_df)
    return _DEFAULT_COLLAB_RECOMMENDER


def recommend_for_user(
    user_id: int,
    n: int = 5,
    movies_df: Optional[pd.DataFrame] = None,
    ratings_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """Top-level convenience function to recommend movies for a user.

    Args:
        user_id: User identifier (integer).
        n: Number of recommendations to return.
        movies_df: Optional custom movies DataFrame.
        ratings_df: Optional custom ratings DataFrame.

    Returns:
        DataFrame with columns ['movie_id', 'title', 'genres', 'predicted_rating', 'is_cold_start'].
    """
    recommender = get_default_collab_recommender(movies_df, ratings_df)
    return recommender.recommend_for_user(user_id, n=n)
