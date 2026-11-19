"""Content-Based Filtering Recommendation Module.

Uses TF-IDF Vectorization on movie genres and Cosine Similarity to find
the most similar movies to a given title.
"""

import difflib
from typing import Optional, List, Tuple, Union
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.data import load_movies


class ContentBasedRecommender:
    """Content-based recommender using TF-IDF on genres and cosine similarity.
    """

    def __init__(self, movies_df: pd.DataFrame):
        self.movies_df = movies_df.copy().reset_index(drop=True)
        self._build_model()

    def _build_model(self):
        """Builds TF-IDF genre vectors and computes cosine similarity matrix.
        """
        # Vectorize genres
        # token_pattern preserves multi-word/hyphenated tokens like 'filmnoir' or 'scifi'
        self.tfidf = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
        self.tfidf_matrix = self.tfidf.fit_transform(self.movies_df["genres_tokens"])

        # Compute cosine similarity matrix
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)

        # Mapping dictionaries
        self.title_to_idx = {
            title: idx for idx, title in enumerate(self.movies_df["title"])
        }
        self.lower_title_to_idx = {
            title.lower(): idx for idx, title in enumerate(self.movies_df["title"])
        }

    def find_movie(self, query: str) -> Tuple[Optional[str], Optional[int], List[str]]:
        """Finds the best matching movie title from a query string.

        Handles case insensitivity, substrings, and fuzzy matching.

        Args:
            query: User input movie title.

        Returns:
            Tuple of (matched_title, index, list_of_close_suggestions).
        """
        query_clean = str(query).strip()
        query_lower = query_clean.lower()

        # 1. Exact match
        if query_clean in self.title_to_idx:
            idx = self.title_to_idx[query_clean]
            return query_clean, idx, []

        # 2. Case-insensitive exact match
        if query_lower in self.lower_title_to_idx:
            idx = self.lower_title_to_idx[query_lower]
            actual_title = self.movies_df.loc[idx, "title"]
            return actual_title, idx, []

        # 3. Substring match
        substring_matches = self.movies_df[
            self.movies_df["title"].str.lower().str.contains(query_lower, regex=False)
        ]
        if not substring_matches.empty:
            idx = substring_matches.index[0]
            matched_title = substring_matches.loc[idx, "title"]
            suggestions = substring_matches["title"].head(5).tolist()
            return matched_title, idx, suggestions

        # 4. Fuzzy match with difflib
        all_titles = self.movies_df["title"].tolist()
        fuzzy_matches = difflib.get_close_matches(query_clean, all_titles, n=5, cutoff=0.4)
        if fuzzy_matches:
            best_match = fuzzy_matches[0]
            idx = self.title_to_idx[best_match]
            return best_match, idx, fuzzy_matches

        return None, None, []

    def recommend_similar(self, movie_title: str, n: int = 5) -> pd.DataFrame:
        """Recommends top n movies most similar to the given title based on genres.

        Args:
            movie_title: Movie title to find similarities for.
            n: Number of recommendations to return.

        Returns:
            DataFrame with columns ['movie_id', 'title', 'genres', 'similarity_score'].

        Raises:
            ValueError: If movie title cannot be found or matched.
        """
        matched_title, idx, suggestions = self.find_movie(movie_title)

        if idx is None:
            suggestion_msg = (
                f" Did you mean: {', '.join(suggestions)}?"
                if suggestions
                else " Please verify the title name or year format (e.g. 'Toy Story (1995)')."
            )
            raise ValueError(f"Movie '{movie_title}' was not found in the dataset.{suggestion_msg}")

        # Get similarity scores for all movies against target
        sim_scores = list(enumerate(self.similarity_matrix[idx]))

        # Sort by similarity score descending
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Exclude the query movie itself (index idx)
        similar_movies = [item for item in sim_scores if item[0] != idx][:n]

        results = []
        for movie_idx, score in similar_movies:
            row = self.movies_df.iloc[movie_idx]
            results.append({
                "movie_id": int(row["movie_id"]),
                "title": row["title"],
                "genres": row["genres"],
                "similarity_score": round(float(score), 4)
            })

        return pd.DataFrame(results)


# Default singleton instance
_DEFAULT_CONTENT_RECOMMENDER: Optional[ContentBasedRecommender] = None


def get_default_content_recommender(movies_df: Optional[pd.DataFrame] = None) -> ContentBasedRecommender:
    """Gets or initializes the cached ContentBasedRecommender instance.
    """
    global _DEFAULT_CONTENT_RECOMMENDER
    if _DEFAULT_CONTENT_RECOMMENDER is None:
        if movies_df is None:
            movies_df = load_movies()
        _DEFAULT_CONTENT_RECOMMENDER = ContentBasedRecommender(movies_df)
    return _DEFAULT_CONTENT_RECOMMENDER


def recommend_similar(
    movie_title: str,
    n: int = 5,
    movies_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """Top-level convenience function to recommend similar movies by title.

    Args:
        movie_title: Name of the movie (e.g. "Toy Story (1995)" or "Star Wars").
        n: Number of recommendations to return.
        movies_df: Optional custom movies DataFrame.

    Returns:
        DataFrame with columns ['movie_id', 'title', 'genres', 'similarity_score'].
    """
    recommender = get_default_content_recommender(movies_df)
    return recommender.recommend_similar(movie_title, n=n)
