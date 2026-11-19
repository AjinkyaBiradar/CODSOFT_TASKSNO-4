"""Data loading and cleaning module for MovieLens 100K dataset.
"""

import os
import io
import zipfile
import requests
import pandas as pd
from typing import Tuple, Optional, List

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "ml-100k")

GENRE_NAMES = [
    "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery",
    "Romance", "Sci-Fi", "Thriller", "War", "Western"
]


def download_movielens_100k(destination_dir: Optional[str] = None) -> str:
    """Downloads and extracts the MovieLens 100K dataset if not already present.

    Args:
        destination_dir: Directory where 'ml-100k' folder should be located.

    Returns:
        Path to the extracted 'ml-100k' directory.
    """
    if destination_dir is None:
        target_dir = DEFAULT_DATA_DIR
    else:
        target_dir = os.path.abspath(destination_dir)

    u_data_path = os.path.join(target_dir, "u.data")
    u_item_path = os.path.join(target_dir, "u.item")

    if os.path.exists(u_data_path) and os.path.exists(u_item_path):
        return target_dir

    parent_dir = os.path.dirname(target_dir)
    os.makedirs(parent_dir, exist_ok=True)

    print(f"Dataset not found at '{target_dir}'. Downloading from {MOVIELENS_URL}...")
    headers = {"User-Agent": "MovieRecommenderSystem/1.0"}
    response = requests.get(MOVIELENS_URL, headers=headers, timeout=60)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        zf.extractall(parent_dir)

    print(f"MovieLens 100K successfully extracted to: {target_dir}")
    return target_dir


def load_movies(data_dir: Optional[str] = None) -> pd.DataFrame:
    """Loads and cleans the movie metadata (u.item).

    Args:
        data_dir: Directory path containing u.item.

    Returns:
        DataFrame with movie metadata, parsed genres, and cleaned titles.
    """
    if data_dir is None:
        data_dir = download_movielens_100k()

    u_item_path = os.path.join(data_dir, "u.item")
    if not os.path.exists(u_item_path):
        data_dir = download_movielens_100k(data_dir)
        u_item_path = os.path.join(data_dir, "u.item")

    item_cols = [
        "movie_id", "title", "release_date", "video_release_date", "imdb_url"
    ] + GENRE_NAMES

    df_movies = pd.read_csv(
        u_item_path,
        sep="|",
        names=item_cols,
        encoding="latin-1",
        engine="python"
    )

    # Clean title and remove excess trailing spaces
    df_movies["title"] = df_movies["title"].astype(str).str.strip()

    # Extract human-readable genre string and clean tokenized genre representation
    def extract_genres(row: pd.Series) -> str:
        active_genres = [g for g in GENRE_NAMES if g != "unknown" and row.get(g) == 1]
        if not active_genres:
            return "Unknown"
        return " | ".join(active_genres)

    def extract_genres_tokens(row: pd.Series) -> str:
        # Standardize for TF-IDF (e.g., Children's -> Children, Film-Noir -> FilmNoir / Sci-Fi -> SciFi)
        active_genres = [
            g.replace("'", "").replace("-", "")
            for g in GENRE_NAMES
            if g != "unknown" and row.get(g) == 1
        ]
        if not active_genres:
            return "Unknown"
        return " ".join(active_genres)

    df_movies["genres"] = df_movies.apply(extract_genres, axis=1)
    df_movies["genres_tokens"] = df_movies.apply(extract_genres_tokens, axis=1)

    # Convert release_date if possible
    df_movies["release_date"] = df_movies["release_date"].fillna("Unknown")

    return df_movies


def load_ratings(data_dir: Optional[str] = None) -> pd.DataFrame:
    """Loads and cleans ratings data (u.data).

    Args:
        data_dir: Directory path containing u.data.

    Returns:
        DataFrame with columns ['user_id', 'movie_id', 'rating', 'timestamp'].
    """
    if data_dir is None:
        data_dir = download_movielens_100k()

    u_data_path = os.path.join(data_dir, "u.data")
    if not os.path.exists(u_data_path):
        data_dir = download_movielens_100k(data_dir)
        u_data_path = os.path.join(data_dir, "u.data")

    data_cols = ["user_id", "movie_id", "rating", "timestamp"]
    df_ratings = pd.read_csv(
        u_data_path,
        sep="\t",
        names=data_cols,
        encoding="latin-1",
        engine="python"
    )

    # Cast types and drop any corrupted rows
    df_ratings["user_id"] = df_ratings["user_id"].astype(int)
    df_ratings["movie_id"] = df_ratings["movie_id"].astype(int)
    df_ratings["rating"] = df_ratings["rating"].astype(float)
    df_ratings["timestamp"] = df_ratings["timestamp"].astype(int)

    return df_ratings


def load_data(data_dir: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads both movies and ratings DataFrames.

    Args:
        data_dir: Directory path containing ml-100k files.

    Returns:
        Tuple of (df_movies, df_ratings).
    """
    resolved_dir = download_movielens_100k(data_dir)
    movies = load_movies(resolved_dir)
    ratings = load_ratings(resolved_dir)
    return movies, ratings


def get_popular_movies(
    movies_df: pd.DataFrame,
    ratings_df: pd.DataFrame,
    min_ratings: int = 30,
    top_n: int = 10
) -> pd.DataFrame:
    """Returns top rated popular movies with weighted scoring for cold-start recommendations.

    Uses IMDB/Bayesian weighted formula:
    WR = (v / (v + m)) * R + (m / (v + m)) * C
    where:
        v = number of ratings for the movie
        m = minimum ratings threshold
        R = average rating of the movie
        C = mean rating across all movies

    Args:
        movies_df: Movies DataFrame.
        ratings_df: Ratings DataFrame.
        min_ratings: Minimum number of ratings required.
        top_n: Number of recommendations to return.

    Returns:
        DataFrame containing top popular movies with average rating and vote count.
    """
    stats = ratings_df.groupby("movie_id").agg(
        vote_count=("rating", "count"),
        vote_average=("rating", "mean")
    ).reset_index()

    m = min_ratings
    C = ratings_df["rating"].mean()

    # Filter candidates with at least m ratings
    qualified = stats[stats["vote_count"] >= m].copy()

    # Weighted rating calculation
    qualified["weighted_score"] = (
        (qualified["vote_count"] / (qualified["vote_count"] + m)) * qualified["vote_average"]
        + (m / (qualified["vote_count"] + m)) * C
    )

    top_movies = qualified.sort_values("weighted_score", ascending=False).head(top_n)

    result = top_movies.merge(
        movies_df[["movie_id", "title", "genres"]],
        on="movie_id",
        how="left"
    )

    result = result.rename(columns={
        "vote_count": "ratings_count",
        "vote_average": "avg_rating",
        "weighted_score": "score"
    })

    return result[["movie_id", "title", "genres", "avg_rating", "ratings_count", "score"]].reset_index(drop=True)
