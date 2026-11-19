"""Root wrapper for data module to support direct import as `import data`.
"""

from src.data import (
    load_data,
    load_movies,
    load_ratings,
    download_movielens_100k,
    get_popular_movies,
    GENRE_NAMES,
    DEFAULT_DATA_DIR,
)

__all__ = [
    "load_data",
    "load_movies",
    "load_ratings",
    "download_movielens_100k",
    "get_popular_movies",
    "GENRE_NAMES",
    "DEFAULT_DATA_DIR",
]

if __name__ == "__main__":
    print("Testing data loading...")
    movies, ratings = load_data()
    print(f"Loaded {len(movies)} movies and {len(ratings)} ratings.")
    print("Top 5 popular movies (cold start baseline):")
    print(get_popular_movies(movies, ratings, top_n=5))
