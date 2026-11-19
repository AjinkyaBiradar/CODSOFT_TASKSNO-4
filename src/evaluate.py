"""Model Evaluation Module for MovieLens 100K Recommender.

Computes RMSE and Precision@K on a train/test split.
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

from src.data import load_data
from src.collaborative import CollaborativeRecommender


def evaluate_model(
    movies_df: Optional[pd.DataFrame] = None,
    ratings_df: Optional[pd.DataFrame] = None,
    test_size: float = 0.2,
    k: int = 5,
    relevance_threshold: float = 4.0,
    n_components: int = 20,
    random_state: int = 42
) -> Dict[str, Any]:
    """Evaluates the Collaborative Filtering SVD model on train/test split.

    Args:
        movies_df: Movies DataFrame.
        ratings_df: Ratings DataFrame.
        test_size: Proportion of ratings used for testing (default 0.2).
        k: Cut-off for Precision@K (default 5).
        relevance_threshold: Minimum rating to consider an item relevant (default 4.0).
        n_components: SVD latent components.
        random_state: Seed for reproducibility.

    Returns:
        Dictionary of evaluation results including RMSE and Precision@K.
    """
    if movies_df is None or ratings_df is None:
        movies_df, ratings_df = load_data()

    # Train / Test split
    train_ratings, test_ratings = train_test_split(
        ratings_df,
        test_size=test_size,
        random_state=random_state
    )

    # Fit collaborative recommender strictly on train set
    train_recommender = CollaborativeRecommender(
        movies_df=movies_df,
        ratings_df=train_ratings,
        n_components=n_components,
        random_state=random_state
    )

    # 1. Compute RMSE and MAE on test set
    y_true = []
    y_pred = []

    for _, row in test_ratings.iterrows():
        uid = int(row["user_id"])
        mid = int(row["movie_id"])
        actual_rating = float(row["rating"])
        predicted_rating = train_recommender.predict_rating(uid, mid)

        y_true.append(actual_rating)
        y_pred.append(predicted_rating)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))

    # 2. Compute Precision@K on test items
    # For each user with at least K ratings in test set:
    # rank test items by predicted score, take top K, check how many have actual rating >= threshold
    precisions = []
    users_evaluated = 0

    for uid, user_group in test_ratings.groupby("user_id"):
        if len(user_group) < k:
            continue

        test_movie_ids = user_group["movie_id"].values
        actual_dict = dict(zip(user_group["movie_id"], user_group["rating"]))

        # Predict ratings for this user on all test movies
        predicted_scores = [
            (mid, train_recommender.predict_rating(uid, mid))
            for mid in test_movie_ids
        ]

        # Rank by predicted rating descending
        predicted_scores.sort(key=lambda x: x[1], reverse=True)
        top_k_candidates = [mid for mid, _ in predicted_scores[:k]]

        # Count how many top-k items are relevant (actual rating >= threshold)
        relevant_count = sum(1 for mid in top_k_candidates if actual_dict[mid] >= relevance_threshold)
        precision_at_k = relevant_count / float(k)
        precisions.append(precision_at_k)
        users_evaluated += 1

    mean_precision_at_k = float(np.mean(precisions)) if precisions else 0.0

    return {
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        f"precision@{k}": round(mean_precision_at_k, 4),
        "k": k,
        "relevance_threshold": relevance_threshold,
        "users_evaluated": users_evaluated,
        "train_samples": len(train_ratings),
        "test_samples": len(test_ratings),
        "total_users": ratings_df["user_id"].nunique(),
        "total_movies": movies_df["movie_id"].nunique()
    }


def print_evaluation_report(results: Dict[str, Any]):
    """Pretty prints the evaluation report.
    """
    k = results.get("k", 5)
    print("=" * 60)
    print("       MOVIELENS 100K RECOMMENDATION MODEL EVALUATION       ")
    print("=" * 60)
    print(f" Dataset Summary:")
    print(f"  • Total Users:       {results['total_users']}")
    print(f"  • Total Movies:      {results['total_movies']}")
    print(f"  • Train Set Size:    {results['train_samples']} ratings (80%)")
    print(f"  • Test Set Size:     {results['test_samples']} ratings (20%)")
    print("-" * 60)
    print(f" Evaluation Metrics:")
    print(f"  • RMSE (Root Mean Squared Error):   {results['rmse']}")
    print(f"  • MAE (Mean Absolute Error):        {results['mae']}")
    print(f"  • Precision@{k} (Rating >= {results['relevance_threshold']}):       {results[f'precision@{k}'] * 100:.2f}% ({results[f'precision@{k}']})")
    print(f"  • Test Users Evaluated for P@{k}:   {results['users_evaluated']}")
    print("=" * 60)


if __name__ == "__main__":
    print("Starting evaluation on MovieLens 100K dataset...")
    metrics = evaluate_model()
    print_evaluation_report(metrics)
