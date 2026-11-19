"""Standalone script to execute evaluation and report RMSE and Precision@5.
"""

import sys
from src.evaluate import evaluate_model, print_evaluation_report

def main():
    print("Loading MovieLens 100K data and computing train/test split...")
    results = evaluate_model(test_size=0.2, k=5, relevance_threshold=4.0)
    print_evaluation_report(results)

if __name__ == "__main__":
    main()
