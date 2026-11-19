"""Utility script to download and extract the MovieLens 100K dataset.
"""

from src.data import download_movielens_100k, load_data

def main():
    print("Initiating MovieLens 100K dataset download and validation...")
    data_dir = download_movielens_100k()
    print(f"Dataset successfully prepared at: {data_dir}")
    movies, ratings = load_data(data_dir)
    print(f"Verified dataset integrity:")
    print(f" - Movies count:  {len(movies)}")
    print(f" - Ratings count: {len(ratings)}")
    print("Ready to run recommender system!")

if __name__ == "__main__":
    main()
