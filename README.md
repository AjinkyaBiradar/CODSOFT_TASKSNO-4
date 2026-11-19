<div align="center">

# 🎬 MovieLens 100K Movie Recommendation System

**A production-ready hybrid recommendation engine in Python featuring Content-Based Filtering, SVD Matrix Factorization, Offline Train/Test Evaluation, and an Interactive Streamlit Web Application.**

[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/pandas-2.0%2B-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![MovieLens](https://img.shields.io/badge/Dataset-MovieLens%20100K-00A859?style=for-the-badge)](https://grouplens.org/datasets/movielens/100k/)

[Features](#-key-features) • [Quick Start](#-quick-start) • [Web App](#-streamlit-web-application) • [Evaluation](#-offline-evaluation--benchmarks) • [API Usage](#-python-api-usage) • [Algorithms](#-mathematical-methodology)

</div>

---

## 🌟 Key Features

| Capability | Implementation | Highlight |
| :--- | :--- | :--- |
| **Content-Based Filtering** | TF-IDF on 19 genre vectors + Cosine Similarity | Accurately groups movies with similar thematic and genre profiles |
| **Collaborative Filtering** | TruncatedSVD Matrix Factorization with user-mean centering | Predicts personal ratings $\hat{r}_{u,i}$ on unseen items |
| **Fuzzy Title Search** | `difflib.get_close_matches` + case-insensitive substring search | Handles typos, partial queries, and unknown movie names gracefully |
| **Cold-Start Handling** | IMDB/Bayesian Weighted Popularity Formula | Automatically suggests top-rated popular films when users have no history |
| **Offline Evaluation** | 80/20 Train/Test split | Evaluates both ranking accuracy (**Precision@5: 72.11%**) and rating error (**RMSE: 0.9885**) |
| **Streamlit Dashboard** | Dual-tab modern Web UI | Tab 1 for movie similarity exploration & Tab 2 for personalized user recommendations |

---

## 📁 Repository Folder Structure

```text
CODSOFT_TASKSNO-4/
│
├── data/
│   └── ml-100k/            # Extracted dataset files (u.data, u.item, u.genre, etc.)
│
├── src/
│   ├── __init__.py         # Package root exposing clean API functions
│   ├── data.py             # Data ingestion, schema cleaning, auto-downloader & Bayesian ranking
│   ├── content_based.py    # TF-IDF vectorizer, cosine similarity, recommend_similar()
│   ├── collaborative.py    # SVD matrix factorization, recommend_for_user(), cold-start logic
│   └── evaluate.py         # Train/test split evaluation: RMSE, MAE, Precision@5
│
├── data.py                 # Root alias for direct `import data` usage
├── app.py                  # Interactive dual-tab Streamlit web application
├── download_data.py        # Standalone dataset verification and auto-download script
├── run_evaluation.py       # Standalone CLI evaluation runner
├── requirements.txt        # Production dependencies
└── README.md               # Complete documentation and user manual
```

---

## 📥 Dataset Download Steps

This project uses the official **MovieLens 100K** dataset published by GroupLens Research at the University of Minnesota (comprising 100,000 ratings from 943 users on 1,682 movies).

### Method 1: Automatic Download (Recommended)
You do **not** need to manually download or unpack anything. All scripts and the web app automatically check for `data/ml-100k/`. If not present, the files are downloaded and extracted on first run.

To trigger the download immediately via command line:
```bash
python download_data.py
```

### Method 2: Manual Download
If you are working offline or prefer manual setup:
1. Download `ml-100k.zip` from GroupLens:
   - **URL:** [https://files.grouplens.org/datasets/movielens/ml-100k.zip](https://files.grouplens.org/datasets/movielens/ml-100k.zip)
2. Extract the contents so that `u.data` and `u.item` are placed directly in `data/ml-100k/`:
   ```text
   CODSOFT_TASKSNO-4/
   └── data/
       └── ml-100k/
           ├── u.data
           ├── u.item
           ├── u.genre
           ├── u.info
           └── u.user
   ```

---

## 🚀 Quick Start

### 1. Clone or Open the Workspace
```bash
cd "CODSOFT_TASKSNO-4"
```

### 2. Create and Activate a Virtual Environment
- **Windows (PowerShell):**
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```
- **macOS / Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

> [!NOTE]
> The primary dependencies are `pandas`, `numpy`, `scikit-learn`, `streamlit`, and `requests`. No complex external C-compilers or heavyweight packages are needed.

---

## 🖥️ Streamlit Web Application

To start the interactive web application, run:

```bash
streamlit run app.py
```

Streamlit will launch locally at `http://localhost:8501`.

```text
 ┌────────────────────────────────────────────────────────────────────────┐
 │ 🎬 MovieLens 100K Recommendation System                                │
 ├────────────────────────────────────────────────────────────────────────┤
 │                                                                        │
 │  ┌─────────────────────────────────┐  ┌─────────────────────────────┐  │
 │  │ 🎯 Tab 1: Similar Movies        │  │ 👤 Tab 2: For a User        │  │
 │  └─────────────────────────────────┘  └─────────────────────────────┘  │
 │                                                                        │
 │  • Search / select 1,682 movies        • Enter User ID (1 - 943)       │
 │  • Fuzzy title matching & typo fixes   • Cold-start fallback detection │
 │  • Genre TF-IDF similarity %           • SVD predicted ratings ⭐      │
 │  • Interactive clean results table     • History preview of past likes │
 └────────────────────────────────────────────────────────────────────────┘
```

### Tab Walkthrough

#### 🎯 Tab 1: "Similar movies" (Content-Based)
1. **Dropdown Selector**: Choose any of the 1,682 movies sorted alphabetically.
2. **Search / Fuzzy Finder**: Type a title with partial spelling (e.g. `"toy story"`, `"star wars"`). The algorithm identifies the exact movie or presents the closest suggestions.
3. **Similarity Table**: Ranks top $N$ movies with genre details and match percentages.

#### 👤 Tab 2: "For a user" (Collaborative SVD)
1. **User ID Input**: Enter any integer user ID (IDs `1` to `943` represent known training users).
2. **User History Preview**: Inspect the user's historical ratings, average score, and top-rated favorites.
3. **Personalized Recommendations**: SVD generates predictions on unseen items, sorting highest predicted ratings first.
4. **Cold-Start Indicator**: Entering an unrecognized user ID (e.g. `99999`) automatically activates the Bayesian popularity fallback, with a clear status notice.

---

## 📊 Offline Evaluation & Benchmarks

To train on the 80% split and evaluate strictly against the 20% holdout test set, run:

```bash
python run_evaluation.py
```

### Exact Terminal Output:
```text
============================================================
       MOVIELENS 100K RECOMMENDATION MODEL EVALUATION       
============================================================
 Dataset Summary:
  • Total Users:       943
  • Total Movies:      1682
  • Train Set Size:    80,000 ratings (80%)
  • Test Set Size:     20,000 ratings (20%)
------------------------------------------------------------
 Evaluation Metrics:
  • RMSE (Root Mean Squared Error):   0.9885
  • MAE (Mean Absolute Error):        0.7832
  • Precision@5 (Rating >= 4.0):       72.11% (0.7211)
  • Test Users Evaluated for P@5:   821
============================================================
```

> [!TIP]
> **Why Precision@5 is 72.11%**: In top-5 recommendation tasks on test users, **over 72%** of items recommended by the SVD model were confirmed to be rated $\ge 4.0$ by the user in the holdout test set.

---

## 💡 Python API Usage

You can import and use the recommendation functions directly in any Python script:

### 1. Content-Based Filtering: `recommend_similar(movie_title, n=5)`
```python
from src.content_based import recommend_similar

# Find 5 movies similar to "Toy Story (1995)" based on genres
recs = recommend_similar("Toy Story (1995)", n=5)
print(recs[["movie_id", "title", "genres", "similarity_score"]])
```

**Output:**
```text
   movie_id                             title                           genres  similarity_score
0       422  Aladdin and the King of Thieves (1996)  Animation | Children's | Comedy            1.0000
1       102  Aristocats, The (1970)                  Animation | Children's | Comedy            0.9372
2       404  Pinocchio (1940)                        Animation | Children's | Comedy            0.9372
3       625  Sword in the Stone, The (1963)          Animation | Children's | Comedy            0.9372
4       946  Fox and the Hound, The (1981)           Animation | Children's | Comedy            0.9372
```

---

### 2. Collaborative Filtering: `recommend_for_user(user_id, n=5)`
```python
from src.collaborative import recommend_for_user

# Personalized top-5 recommendations for User 1 (excluding already-rated movies)
user_recs = recommend_for_user(user_id=1, n=5)
print(user_recs[["movie_id", "title", "genres", "predicted_rating", "is_cold_start"]])
```

**Output:**
```text
   movie_id                              title                          genres  predicted_rating  is_cold_start
0       276           Leaving Las Vegas (1995)                 Drama | Romance              4.61          False
1       455  Jackie Chan's First Strike (1996)                 Action | Comedy              4.56          False
2       318            Schindler's List (1993)                     Drama | War              4.43          False
3       408              Close Shave, A (1995)   Animation | Comedy | Thriller              4.41          False
4       919  City of Lost Children, The (1995)              Adventure | Sci-Fi              4.23          False
```

---

### 3. Cold-Start Demonstration
```python
from src.collaborative import recommend_for_user

# New user with no interaction history
cold_recs = recommend_for_user(user_id=99999, n=3)
print(cold_recs[["title", "predicted_rating", "is_cold_start"]])
```

**Output:**
```text
                              title  predicted_rating  is_cold_start
0           Schindler's List (1993)              4.38           True
1  Shawshank Redemption, The (1994)              4.36           True
2                 Casablanca (1942)              4.35           True
```

---

## 🧮 Mathematical Methodology

### 1. Content-Based TF-IDF & Cosine Similarity
Each movie $i$ is represented by a term-frequency inverse-document-frequency vector $\mathbf{v}_i$ over the 19 genre categories:
$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \log\left(\frac{1 + |D|}{1 + \text{DF}(t, D)}\right) + 1$$
Similarity between movies $i$ and $j$ is computed as:
$$\text{Sim}(i, j) = \frac{\mathbf{v}_i \cdot \mathbf{v}_j}{\|\mathbf{v}_i\| \|\mathbf{v}_j\|}$$

### 2. SVD Collaborative Matrix Factorization
The sparse User-Item matrix $R \in \mathbb{R}^{M \times N}$ is normalized by subtracting user baseline means $\mu_u$:
$$R_{\text{norm}}(u, i) = R(u, i) - \mu_u$$
Singular Value Decomposition approximates $R_{\text{norm}} \approx U_k \Sigma_k V_k^T$. Predictions on unseen movie $i$ for user $u$ are restored via:
$$\hat{r}_{u, i} = \text{clip}\left(\left(U_k \Sigma_k V_k^T\right)_{u, i} + \mu_u, \; 1.0, \; 5.0\right)$$

### 3. Bayesian Weighted Score (Cold-Start Users)
To avoid recommending movies with only a single 5-star rating to new users, the cold-start fallback computes the IMDB weighted score:
$$\text{WR} = \left(\frac{v}{v + m}\right) R + \left(\frac{m}{v + m}\right) C$$
Where:
- $v$: Number of ratings for the movie
- $m$: Minimum vote threshold ($m = 30$)
- $R$: Average rating of the movie
- $C$: Mean rating across all movies ($C \approx 3.53$)

---

## 🛡️ Edge Cases Handled

1. **Unknown or Misspelled Movie Titles**:
   - Multi-stage title resolution: Exact match $\rightarrow$ Case-insensitive match $\rightarrow$ Substring search $\rightarrow$ Levenshtein/Gestalt fuzzy matching via `difflib.get_close_matches`.
   - Recommends the best match or returns helpful close suggestions when input is ambiguous.
2. **Cold-Start Users**:
   - Any unknown user ID, non-numeric input, or user with zero ratings automatically triggers the Bayesian popularity baseline with clear feedback (`is_cold_start = True`).
3. **Repeated Recommendations**:
   - `exclude_rated = True` filters out items the user has already rated so recommendations always offer new discoveries.
4. **Rating Scale Boundaries**:
   - Reconstructed matrix predictions are bounded to the valid range $[1.0, 5.0]$.

---

## 📜 Citation & Credits
- **Dataset**: F. Maxwell Harper and Joseph A. Konstan. 2015. *The MovieLens Datasets: History and Context.* ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4, Article 19. [GroupLens Research](https://grouplens.org/datasets/movielens/100k/).
- **License**: MIT License. Free for educational and portfolio use.
#   C O D S O F T _ T A S K S N O - 4  
 