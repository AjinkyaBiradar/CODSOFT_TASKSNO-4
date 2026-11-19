<div align="center">

# 🎬 Movie Recommendation System
### 🚀 CODSOFT Data Science Internship — Task 4

**A production-grade Movie Recommendation Engine built in Python using the canonical MovieLens 100K dataset. Features Content-Based Filtering, SVD Matrix Factorization, Offline Train/Test Evaluation (RMSE & Precision@5), and a modern dual-tab Streamlit Web Application.**

---

[![CodSoft Internship](https://img.shields.io/badge/CODSOFT-Internship%20Task%204-darkblue?style=for-the-badge&logo=google-classroom&logoColor=white)](https://www.codsoft.in/)
[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/pandas-2.0%2B-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Dataset](https://img.shields.io/badge/Dataset-MovieLens%20100K-00A859?style=for-the-badge)](https://grouplens.org/datasets/movielens/100k/)

<br>

**Author:** [Ajinkya Biradar](https://github.com/AjinkyaBiradar) • **Repository:** [CODSOFT_TASKSNO-4](https://github.com/AjinkyaBiradar/CODSOFT_TASKSNO-4)

<br>

[Project Overview](#-project-overview) • [Task Deliverables](#-task-requirements--deliverables) • [Quick Start](#-quick-start--installation) • [Streamlit App](#-streamlit-web-application) • [Evaluation](#-model-evaluation--benchmarks) • [API Guide](#-python-api-usage) • [Algorithms](#-mathematical-methodology)

</div>

---

## 📌 Project Overview

This project implements a complete, end-to-end **Movie Recommendation System** leveraging the benchmark **MovieLens 100K** dataset (100,000 ratings from 943 users across 1,682 movies).

The system combines two complementary recommendation strategies:
1. **Content-Based Filtering**: Extracts genre profiles across 19 categories using **TF-IDF Vectorization** and computes item-to-item **Cosine Similarity**. Includes intelligent **fuzzy title matching** for typo tolerance.
2. **Collaborative Filtering**: Deploys **Singular Value Decomposition (SVD)** matrix factorization to capture latent user-movie interactions with user-mean centering. Handles **cold-start users** via an IMDB/Bayesian weighted popularity algorithm.
3. **Rigorous Offline Evaluation**: Evaluated using an 80/20 train/test split reporting **RMSE (0.9885)**, **MAE (0.7832)**, and **Precision@5 (72.11%)**.
4. **Interactive Streamlit Web Dashboard**: A clean, responsive web interface with dedicated tabs for finding similar titles and generating personalized user recommendations.

---

## ✅ Task Requirements & Deliverables

Every requirement specified for CODSOFT Task 4 has been fulfilled:

| # | Requirement | Implementation | Status |
| :---: | :--- | :--- | :---: |
| **1** | **Data loading and cleaning in `data.py` with `pandas`** | Handled in `data.py` and `src/data.py`: parses binary genre flags, cleans titles, auto-downloads data, and formats dataframes. | ✅ Complete |
| **2** | **Content-based filtering: TF-IDF on genres, cosine similarity** | Implemented in `src/content_based.py`: includes `recommend_similar(movie_title, n=5)`. | ✅ Complete |
| **3** | **Collaborative filtering: SVD / item-item similarity** | Implemented in `src/collaborative.py`: includes `recommend_for_user(user_id, n=5)`. | ✅ Complete |
| **4** | **Evaluation script: train/test split, report Precision@5 and RMSE** | Implemented in `src/evaluate.py` & `run_evaluation.py`: 80/20 split reporting **RMSE: 0.9885** & **Precision@5: 72.11%**. | ✅ Complete |
| **5** | **Streamlit app with two tabs: "Similar movies" & "For a user"** | Implemented in `app.py`: dropdown selection, numeric user ID input, history preview, and clean interactive tables. | ✅ Complete |
| **6** | **Handle edge cases (unknown titles, cold-start users)** | Fuzzy matching via `difflib` for misspelled titles; Bayesian weighted popularity fallback for cold-start users. | ✅ Complete |
| **7** | **Folder structure, full code, requirements.txt, and README** | Clean modular architecture, dependencies documented, and comprehensive setup guide provided. | ✅ Complete |

---

## 🏛️ System Architecture

```text
                               ┌───────────────────────────────────────────────┐
                               │           MovieLens 100K Dataset              │
                               │        100,000 ratings | 1,682 movies         │
                               └───────────────────────┬───────────────────────┘
                                                       │
                                            src/data.py / data.py
                                  (Schema Ingestion, Cleaning & Pipeline)
                                                       │
                       ┌───────────────────────────────┴───────────────────────────────┐
                       ▼                                                               ▼
        ┌─────────────────────────────┐                                 ┌─────────────────────────────┐
        │    Content-Based Filter     │                                 │    Collaborative Filter     │
        │   TF-IDF on 19 Categories   │                                 │     TruncatedSVD Latent     │
        │    + Cosine Similarity      │                                 │     Matrix Factorization    │
        └──────────────┬──────────────┘                                 └──────────────┬──────────────┘
                       │                                                               │
                       ▼                                                               ▼
        recommend_similar(title, n)                                     recommend_for_user(user_id, n)
        • Exact & Substring Match                                       • User-mean Centered SVD
        • Fuzzy Levenshtein Distance                                    • Bayesian Cold-Start Fallback
        • Deduplication of Query                                        • Excludes Watched Films
                       │                                                               │
                       └───────────────────────────────┬───────────────────────────────┘
                                                       ▼
                                      ┌─────────────────────────────────┐
                                      │       Streamlit Dashboard       │
                                      │   Tab 1: Similar Movies Search  │
                                      │   Tab 2: Personalized User Recs │
                                      └─────────────────────────────────┘
```

---

## 📁 Repository Structure

```text
CODSOFT_TASKSNO-4/
├── data/
│   └── ml-100k/            # MovieLens 100K data files (u.data, u.item, u.genre, etc.)
├── src/
│   ├── __init__.py         # Package root exposing core functions
│   ├── data.py             # Data loading, schema cleaning, auto-downloader & Bayesian ranking
│   ├── content_based.py    # TF-IDF vectorizer, cosine similarity, recommend_similar()
│   ├── collaborative.py    # SVD matrix factorization, recommend_for_user(), cold-start logic
│   └── evaluate.py         # Train/test split evaluation: RMSE, MAE, Precision@5
├── data.py                 # Root wrapper supporting direct `import data`
├── app.py                  # Two-tab Streamlit web application
├── download_data.py        # Standalone dataset verification and auto-download script
├── run_evaluation.py       # Standalone CLI evaluation runner
├── requirements.txt        # Production dependencies
├── .gitignore              # Ignores byte-cache, virtual environments, and temp files
└── README.md               # Project documentation and manual
```

---

## 📥 Dataset Download Steps

The project uses the canonical **MovieLens 100K** dataset published by GroupLens Research at the University of Minnesota.

### Option A: Automatic Download (Recommended)
You do **not** need to manually download or unpack anything. All scripts and the web app automatically check for `data/ml-100k/`. If not present, the files are downloaded and extracted automatically on first run.

To trigger the download immediately via command line:
```bash
python download_data.py
```

### Option B: Manual Download
If you are working offline:
1. Download the archive directly from GroupLens:
   - **Download Link:** [https://files.grouplens.org/datasets/movielens/ml-100k.zip](https://files.grouplens.org/datasets/movielens/ml-100k.zip)
2. Extract the archive into your project directory under `data/ml-100k/`:
   ```text
   CODSOFT_TASKSNO-4/
   └── data/
       └── ml-100k/
           ├── u.data          # User ID | Movie ID | Rating | Timestamp
           ├── u.item          # Movie ID | Title | Release Date | Genres...
           ├── u.genre         # 19 Genre definitions
           └── ...
   ```

---

## ⚙️ Quick Start & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/AjinkyaBiradar/CODSOFT_TASKSNO-4.git
cd CODSOFT_TASKSNO-4
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

---

## 🖥️ Streamlit Web Application

Launch the interactive web application with:

```bash
streamlit run app.py
```

The web dashboard will open in your browser at:
👉 **`http://localhost:8501`**

### UI Features

#### 🎯 Tab 1: "Similar movies" (Content-Based)
- **Alphabetical Dropdown**: Select any of the 1,682 movies.
- **Fuzzy Search Box**: Test queries with typos or partial names (e.g. `"toy story"`, `"star wars"`). The algorithm finds the best match or suggests alternatives.
- **Clean Results Table**: Displays **Rank**, **Movie Title**, **Genres**, **Similarity Match %**, and **Movie ID**.

#### 👤 Tab 2: "For a user" (Collaborative SVD)
- **User ID Input**: Input user IDs from `1` to `943`.
- **User History Preview**: Inspect the user's historical ratings, average score, and top-rated favorites.
- **Personalized Recommendations**: SVD predicts ratings on unrated items, displaying predicted stars (e.g. `4.61 ⭐`).
- **Cold-Start Indicator**: Entering an unrecognized user ID (e.g. `99999`) automatically activates the Bayesian popularity fallback, accompanied by a clear status banner.

---

## 📊 Model Evaluation & Benchmarks

The model is evaluated using an offline **80% Train / 20% Holdout Test** split (`test_size=0.2, random_state=42`). The SVD model is fitted strictly on the training set to prevent data leakage.

### Run the Evaluation:
```bash
python run_evaluation.py
```

### Terminal Output:
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

### Metric Definitions:
1. **RMSE (Root Mean Squared Error)**: Standard metric measuring deviation between predicted rating and true rating across all 20,000 test interactions:
   $$\text{RMSE} = \sqrt{\frac{1}{|\mathcal{D}_{\text{test}}|} \sum_{(u,i) \in \mathcal{D}_{\text{test}}} (r_{u,i} - \hat{r}_{u,i})^2} = \mathbf{0.9885}$$
2. **MAE (Mean Absolute Error)**: Average absolute rating difference is only **0.78** stars.
3. **Precision@5**: Out of the top-5 recommended test items for users, **72.11%** were confirmed to be movies the user genuinely liked (rated $\ge 4.0$).

---

## 💡 Python API Usage

The modules can be imported directly into Python scripts or Jupyter notebooks:

### 1. Content-Based Recommendations (`recommend_similar`)
```python
from src.content_based import recommend_similar

# Find top 5 movies similar to "Toy Story (1995)" based on genres
results = recommend_similar(movie_title="Toy Story (1995)", n=5)
print(results[["movie_id", "title", "genres", "similarity_score"]])
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

### 2. Collaborative Recommendations (`recommend_for_user`)
```python
from src.collaborative import recommend_for_user

# Recommend top 5 movies for User 1 (excluding previously rated movies)
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

### 3. Cold-Start Demonstration
```python
from src.collaborative import recommend_for_user

# Query with an unseen user ID
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

### 1. Genre TF-IDF Vectorization & Cosine Similarity
Each movie $i$ is converted into a vector $\mathbf{v}_i$ over the 19 genre categories:
$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \log\left(\frac{1 + |D|}{1 + \text{DF}(t, D)}\right) + 1$$
Cosine similarity between movies $i$ and $j$ measures alignment:
$$\text{Sim}(i, j) = \frac{\mathbf{v}_i \cdot \mathbf{v}_j}{\|\mathbf{v}_i\| \|\mathbf{v}_j\|}$$

### 2. SVD Matrix Factorization with User-Mean Centering
The sparse User-Item matrix $R \in \mathbb{R}^{M \times N}$ is centered by subtracting each user's mean rating $\mu_u$:
$$R_{\text{norm}}(u, i) = R(u, i) - \mu_u$$
`TruncatedSVD` factors $R_{\text{norm}} \approx U_k \Sigma_k V_k^T$. Predictions on unrated movie $i$ for user $u$ are reconstructed as:
$$\hat{r}_{u, i} = \text{clip}\left(\left(U_k \Sigma_k V_k^T\right)_{u, i} + \mu_u, \; 1.0, \; 5.0\right)$$

### 3. Bayesian Weighted Popularity (Cold-Start Solution)
To recommend high-quality, proven movies to new users, the system computes the IMDB/Bayesian formula:
$$\text{WR} = \left(\frac{v}{v + m}\right) R + \left(\frac{m}{v + m}\right) C$$
Where:
- $v$: Number of ratings for the movie
- $m$: Minimum ratings threshold ($m = 30$)
- $R$: Average rating of the movie
- $C$: Global mean rating across all movies ($C \approx 3.53$)

---

## 🛡️ Edge Cases Handled

| Scenario | Edge Case Challenge | Engineering Solution |
| :--- | :--- | :--- |
| **Misspelled / Partial Title** | User searches with typos (e.g. `"toy stori"` or `"matrix"`). | Multi-stage resolution: Exact match $\rightarrow$ Case-insensitive match $\rightarrow$ Substring search $\rightarrow$ Levenshtein fuzzy distance (`difflib.get_close_matches`). Returns best match or suggested alternatives. |
| **Cold-Start Users** | Unregistered or new user enters the platform without any rating history. | Triggers Bayesian weighted popularity ranking, recommending proven high-rated films with explicit `is_cold_start = True` flag. |
| **Already Rated Items** | Recommending movies the user has already watched ruins user experience. | `exclude_rated = True` dynamically filters out past user interactions before ranking predicted scores. |
| **Rating Boundary Excursions** | Unconstrained matrix factorization can predict ratings outside $[1.0, 5.0]$. | All predicted ratings are clipped to valid bounds $[1.0, 5.0]$. |

---

## 👨‍💻 Author & Acknowledgments

- **Author:** [Ajinkya Biradar](https://github.com/AjinkyaBiradar)
- **Internship:** [CODSOFT Data Science Internship](https://www.codsoft.in/)
- **Task:** Task 4 — Movie Recommendation System
- **Dataset:** MovieLens 100K provided by GroupLens Research, University of Minnesota.
- **License:** Released under the [MIT License](LICENSE).
