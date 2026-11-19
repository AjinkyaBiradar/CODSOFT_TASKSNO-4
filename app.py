"""Streamlit Web Application for MovieLens 100K Recommendation System.

Features two interactive tabs:
1. Similar Movies: Content-Based Filtering using TF-IDF and Cosine Similarity on genres.
2. For a User: Collaborative Filtering using SVD Matrix Factorization with cold-start support.
"""

import streamlit as st
import pandas as pd
import numpy as np

from src.data import load_data, get_popular_movies
from src.content_based import ContentBasedRecommender
from src.collaborative import CollaborativeRecommender

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="MovieLens Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        color: #1E293B;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 12px;
        border: 1px solid #E2E8F0;
        text-align: center;
    }
    .badge-cold {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-svd {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)


@st.cache_resource(show_spinner="Loading MovieLens 100K dataset and initializing recommendation models...")
def init_models():
    """Caches data and recommendation model initialization for fast reload.
    """
    movies_df, ratings_df = load_data()
    content_model = ContentBasedRecommender(movies_df)
    collab_model = CollaborativeRecommender(movies_df, ratings_df)
    return movies_df, ratings_df, content_model, collab_model


# Initialize models and data
movies_df, ratings_df, content_model, collab_model = init_models()

# App Header
st.markdown('<div class="main-header">🎬 MovieLens 100K Recommendation System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Explore content-based genre similarities and collaborative SVD personalized recommendations.</div>',
    unsafe_allow_html=True
)

# Sidebar: Dataset stats & controls
with st.sidebar:
    st.header("📊 Dataset Statistics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Movies", f"{len(movies_df):,}")
        st.metric("Total Users", f"{ratings_df['user_id'].nunique():,}")
    with col2:
        st.metric("Total Ratings", f"{len(ratings_df):,}")
        st.metric("Avg Rating", f"{ratings_df['rating'].mean():.2f} ⭐")

    st.divider()
    st.header("⚙️ General Settings")
    n_recommendations = st.slider("Number of Recommendations (N):", min_value=1, max_value=20, value=5, step=1)

    st.divider()
    st.markdown("### 📈 Evaluation Benchmarks")
    st.markdown("- **Algorithm:** SVD (scikit-learn)")
    st.markdown("- **Test RMSE:** `0.9885`")
    st.markdown("- **Precision@5:** `72.11%` (Ratings $\ge$ 4.0)")

# Main Tabs
tab1, tab2 = st.tabs(["🎯 Similar Movies (Content-Based)", "👤 For a User (Collaborative SVD)"])

# -------------------------------------------------------------
# TAB 1: Content-Based Filtering
# -------------------------------------------------------------
with tab1:
    st.subheader("Discover Movies Similar to Your Favorite Film")
    st.caption("Matches movies based on genre TF-IDF representation and Cosine Similarity.")

    col_input, col_action = st.columns([3, 1])

    all_titles = sorted(movies_df["title"].unique().tolist())

    with col_input:
        selected_title = st.selectbox(
            "Select or search for a movie title:",
            options=all_titles,
            index=all_titles.index("Toy Story (1995)") if "Toy Story (1995)" in all_titles else 0,
            help="Choose from the list or start typing to search titles with release year."
        )

    # Optional manual title test for edge cases (unknown title handling)
    with st.expander("🔍 Test Unknown Title / Custom Search (Edge Case Handler)"):
        custom_query = st.text_input(
            "Enter any title (e.g., 'Star Wars', 'toy story', 'Matrix', 'Unknown Movie XYZ'):",
            value=""
        )

    query_to_run = custom_query.strip() if custom_query.strip() else selected_title

    # Show metadata for chosen query
    matched_title, idx, suggestions = content_model.find_movie(query_to_run)

    if idx is not None:
        target_movie = movies_df.iloc[idx]
        st.info(f"🎞️ **Selected Movie:** **{target_movie['title']}** | **Genres:** {target_movie['genres']}")

        try:
            similar_df = content_model.recommend_similar(query_to_run, n=n_recommendations)

            if not similar_df.empty:
                # Format dataframe for display
                display_df = similar_df.copy()
                display_df.index = range(1, len(display_df) + 1)
                display_df.index.name = "Rank"
                display_df["Similarity Match"] = display_df["similarity_score"].apply(lambda s: f"{s * 100:.1f}%")
                
                # Reorder columns
                display_cols = ["title", "genres", "Similarity Match", "movie_id"]
                display_df = display_df.rename(columns={
                    "title": "Movie Title",
                    "genres": "Genres",
                    "movie_id": "Movie ID"
                })

                st.markdown(f"#### Top {n_recommendations} Recommendations for *{target_movie['title']}*")
                st.dataframe(
                    display_df[["Movie Title", "Genres", "Similarity Match", "Movie ID"]],
                    use_container_width=True
                )
            else:
                st.warning("No similar movies found for the selected title.")
        except Exception as e:
            st.error(f"Error generating recommendations: {str(e)}")
    else:
        st.error(f"❌ Movie **'{query_to_run}'** not found in the dataset.")
        if suggestions:
            st.warning(f"💡 Did you mean one of these? **{', '.join(suggestions)}**")
        else:
            st.info("Tip: Try searching by main keywords or checking spelling.")

# -------------------------------------------------------------
# TAB 2: Collaborative Filtering
# -------------------------------------------------------------
with tab2:
    st.subheader("Personalized Movie Recommendations for a User")
    st.caption("Leverages Singular Value Decomposition (SVD) matrix factorization with automatic cold-start fallback.")

    c_user, c_filter = st.columns([2, 2])
    with c_user:
        user_id_input = st.number_input(
            "Enter User ID (Known users: 1 to 943; or try 99999 for Cold-Start):",
            min_value=1,
            max_value=100000,
            value=1,
            step=1,
            help="Enter an integer user ID. IDs outside the range 1-943 trigger cold-start logic."
        )

    with c_filter:
        exclude_rated = st.checkbox(
            "Exclude movies already watched/rated by this user",
            value=True,
            help="When checked, filters out movies the user has already seen to provide fresh discoveries."
        )

    # Check user status
    is_cold = collab_model.is_cold_start_user(user_id_input)

    if is_cold:
        st.warning(
            f"❄️ **Cold-Start User Detected (User ID: {user_id_input})**\n\n"
            "This user has no prior rating history. System automatically falls back to "
            "**Bayesian Weighted Top-Rated & Popular Movies**."
        )
    else:
        user_ratings_dict = collab_model.get_user_rated_movies(user_id_input)
        user_avg = float(np.mean(list(user_ratings_dict.values()))) if user_ratings_dict else 0.0
        
        st.success(
            f"👤 **Known User Profile (User ID: {user_id_input})** | "
            f"**Rated Movies:** {len(user_ratings_dict)} | **User Average Rating:** {user_avg:.2f} ⭐"
        )

        with st.expander(f"📜 View Top Rated Movies by User {user_id_input} in Dataset"):
            user_ratings_df = ratings_df[ratings_df["user_id"] == user_id_input].merge(
                movies_df[["movie_id", "title", "genres"]],
                on="movie_id",
                how="left"
            ).sort_values("rating", ascending=False).head(10)

            user_ratings_display = user_ratings_df[["title", "genres", "rating"]].rename(columns={
                "title": "Movie Title",
                "genres": "Genres",
                "rating": "User's Rating"
            }).reset_index(drop=True)
            user_ratings_display.index += 1
            st.dataframe(user_ratings_display, use_container_width=True)

    # Get recommendations
    rec_df = collab_model.recommend_for_user(
        user_id=user_id_input,
        n=n_recommendations,
        exclude_rated=exclude_rated
    )

    if not rec_df.empty:
        display_rec = rec_df.copy()
        display_rec.index = range(1, len(display_rec) + 1)
        display_rec.index.name = "Rank"

        # Format rating with stars
        display_rec["Predicted Rating"] = display_rec["predicted_rating"].apply(lambda r: f"{r:.2f} ⭐")
        display_rec["Recommendation Method"] = display_rec["is_cold_start"].apply(
            lambda cs: "Popularity Weighted (Cold-Start)" if cs else "SVD Matrix Factorization"
        )

        display_rec = display_rec.rename(columns={
            "title": "Movie Title",
            "genres": "Genres",
            "movie_id": "Movie ID"
        })

        st.markdown(f"#### Top {n_recommendations} Recommendations for User {user_id_input}")
        st.dataframe(
            display_rec[["Movie Title", "Genres", "Predicted Rating", "Recommendation Method", "Movie ID"]],
            use_container_width=True
        )
    else:
        st.info("No recommendations could be generated for the given parameters.")

st.divider()
st.caption("Built with Python, Pandas, Scikit-learn, and Streamlit using the MovieLens 100K Dataset.")
