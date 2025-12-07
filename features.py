# features.py

import numpy as np
import pandas as pd
from collections import Counter
from scipy.sparse import csr_matrix
from sklearn.preprocessing import MinMaxScaler

from config import INTEGRATED_MOVIES_PATH, ML_RATINGS_INTEGRATED_PATH


def load_integrated_data():
    """
    Load integrated movies and integrated MovieLens ratings
    from the CSVs created by your prep_data.py pipeline.
    """
    movies = pd.read_csv(INTEGRATED_MOVIES_PATH)
    ratings = pd.read_csv(ML_RATINGS_INTEGRATED_PATH)

    # Basic sanity checks
    if "movie_id" not in movies.columns:
        raise ValueError("integrated_movies.csv is missing 'movie_id' column.")
    if "movie_id" not in ratings.columns or "userId" not in ratings.columns:
        raise ValueError("movielens_ratings_integrated.csv must have 'movie_id' and 'userId' columns.")

    print(f"[Features] Loaded {len(movies)} integrated movies")
    print(f"[Features] Loaded {len(ratings)} integrated ratings")

    return movies, ratings


def build_movie_feature_matrix(movies: pd.DataFrame):
    """
    Build a numeric feature matrix for movies for content-based recommendation.

    Features:
      - Genres (multi-hot)
      - Year (scaled)
      - Rating_value (scaled)
      - Popularity (scaled)

    Returns:
      X_movie: (num_movies x num_features) numpy array
      movie_id_to_idx: dict movie_id -> row index in X_movie
      idx_to_movie_id: dict row index -> movie_id
      genres_vocab: list of genre tokens
      scaler: fitted MinMaxScaler for numeric features
    """
    # ---------------------------
    # 1) Build genre vocabulary
    # ---------------------------
    genre_counts = Counter()
    for g in movies["genres_norm"].fillna(""):
        if not g:
            continue
        for genre in str(g).split("|"):
            genre = genre.strip().lower()
            if genre:
                genre_counts[genre] += 1

    # Keep genres that appear more than a few times to avoid crazy sparsity
    genres_vocab = sorted([g for g, c in genre_counts.items() if c > 5])
    genre_to_idx = {g: i for i, g in enumerate(genres_vocab)}
    num_movies = len(movies)
    num_genres = len(genres_vocab)

    X_genres = np.zeros((num_movies, num_genres), dtype=np.float32)

    for i, gstr in enumerate(movies["genres_norm"].fillna("")):
        if not gstr:
            continue
        for g in str(gstr).split("|"):
            g = g.strip().lower()
            if g in genre_to_idx:
                X_genres[i, genre_to_idx[g]] = 1.0

    # ---------------------------
    # 2) Numeric features
    # ---------------------------
    numeric_cols = []

    # Ensure columns exist before using them
    for col in ["year", "rating_value", "popularity"]:
        if col in movies.columns:
            numeric_cols.append(col)

    if not numeric_cols:
        # Fallback: no numeric features, just genres
        X_movie = X_genres
        scaler = None
    else:
        numeric = movies[numeric_cols].copy()
        numeric = numeric.fillna(numeric.mean())
        scaler = MinMaxScaler()
        X_num = scaler.fit_transform(numeric.values.astype(float))
        X_movie = np.hstack([X_genres, X_num])

    # ---------------------------
    # 3) ID mappings
    # ---------------------------
    movie_ids = movies["movie_id"].values
    movie_id_to_idx = {mid: i for i, mid in enumerate(movie_ids)}
    idx_to_movie_id = {i: mid for mid, i in movie_id_to_idx.items()}

    print(f"[Features] Movie feature matrix shape: {X_movie.shape}")
    print(f"[Features] Genres used: {len(genres_vocab)}")

    return X_movie, movie_id_to_idx, idx_to_movie_id, genres_vocab, scaler


def build_user_item_matrix(ratings: pd.DataFrame):
    """
    Build a sparse user-item rating matrix for collaborative filtering.

    Uses:
      - rows: MovieLens userId
      - cols: integrated movie_id

    Returns:
      R: scipy.sparse.csr_matrix (num_users x num_movies)
      user_id_to_idx: dict userId -> row index
      idx_to_user_id: dict row index -> userId
      movie_id_to_idx_cf: dict movie_id -> col index
      idx_to_movie_id_cf: dict col index -> movie_id
    """
    user_ids = ratings["userId"].unique()
    movie_ids = ratings["movie_id"].unique()

    user_id_to_idx = {uid: i for i, uid in enumerate(user_ids)}
    idx_to_user_id = {i: uid for uid, i in user_id_to_idx.items()}

    movie_id_to_idx_cf = {mid: i for i, mid in enumerate(movie_ids)}
    idx_to_movie_id_cf = {i: mid for mid, i in movie_id_to_idx_cf.items()}

    rows = ratings["userId"].map(user_id_to_idx).values
    cols = ratings["movie_id"].map(movie_id_to_idx_cf).values
    data = ratings["rating"].astype(np.float32).values

    num_users = len(user_ids)
    num_movies = len(movie_ids)

    R = csr_matrix((data, (rows, cols)), shape=(num_users, num_movies))

    print(f"[Features] Rating matrix shape: {R.shape} (users x movies)")

    return R, user_id_to_idx, idx_to_user_id, movie_id_to_idx_cf, idx_to_movie_id_cf
