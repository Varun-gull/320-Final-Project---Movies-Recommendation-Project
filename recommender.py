# recommender.py

from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from features import (
    load_integrated_data,
    build_movie_feature_matrix,
    build_user_item_matrix,
)
from utils import normalize_title


class MovieRecommender:
    """
    Wraps:
      - Integrated movies
      - Ratings
      - Content features
      - User-item matrix

    Provides:
      - recommend_for_existing_user(userId)
      - recommend_for_new_user([titles])
    """

    def __init__(self):
        self.movies_df = None
        self.ratings_df = None

        self.X_movie = None
        self.movie_id_to_idx_features = None
        self.idx_to_movie_id_features = None

        self.R = None
        self.user_id_to_idx = None
        self.idx_to_user_id = None
        self.movie_id_to_idx_cf = None
        self.idx_to_movie_id_cf = None

    def fit(self):
        """Load data, build feature matrices, and be ready to recommend."""
        # Load data from CSVs
        movies, ratings = load_integrated_data()
        self.movies_df = movies
        self.ratings_df = ratings

        # Build content-based feature matrix
        (
            self.X_movie,
            self.movie_id_to_idx_features,
            self.idx_to_movie_id_features,
            genres_vocab,
            scaler,
        ) = build_movie_feature_matrix(movies)

        # Build collaborative filtering rating matrix
        (
            self.R,
            self.user_id_to_idx,
            self.idx_to_user_id,
            self.movie_id_to_idx_cf,
            self.idx_to_movie_id_cf,
        ) = build_user_item_matrix(ratings)

        print("[Recommender] Fit complete. Ready to recommend.")

    # -----------------------------
    # Existing user: CF-based
    # -----------------------------

    def recommend_for_existing_user(self, user_id: int, top_k: int = 10, neighbor_k: int = 50):
        """
        User-based collaborative filtering for an existing MovieLens userId.

        Steps:
          1. Find user row in rating matrix
          2. Compute cosine similarity with all other users
          3. Aggregate neighbors' ratings into scores for unseen movies
          4. Return top_k recommended movie_ids and print them nicely
        """
        if self.R is None:
            raise RuntimeError("Model not fit yet. Call .fit() first.")

        if user_id not in self.user_id_to_idx:
            print(f"[Recommender] userId {user_id} not found in ratings.")
            return []

        u_idx = self.user_id_to_idx[user_id]
        user_row = self.R[u_idx]  # 1 x num_movies sparse

        # Movies the user has already rated
        user_ratings_dense = user_row.toarray()[0]
        seen_movie_indices = set(np.where(user_ratings_dense > 0)[0])

        # Compute similarities on the fly (do NOT store full user-user matrix)
        sims = cosine_similarity(user_row, self.R)[0]
        sims[u_idx] = 0.0  # ignore self

        # Top neighbors
        neighbor_indices = np.argsort(sims)[::-1][:neighbor_k]
        neighbor_sims = sims[neighbor_indices]

        candidate_scores = defaultdict(float)
        sim_sums = defaultdict(float)

        for n_idx, sim in zip(neighbor_indices, neighbor_sims):
            if sim <= 0:
                continue
            neighbor_row = self.R[n_idx].toarray()[0]
            rated_indices = np.where(neighbor_row > 0)[0]

            for m_idx in rated_indices:
                if m_idx in seen_movie_indices:
                    continue
                rating = neighbor_row[m_idx]
                candidate_scores[m_idx] += sim * rating
                sim_sums[m_idx] += sim

        scored_movies = []
        for m_idx, score in candidate_scores.items():
            if sim_sums[m_idx] > 0:
                scored_movies.append((m_idx, score / sim_sums[m_idx]))

        scored_movies.sort(key=lambda x: x[1], reverse=True)
        top = scored_movies[:top_k]

        top_movie_ids = [self.idx_to_movie_id_cf[m_idx] for m_idx, _ in top]

        self._print_movie_list(top_movie_ids, header=f"Recommendations for existing user {user_id}")
        return top_movie_ids

    # -----------------------------
    # New user: Content-based
    # -----------------------------

    def recommend_for_new_user(self, favorite_titles, top_k: int = 10):
        """
        Content-based recommendation for a new user with no ratings.
        Input is a list of raw favorite movie titles (as strings).

        Steps:
          1. Normalize the titles
          2. Find matching movie_ids in integrated catalog
          3. Average their feature vectors to make a 'user profile'
          4. Rank all movies by cosine similarity to that profile
          5. Return top_k recommendations, excluding the favorites
        """
        if self.X_movie is None:
            raise RuntimeError("Model not fit yet. Call .fit() first.")

        if not favorite_titles:
            print("[Recommender] No favorite titles provided.")
            return []

        # Map normalized title -> list of movie_ids
        title_map = defaultdict(list)
        for _, row in self.movies_df.iterrows():
            t = row.get("title_norm", None)
            mid = row["movie_id"]
            if pd.isna(t):
                continue
            title_map[str(t)].append(mid)

        # Map favorites to movie_ids
        fav_ids = []
        for raw_title in favorite_titles:
            norm = normalize_title(raw_title)
            if norm in title_map:
                # Just pick the first match for now
                fav_ids.append(title_map[norm][0])
            else:
                print(f"[Recommender] Could not find a match for title '{raw_title}'")

        if not fav_ids:
            print("[Recommender] No favorite titles matched any movies.")
            return []

        # Build user profile as average of favorite movie feature vectors
        fav_vecs = []
        for mid in fav_ids:
            if mid in self.movie_id_to_idx_features:
                idx = self.movie_id_to_idx_features[mid]
                fav_vecs.append(self.X_movie[idx])

        if not fav_vecs:
            print("[Recommender] No feature vectors found for favorite titles.")
            return []

        user_profile = np.mean(np.stack(fav_vecs, axis=0), axis=0, keepdims=True)

        # Compute similarity to all movies
        sims = cosine_similarity(user_profile, self.X_movie)[0]

        # Exclude favorites from recommendations
        excluded_indices = {self.movie_id_to_idx_features[mid]
                            for mid in fav_ids
                            if mid in self.movie_id_to_idx_features}

        ranked_indices = np.argsort(sims)[::-1]
        rec_ids = []
        for idx in ranked_indices:
            if idx in excluded_indices:
                continue
            mid = self.idx_to_movie_id_features[idx]
            rec_ids.append(mid)
            if len(rec_ids) == top_k:
                break

        self._print_movie_list(rec_ids, header="Recommendations for new user (content-based)")
        return rec_ids

    # -----------------------------
    # Helper for pretty-printing
    # -----------------------------

    def _print_movie_list(self, movie_ids, header="Recommendations"):
        print(f"\n[{header}]")
        im = self.movies_df.set_index("movie_id")
        for rank, mid in enumerate(movie_ids, start=1):
            if mid not in im.index:
                continue
            row = im.loc[mid]
            title_norm = row.get("title_norm", "")
            title = (title_norm or "").title()
            year = row.get("year", None)
            year_str = "N/A" if pd.isna(year) else str(int(year))
            genres = row.get("genres_norm", "")
            genres_str = genres.replace("|", ", ") if isinstance(genres, str) else ""
            print(f"{rank}. {title} ({year_str}) | Genres: {genres_str} | movie_id={mid}")
