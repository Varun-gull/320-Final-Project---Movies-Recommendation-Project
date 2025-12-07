# integration.py
import pandas as pd
import numpy as np

from config import (
    DATA_PROC,
    MEDIATED_PATH,
    INTEGRATED_MOVIES_PATH,
    ML_RATINGS_INTEGRATED_PATH
)
from data_load import load_movielens_raw

def integrate_movies(mediated: pd.DataFrame) -> pd.DataFrame:
    """
    Integrate movies by grouping on (title_norm, year).
    Simple deterministic integration for now.
    """
    grouped = mediated.groupby(["title_norm", "year"], dropna=False)

    integrated_rows = []
    integrated_id = 1

    for (title_norm, year), group in grouped:
        # collect source ids
        imdb_ids = group[group["source"] == "imdb"]["source_id"].tolist()
        ml_ids = group[group["source"] == "movielens"]["source_id"].tolist()
        tmdb_ids = group[group["source"] == "tmdb"]["source_id"].tolist()

        # genres union
        all_genres = []
        for g in group["genres_norm"].dropna():
            if not g:
                continue
            all_genres.extend(g.split("|"))
        genres_norm = "|".join(sorted(set(all_genres))) if all_genres else ""

        # weighted rating
        ratings = group["rating_value"].values
        counts = group["rating_count"].fillna(0).values.astype(float)

        if np.sum(counts) > 0:
            weighted_rating = np.nansum(ratings * counts) / np.sum(counts)
            total_count = int(np.sum(counts))
        else:
            weighted_rating = np.nan
            total_count = int(np.nansum(counts))

        popularity = group["popularity"].max()

        integrated_rows.append({
            "movie_id": integrated_id,
            "title_norm": title_norm,
            "year": year,
            "genres_norm": genres_norm,
            "rating_value": weighted_rating,
            "rating_count": total_count,
            "popularity": popularity,
            "imdb_ids": "|".join(map(str, imdb_ids)),
            "movielens_ids": "|".join(map(str, ml_ids)),
            "tmdb_ids": "|".join(map(str, tmdb_ids))
        })
        integrated_id += 1

    integrated_df = pd.DataFrame(integrated_rows)
    integrated_df.to_csv(INTEGRATED_MOVIES_PATH, index=False)
    print(f"[Integrated] Saved {len(integrated_df)} unique movies -> {INTEGRATED_MOVIES_PATH}")
    return integrated_df


def map_movielens_ratings_to_integrated(integrated: pd.DataFrame) -> pd.DataFrame:
    """
    Join MovieLens ratings with integrated movie_ids.
    """
    ratings_raw_path = DATA_PROC / "movielens_ratings_raw.csv"
    if not ratings_raw_path.exists():
        _, ratings = load_movielens_raw()
        ratings.to_csv(ratings_raw_path, index=False)
    ratings = pd.read_csv(ratings_raw_path)

    # MovieLens movieId -> integrated movie_id
    ml_to_integrated = {}
    for _, row in integrated.iterrows():
        ml_ids_str = row["movielens_ids"]
        if pd.isna(ml_ids_str) or ml_ids_str == "":
            continue
        for mid in ml_ids_str.split("|"):
            if not mid:
                continue
            ml_to_integrated[int(mid)] = row["movie_id"]

    mapped = ratings[ratings["movieId"].isin(ml_to_integrated.keys())].copy()
    mapped["movie_id"] = mapped["movieId"].map(ml_to_integrated)

    mapped.to_csv(ML_RATINGS_INTEGRATED_PATH, index=False)
    print(f"[Ratings] Mapped {len(mapped)} ratings to integrated movie IDs -> {ML_RATINGS_INTEGRATED_PATH}")
    return mapped
