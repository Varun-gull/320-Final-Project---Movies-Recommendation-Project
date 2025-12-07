# mediated.py

import numpy as np
import pandas as pd

from config import DATA_PROC, MEDIATED_PATH
from data_load import load_imdb_raw, load_movielens_raw, load_tmdb_raw
from utils import (
    normalize_title,
    normalize_genres,
    scale_rating_to_10,
    extract_year_from_title,
    extract_year_from_date,
)


# ---------------------------------------------------------
# IMDb → Mediated
# ---------------------------------------------------------

def map_imdb_to_mediated(df_imdb: pd.DataFrame) -> pd.DataFrame:
    """Transform IMDb CSV into mediated schema."""
    rows = []
    for _, row in df_imdb.iterrows():
        # startYear may be missing or non-numeric
        try:
            year = int(row["startYear"])
        except Exception:
            year = np.nan

        rows.append(
            {
                "movie_temp_id": f"imdb:{row['tconst']}",
                "source": "imdb",
                "source_id": row["tconst"],
                "title_norm": normalize_title(row["primaryTitle"]),
                "year": year,
                "genres_norm": normalize_genres(row["genres"], "imdb"),
                "rating_value": scale_rating_to_10(row["averageRating"], "imdb"),
                "rating_count": row["numVotes"],
                "popularity": np.nan,
                "budget": np.nan,
                "revenue": np.nan,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------
# MovieLens → Mediated
# ---------------------------------------------------------

def map_movielens_to_mediated(df_ml_movies: pd.DataFrame) -> pd.DataFrame:
    """Transform MovieLens movies CSV into mediated schema."""
    rows = []
    for _, row in df_ml_movies.iterrows():
        year = extract_year_from_title(row["title"])

        rows.append(
            {
                "movie_temp_id": f"ml:{row['movieId']}",
                "source": "movielens",
                "source_id": row["movieId"],
                "title_norm": normalize_title(row["title"]),
                "year": year,
                "genres_norm": normalize_genres(row["genres"], "movielens"),
                "rating_value": scale_rating_to_10(row["rating_mean"], "movielens"),
                "rating_count": row["rating_count"],
                "popularity": np.nan,
                "budget": np.nan,
                "revenue": np.nan,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------
# TMDb → Mediated (robust to different column names)
# ---------------------------------------------------------

def map_tmdb_to_mediated(df_tmdb: pd.DataFrame) -> pd.DataFrame:
    """
    Transform TMDb CSV into mediated schema.

    Handles:
    - genres stored in 'genre_names' OR 'genres'
    - id column named 'id' or 'tmdbId'
    - release year from 'release_date' or 'year' if present
    - missing vote_average / vote_count / popularity / budget / revenue
    """
    # If empty, return an empty-but-correct-shaped DataFrame
    if df_tmdb is None or df_tmdb.empty:
        return pd.DataFrame(
            columns=[
                "movie_temp_id",
                "source",
                "source_id",
                "title_norm",
                "year",
                "genres_norm",
                "rating_value",
                "rating_count",
                "popularity",
                "budget",
                "revenue",
            ]
        )

    # Decide which column holds genres
    if "genre_names" in df_tmdb.columns:
        def get_genres(row):
            return row["genre_names"]
    elif "genres" in df_tmdb.columns:
        def get_genres(row):
            return row["genres"]
    else:
        def get_genres(row):
            return ""

    # Decide ID column
    if "id" in df_tmdb.columns:
        id_col = "id"
    elif "tmdbId" in df_tmdb.columns:
        id_col = "tmdbId"
    else:
        # Fallback: first column
        id_col = df_tmdb.columns[0]

    # Decide how to get year
    if "release_date" in df_tmdb.columns:
        def get_year(row):
            return extract_year_from_date(row["release_date"])
    elif "year" in df_tmdb.columns:
        def get_year(row):
            try:
                return int(row["year"])
            except Exception:
                return np.nan
    else:
        def get_year(row):
            return np.nan

    rows = []
    for _, row in df_tmdb.iterrows():
        year = get_year(row)

        # Safe gets with defaults
        vote_avg = row["vote_average"] if "vote_average" in df_tmdb.columns else np.nan
        vote_cnt = row["vote_count"] if "vote_count" in df_tmdb.columns else np.nan
        popularity = row["popularity"] if "popularity" in df_tmdb.columns else np.nan
        budget = row["budget"] if "budget" in df_tmdb.columns else np.nan
        revenue = row["revenue"] if "revenue" in df_tmdb.columns else np.nan

        rows.append(
            {
                "movie_temp_id": f"tmdb:{row[id_col]}",
                "source": "tmdb",
                "source_id": row[id_col],
                "title_norm": normalize_title(row["title"]),
                "year": year,
                "genres_norm": normalize_genres(get_genres(row), "tmdb"),
                "rating_value": scale_rating_to_10(vote_avg, "tmdb"),
                "rating_count": vote_cnt,
                "popularity": popularity,
                "budget": budget,
                "revenue": revenue,
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------
# Build mediated table from all sources
# ---------------------------------------------------------

def build_mediated_table() -> pd.DataFrame:
    """
    Build mediated table from IMDb, MovieLens, and TMDb
    and save to MEDIATED_PATH.
    """
    imdb_raw_path = DATA_PROC / "imdb_movies_raw.csv"
    ml_movies_raw_path = DATA_PROC / "movielens_movies_raw.csv"

    # IMDb
    if imdb_raw_path.exists():
        df_imdb = pd.read_csv(imdb_raw_path)
        print(f"[Mediated] Loaded IMDb raw from {imdb_raw_path}")
    else:
        df_imdb = load_imdb_raw()

    # MovieLens
    if ml_movies_raw_path.exists():
        df_ml_movies = pd.read_csv(ml_movies_raw_path)
        print(f"[Mediated] Loaded MovieLens raw from {ml_movies_raw_path}")
    else:
        df_ml_movies, _ = load_movielens_raw()

    # TMDb (your CSV)
    try:
        df_tmdb = load_tmdb_raw()
    except FileNotFoundError:
        print("[Mediated] TMDb CSV not found – proceeding without TMDb.")
        df_tmdb = pd.DataFrame()

    # Map each source into mediated schema
    imdb_m = map_imdb_to_mediated(df_imdb)
    ml_m = map_movielens_to_mediated(df_ml_movies)
    tmdb_m = map_tmdb_to_mediated(df_tmdb)

    mediated = pd.concat([imdb_m, ml_m, tmdb_m], ignore_index=True)
    mediated.to_csv(MEDIATED_PATH, index=False)
    print(f"[Mediated] Saved {len(mediated)} rows -> {MEDIATED_PATH}")
    return mediated
