import pandas as pd

from config import (
    IMDB_BASICS, IMDB_RATINGS,
    ML_MOVIES, ML_RATINGS,
    TMDB_MOVIES, DATA_PROC
)

print("IMDB_BASICS:", IMDB_BASICS, "exists?", IMDB_BASICS.exists())
print("IMDB_RATINGS:", IMDB_RATINGS, "exists?", IMDB_RATINGS.exists())

def load_imdb_raw():
    """Load IMDb basics + ratings from your CSV files."""
    if not IMDB_BASICS.exists() or not IMDB_RATINGS.exists():
        raise FileNotFoundError("IMDb CSVs not found in data_raw/.")

    basics = pd.read_csv(IMDB_BASICS)   # no sep="\t" now
    ratings = pd.read_csv(IMDB_RATINGS)

    # If you preserved original column names from TSV, this will still work:
    # basics should have: tconst, primaryTitle, startYear, genres
    # ratings should have: tconst, averageRating, numVotes
    df = basics.merge(ratings, on="tconst", how="left")

    df = df[["tconst", "primaryTitle", "startYear", "genres", "averageRating", "numVotes"]]
    out_path = DATA_PROC / "imdb_movies_raw.csv"
    df.to_csv(out_path, index=False)
    print(f"[IMDb] Saved {len(df)} movies -> {out_path}")
    return df


def load_movielens_raw():
    if not ML_MOVIES.exists() or not ML_RATINGS.exists():
        raise FileNotFoundError("MovieLens CSVs not found in data_raw/.")

    movies = pd.read_csv(ML_MOVIES)
    ratings = pd.read_csv(ML_RATINGS)

    movie_stats = ratings.groupby("movieId").agg(
        rating_mean=("rating", "mean"),
        rating_count=("rating", "count")
    ).reset_index()

    movies_agg = movies.merge(movie_stats, on="movieId", how="left")

    movies_agg.to_csv(DATA_PROC / "movielens_movies_raw.csv", index=False)
    ratings.to_csv(DATA_PROC / "movielens_ratings_raw.csv", index=False)

    print(f"[MovieLens] Saved {len(movies_agg)} movies and {len(ratings)} ratings")
    return movies_agg, ratings



def load_tmdb_raw():
    if not TMDB_MOVIES.exists():
        raise FileNotFoundError("TMDb CSV movies_tmdb.csv not found in data_raw/.")
    df = pd.read_csv(TMDB_MOVIES)
    print(f"[TMDb] Loaded {len(df)} rows from {TMDB_MOVIES}")
    return df
