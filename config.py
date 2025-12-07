# config.py
import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent

DATA_RAW = BASE_DIR / "Raw Data"   # <-- added "Raw Data"
DATA_PROC = BASE_DIR / "data_processed"
DATA_RAW.mkdir(exist_ok=True, parents=True)
DATA_PROC.mkdir(exist_ok=True, parents=True)

IMDB_BASICS  = DATA_RAW / "basics_imdb.csv"
IMDB_RATINGS = DATA_RAW / "ratings_imdb.csv"

ML_MOVIES  = DATA_RAW / "movies_ml.csv"
ML_RATINGS = DATA_RAW / "ratings_ml.csv"

TMDB_MOVIES = DATA_RAW / "movies_tmdb.csv"

MEDIATED_PATH               = DATA_PROC / "mediated_movies_all_sources.csv"
INTEGRATED_MOVIES_PATH      = DATA_PROC / "integrated_movies.csv"
ML_RATINGS_INTEGRATED_PATH  = DATA_PROC / "movielens_ratings_integrated.csv"

load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
