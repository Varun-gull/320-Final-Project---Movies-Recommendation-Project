# utils.py
import re
import numpy as np
import pandas as pd

def normalize_title(title: str) -> str:
    """Normalize titles across sources (lowercase, strip year, remove punctuation)."""
    if pd.isna(title):
        return None
    title = str(title)

    # Remove year "(YYYY)" at end
    title = re.sub(r"\s*\(\d{4}\)$", "", title)

    # Lowercase
    title = title.lower()

    # Remove non-alphanumeric except spaces
    title = re.sub(r"[^a-z0-9\s]", "", title)

    # Collapse whitespace
    title = " ".join(title.split())

    # Remove leading articles optionally
    title = re.sub(r"^(the|a|an)\s+", "", title)

    return title


def normalize_genres(genres, source: str) -> str:
    """Normalize genre field into pipe-separated lowercase names."""
    if pd.isna(genres):
        return ""

    if source == "imdb":
        parts = [g.strip().lower() for g in str(genres).split(",")]
    elif source == "movielens":
        parts = [g.strip().lower() for g in str(genres).split("|")]
    elif source == "tmdb":
        if isinstance(genres, list):
            parts = [g.strip().lower() for g in genres]
        else:
            g_str = str(genres).strip("[]")
            parts = [g.strip(" '\"").lower() for g in g_str.split(",") if g.strip()]
    else:
        parts = []

    mapping = {
        "sci-fi": "science fiction",
        "scifi": "science fiction"
    }

    norm = [mapping.get(g, g) for g in parts]
    if not norm:
        return ""
    return "|".join(sorted(set(norm)))


def scale_rating_to_10(rating, source: str):
    """Scale MovieLens ratings from 0.5–5 to 0–10; IMDb/TMDb already 0–10."""
    if pd.isna(rating):
        return np.nan
    r = float(rating)
    if source == "movielens":
        return r * 2.0
    return r


def extract_year_from_title(title: str):
    """Extract year from MovieLens titles like 'Toy Story (1995)'."""
    if pd.isna(title):
        return np.nan
    m = re.search(r"\((\d{4})\)", str(title))
    if m:
        return int(m.group(1))
    return np.nan


def extract_year_from_date(date_str: str):
    """Extract year from 'YYYY-MM-DD' or 'YYYY'."""
    if pd.isna(date_str):
        return np.nan
    s = str(date_str)
    if len(s) >= 4 and s[:4].isdigit():
        return int(s[:4])
    return np.nan
