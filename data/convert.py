import csv
import os

print("Running in:", os.getcwd())

# Paths
basics_path = "data/title.basics.tsv"
ratings_path = "data/title.ratings.tsv"

basics_out = "data/basics_clean.csv"
ratings_out = "data/ratings_clean.csv"

# How many movies to keep (tune this if needed)
MAX_MOVIES = 1_000_000   # try 1_000_000 first; if still too big, go to 500_000, etc.

# ----------------------------------------------------
# Step 1: Create smaller basics_clean.csv
#   - only titleType == "movie"
#   - only first MAX_MOVIES movies
#   - store tconsts so ratings can be filtered to match
# ----------------------------------------------------

selected_tconsts = set()
movie_count = 0

with open(basics_path, "r", encoding="utf-8") as infile, \
     open(basics_out, "w", newline="", encoding="utf-8") as outfile:

    reader = csv.DictReader(infile, delimiter="\t")
    writer = csv.writer(outfile)

    # basics header we want
    writer.writerow(["tconst", "primaryTitle", "startYear", "genres"])

    for row in reader:
        # keep only movies
        if row.get("titleType") != "movie":
            continue

        tconst = row["tconst"]

        writer.writerow([
            tconst,
            row.get("primaryTitle", ""),
            row.get("startYear", ""),
            row.get("genres", ""),
        ])

        selected_tconsts.add(tconst)
        movie_count += 1

        if movie_count >= MAX_MOVIES:
            break

print(f"Created: {basics_out} with {movie_count} movies")

# ----------------------------------------------------
# Step 2: Create smaller ratings_clean.csv
#   - only ratings whose tconst is in selected_tconsts
# ----------------------------------------------------

rating_count = 0

with open(ratings_path, "r", encoding="utf-8") as infile, \
     open(ratings_out, "w", newline="", encoding="utf-8") as outfile:

    reader = csv.DictReader(infile, delimiter="\t")
    writer = csv.writer(outfile)

    writer.writerow(["tconst", "averageRating", "numVotes"])

    for row in reader:
        tconst = row["tconst"]
        if tconst not in selected_tconsts:
            continue

        writer.writerow([
            tconst,
            row.get("averageRating", ""),
            row.get("numVotes", ""),
        ])
        rating_count += 1

print(f"Created: {ratings_out} with {rating_count} ratings")