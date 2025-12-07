import csv
import os

print("Running in:", os.getcwd())

basics_path = "data/title.basics.tsv"
ratings_path = "data/title.ratings.tsv"

basics_out = "data/basics_clean.csv"
ratings_out = "data/ratings_clean.csv"


with open(basics_path, "r", encoding="utf-8") as infile, \
     open(basics_out, "w", newline="", encoding="utf-8") as outfile:

    reader = csv.DictReader(infile, delimiter="\t")
    writer = csv.writer(outfile)

    writer.writerow(["tconst", "primaryTitle", "startYear", "genres"])

    for row in reader:
        writer.writerow([
            row["tconst"],
            row["primaryTitle"],
            row["startYear"],
            row["genres"],
        ])

print("Created:", basics_out)

with open(ratings_path, "r", encoding="utf-8") as infile, \
     open(ratings_out, "w", newline="", encoding="utf-8") as outfile:

    reader = csv.DictReader(infile, delimiter="\t")
    writer = csv.writer(outfile)

    writer.writerow(["tconst", "averageRating", "numVotes"])

    for row in reader:
        writer.writerow([
            row["tconst"],
            row["averageRating"],
            row["numVotes"],
        ])

print("Created:", ratings_out)