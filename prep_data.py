# prep_data_until_features.py

import pandas as pd

from config import (
    MEDIATED_PATH,
    INTEGRATED_MOVIES_PATH,
    ML_RATINGS_INTEGRATED_PATH,
)
from mediated import build_mediated_table
from integration import integrate_movies, map_movielens_ratings_to_integrated


def main():
    # 1. Mediated table
    if MEDIATED_PATH.exists():
        mediated = pd.read_csv(MEDIATED_PATH)
        print(f"[Prep] Loaded mediated table ({len(mediated)} rows)")
    else:
        mediated = build_mediated_table()

    # 2. Integrated movies
    if INTEGRATED_MOVIES_PATH.exists():
        integrated = pd.read_csv(INTEGRATED_MOVIES_PATH)
        print(f"[Prep] Loaded integrated movies ({len(integrated)} rows)")
    else:
        integrated = integrate_movies(mediated)

    # 3. Integrated ratings
    if ML_RATINGS_INTEGRATED_PATH.exists():
        ratings_integrated = pd.read_csv(ML_RATINGS_INTEGRATED_PATH)
        print(f"[Prep] Loaded integrated ratings ({len(ratings_integrated)} rows)")
    else:
        ratings_integrated = map_movielens_ratings_to_integrated(integrated)

    print("\n[Prep] DONE. All CSVs are ready in data_processed/.")


if __name__ == "__main__":
    main()
