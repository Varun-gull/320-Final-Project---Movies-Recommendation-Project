# run_recommender_demo.py

from recommender import MovieRecommender


def main():
    rec = MovieRecommender()
    rec.fit()  # builds matrices from your CSVs

    while True:
        print("\n==== Movie Recommender Demo ====")
        print("1. Recommend for existing MovieLens userId")
        print("2. Recommend for new user (enter favorite titles)")
        print("3. Quit")
        choice = input("Choice: ").strip()

        if choice == "1":
            uid_str = input("Enter MovieLens userId (integer): ").strip()
            try:
                uid = int(uid_str)
            except ValueError:
                print("Invalid userId.")
                continue
            rec.recommend_for_existing_user(uid, top_k=10, neighbor_k=50)

        elif choice == "2":
            fav_str = input("Enter 3–5 favorite movie titles, comma-separated:\n")
            favorites = [t.strip() for t in fav_str.split(",") if t.strip()]
            rec.recommend_for_new_user(favorites, top_k=10)

        elif choice == "3":
            print("Exiting demo.")
            break
        else:
            print("Invalid choice.")


if __name__ == "__main__":
    main()
