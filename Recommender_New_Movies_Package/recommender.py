#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

def load_data(path="movies.csv"):
    df = pd.read_csv(path)
    required = {"user_id", "movie_title", "rating"}
    if not required.issubset(df.columns):
        raise ValueError("CSV must contain: user_id, movie_title, rating")
    return df

def build_user_item(df):
    return df.pivot_table(index="user_id", columns="movie_title", values="rating")

def user_user_recommend(user_item, target_user_id, top_k=5):
    if target_user_id not in user_item.index:
        raise ValueError(f"user_id {target_user_id} not found")

    filled = user_item.fillna(0).to_numpy()
    sim = cosine_similarity(filled)
    user_index = list(user_item.index).index(target_user_id)

    sims = sim[user_index]
    sims[user_index] = -np.inf

    ratings = user_item.to_numpy()
    ratings_filled = np.nan_to_num(ratings, nan=0)

    w = sims.reshape(-1, 1)
    preds = (w * ratings_filled).sum(axis=0) / (np.abs(w).sum(axis=0) + 1e-8)

    already_rated = ~np.isnan(ratings[user_index])
    preds[already_rated] = -np.inf

    movie_titles = list(user_item.columns)
    top_idx = np.argsort(preds)[-top_k:][::-1]

    return [(movie_titles[i], preds[i]) for i in top_idx if preds[i] != -np.inf]

def content_based(df, target_user_id, top_k=5):
    if "genre" not in df.columns:
        return []

    df["genre_text"] = df["genre"].astype(str).str.replace("|", " ")
    tfidf = TfidfVectorizer()
    item_tfidf = tfidf.fit_transform(df["genre_text"])

    rows_u = df[df["user_id"] == target_user_id]
    if rows_u.empty:
        return []

    thr = rows_u["rating"].mean()
    liked = rows_u[rows_u["rating"] >= thr]
    if liked.empty:
        liked = rows_u

    user_profile = item_tfidf[liked.index.tolist()].mean(axis=0)
    U = normalize(user_profile)
    I = normalize(item_tfidf)
    sims = (I @ U.T).toarray().ravel()

    rated_titles = set(rows_u["movie_title"])
    candidates = {title: s for title, s in zip(df["movie_title"], sims) if title not in rated_titles}
    return sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:top_k]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="movies.csv")
    ap.add_argument("--user-id", type=int, default=1)
    ap.add_argument("--top-k", type=int, default=3)
    args = ap.parse_args()

    df = load_data(args.csv)
    user_item = build_user_item(df)

    print("\nCollaborative Filtering Recommendations:")
    for m, s in user_user_recommend(user_item, args.user_id, args.top_k):
        print(f"- {m} (score={s:.3f})")

    print("\nContent-Based Recommendations:")
    for m, s in content_based(df, args.user_id, args.top_k):
        print(f"- {m} (similarity={s:.3f})")

if __name__ == "__main__":
    main()
