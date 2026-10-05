"""
Step 6 – Estrazione caratteristiche strutturali e collaborative (MovieLens 10M).

Per ogni coppia (utente, film) costruisce incrementalmente la memoria storica
degli eventi passati (timestamp < corrente) senza alcun data leakage temporale.

Caratteristiche estratte:
  1. Attività & Topologia Utente:
     - user_degree, user_n_reviews, user_mean_rating, user_n_pos, user_n_neg,
       user_frac_pos, user_frac_neg, user_days_since_last, log_user_degree
  2. Popolarità & Topologia Film:
     - movie_degree, movie_n_reviews, movie_mean_rating, movie_n_pos, movie_n_neg,
       movie_frac_pos, movie_frac_neg, movie_days_since_last, log_movie_degree
  3. Feature di Coppia & Topologiche:
     - preferential_attachment (deg_u * deg_m)
     - already_connected (1 se l'utente ha già valutato il film in passato)
  4. Affinità Contenutistica & Collaborativa (cruciale per Link Prediction):
     - genre_match_score: sovrapposizione tra generi del film e gusti storici dell'utente
     - svd_similarity: allineamento dei fattori latenti (TruncatedSVD sulla matrice d'interazione)
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

from config import (
    DATA_PROCESSED,
    CSV_PROCESSED,
    EVENTS_CSV,
    TRAIN_CSV,
    VAL_CSV,
    TEST_CSV,
    TRAIN_FILE,
    VAL_FILE,
    TEST_FILE,
)

# ── Feature generate ────────────────────────────────────────────────────
STRUCTURAL_FEATURES = [
    # Utente
    "user_degree",
    "user_n_reviews",
    "user_mean_rating",
    "user_n_pos",
    "user_n_neg",
    "user_frac_pos",
    "user_frac_neg",
    "user_days_since_last",
    "log_user_degree",
    # Film
    "movie_degree",
    "movie_n_reviews",
    "movie_mean_rating",
    "movie_n_pos",
    "movie_n_neg",
    "movie_frac_pos",
    "movie_frac_neg",
    "movie_days_since_last",
    "log_movie_degree",
    # Coppia & Topologia
    "preferential_attachment",
    "already_connected",
    # Affinità avanzata
    "genre_match_score",
    "svd_similarity",
]


class FastUserStats:
    """Contatori incrementali ad alte prestazioni per un utente."""
    __slots__ = ("n_reviews", "rating_sum", "n_pos", "n_neg", "last_ts", "rated_movies", "genres")

    def __init__(self):
        self.n_reviews = 0
        self.rating_sum = 0.0
        self.n_pos = 0
        self.n_neg = 0
        self.last_ts = None
        self.rated_movies = set()
        self.genres = set()

    def update(self, movie_id: int, rating: float, label: str, ts, genres_str: str):
        self.n_reviews += 1
        self.rating_sum += rating
        if label == "positiva":
            self.n_pos += 1
        elif label == "negativa":
            self.n_neg += 1
        if self.last_ts is None or ts > self.last_ts:
            self.last_ts = ts
        self.rated_movies.add(movie_id)
        if genres_str and genres_str != "":
            for g in genres_str.split("|"):
                if g:
                    self.genres.add(g)


class FastMovieStats:
    """Contatori incrementali ad alte prestazioni per un film."""
    __slots__ = ("n_reviews", "rating_sum", "n_pos", "n_neg", "last_ts", "rating_users")

    def __init__(self):
        self.n_reviews = 0
        self.rating_sum = 0.0
        self.n_pos = 0
        self.n_neg = 0
        self.last_ts = None
        self.rating_users = set()

    def update(self, user_id: int, rating: float, label: str, ts):
        self.n_reviews += 1
        self.rating_sum += rating
        if label == "positiva":
            self.n_pos += 1
        elif label == "negativa":
            self.n_neg += 1
        if self.last_ts is None or ts > self.last_ts:
            self.last_ts = ts
        self.rating_users.add(user_id)


def train_svd_embeddings(train_df: pd.DataFrame, n_components: int = 16):
    """
    Allena TruncatedSVD sul training set per ottenere vettori latenti utente e film.
    Ritorna due dizionari: user_vecs {user_id: vector} e movie_vecs {movie_id: vector}.
    """
    print(f"  Calcolo embeddings collaborativi SVD ({n_components} componenti su train)...")
    unique_users = train_df["user_id"].unique()
    unique_movies = train_df["movie_id"].unique()

    u_map = {uid: i for i, uid in enumerate(unique_users)}
    m_map = {mid: i for i, mid in enumerate(unique_movies)}

    row_indices = train_df["user_id"].map(u_map).values
    col_indices = train_df["movie_id"].map(m_map).values
    data_vals = np.ones(len(train_df), dtype=np.float32)

    adj = csr_matrix((data_vals, (row_indices, col_indices)), shape=(len(unique_users), len(unique_movies)))

    svd = TruncatedSVD(n_components=min(n_components, len(unique_movies) - 1, len(unique_users) - 1), random_state=42)
    user_factors = svd.fit_transform(adj)
    movie_factors = svd.components_.T

    # Normalizzazione L2 per cos-similarity immediata
    u_norms = np.linalg.norm(user_factors, axis=1, keepdims=True)
    u_norms[u_norms == 0] = 1.0
    user_factors = user_factors / u_norms

    m_norms = np.linalg.norm(movie_factors, axis=1, keepdims=True)
    m_norms[m_norms == 0] = 1.0
    movie_factors = movie_factors / m_norms

    user_vecs = {uid: user_factors[i] for uid, i in u_map.items()}
    movie_vecs = {mid: movie_factors[i] for mid, i in m_map.items()}
    return user_vecs, movie_vecs


def extract_features_for_split(
    split_df: pd.DataFrame,
    events_df: pd.DataFrame,
    user_vecs: dict,
    movie_vecs: dict,
) -> pd.DataFrame:
    """
    Estrae le feature strutturali e collaborative per ogni riga dello split,
    aggiornando lo stato storico con gli eventi passati (ts < ts_corrente).
    """
    events_df = events_df.copy()
    split_df = split_df.copy()

    events_df["timestamp"] = pd.to_datetime(events_df["timestamp"])
    split_df["timestamp"] = pd.to_datetime(split_df["timestamp"])

    events_df = events_df.sort_values("timestamp").reset_index(drop=True)
    split_df = split_df.sort_values("timestamp").reset_index(drop=True)

    ev_user = events_df["user_id"].values
    ev_movie = events_df["movie_id"].values
    ev_rating = events_df["rating"].values
    ev_label = events_df["label"].values
    ev_ts = events_df["timestamp"].values
    ev_genres = events_df["genres"].values if "genres" in events_df.columns else [""] * len(events_df)

    split_user = split_df["user_id"].values
    split_movie = split_df["movie_id"].values
    split_ts = split_df["timestamp"].values
    split_genres = split_df["genres"].values if "genres" in split_df.columns else [""] * len(split_df)

    user_stats = {}
    movie_stats = {}
    n_events = len(events_df)
    n_split = len(split_df)
    events_idx = 0

    records = []

    for i in range(n_split):
        curr_ts = split_ts[i]

        # Ingestione eventi storici strettamente antecedenti
        while events_idx < n_events:
            t = ev_ts[events_idx]
            if t >= curr_ts:
                break
            u = ev_user[events_idx]
            m = ev_movie[events_idx]

            if u not in user_stats:
                user_stats[u] = FastUserStats()
            user_stats[u].update(m, ev_rating[events_idx], ev_label[events_idx], t, ev_genres[events_idx])

            if m not in movie_stats:
                movie_stats[m] = FastMovieStats()
            movie_stats[m].update(u, ev_rating[events_idx], ev_label[events_idx], t)

            events_idx += 1

        u_id = split_user[i]
        m_id = split_movie[i]
        m_genre_str = split_genres[i]

        # ── 1. Feature Utente ──
        if u_id in user_stats:
            us = user_stats[u_id]
            u_deg = len(us.rated_movies)
            u_rev = us.n_reviews
            u_mean = round(us.rating_sum / u_rev, 4)
            u_pos = us.n_pos
            u_neg = us.n_neg
            u_frac_pos = round(u_pos / u_rev, 4)
            u_frac_neg = round(u_neg / u_rev, 4)
            u_days = round((pd.Timestamp(curr_ts) - pd.Timestamp(us.last_ts)).total_seconds() / 86400.0, 2)
            u_already = 1 if m_id in us.rated_movies else 0
            u_genres = us.genres
        else:
            u_deg, u_rev, u_mean, u_pos, u_neg = 0, 0, np.nan, 0, 0
            u_frac_pos, u_frac_neg, u_days, u_already = np.nan, np.nan, np.nan, 0
            u_genres = set()

        # ── 2. Feature Film ──
        if m_id in movie_stats:
            ms = movie_stats[m_id]
            m_deg = len(ms.rating_users)
            m_rev = ms.n_reviews
            m_mean = round(ms.rating_sum / m_rev, 4)
            m_pos = ms.n_pos
            m_neg = ms.n_neg
            m_frac_pos = round(m_pos / m_rev, 4)
            m_frac_neg = round(m_neg / m_rev, 4)
            m_days = round((pd.Timestamp(curr_ts) - pd.Timestamp(ms.last_ts)).total_seconds() / 86400.0, 2)
        else:
            m_deg, m_rev, m_mean, m_pos, m_neg = 0, 0, np.nan, 0, 0
            m_frac_pos, m_frac_neg, m_days = np.nan, np.nan, np.nan

        # ── 3. Feature di Coppia & Topologia ──
        pref_att = float(u_deg * m_deg)
        log_u_deg = float(np.log1p(u_deg))
        log_m_deg = float(np.log1p(m_deg))

        # ── 4. Affinità di Genere (Jaccard-like overlap) ──
        if m_genre_str and isinstance(m_genre_str, str) and u_genres:
            movie_genre_set = set(g for g in m_genre_str.split("|") if g)
            if movie_genre_set:
                overlap = len(u_genres.intersection(movie_genre_set))
                genre_score = round(overlap / len(movie_genre_set), 4)
            else:
                genre_score = 0.0
        else:
            genre_score = 0.0

        # ── 5. Similarità SVD (Collaborative Filtering) ──
        if u_id in user_vecs and m_id in movie_vecs:
            svd_sim = round(float(np.dot(user_vecs[u_id], movie_vecs[m_id])), 4)
        else:
            svd_sim = 0.0

        records.append({
            "user_degree": u_deg,
            "user_n_reviews": u_rev,
            "user_mean_rating": u_mean,
            "user_n_pos": u_pos,
            "user_n_neg": u_neg,
            "user_frac_pos": u_frac_pos,
            "user_frac_neg": u_frac_neg,
            "user_days_since_last": u_days,
            "log_user_degree": log_u_deg,
            "movie_degree": m_deg,
            "movie_n_reviews": m_rev,
            "movie_mean_rating": m_mean,
            "movie_n_pos": m_pos,
            "movie_n_neg": m_neg,
            "movie_frac_pos": m_frac_pos,
            "movie_frac_neg": m_frac_neg,
            "movie_days_since_last": m_days,
            "log_movie_degree": log_m_deg,
            "preferential_attachment": pref_att,
            "already_connected": u_already,
            "genre_match_score": genre_score,
            "svd_similarity": svd_sim,
        })

        if (i + 1) % 25_000 == 0 or (i + 1) == n_split:
            print(f"    ... {i + 1:,}/{n_split:,} righe elaborate")

    # Rimuovi eventuali colonne preesistenti per evitare duplicazioni
    cols_to_drop = [c for c in STRUCTURAL_FEATURES if c in split_df.columns]
    if cols_to_drop:
        split_df = split_df.drop(columns=cols_to_drop)

    features_df = pd.DataFrame(records, index=split_df.index)
    return pd.concat([split_df, features_df], axis=1)


def save_enriched_splits(splits: dict) -> None:
    """Salva i tre set arricchiti in CSV e Parquet."""
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    CSV_PROCESSED.mkdir(parents=True, exist_ok=True)
    file_map = {
        "train": (TRAIN_FILE, TRAIN_CSV),
        "val":   (VAL_FILE, VAL_CSV),
        "test":  (TEST_FILE, TEST_CSV),
    }
    for name, split_df in splits.items():
        pq, csv = file_map[name]
        split_df.to_parquet(pq, index=False)
        split_df.to_csv(csv, index=False)
        print(f"  {name:>5s} arricchito salvato: {csv}")


def run(
    train_path: Path = TRAIN_CSV,
    val_path: Path = VAL_CSV,
    test_path: Path = TEST_CSV,
    events_path: Path = EVENTS_CSV,
):
    """Esegue lo Step 6 completo per train, validation e test."""
    print("=" * 60)
    print("STEP 6 – Estrazione Caratteristiche Strutturali e Collaborative")
    print("=" * 60)

    splits = {
        "train": pd.read_csv(train_path, parse_dates=["timestamp"]),
        "val":   pd.read_csv(val_path, parse_dates=["timestamp"]),
        "test":  pd.read_csv(test_path, parse_dates=["timestamp"]),
    }
    events_df = pd.read_csv(events_path, parse_dates=["timestamp"])

    # Allena embeddings SVD sui dati di training per evitare data leakage
    user_vecs, movie_vecs = train_svd_embeddings(splits["train"])

    enriched = {}
    for name in ["train", "val", "test"]:
        print(f"\n  Elaborazione set {name.upper()} ({len(splits[name]):,} righe)...")
        enriched[name] = extract_features_for_split(splits[name], events_df, user_vecs, movie_vecs)

    save_enriched_splits(enriched)
    print("\nStep 6 completato con successo.\n")
    return enriched


if __name__ == "__main__":
    enriched = run()
