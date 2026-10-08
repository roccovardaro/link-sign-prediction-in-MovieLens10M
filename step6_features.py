"""
Step 6 – Estrazione caratteristiche strutturali e collaborative (MovieLens 10M).

Modulo compatto e modulare per:
  1. Negative sampling e creazione dataset Link Prediction (train_link, val_link, test_link).
  2. Estrazione causale online delle 7 feature link essenziali e statistiche di rating.
  3. Calcolo dei fattori latenti SVD e preparazione dataset Sign Prediction (train, val, test).
  4. Salvataggio compatto dei dataset pronti per lo Stadio 1 e lo Stadio 2.
"""

import time
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

from config import (
    DATA_PROCESSED,
    CSV_PROCESSED,
    EVENTS_CSV,
    EVENTS_TABLE_FILE,
    TRAIN_CSV,
    VAL_CSV,
    TEST_CSV,
    TRAIN_FILE,
    VAL_FILE,
    TEST_FILE,
    TRAIN_LINK_CSV,
    VAL_LINK_CSV,
    TEST_LINK_CSV,
    TRAIN_LINK_FILE,
    VAL_LINK_FILE,
    TEST_LINK_FILE,
)

# ── 7 Feature essenziali e non ridondanti per la Link Prediction (Stadio 1) ──
LINK_FEATURE_COLS = [
    # Topologia & Popolarità
    "user_degree",           # Grado storico utente nel grafo bipartito
    "movie_degree",          # Popolarità storica del film
    # Recency (giorni dall'ultima interazione)
    "user_days_since_last",  # Recency utente (-1 se primo evento)
    "movie_days_since_last", # Recency film (-1 se primo evento)
    # Attività recente (finestra temporale a 30 giorni)
    "user_events_30d",       # Numero recensioni utente negli ultimi 30 giorni
    "movie_events_30d",      # Numero recensioni film negli ultimi 30 giorni
    # Affinità tematica
    "genre_match_score",     # Quota di generi del film già visti dall'utente
]

# ── Feature storiche di gradimento necessarie allo Stadio 2 (Sign Prediction) ─
EXTRA_SENTIMENT_COLS = [
    "user_mean_rating",      # Media storica voti utente
    "movie_mean_rating",     # Media storica voti film
    "user_frac_pos",         # Quota valutazioni positive (rating >= 4.0) utente
    "movie_frac_pos",        # Quota valutazioni positive film
]

ALL_FEATURE_COLS = LINK_FEATURE_COLS + EXTRA_SENTIMENT_COLS

GENRE_CACHE = {}


def get_genres_tuple(val) -> tuple:
    """Restituisce la tupla di generi con memorizzazione in cache per evitare riallocazioni."""
    if not isinstance(val, str) or not val or val == "(no genres listed)":
        return ()
    res = GENRE_CACHE.get(val)
    if res is None:
        res = tuple(g for g in val.split("|") if g and g != "(no genres listed)")
        GENRE_CACHE[val] = res
    return res


def compute_genre_match(user_genres: dict, movie_genres_tuple: tuple) -> float:
    """Calcola la percentuale di generi del film presenti nello storico dell'utente."""
    if not user_genres or not movie_genres_tuple:
        return 0.0
    return sum(1 for g in movie_genres_tuple if g in user_genres) / len(movie_genres_tuple)


class NodeState:
    """Traccia in tempo reale la topologia, la recency e le statistiche di un nodo (utente o film)."""
    __slots__ = ("neighbors", "last_ts", "recent_30d", "genres", "rating_sum", "pos_count", "count")

    def __init__(self):
        self.neighbors = set()
        self.last_ts = -1.0
        self.recent_30d = deque()
        self.genres = {}
        self.rating_sum = 0.0
        self.pos_count = 0
        self.count = 0

    def update(self, neighbor: int, t: float, r: float):
        """Aggiorna lo stato del nodo con una nuova interazione temporale."""
        self.neighbors.add(neighbor)
        self.last_ts = t
        self.recent_30d.append(t)
        self.rating_sum += r
        self.pos_count += int(r >= 4.0)
        self.count += 1

    def get_features(self, t: float, cutoff_30d: float):
        """Restituisce (degree, days_since_last, events_30d, mean_rating, frac_pos) in O(1)."""
        q = self.recent_30d
        while q and q[0] < cutoff_30d:
            q.popleft()

        days_since = (t - self.last_ts) / 86400.0 if self.last_ts >= 0 else -1.0
        mean_rating = (self.rating_sum / self.count) if self.count > 0 else np.nan
        frac_pos = (self.pos_count / self.count) if self.count > 0 else 0.0

        return len(self.neighbors), days_since, len(q), mean_rating, frac_pos


def load_dataset(parquet_path: Path, csv_path: Path) -> pd.DataFrame:
    """Carica velocemente da Parquet se presente, altrimenti da CSV."""
    df = pd.read_parquet(parquet_path) if parquet_path.exists() else pd.read_csv(csv_path)
    if "timestamp" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def extract_link_features(queries: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """
    Estrae le feature online per le query usando solo eventi con t_evento < t_query.
    Calcola le feature topologiche e di rating in un solo passaggio lineare O(N).
    """
    n_queries = len(queries)
    if n_queries == 0:
        empty_df = pd.DataFrame(columns=ALL_FEATURE_COLS, dtype=np.float32)
        return pd.concat([queries.drop(columns=ALL_FEATURE_COLS, errors="ignore"), empty_df], axis=1)

    # Eventi storici in array NumPy
    events_sorted = events.sort_values("timestamp", kind="stable")
    h_users = events_sorted["user_id"].to_numpy()
    h_movies = events_sorted["movie_id"].to_numpy()
    h_ts = pd.to_datetime(events_sorted["timestamp"]).astype("int64").to_numpy() / 1e9
    h_ratings = events_sorted["rating"].to_numpy(dtype=np.float32)
    h_genres = events_sorted["genres"].to_numpy() if "genres" in events_sorted.columns else None

    # Query in array NumPy
    q_users = queries["user_id"].to_numpy()
    q_movies = queries["movie_id"].to_numpy()
    q_ts = pd.to_datetime(queries["timestamp"]).astype("int64").to_numpy() / 1e9
    q_genres = queries["genres"].to_numpy() if "genres" in queries.columns else None
    order = np.argsort(q_ts, kind="stable")

    users, movies = {}, {}
    n_events, ev_idx = len(events_sorted), 0

    # Matrice pre-allocata: 7 link features + 4 sentiment = 11 colonne
    feat_mat = np.zeros((n_queries, 11), dtype=np.float32)
    feat_mat[:, 2] = -1.0    # user_days_since_last
    feat_mat[:, 3] = -1.0    # movie_days_since_last
    feat_mat[:, 7] = np.nan  # user_mean_rating
    feat_mat[:, 8] = np.nan  # movie_mean_rating

    log_step = max(500_000, n_queries // 10)

    for step, idx in enumerate(order, 1):
        t = q_ts[idx]
        uid, mid = q_users[idx], q_movies[idx]

        # Ingestione causale: t_evento < t_query
        while ev_idx < n_events and h_ts[ev_idx] < t:
            ev_u, ev_m, ev_t = h_users[ev_idx], h_movies[ev_idx], h_ts[ev_idx]
            r = float(h_ratings[ev_idx])

            u_node = users.get(ev_u)
            if u_node is None:
                u_node = NodeState()
                users[ev_u] = u_node
            u_node.update(ev_m, ev_t, r)

            m_node = movies.get(ev_m)
            if m_node is None:
                m_node = NodeState()
                movies[ev_m] = m_node
            m_node.update(ev_u, ev_t, r)

            if h_genres is not None:
                gs = get_genres_tuple(h_genres[ev_idx])
                if gs:
                    w = 1.0 / len(gs)
                    for g in gs:
                        u_node.genres[g] = u_node.genres.get(g, 0.0) + w

            ev_idx += 1

        cutoff_30 = t - 2592000.0  # 30 giorni in secondi

        # Feature utente: degree(0), days_since(2), events_30d(4), mean_rating(7), frac_pos(9)
        u = users.get(uid)
        if u is not None:
            feat_mat[idx, 0], feat_mat[idx, 2], feat_mat[idx, 4], feat_mat[idx, 7], feat_mat[idx, 9] = (
                u.get_features(t, cutoff_30)
            )

        # Feature film: degree(1), days_since(3), events_30d(5), mean_rating(8), frac_pos(10)
        m = movies.get(mid)
        if m is not None:
            feat_mat[idx, 1], feat_mat[idx, 3], feat_mat[idx, 5], feat_mat[idx, 8], feat_mat[idx, 10] = (
                m.get_features(t, cutoff_30)
            )

        # Feature affinità genere: genre_match_score(6)
        if u is not None and u.genres and q_genres is not None:
            feat_mat[idx, 6] = compute_genre_match(u.genres, get_genres_tuple(q_genres[idx]))

        if n_queries >= 200_000 and step % log_step == 0:
            print(f"      ... estratte {step:,}/{n_queries:,} feature ({step * 100 // n_queries}%)")

    feature_df = pd.DataFrame(feat_mat, index=queries.index, columns=ALL_FEATURE_COLS)
    return pd.concat([queries.drop(columns=ALL_FEATURE_COLS, errors="ignore"), feature_df], axis=1)


def sample_negative_pairs(df: pd.DataFrame, events_df: pd.DataFrame, ratio: float = 1.0, seed: int = 42) -> pd.DataFrame:
    """
    Campiona coppie negative (link_exists = 0) per il training set.
    Mantiene utente e timestamp del positivo reale, associando un film storico valido
    mai visto prima dall'utente.
    """
    n_neg = int(len(df) * ratio)
    if n_neg <= 0:
        return pd.DataFrame(columns=["user_id", "movie_id", "timestamp", "genres", "link_exists"])

    rng = np.random.default_rng(seed)
    events = events_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(events["timestamp"]):
        events["timestamp"] = pd.to_datetime(events["timestamp"])

    catalog = events.sort_values("timestamp").drop_duplicates("movie_id")
    first_seen = catalog["timestamp"].to_numpy()
    movie_ids = catalog["movie_id"].to_numpy()
    movie_genres = catalog["genres"].to_numpy() if "genres" in catalog else np.full(len(catalog), "")

    known = {}
    for source in (events, df):
        for uid, mid in zip(source["user_id"].to_numpy(), source["movie_id"].to_numpy()):
            known.setdefault(uid, set()).add(mid)

    df_ts = pd.to_datetime(df["timestamp"]).to_numpy()
    counts = np.searchsorted(first_seen, df_ts, side="left")

    first_available = {
        uid: next((j for j, mid in enumerate(movie_ids) if mid not in known.get(uid, set())), len(movie_ids))
        for uid in df["user_id"].unique()
    }

    eligible = np.flatnonzero(counts > df["user_id"].map(first_available).to_numpy())
    if len(eligible) == 0:
        raise ValueError("Nessun film storico disponibile per il negative sampling")

    records = []
    df_u = df["user_id"].to_numpy()
    for source_idx in rng.choice(eligible, size=n_neg, replace=True):
        uid = df_u[source_idx]
        ts = df_ts[source_idx]
        limit = counts[source_idx]
        blocked = known.get(uid, set())

        chosen = None
        for _ in range(64):
            candidate = int(rng.integers(limit))
            if movie_ids[candidate] not in blocked:
                chosen = candidate
                break
        if chosen is None:
            candidates = [j for j in range(limit) if movie_ids[j] not in blocked]
            if not candidates:
                raise ValueError(f"Nessun negativo disponibile per utente {uid} a {ts}")
            chosen = int(rng.choice(candidates))

        records.append((uid, movie_ids[chosen], ts, movie_genres[chosen], 0))

    return pd.DataFrame(records, columns=["user_id", "movie_id", "timestamp", "genres", "link_exists"])


def train_svd_embeddings(train_df: pd.DataFrame, n_components: int = 16):
    """Allena TruncatedSVD sul training set per ottenere fattori latenti normalizzati L2."""
    t0 = time.time()
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

    # Normalizzazione L2 vettoriale
    u_norms = np.linalg.norm(user_factors, axis=1, keepdims=True)
    user_factors /= np.where(u_norms == 0, 1.0, u_norms)

    m_norms = np.linalg.norm(movie_factors, axis=1, keepdims=True)
    movie_factors /= np.where(m_norms == 0, 1.0, m_norms)

    user_vecs = {uid: user_factors[i] for uid, i in u_map.items()}
    movie_vecs = {mid: movie_factors[i] for mid, i in m_map.items()}
    print(f"  --> SVD completata in {time.time() - t0:.1f}s")
    return user_vecs, movie_vecs


def compute_svd_similarity(df: pd.DataFrame, user_vecs: dict, movie_vecs: dict) -> np.ndarray:
    """Calcola la cosine similarity tra i fattori SVD di utente e film in modo diretto e scalabile."""
    n = len(df)
    if not user_vecs or not movie_vecs or n == 0:
        return np.zeros(n, dtype=np.float32)

    u_ids = df["user_id"].to_numpy()
    m_ids = df["movie_id"].to_numpy()
    sims = np.zeros(n, dtype=np.float32)

    for i in range(n):
        uv = user_vecs.get(u_ids[i])
        mv = movie_vecs.get(m_ids[i])
        if uv is not None and mv is not None:
            sims[i] = float(np.dot(uv, mv))

    return np.round(sims, 4)


def build_split_datasets(
    name: str,
    split_df: pd.DataFrame,
    events_df: pd.DataFrame,
    user_vecs: dict,
    movie_vecs: dict,
    ratio: float = 1.0,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Costruisce e arricchisce i dataset per Link Prediction e Sign Prediction per un singolo split.
    Riusa istantaneamente le prime N righe reali calcolate per Link Prediction senza ricalcoli.
    """
    t0 = time.time()
    n_pos = len(split_df)
    info = f"ratio={ratio} negativi" if ratio > 0 else "solo dati veri"
    print(f"  Elaborazione {name.upper()} ({n_pos:,} link reali, {info})...")

    # 1. Dataset Link Prediction
    positives = split_df.assign(link_exists=1)
    if ratio > 0:
        negatives = sample_negative_pairs(split_df, events_df, ratio=ratio, seed=seed)
        combined = pd.concat([positives, negatives], ignore_index=True)
    else:
        combined = positives

    link_df = extract_link_features(combined, events_df)

    # 2. Dataset Sign Prediction (riuso istantaneo delle sole coppie reali)
    sign_df = link_df.iloc[:n_pos].copy()
    sign_df["svd_similarity"] = compute_svd_similarity(sign_df, user_vecs, movie_vecs)

    print(f"  --> {name.upper()} completato in {time.time() - t0:.1f}s (Link: {len(link_df):,} | Sign: {len(sign_df):,})")
    return link_df, sign_df


def save_all_datasets(link_splits: dict, sign_splits: dict) -> None:
    """Salva i dataset per Link Prediction e Sign Prediction in formato Parquet e CSV."""
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    CSV_PROCESSED.mkdir(parents=True, exist_ok=True)

    file_maps = [
        ("Link Stage", link_splits, {
            "train": (TRAIN_LINK_FILE, TRAIN_LINK_CSV),
            "val":   (VAL_LINK_FILE, VAL_LINK_CSV),
            "test":  (TEST_LINK_FILE, TEST_LINK_CSV),
        }),
        ("Sign Stage", sign_splits, {
            "train": (TRAIN_FILE, TRAIN_CSV),
            "val":   (VAL_FILE, VAL_CSV),
            "test":  (TEST_FILE, TEST_CSV),
        }),
    ]

    for stage_name, splits_dict, paths_map in file_maps:
        for name, df in splits_dict.items():
            t0 = time.time()
            pq, csv = paths_map[name]
            df.to_parquet(pq, index=False)
            df.to_csv(csv, index=False)
            print(f"  [{stage_name}] {name:>5s} salvato ({len(df):,} righe) in {time.time() - t0:.1f}s: {pq.name} / {csv.name}")


def run(
    train_path: Path = TRAIN_CSV,
    val_path: Path = VAL_CSV,
    test_path: Path = TEST_CSV,
    events_path: Path = EVENTS_CSV,
    ratio: float = 1.0,
):
    """Esegue la pipeline completa di Step 6: creazione dataset Link e Sign."""
    t_total = time.time()
    print("=" * 60)
    print("STEP 6 – Estrazione Feature e Creazione Dataset (Link & Sign)")
    print("=" * 60)

    print("\n1. Caricamento split di base ed eventi...")
    splits = {
        "train": load_dataset(TRAIN_FILE, train_path),
        "val":   load_dataset(VAL_FILE, val_path),
        "test":  load_dataset(TEST_FILE, test_path),
    }
    events_df = load_dataset(EVENTS_TABLE_FILE, events_path)

    print("\n2. Calcolo fattori latenti SVD su train...")
    user_vecs, movie_vecs = train_svd_embeddings(splits["train"])

    print("\n3. Costruzione dataset Link Prediction e Sign Prediction...")
    link_datasets = {}
    sign_datasets = {}
    seeds = {"train": 42, "val": 43, "test": 44}

    for name in ["train", "val", "test"]:
        # Train usa negative sampling; Val e Test usano esclusivamente dati veri
        curr_ratio = ratio if name == "train" else 0.0
        link_df, sign_df = build_split_datasets(
            name=name,
            split_df=splits[name],
            events_df=events_df,
            user_vecs=user_vecs,
            movie_vecs=movie_vecs,
            ratio=curr_ratio,
            seed=seeds[name],
        )
        link_datasets[name] = link_df
        sign_datasets[name] = sign_df

    print("\n4. Salvataggio su disco...")
    save_all_datasets(link_datasets, sign_datasets)

    print(f"\nStep 6 completato con successo in {time.time() - t_total:.1f}s!\n")
    return link_datasets, sign_datasets


if __name__ == "__main__":
    link_datasets, sign_datasets = run()
