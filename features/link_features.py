"""
features/link_features.py – Estrazione Caratteristiche per la Link Prediction (Stadio 1).

Modella la formazione del legame (arco nel grafo bipartito) attraverso:
  1. Topologia & Popolarità (user_degree, movie_degree)
  2. Recency temporale (user_days_since_last, movie_days_since_last)
  3. Attività recente a 30 giorni (user_events_30d, movie_events_30d)
  4. Affinità tematica per genere (genre_match_score)
"""

from collections import deque
from pathlib import Path
import time
import numpy as np
import pandas as pd

from features.common import (
    SECONDS_PER_DAY,
    WINDOW_30D_SEC,
    parse_genres_tuple,
    compute_genre_match,
    load_dataset,
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

TARGET_COL = "link_exists"


class LinkNodeState:
    """Traccia in tempo reale la topologia, la recency e i generi di un nodo (utente o film)."""
    __slots__ = ("neighbors", "last_ts", "recent_30d", "genres")

    def __init__(self):
        self.neighbors = set()
        self.last_ts = -1.0
        self.recent_30d = deque()
        self.genres = {}

    def update(self, neighbor: int, t: float, genres_tuple: tuple[str, ...] = None):
        """Aggiorna lo stato del nodo con una nuova interazione temporale."""
        self.neighbors.add(neighbor)
        self.last_ts = t
        self.recent_30d.append(t)
        if genres_tuple:
            w = 1.0 / len(genres_tuple)
            for g in genres_tuple:
                self.genres[g] = self.genres.get(g, 0.0) + w

    def get_features(self, t: float, cutoff_30d: float) -> tuple[int, float, int]:
        """Restituisce (degree, days_since_last, events_30d) in O(1)."""
        q = self.recent_30d
        while q and q[0] < cutoff_30d:
            q.popleft()

        days_since = (t - self.last_ts) / SECONDS_PER_DAY if self.last_ts >= 0 else -1.0
        return len(self.neighbors), days_since, len(q)


def sample_negative_pairs(
    df: pd.DataFrame,
    events_df: pd.DataFrame,
    ratio: float = 1.0,
    seed: int = 42,
) -> pd.DataFrame:
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


def extract_link_features(queries: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """
    Estrae online in modo causale le 7 feature per le query usando solo eventi con t_evento < t_query.
    Garantisce assenza di data leakage e pulizia di colonne superflue.
    """
    n_queries = len(queries)
    base_cols = [c for c in ["user_id", "movie_id", "timestamp", "title", "genres", "link_exists"] if c in queries.columns]
    for extra in ["rating", "label", "binary_label"]:
        if extra in queries.columns and extra not in base_cols:
            base_cols.append(extra)

    if n_queries == 0:
        empty_df = pd.DataFrame(columns=LINK_FEATURE_COLS, dtype=np.float32)
        return pd.concat([queries[base_cols].copy(), empty_df], axis=1)

    # Eventi storici ordinati cronologicamente
    events_sorted = events.sort_values("timestamp", kind="stable")
    h_users = events_sorted["user_id"].to_numpy()
    h_movies = events_sorted["movie_id"].to_numpy()
    h_ts = pd.to_datetime(events_sorted["timestamp"]).astype("int64").to_numpy() / 1e9
    h_genres = events_sorted["genres"].to_numpy() if "genres" in events_sorted.columns else None

    # Query in array NumPy
    q_users = queries["user_id"].to_numpy()
    q_movies = queries["movie_id"].to_numpy()
    q_ts = pd.to_datetime(queries["timestamp"]).astype("int64").to_numpy() / 1e9
    q_genres = queries["genres"].to_numpy() if "genres" in queries.columns else None
    order = np.argsort(q_ts, kind="stable")

    users: dict[int, LinkNodeState] = {}
    movies: dict[int, LinkNodeState] = {}
    n_events, ev_idx = len(events_sorted), 0

    # Matrice pre-allocata per le 7 feature di Link Prediction
    feat_mat = np.zeros((n_queries, 7), dtype=np.float32)
    feat_mat[:, 2] = -1.0  # user_days_since_last default
    feat_mat[:, 3] = -1.0  # movie_days_since_last default

    log_step = max(500_000, n_queries // 10)

    for step, idx in enumerate(order, 1):
        t = q_ts[idx]
        uid, mid = q_users[idx], q_movies[idx]

        # Ingestione causale: t_evento < t_query
        while ev_idx < n_events and h_ts[ev_idx] < t:
            ev_u, ev_m, ev_t = h_users[ev_idx], h_movies[ev_idx], h_ts[ev_idx]
            gs = parse_genres_tuple(h_genres[ev_idx]) if h_genres is not None else None

            u_node = users.get(ev_u)
            if u_node is None:
                u_node = LinkNodeState()
                users[ev_u] = u_node
            u_node.update(ev_m, ev_t, gs)

            m_node = movies.get(ev_m)
            if m_node is None:
                m_node = LinkNodeState()
                movies[ev_m] = m_node
            m_node.update(ev_u, ev_t)

            ev_idx += 1

        cutoff_30 = t - WINDOW_30D_SEC

        # Feature utente: degree(0), days_since(2), events_30d(4)
        u = users.get(uid)
        if u is not None:
            feat_mat[idx, 0], feat_mat[idx, 2], feat_mat[idx, 4] = u.get_features(t, cutoff_30)

        # Feature film: degree(1), days_since(3), events_30d(5)
        m = movies.get(mid)
        if m is not None:
            feat_mat[idx, 1], feat_mat[idx, 3], feat_mat[idx, 5] = m.get_features(t, cutoff_30)

        # Feature affinità genere: genre_match_score(6)
        if u is not None and u.genres and q_genres is not None:
            feat_mat[idx, 6] = compute_genre_match(u.genres, parse_genres_tuple(q_genres[idx]))

        if n_queries >= 200_000 and step % log_step == 0:
            print(f"      ... estratte {step:,}/{n_queries:,} feature ({step * 100 // n_queries}%)")

    feature_df = pd.DataFrame(feat_mat, index=queries.index, columns=LINK_FEATURE_COLS)
    return pd.concat([queries[base_cols].copy(), feature_df], axis=1)


def build_link_dataset(
    name: str,
    split_df: pd.DataFrame,
    events_df: pd.DataFrame,
    ratio: float = 1.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Costruisce e arricchisce il dataset per Link Prediction per un singolo split.
    Train include negative sampling (coppie negative); Val e Test utilizzano solo dati veri.
    """
    t0 = time.time()
    n_pos = len(split_df)
    info = f"ratio={ratio} negativi" if ratio > 0 else "solo dati veri"
    print(f"  Elaborazione {name.upper()} ({n_pos:,} link reali, {info})...")

    positives = split_df.assign(link_exists=1)
    if ratio > 0:
        negatives = sample_negative_pairs(split_df, events_df, ratio=ratio, seed=seed)
        combined = pd.concat([positives, negatives], ignore_index=True)
    else:
        combined = positives

    link_df = extract_link_features(combined, events_df)
    print(f"  --> {name.upper()} completato in {time.time() - t0:.1f}s (Link: {len(link_df):,})")
    return link_df


def load_link_data(csv_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Carica il dataset preparato da Step 6 ed estrae matrice X e target binario link_exists."""
    df = load_dataset(csv_path)
    X = df[LINK_FEATURE_COLS].fillna(0).copy()
    y = df[TARGET_COL].astype(int)
    return X, y
