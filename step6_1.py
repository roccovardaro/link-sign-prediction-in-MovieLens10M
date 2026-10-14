"""
Step 6.1 – Estrazione Caratteristiche Specifiche per la Sign Prediction (MovieLens 10M).

Costruisce il dataset dedicato per lo Stadio 2 (Sign Prediction),
sviluppato sulla falsa riga di quanto realizzato per la Link Prediction in Step 6.

Differenze concettuali rispetto allo Stadio 1:
  - Link Prediction (Step 6/7): modella l'esistenza dell'arco (popolarità, grado, recency, attività).
  - Sign Prediction (Step 6.1/8): modella la polarità dell'interazione (gradimento vs scontento)
    condizionata all'esistenza del link:
      1 = Valutazione Positiva (rating 4.0 - 5.0)
      0 = Valutazione Negativa (rating 0.5 - 2.5)

Caratteristiche specifiche per il Segno:
  1. Storico e Attitudine Utente:
     - user_mean_rating: media storica dei voti assegnati (severità vs generosità)
     - user_rating_std: deviazione standard storica (coerenza vs ampiezza d'uso della scala)
     - user_frac_pos: percentuale di recensioni positive (>= 4.0)
     - user_frac_neg: percentuale di recensioni negative (<= 2.5)
     - user_review_count: volume di recensioni storiche (affidabilità statistica)
  2. Storico e Percezione del Film:
     - movie_mean_rating: voto medio storico ricevuto dal film
     - movie_rating_std: deviazione standard dei voti (consenso unanime vs polarizzazione/controversia)
     - movie_frac_pos: quota di giudizi positivi ricevuti
     - movie_frac_neg: quota di giudizi negativi ricevuti
     - movie_review_count: volume di recensioni storiche del film
  3. Bias Relativo e Baseline di Gradimento (Koren Baseline Predictor):
     - rating_baseline_score: mu + (user_mean - mu) + (movie_mean - mu)
     - user_movie_rating_diff: user_mean_rating - movie_mean_rating
     - user_movie_optimism: (user_frac_pos - 0.5) + (movie_frac_pos - 0.5)
  4. Affinità di Gusto Firmata per Genere:
     - user_genre_mean_rating: voto medio storico dato dall'utente specificamente a questo genere
  5. Filtro Collaborativo Firmato (Signed SVD):
     - signed_svd_similarity: cosine similarity tra fattori latenti SVD addestrati su rating centrati (r - mu)
"""

import time
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
    TRAIN_SIGN_CSV,
    VAL_SIGN_CSV,
    TEST_SIGN_CSV,
    TRAIN_SIGN_FILE,
    VAL_SIGN_FILE,
    TEST_SIGN_FILE,
)

# ── 15 Feature dedicate alla Sign Prediction ────────────────────────────
SIGN_FEATURE_COLS = [
    # Attitudine e storico utente
    "user_mean_rating",        # Media storica voti utente
    "user_rating_std",         # Varianza/dispersione storica voti utente
    "user_frac_pos",           # Quota voti positivi (>= 4.0) utente
    "user_frac_neg",           # Quota voti negativi (<= 2.5) utente
    "user_review_count",       # Log-count recensioni passate dell'utente

    # Storico e qualità percepita del film
    "movie_mean_rating",       # Media storica voti film
    "movie_rating_std",        # Polarizzazione film (divisivo vs consenso)
    "movie_frac_pos",          # Quota valutazioni positive ricevute
    "movie_frac_neg",          # Quota valutazioni negative ricevute
    "movie_review_count",      # Log-count recensioni passate del film

    # Baseline e combinazione bias (Koren Predictor)
    "rating_baseline_score",   # Baseline stima voto: mu + b_u + b_m
    "user_movie_rating_diff",  # Scarto generosità utente - qualità film
    "user_movie_optimism",     # Indice congiunto di ottimismo

    # Affinità di genere firmata (voto medio sui generi specifici)
    "user_genre_mean_rating",  # Voto medio storico dell'utente sui generi del film

    # Filtro collaborativo latente firmato
    "signed_svd_similarity",   # Cosine similarity su SVD da rating centrati
]

TARGET_COL = "binary_label"
GLOBAL_MEAN_RATING = 3.5

# Mappatura dei 19 generi unici di MovieLens a ID interi compatti (0..18)
GENRE_MAP = {
    "Action": 0, "Adventure": 1, "Animation": 2, "Children": 3, "Comedy": 4,
    "Crime": 5, "Documentary": 6, "Drama": 7, "Fantasy": 8, "Film-Noir": 9,
    "Horror": 10, "IMAX": 11, "Musical": 12, "Mystery": 13, "Romance": 14,
    "Sci-Fi": 15, "Thriller": 16, "War": 17, "Western": 18,
}
N_GENRES = len(GENRE_MAP)
GENRE_ID_CACHE = {}


def parse_genres_to_ids(val) -> tuple:
    """Converte la stringa dei generi in una tupla di ID interi con memorizzazione in cache."""
    if not isinstance(val, str) or not val or val == "(no genres listed)":
        return ()
    res = GENRE_ID_CACHE.get(val)
    if res is None:
        res = tuple(GENRE_MAP[g] for g in val.split("|") if g in GENRE_MAP)
        GENRE_ID_CACHE[val] = res
    return res


class UserSignState:
    """Traccia in tempo reale le statistiche valutative e di genere dell'utente in O(1)."""
    __slots__ = ("rsum", "rsq_sum", "pos_cnt", "neg_cnt", "cnt", "genre_rsum", "genre_cnt")

    def __init__(self):
        self.rsum = 0.0
        self.rsq_sum = 0.0
        self.pos_cnt = 0
        self.neg_cnt = 0
        self.cnt = 0
        self.genre_rsum = [0.0] * N_GENRES
        self.genre_cnt = [0] * N_GENRES

    def update(self, r: float, gids: tuple):
        self.rsum += r
        self.rsq_sum += r * r
        self.pos_cnt += int(r >= 4.0)
        self.neg_cnt += int(r <= 2.5)
        self.cnt += 1
        for gid in gids:
            self.genre_rsum[gid] += r
            self.genre_cnt[gid] += 1

    def get_features(self, query_gids: tuple):
        if self.cnt == 0:
            return GLOBAL_MEAN_RATING, 0.0, 0.5, 0.25, 0.0, GLOBAL_MEAN_RATING

        mean_r = self.rsum / self.cnt
        var_r = max(0.0, (self.rsq_sum / self.cnt) - (mean_r * mean_r))
        std_r = var_r ** 0.5
        frac_pos = self.pos_cnt / self.cnt
        frac_neg = self.neg_cnt / self.cnt
        log_cnt = float(np.log1p(self.cnt))

        # Affinità di genere firmata: media voti dell'utente sui generi del film
        if query_gids:
            g_rsum = 0.0
            g_cnt = 0
            for gid in query_gids:
                g_rsum += self.genre_rsum[gid]
                g_cnt += self.genre_cnt[gid]
            genre_mean = (g_rsum / g_cnt) if g_cnt > 0 else mean_r
        else:
            genre_mean = mean_r

        return mean_r, std_r, frac_pos, frac_neg, log_cnt, genre_mean


class MovieSignState:
    """Traccia in tempo reale le statistiche valutative e di polarizzazione del film in O(1)."""
    __slots__ = ("rsum", "rsq_sum", "pos_cnt", "neg_cnt", "cnt")

    def __init__(self):
        self.rsum = 0.0
        self.rsq_sum = 0.0
        self.pos_cnt = 0
        self.neg_cnt = 0
        self.cnt = 0

    def update(self, r: float):
        self.rsum += r
        self.rsq_sum += r * r
        self.pos_cnt += int(r >= 4.0)
        self.neg_cnt += int(r <= 2.5)
        self.cnt += 1

    def get_features(self):
        if self.cnt == 0:
            return GLOBAL_MEAN_RATING, 0.0, 0.5, 0.25, 0.0

        mean_r = self.rsum / self.cnt
        var_r = max(0.0, (self.rsq_sum / self.cnt) - (mean_r * mean_r))
        std_r = var_r ** 0.5
        frac_pos = self.pos_cnt / self.cnt
        frac_neg = self.neg_cnt / self.cnt
        log_cnt = float(np.log1p(self.cnt))

        return mean_r, std_r, frac_pos, frac_neg, log_cnt


def load_dataset(parquet_path: Path, csv_path: Path) -> pd.DataFrame:
    """Carica velocemente da Parquet se presente, altrimenti da CSV."""
    df = pd.read_parquet(parquet_path) if parquet_path.exists() else pd.read_csv(csv_path)
    if "timestamp" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def load_sign_data(parquet_path: Path, csv_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Carica il dataset preparato da Step 6.1 ed estrae matrice X e target binario del segno."""
    df = load_dataset(parquet_path, csv_path)
    for col in SIGN_FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0
    X = df[SIGN_FEATURE_COLS].fillna(0.0).copy()
    y = df[TARGET_COL].astype(int)
    return X, y


def train_signed_svd_embeddings(train_df: pd.DataFrame, n_components: int = 16):
    """
    Allena TruncatedSVD sui voti centrati (r - mu) per ottenere fattori latenti firmati.
    A differenza dell'SVD su soli 1 (link formation), i fattori riflettono concordanza/discordanza di gradimento.
    """
    t0 = time.time()
    print(f"  Calcolo embeddings collaborativi Signed SVD ({n_components} componenti su train)...")
    unique_users = train_df["user_id"].unique()
    unique_movies = train_df["movie_id"].unique()

    u_map = {uid: i for i, uid in enumerate(unique_users)}
    m_map = {mid: i for i, mid in enumerate(unique_movies)}

    row_indices = train_df["user_id"].map(u_map).values
    col_indices = train_df["movie_id"].map(m_map).values

    # Voti centrati rispetto alla media neutra (3.5):
    # rating 5.0 -> +1.5 | rating 4.0 -> +0.5 | rating 2.0 -> -1.5 | rating 1.0 -> -2.5
    raw_ratings = train_df["rating"].to_numpy(dtype=np.float32) if "rating" in train_df.columns else np.where(
        train_df["binary_label"] == 1, 1.0, -1.0
    ).astype(np.float32)
    centered_ratings = raw_ratings - float(GLOBAL_MEAN_RATING)

    adj = csr_matrix((centered_ratings, (row_indices, col_indices)), shape=(len(unique_users), len(unique_movies)))
    svd = TruncatedSVD(n_components=min(n_components, len(unique_movies) - 1, len(unique_users) - 1), random_state=42)
    user_factors = svd.fit_transform(adj)
    movie_factors = svd.components_.T

    # Normalizzazione L2 vettoriale per cosine similarity diretta
    u_norms = np.linalg.norm(user_factors, axis=1, keepdims=True)
    user_factors /= np.where(u_norms == 0, 1.0, u_norms)

    m_norms = np.linalg.norm(movie_factors, axis=1, keepdims=True)
    movie_factors /= np.where(m_norms == 0, 1.0, m_norms)

    user_vecs = {uid: user_factors[i] for uid, i in u_map.items()}
    movie_vecs = {mid: movie_factors[i] for mid, i in m_map.items()}
    print(f"  --> Signed SVD completata in {time.time() - t0:.1f}s")
    return user_vecs, movie_vecs


def compute_signed_svd_similarity(df: pd.DataFrame, user_vecs: dict, movie_vecs: dict) -> np.ndarray:
    """Calcola la similarità coseno vettoriale tra i fattori SVD firmati di utente e film."""
    n = len(df)
    if not user_vecs or not movie_vecs or n == 0:
        return np.zeros(n, dtype=np.float32)

    dim = next(iter(user_vecs.values())).shape[0]
    zero_vec = np.zeros(dim, dtype=np.float32)

    u_ids = df["user_id"].to_numpy()
    m_ids = df["movie_id"].to_numpy()

    u_mat = np.array([user_vecs.get(uid, zero_vec) for uid in u_ids], dtype=np.float32)
    m_mat = np.array([movie_vecs.get(mid, zero_vec) for mid in m_ids], dtype=np.float32)

    sims = np.sum(u_mat * m_mat, axis=1)
    return np.round(sims, 4).astype(np.float32)


def extract_sign_features(queries: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """
    Estrae online in modo causale le 14 feature valutative per le query,
    garantendo l'assoluta assenza di data leakage temporale (t_evento < t_query).
    """
    n_queries = len(queries)
    feature_names = [f for f in SIGN_FEATURE_COLS if f != "signed_svd_similarity"]
    n_features = len(feature_names)

    if n_queries == 0:
        empty_df = pd.DataFrame(columns=feature_names, dtype=np.float32)
        return pd.concat([queries.drop(columns=feature_names, errors="ignore"), empty_df], axis=1)

    # Costruzione mappa statica film -> generi (ID compatti) in O(num_movies)
    movie_genres_map = {}
    for source in (events, queries):
        if "movie_id" in source.columns and "genres" in source.columns:
            m_sub = source[["movie_id", "genres"]].drop_duplicates("movie_id")
            for mid, g in zip(m_sub["movie_id"].to_numpy(), m_sub["genres"].to_numpy()):
                if mid not in movie_genres_map:
                    movie_genres_map[int(mid)] = parse_genres_to_ids(g)

    # Eventi storici in array NumPy
    events_sorted = events.sort_values("timestamp", kind="stable")
    h_users = events_sorted["user_id"].to_numpy()
    h_movies = events_sorted["movie_id"].to_numpy()
    h_ts = pd.to_datetime(events_sorted["timestamp"]).astype("int64").to_numpy() / 1e9
    h_ratings = events_sorted["rating"].to_numpy(dtype=np.float32)

    # Query in array NumPy
    q_users = queries["user_id"].to_numpy()
    q_movies = queries["movie_id"].to_numpy()
    q_ts = pd.to_datetime(queries["timestamp"]).astype("int64").to_numpy() / 1e9

    order = np.argsort(q_ts, kind="stable")

    users: dict[int, UserSignState] = {}
    movies: dict[int, MovieSignState] = {}
    n_events, ev_idx = len(events_sorted), 0

    # Matrice pre-allocata per le 14 feature
    feat_mat = np.zeros((n_queries, n_features), dtype=np.float32)
    mu = float(GLOBAL_MEAN_RATING)

    log_step = max(500_000, n_queries // 10)

    for step, idx in enumerate(order, 1):
        t = q_ts[idx]
        uid, mid = q_users[idx], q_movies[idx]
        query_gids = movie_genres_map.get(mid, ())

        # Ingestione causale: t_evento < t_query
        while ev_idx < n_events and h_ts[ev_idx] < t:
            ev_u = h_users[ev_idx]
            ev_m = h_movies[ev_idx]
            r = float(h_ratings[ev_idx])
            ev_g = movie_genres_map.get(ev_m, ())

            u_node = users.get(ev_u)
            if u_node is None:
                u_node = UserSignState()
                users[ev_u] = u_node
            u_node.update(r, ev_g)

            m_node = movies.get(ev_m)
            if m_node is None:
                m_node = MovieSignState()
                movies[ev_m] = m_node
            m_node.update(r)

            ev_idx += 1

        # Estrazione feature utente
        u = users.get(uid)
        if u is not None:
            u_mean, u_std, u_pos, u_neg, u_cnt, u_genre_mean = u.get_features(query_gids)
        else:
            u_mean, u_std, u_pos, u_neg, u_cnt, u_genre_mean = mu, 0.0, 0.5, 0.25, 0.0, mu

        # Estrazione feature film
        m = movies.get(mid)
        if m is not None:
            m_mean, m_std, m_pos, m_neg, m_cnt = m.get_features()
        else:
            m_mean, m_std, m_pos, m_neg, m_cnt = mu, 0.0, 0.5, 0.25, 0.0

        # Baseline Koren Predictor: mu + b_u + b_m
        baseline_score = mu + (u_mean - mu) + (m_mean - mu)
        rating_diff = u_mean - m_mean
        optimism = (u_pos - 0.5) + (m_pos - 0.5)

        # Inserimento nella riga idx
        feat_mat[idx, 0]  = u_mean          # user_mean_rating
        feat_mat[idx, 1]  = u_std           # user_rating_std
        feat_mat[idx, 2]  = u_pos           # user_frac_pos
        feat_mat[idx, 3]  = u_neg           # user_frac_neg
        feat_mat[idx, 4]  = u_cnt           # user_review_count
        feat_mat[idx, 5]  = m_mean          # movie_mean_rating
        feat_mat[idx, 6]  = m_std           # movie_rating_std
        feat_mat[idx, 7]  = m_pos           # movie_frac_pos
        feat_mat[idx, 8]  = m_neg           # movie_frac_neg
        feat_mat[idx, 9]  = m_cnt           # movie_review_count
        feat_mat[idx, 10] = baseline_score  # rating_baseline_score
        feat_mat[idx, 11] = rating_diff     # user_movie_rating_diff
        feat_mat[idx, 12] = optimism        # user_movie_optimism
        feat_mat[idx, 13] = u_genre_mean    # user_genre_mean_rating

        if n_queries >= 200_000 and step % log_step == 0:
            print(f"      ... estratte {step:,}/{n_queries:,} feature ({step * 100 // n_queries}%)")

    feature_df = pd.DataFrame(feat_mat, index=queries.index, columns=feature_names)
    return pd.concat([queries.drop(columns=feature_names, errors="ignore"), feature_df], axis=1)


def build_sign_split_datasets(
    name: str,
    split_df: pd.DataFrame,
    events_df: pd.DataFrame,
    user_vecs: dict,
    movie_vecs: dict,
) -> pd.DataFrame:
    """Costruisce e arricchisce il dataset per Sign Prediction per un singolo split."""
    t0 = time.time()
    n_samples = len(split_df)
    print(f"  Elaborazione {name.upper()} ({n_samples:,} istanze con segno)...")

    sign_df = extract_sign_features(split_df, events_df)
    sign_df["signed_svd_similarity"] = compute_signed_svd_similarity(sign_df, user_vecs, movie_vecs)

    print(f"  --> {name.upper()} completato in {time.time() - t0:.1f}s ({len(sign_df):,} righe con {len(SIGN_FEATURE_COLS)} feature)")
    return sign_df


def save_all_sign_datasets(sign_splits: dict) -> None:
    """Salva i dataset dedicati a Sign Prediction su disco sia in Parquet che in CSV."""
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    CSV_PROCESSED.mkdir(parents=True, exist_ok=True)

    file_maps = [
        ("Sign Stage Dedicato", sign_splits, {
            "train": (TRAIN_SIGN_FILE, TRAIN_SIGN_CSV),
            "val":   (VAL_SIGN_FILE, VAL_SIGN_CSV),
            "test":  (TEST_SIGN_FILE, TEST_SIGN_CSV),
        }),
        ("Sign Stage Compatibilità", sign_splits, {
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
):
    """Esegue la pipeline completa di Step 6.1: creazione dataset per Sign Prediction."""
    t_total = time.time()
    print("=" * 60)
    print("STEP 6.1 – Creazione Dataset Specifico per Sign Prediction")
    print("=" * 60)
    print(f"  Feature specifiche per il segno ({len(SIGN_FEATURE_COLS)}):")
    for f in SIGN_FEATURE_COLS:
        print(f"    - {f}")

    print("\n1. Caricamento split di base ed eventi...")
    splits = {
        "train": load_dataset(TRAIN_FILE, train_path),
        "val":   load_dataset(VAL_FILE, val_path),
        "test":  load_dataset(TEST_FILE, test_path),
    }
    events_df = load_dataset(EVENTS_TABLE_FILE, events_path)

    print("\n2. Calcolo fattori latenti Signed SVD su train...")
    user_vecs, movie_vecs = train_signed_svd_embeddings(splits["train"])

    print("\n3. Estrazione causale feature di segno su train, val e test...")
    sign_datasets = {}
    for name in ["train", "val", "test"]:
        sign_datasets[name] = build_sign_split_datasets(
            name=name,
            split_df=splits[name],
            events_df=events_df,
            user_vecs=user_vecs,
            movie_vecs=movie_vecs,
        )

    print("\n4. Salvataggio su disco (Parquet e CSV)...")
    save_all_sign_datasets(sign_datasets)

    print(f"\nStep 6.1 completato con successo in {time.time() - t_total:.1f}s!\n")
    return sign_datasets


if __name__ == "__main__":
    sign_datasets = run()
