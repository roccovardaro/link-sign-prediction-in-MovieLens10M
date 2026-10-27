"""
traditional/dataset.py – Macrofase Preparazione e Gestione del Dataset (MovieLens 10M).

Unifica le funzionalità di gestione e preparazione dei dati precedentemente suddivise in step:
  - Acquisizione dati MovieLens 10M (ratings.dat, movies.dat -> raw_ratings.csv)
  - Pulizia, validazione ed esplorazione dati (clean_ratings.csv, exploration_stats.json)
  - Grafo bipartito, sentiment e mapping etichette (events.csv)
  - Split temporale rigido train / val / test (train.csv, val.csv, test.csv, split_info.json)
  - Creazione dataset dedicati per Link Prediction (train_link.csv, val_link.csv, test_link.csv)
  - Creazione dataset dedicati per Sign Prediction (train_sign.csv, val_sign.csv, test_sign.csv)
"""

import json
from pathlib import Path
import time
from typing import Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd

from config import (
    RATINGS_FILE,
    MOVIES_FILE,
    SAMPLE_SIZE,
    KEEP_COLUMNS,
    RAW_RATINGS_CSV,
    CLEAN_RATINGS_CSV,
    STATS_FILE,
    EVENTS_CSV,
    TRAIN_CSV,
    VAL_CSV,
    TEST_CSV,
    TRAIN_LINK_CSV,
    VAL_LINK_CSV,
    TEST_LINK_CSV,
    TRAIN_SIGN_CSV,
    VAL_SIGN_CSV,
    TEST_SIGN_CSV,
    SPLIT_RATIOS,
    SPLIT_INFO_FILE,
    CSV_PROCESSED,
    RESULTS_DIR,
    rating_to_sentiment,
    BINARY_LABEL_MAP,
)
from traditional.features.common import load_dataset, save_split_csv
from traditional.features.link_features import (
    LINK_FEATURE_COLS,
    extract_link_features,
    sample_negative_pairs,
    build_link_dataset,
    load_link_data,
)
from traditional.features.sign_features import (
    SIGN_FEATURE_COLS,
    extract_sign_features,
    train_signed_svd_embeddings,
    compute_signed_svd_similarity,
    build_sign_dataset,
    load_sign_data,
)


# ══════════════════════════════════════════════════════════════════════════════
# 1. ACQUISIZIONE DATI
# ══════════════════════════════════════════════════════════════════════════════

def load_movies(movies_path: Path = MOVIES_FILE) -> pd.DataFrame:
    """Carica movies.dat contenente [movie_id, title, genres]."""
    if not movies_path.exists():
        raise FileNotFoundError(f"File film non trovato: {movies_path.resolve()}")

    df_movies = pd.read_csv(
        movies_path,
        sep="::",
        names=["movie_id", "title", "genres"],
        engine="python",
        encoding="utf-8",
    )
    df_movies["movie_id"] = df_movies["movie_id"].astype(int)
    df_movies["title"] = df_movies["title"].fillna("").astype(str).str.strip()
    df_movies["genres"] = df_movies["genres"].fillna("").astype(str).str.strip()
    return df_movies


def load_ratings(
    ratings_path: Path = RATINGS_FILE,
    sample_size: Optional[int] = SAMPLE_SIZE,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Carica ratings.dat contenente [user_id, movie_id, rating, timestamp].
    Se sample_size è specificato, estrae un campione casuale riproducibile.
    """
    if not ratings_path.exists():
        raise FileNotFoundError(f"File rating non trovato: {ratings_path.resolve()}")

    print(f"  Lettura ratings da: {ratings_path.name}...")
    df = pd.read_csv(
        ratings_path,
        sep="::",
        names=["user_id", "movie_id", "rating", "timestamp"],
        engine="python",
        dtype={"user_id": int, "movie_id": int, "rating": float, "timestamp": int},
    )
    print(f"  Valutazioni totali caricate: {len(df):,}")

    if sample_size is not None and len(df) > sample_size:
        print(f"  Campionamento di {sample_size:,} valutazioni (random_state={random_state})...")
        df = df.sample(n=sample_size, random_state=random_state)

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def acquire_data(
    ratings_path: Path = RATINGS_FILE,
    movies_path: Path = MOVIES_FILE,
    output_csv: Path = RAW_RATINGS_CSV,
    sample_size: Optional[int] = SAMPLE_SIZE,
) -> pd.DataFrame:
    """Esegue l'acquisizione dei dati grezzi e unisce i metadati dei film."""
    print("=" * 60)
    print("ACQUISIZIONE DATI (MovieLens 10M)")
    print("=" * 60)

    df_ratings = load_ratings(ratings_path, sample_size=sample_size)
    df_movies = load_movies(movies_path)

    df = df_ratings.merge(df_movies, on="movie_id", how="left")
    cols = [c for c in KEEP_COLUMNS if c in df.columns]
    df = df[cols].copy()

    print(f"  Righe finali acquisite: {len(df):,}")
    print(f"  Utenti unici:           {df['user_id'].nunique():,}")
    print(f"  Film unici:             {df['movie_id'].nunique():,}")
    print(f"  Colonne:                {list(df.columns)}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"  Dati grezzi salvati in CSV: {output_csv}\n")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2. PULIZIA ED ESPLORAZIONE
# ══════════════════════════════════════════════════════════════════════════════

def validate_types(df: pd.DataFrame) -> pd.DataFrame:
    """Assicura tipi corretti per le colonne chiave."""
    df["user_id"] = pd.to_numeric(df["user_id"], errors="coerce").fillna(-1).astype(int)
    df["movie_id"] = pd.to_numeric(df["movie_id"], errors="coerce").fillna(-1).astype(int)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if "title" in df.columns:
        df["title"] = df["title"].fillna("").astype(str)
    if "genres" in df.columns:
        df["genres"] = df["genres"].fillna("").astype(str)
    return df


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Rimuove record invalidi o con rating fuori intervallo [0.5, 5.0]."""
    before = len(df)
    valid_mask = (
        (df["user_id"] > 0)
        & (df["movie_id"] > 0)
        & df["rating"].between(0.5, 5.0)
        & df["timestamp"].notna()
    )
    df = df[valid_mask].copy()
    after = len(df)
    print(f"  Valori mancanti/invalidi rimossi: {before - after:,}")
    return df


def remove_exact_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Rimuove duplicati esatti per coppia (user_id, movie_id, timestamp)."""
    before = len(df)
    df = df.drop_duplicates(subset=["user_id", "movie_id", "timestamp"]).copy()
    after = len(df)
    print(f"  Duplicati rimossi: {before - after:,}")
    return df


def sort_chronologically(df: pd.DataFrame) -> pd.DataFrame:
    """Ordina il DataFrame per timestamp crescente."""
    df = df.sort_values("timestamp").reset_index(drop=True)
    print("  Ordinamento cronologico completato")
    return df


def compute_stats(df: pd.DataFrame) -> dict:
    """Calcola statistiche descrittive ed esplorative sul dataset pulito."""
    user_counts = df.groupby("user_id").size()
    movie_counts = df.groupby("movie_id").size()
    rating_dist = df["rating"].value_counts().sort_index().to_dict()

    return {
        "totale_valutazioni": int(len(df)),
        "utenti_unici": int(df["user_id"].nunique()),
        "film_unici": int(df["movie_id"].nunique()),
        "distribuzione_valutazioni": {str(k): int(v) for k, v in rating_dist.items()},
        "valutazioni_per_utente": {
            "media": round(float(user_counts.mean()), 2),
            "mediana": float(user_counts.median()),
            "max": int(user_counts.max()),
            "min": int(user_counts.min()),
        },
        "valutazioni_per_film": {
            "media": round(float(movie_counts.mean()), 2),
            "mediana": float(movie_counts.median()),
            "max": int(movie_counts.max()),
            "min": int(movie_counts.min()),
        },
        "range_temporale": {
            "inizio": str(df["timestamp"].min()),
            "fine": str(df["timestamp"].max()),
        },
    }


def save_stats(stats: dict, path: Path = STATS_FILE) -> None:
    """Salva le statistiche in JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"  Statistiche salvate in: {path}")


def print_stats(stats: dict) -> None:
    """Stampa a video il report esplorativo."""
    print("\n  ── Report Esplorativo MovieLens ──")
    print(f"  Totale valutazioni:   {stats['totale_valutazioni']:,}")
    print(f"  Utenti unici:         {stats['utenti_unici']:,}")
    print(f"  Film unici:           {stats['film_unici']:,}")
    print(f"  Periodo temporale:    {stats['range_temporale']['inizio']} → {stats['range_temporale']['fine']}")
    print(f"  Valutazioni per user: media={stats['valutazioni_per_utente']['media']}, max={stats['valutazioni_per_utente']['max']}")
    print(f"  Valutazioni per film: media={stats['valutazioni_per_film']['media']}, max={stats['valutazioni_per_film']['max']}")
    print("  Distribuzione voti:")
    for star, count in sorted(stats["distribuzione_valutazioni"].items(), key=lambda x: float(x[0])):
        print(f"    {star:>3s} ★: {count:,}")
    print()


def clean_data(
    input_path: Path = RAW_RATINGS_CSV,
    output_path: Path = CLEAN_RATINGS_CSV,
) -> tuple[pd.DataFrame, dict]:
    """Esegue la pulizia e il calcolo delle statistiche esplorative."""
    print("=" * 60)
    print("PULIZIA ED ESPLORAZIONE DATI")
    print("=" * 60)
    print(f"  Caricamento dati da: {input_path}")
    df = pd.read_csv(input_path)

    df = validate_types(df)
    df = handle_missing(df)
    df = remove_exact_duplicates(df)
    df = sort_chronologically(df)

    stats = compute_stats(df)
    print_stats(stats)
    save_stats(stats)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"  Dati puliti salvati in: {output_path}\n")
    return df, stats


# ══════════════════════════════════════════════════════════════════════════════
# 3. GRAFO BIPARTITO ED ETICHETTE
# ══════════════════════════════════════════════════════════════════════════════

def assign_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Assegna l'etichetta sentiment (negativa, neutrale, positiva) in base al voto."""
    df["label"] = df["rating"].apply(rating_to_sentiment)
    counts = df["label"].value_counts()
    print("  Distribuzione etichette:")
    for lbl in ["negativa", "neutrale", "positiva"]:
        print(f"    {lbl:<10s}: {counts.get(lbl, 0):,}")
    return df


def build_events_table(df: pd.DataFrame) -> pd.DataFrame:
    """Costruisce la tabella degli eventi ordinata per timestamp."""
    cols = ["user_id", "movie_id", "rating", "label", "timestamp"]
    if "title" in df.columns:
        cols.append("title")
    if "genres" in df.columns:
        cols.append("genres")

    events = df[cols].copy()
    events = events.sort_values("timestamp").reset_index(drop=True)
    return events


def build_graph_edges(df: pd.DataFrame) -> pd.DataFrame:
    """Costruisce la vista degli archi del grafo bipartito (src=user_id, dst=movie_id)."""
    edges = df[["user_id", "movie_id", "rating", "label"]].copy()
    edges = edges.rename(columns={"user_id": "src", "movie_id": "dst"})
    print(f"\n  ── Grafo Bipartito Utente-Film ──")
    print(f"  Nodi utente: {edges['src'].nunique():,}")
    print(f"  Nodi film:   {edges['dst'].nunique():,}")
    print(f"  Archi totali:{len(edges):,}")
    return edges


def build_binary_subset(df: pd.DataFrame, binary_map: dict = BINARY_LABEL_MAP) -> pd.DataFrame:
    """Estrae il subset binario escludendo i voti neutrali (negativa=0, positiva=1)."""
    binary_df = df[df["label"] != "neutrale"].copy()
    binary_df["binary_label"] = binary_df["label"].map(binary_map).astype(int)

    n_neg = (binary_df["binary_label"] == 0).sum()
    n_pos = (binary_df["binary_label"] == 1).sum()
    print(f"\n  ── Subset Binario (no neutrale) ──")
    print(f"  Totale istanze: {len(binary_df):,}")
    print(f"  Negativi (0):   {n_neg:,} ({n_neg / len(binary_df) * 100:.1f}%)")
    print(f"  Positivi (1):   {n_pos:,} ({n_pos / len(binary_df) * 100:.1f}%)")
    return binary_df


def build_graph_and_labels(input_path: Path = CLEAN_RATINGS_CSV):
    """Esegue la definizione del grafo bipartito e genera events.csv."""
    print("=" * 60)
    print("DEFINIZIONE GRAFO BIPARTITO ED ETICHETTE")
    print("=" * 60)
    print(f"  Caricamento valutazioni pulite da: {input_path}")
    df = pd.read_csv(input_path, parse_dates=["timestamp"])

    df = assign_labels(df)
    events = build_events_table(df)
    edges = build_graph_edges(df)
    binary_df = build_binary_subset(df)

    CSV_PROCESSED.mkdir(parents=True, exist_ok=True)
    events.to_csv(EVENTS_CSV, index=False)
    print(f"  Tabella eventi CSV salvata: {EVENTS_CSV}\n")
    return df, events, edges, binary_df


# ══════════════════════════════════════════════════════════════════════════════
# 4. SPLIT TEMPORALE (TRAIN / VAL / TEST)
# ══════════════════════════════════════════════════════════════════════════════

def temporal_split(df: pd.DataFrame, ratios: dict = SPLIT_RATIOS) -> tuple[dict, dict]:
    """Suddivide cronologicamente il DataFrame in train, val e test."""
    assert abs(sum(ratios.values()) - 1.0) < 1e-9, f"Proporzioni non sommano a 1.0: {sum(ratios.values())}"
    df = df.sort_values("timestamp").reset_index(drop=True)
    n = len(df)

    n_train = int(n * ratios["train"])
    n_val = int(n * ratios["val"])

    splits = {
        "train": df.iloc[:n_train].copy(),
        "val":   df.iloc[n_train:n_train + n_val].copy(),
        "test":  df.iloc[n_train + n_val:].copy(),
    }

    split_info = {}
    for name, split_df in splits.items():
        split_info[name] = {
            "n_samples": len(split_df),
            "percentuale": round(len(split_df) / n * 100, 2),
            "timestamp_inizio": str(split_df["timestamp"].min()),
            "timestamp_fine": str(split_df["timestamp"].max()),
        }
        if "binary_label" in split_df.columns:
            split_info[name]["n_negativi"] = int((split_df["binary_label"] == 0).sum())
            split_info[name]["n_positivi"] = int((split_df["binary_label"] == 1).sum())

    return splits, split_info


def verify_no_leakage(splits: dict) -> None:
    """Verifica rigorosa sull'assenza di sovrapposizione temporale tra gli split."""
    order = ["train", "val", "test"]
    for i in range(len(order) - 1):
        curr_set = order[i]
        next_set = order[i + 1]
        max_curr = splits[curr_set]["timestamp"].max()
        min_next = splits[next_set]["timestamp"].min()
        if max_curr > min_next:
            raise ValueError(
                f"Data leakage temporale! {curr_set} max ({max_curr}) > {next_set} min ({min_next})"
            )
    print("  ✓ Nessun data leakage temporale rilevato tra i set")


def save_splits(splits: dict, split_info: dict) -> None:
    """Salva i tre split su disco in CSV e le relative informazioni in JSON."""
    CSV_PROCESSED.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    file_map = {
        "train": TRAIN_CSV,
        "val":   VAL_CSV,
        "test":  TEST_CSV,
    }
    for name, split_df in splits.items():
        csv_path = file_map[name]
        split_df.to_csv(csv_path, index=False)
        print(f"  {name:>5s} salvato: {csv_path}  ({len(split_df):,} righe)")

    with open(SPLIT_INFO_FILE, "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2, ensure_ascii=False)
    print(f"  Info split salvate: {SPLIT_INFO_FILE}")


def print_split_summary(split_info: dict) -> None:
    """Stampa a video il prospetto temporale dei tre set."""
    print("\n  ── Riepilogo Split Temporale MovieLens ──")
    print(f"  {'Set':<8} {'Righe':>8}  {'%':>6}  {'Da':<22} {'A':<22}  {'Neg (0)':>8}  {'Pos (1)':>8}")
    print("  " + "-" * 90)
    for name in ["train", "val", "test"]:
        info = split_info[name]
        neg = info.get("n_negativi", "—")
        pos = info.get("n_positivi", "—")
        print(
            f"  {name:<8} {info['n_samples']:>8,}  "
            f"{info['percentuale']:>5.1f}%  "
            f"{info['timestamp_inizio'][:19]:<22} "
            f"{info['timestamp_fine'][:19]:<22}  "
            f"{neg:>8,}  {pos:>8,}"
        )
    print()


def split_temporal_dataset(events_path: Path = EVENTS_CSV):
    """Esegue la separazione temporale del subset binario."""
    print("=" * 60)
    print("SEPARAZIONE TEMPORALE (TRAIN / VAL / TEST)")
    print("=" * 60)
    print(f"  Caricamento eventi da: {events_path}")
    df = pd.read_csv(events_path, parse_dates=["timestamp"])

    binary_df = df[df["label"] != "neutrale"].copy()
    if "binary_label" not in binary_df.columns:
        binary_df["binary_label"] = binary_df["label"].map(BINARY_LABEL_MAP).astype(int)

    print(f"  Subset binario estratto da events: {len(binary_df):,} righe")
    splits, split_info = temporal_split(binary_df)
    verify_no_leakage(splits)
    print_split_summary(split_info)
    save_splits(splits, split_info)
    print()
    return splits


# ══════════════════════════════════════════════════════════════════════════════
# 5. COSTRUZIONE DATASET FEATURE (LINK & SIGN)
# ══════════════════════════════════════════════════════════════════════════════

def build_link_datasets(
    train_path: Path = TRAIN_CSV,
    val_path: Path = VAL_CSV,
    test_path: Path = TEST_CSV,
    events_path: Path = EVENTS_CSV,
    ratio: float = 1.0,
) -> dict:
    """Costruisce i dataset per Link Prediction con negative sampling causale."""
    t_start = time.time()
    print("=" * 60)
    print("CREAZIONE DATASET LINK PREDICTION (Stadio 1)")
    print("=" * 60)
    print(f"  Feature impiegate ({len(LINK_FEATURE_COLS)}):")
    for f in LINK_FEATURE_COLS:
        print(f"    - {f}")

    print("\n1. Caricamento split di base ed eventi...")
    splits = {
        "train": load_dataset(train_path),
        "val":   load_dataset(val_path),
        "test":  load_dataset(test_path),
    }
    events_df = load_dataset(events_path)

    print("\n2. Costruzione dataset Link Prediction...")
    link_datasets = {}
    seeds = {"train": 42, "val": 43, "test": 44}

    for name in ["train", "val", "test"]:
        curr_ratio = ratio if name == "train" else 0.0
        link_datasets[name] = build_link_dataset(
            name=name,
            split_df=splits[name],
            events_df=events_df,
            ratio=curr_ratio,
            seed=seeds[name],
        )

    print("\n3. Salvataggio su disco (CSV)...")
    save_map = {
        "train": TRAIN_LINK_CSV,
        "val":   VAL_LINK_CSV,
        "test":  TEST_LINK_CSV,
    }
    for name, df in link_datasets.items():
        save_split_csv(df, save_map[name], "Link Stage", name)

    print(f"\nDataset Link Prediction completati in {time.time() - t_start:.1f}s!\n")
    return link_datasets


def build_sign_datasets(
    train_path: Path = TRAIN_CSV,
    val_path: Path = VAL_CSV,
    test_path: Path = TEST_CSV,
    events_path: Path = EVENTS_CSV,
) -> dict:
    """Costruisce i dataset per Sign Prediction con Signed SVD e feature valutative."""
    t_start = time.time()
    print("=" * 60)
    print("CREAZIONE DATASET SIGN PREDICTION (Stadio 2)")
    print("=" * 60)
    print(f"  Feature specifiche per il segno ({len(SIGN_FEATURE_COLS)}):")
    for f in SIGN_FEATURE_COLS:
        print(f"    - {f}")

    print("\n1. Caricamento split di base ed eventi...")
    splits = {
        "train": load_dataset(train_path),
        "val":   load_dataset(val_path),
        "test":  load_dataset(test_path),
    }
    events_df = load_dataset(events_path)

    print("\n2. Calcolo fattori latenti Signed SVD su train...")
    user_vecs, movie_vecs = train_signed_svd_embeddings(splits["train"])

    print("\n3. Estrazione causale feature di segno su train, val e test...")
    sign_datasets = {}
    for name in ["train", "val", "test"]:
        sign_datasets[name] = build_sign_dataset(
            name=name,
            split_df=splits[name],
            events_df=events_df,
            user_vecs=user_vecs,
            movie_vecs=movie_vecs,
        )

    print("\n4. Salvataggio su disco (CSV)...")
    save_map = {
        "train": TRAIN_SIGN_CSV,
        "val":   VAL_SIGN_CSV,
        "test":  TEST_SIGN_CSV,
    }
    for name, df in sign_datasets.items():
        save_split_csv(df, save_map[name], "Sign Stage Dedicato", name)

    print(f"\nDataset Sign Prediction completati in {time.time() - t_start:.1f}s!\n")
    return sign_datasets


# ══════════════════════════════════════════════════════════════════════════════
# 6. PIPELINE COMPLETA DATASET E CLASSE TRADITIONALDATASET
# ══════════════════════════════════════════════════════════════════════════════

def build_all_datasets():
    """Esegue l'intera pipeline di preparazione del dataset dall'acquisizione alle feature."""
    start_total = time.time()
    print("=" * 60)
    print("AVVIO PIPELINE COMPLETA DATASET TRADIZIONALE (MovieLens 10M)")
    print("=" * 60)

    # 1. Acquisizione
    t0 = time.time()
    acquire_data()
    print(f"--> Fase 1 (Acquisizione) completata in {time.time() - t0:.1f}s\n")

    # 2. Pulizia ed esplorazione
    t0 = time.time()
    clean_data()
    print(f"--> Fase 2 (Pulizia) completata in {time.time() - t0:.1f}s\n")

    # 3. Grafo ed etichette
    t0 = time.time()
    build_graph_and_labels()
    print(f"--> Fase 3 (Grafo & Etichette) completata in {time.time() - t0:.1f}s\n")

    # 4. Split temporale
    t0 = time.time()
    split_temporal_dataset()
    print(f"--> Fase 4 (Split Temporale) completata in {time.time() - t0:.1f}s\n")

    # 5. Feature Link Prediction
    t0 = time.time()
    build_link_datasets()
    print(f"--> Fase 5 (Feature Link Prediction) completata in {time.time() - t0:.1f}s\n")

    # 6. Feature Sign Prediction
    t0 = time.time()
    build_sign_datasets()
    print(f"--> Fase 6 (Feature Sign Prediction) completata in {time.time() - t0:.1f}s\n")

    elapsed = time.time() - start_total
    print("=" * 60)
    print(f"PIPELINE DATASET COMPLETATA CON SUCCESSO in {elapsed:.1f}s")
    print("=" * 60)
    print("\nFile generati in data/processed/csv/:")
    print("  1. raw_ratings.csv   (Acquisizione)")
    print("  2. clean_ratings.csv (Pulizia)")
    print("  3. events.csv        (Grafo ed etichette)")
    print("  4. train.csv, val.csv, test.csv (Split temporale)")
    print("  5. train_link.csv, val_link.csv, test_link.csv (Link Prediction)")
    print("  6. train_sign.csv, val_sign.csv, test_sign.csv (Sign Prediction)")


class TraditionalDataset:
    """
    Gestore unificato del dataset per l'approccio tradizionale tabulare a due stadi.
    Analogamente a BipartiteGraphDataset per GNN, incapsula caricamento e costruzione dei dati.
    """

    def __init__(
        self,
        train_link_csv: Path = TRAIN_LINK_CSV,
        val_link_csv: Path = VAL_LINK_CSV,
        test_link_csv: Path = TEST_LINK_CSV,
        train_sign_csv: Path = TRAIN_SIGN_CSV,
        val_sign_csv: Path = VAL_SIGN_CSV,
        test_sign_csv: Path = TEST_SIGN_CSV,
    ):
        self.train_link_csv = Path(train_link_csv)
        self.val_link_csv = Path(val_link_csv)
        self.test_link_csv = Path(test_link_csv)
        self.train_sign_csv = Path(train_sign_csv)
        self.val_sign_csv = Path(val_sign_csv)
        self.test_sign_csv = Path(test_sign_csv)

    def is_prepared(self) -> bool:
        """Verifica se tutti i file CSV elaborati sono presenti su disco."""
        files = [
            self.train_link_csv, self.val_link_csv, self.test_link_csv,
            self.train_sign_csv, self.val_sign_csv, self.test_sign_csv,
        ]
        return all(f.exists() for f in files)

    def load_link_data(self) -> Tuple[Tuple[pd.DataFrame, pd.Series], Tuple[pd.DataFrame, pd.Series], Tuple[pd.DataFrame, pd.Series]]:
        """Carica train, val e test per lo Stadio 1 (Link Prediction)."""
        return (
            load_link_data(self.train_link_csv),
            load_link_data(self.val_link_csv),
            load_link_data(self.test_link_csv),
        )

    def load_sign_data(self) -> Tuple[Tuple[pd.DataFrame, pd.Series], Tuple[pd.DataFrame, pd.Series], Tuple[pd.DataFrame, pd.Series]]:
        """Carica train, val e test per lo Stadio 2 (Sign Prediction)."""
        return (
            load_sign_data(self.train_sign_csv),
            load_sign_data(self.val_sign_csv),
            load_sign_data(self.test_sign_csv),
        )

    def build_all(self):
        """Costruisce tutti i file della pipeline."""
        build_all_datasets()
