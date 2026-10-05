"""
Step 2 – Pulizia ed esplorazione (MovieLens 10M).

Operazioni:
  - Validazione dei tipi di dati (user_id, movie_id, rating, timestamp)
  - Gestione valori mancanti o rating fuori intervallo [0.5, 5.0]
  - Rimozione duplicati esatti
  - Ordinamento cronologico
  - Calcolo e salvataggio statistiche esplorative del dataset
"""

import json
from pathlib import Path
import pandas as pd

from config import (
    RAW_RATINGS_CSV,
    CLEAN_RATINGS_CSV,
    STATS_FILE,
)


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
    """
    Rimuove record invalidi:
      - user_id o movie_id mancanti (<= 0)
      - rating mancante o non compreso tra 0.5 e 5.0
      - timestamp mancante
    """
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
    """Rimuove eventuali duplicati esatti mantenendo l'ultima occorrenza."""
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

    stats = {
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
    return stats


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


def run(
    input_path: Path = RAW_RATINGS_CSV,
    output_path: Path = CLEAN_RATINGS_CSV,
) -> tuple[pd.DataFrame, dict]:
    """Esegue lo Step 2 completo: pulizia + esplorazione."""
    print("=" * 60)
    print("STEP 2 – Pulizia ed esplorazione")
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


if __name__ == "__main__":
    df, stats = run()
