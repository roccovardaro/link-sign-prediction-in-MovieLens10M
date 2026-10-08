"""
Step 1 – Acquisizione dei dati (MovieLens 10M).

Carica ratings.dat e movies.dat, unisce i metadati dei film (titolo e generi),
converte il timestamp in formato datetime e salva raw_ratings.csv.
"""

from pathlib import Path
import pandas as pd

from config import (
    RATINGS_FILE,
    MOVIES_FILE,
    RAW_RATINGS_CSV,
    SAMPLE_SIZE,
    KEEP_COLUMNS)


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
    sample_size: int = SAMPLE_SIZE,
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

    # Converti timestamp unix in datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def run(
    ratings_path: Path = RATINGS_FILE,
    movies_path: Path = MOVIES_FILE,
    output_csv: Path = RAW_RATINGS_CSV,
    sample_size: int = SAMPLE_SIZE,
) -> pd.DataFrame:
    """Esegue lo Step 1 completo e salva il DataFrame unito in CSV."""
    print("=" * 60)
    print("STEP 1 – Acquisizione dei dati (MovieLens 10M)")
    print("=" * 60)

    df_ratings = load_ratings(ratings_path, sample_size=sample_size)
    df_movies = load_movies(movies_path)

    # Merge delle valutazioni con i metadati dei film
    df = df_ratings.merge(df_movies, on="movie_id", how="left")

    # Ordina colonne secondo KEEP_COLUMNS
    cols = [c for c in KEEP_COLUMNS if c in df.columns]
    df = df[cols].copy()

    print(f"  Righe finali acquisite: {len(df):,}")
    print(f"  Utenti unici:           {df['user_id'].nunique():,}")
    print(f"  Film unici:             {df['movie_id'].nunique():,}")
    print(f"  Colonne:                {list(df.columns)}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"  Dati grezzi salvati in CSV: {output_csv}")
    print()

    return df


if __name__ == "__main__":
    df = run()
    print(df.head())
