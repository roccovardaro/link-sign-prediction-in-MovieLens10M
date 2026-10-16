"""
features/common.py – Utilità comuni per l'ingegneria delle feature (MovieLens 10M).
Contiene costanti condivise, caricamento/salvataggio dati e parsing dei generi.
"""

from pathlib import Path
import time
import numpy as np
import pandas as pd

# Costanti temporali e statistiche
GLOBAL_MEAN_RATING = 3.5
SECONDS_PER_DAY = 86400.0
WINDOW_30D_SEC = 30.0 * SECONDS_PER_DAY

# Mapping dei 19 generi unici di MovieLens a ID interi compatti (0..18)
GENRE_MAP = {
    "Action": 0, "Adventure": 1, "Animation": 2, "Children": 3, "Comedy": 4,
    "Crime": 5, "Documentary": 6, "Drama": 7, "Fantasy": 8, "Film-Noir": 9,
    "Horror": 10, "IMAX": 11, "Musical": 12, "Mystery": 13, "Romance": 14,
    "Sci-Fi": 15, "Thriller": 16, "War": 17, "Western": 18,
}
N_GENRES = len(GENRE_MAP)

_GENRE_TUPLE_CACHE: dict[str, tuple[str, ...]] = {}
_GENRE_ID_CACHE: dict[str, tuple[int, ...]] = {}


def parse_genres_tuple(val) -> tuple[str, ...]:
    """Restituisce la tupla di generi (stringhe) con memorizzazione in cache per evitare riallocazioni."""
    if not isinstance(val, str) or not val or val == "(no genres listed)":
        return ()
    res = _GENRE_TUPLE_CACHE.get(val)
    if res is None:
        res = tuple(g for g in val.split("|") if g and g != "(no genres listed)")
        _GENRE_TUPLE_CACHE[val] = res
    return res


def parse_genres_to_ids(val) -> tuple[int, ...]:
    """Converte la stringa dei generi in una tupla di ID interi (0..18) con memorizzazione in cache."""
    if not isinstance(val, str) or not val or val == "(no genres listed)":
        return ()
    res = _GENRE_ID_CACHE.get(val)
    if res is None:
        res = tuple(GENRE_MAP[g] for g in val.split("|") if g in GENRE_MAP)
        _GENRE_ID_CACHE[val] = res
    return res


def compute_genre_match(user_genres: dict, movie_genres_tuple: tuple[str, ...]) -> float:
    """Calcola la percentuale di generi del film presenti nello storico dell'utente."""
    if not user_genres or not movie_genres_tuple:
        return 0.0
    return sum(1 for g in movie_genres_tuple if g in user_genres) / len(movie_genres_tuple)


def load_dataset(csv_path: Path) -> pd.DataFrame:
    """
    Carica un file CSV in DataFrame Pandas in modo sicuro e ottimizzato.
    Usa low_memory=False per evitare avvisi di tipo misto e converte il timestamp in datetime.
    """
    df = pd.read_csv(csv_path, low_memory=False)
    if "timestamp" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def save_split_csv(df: pd.DataFrame, output_path: Path, stage_label: str, split_name: str) -> None:
    """Salva il DataFrame su disco in formato CSV misurando il tempo impiegato."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    df.to_csv(output_path, index=False)
    elapsed = time.time() - t0
    print(f"  [{stage_label}] {split_name:>5s} salvato ({len(df):,} righe) in {elapsed:.1f}s: {output_path.name}")
