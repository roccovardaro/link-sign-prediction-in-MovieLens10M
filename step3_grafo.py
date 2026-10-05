"""
Step 3 – Definizione del grafo bipartito e delle etichette (MovieLens 10M).

Costruisce:
  1. Tabella degli eventi (tutte le interazioni utente-film con voto, data, titolo, generi)
  2. Grafo bipartito utente-film con etichette di gradimento
  3. Subset binario per addestramento (esclude neutrale: negativa=0, positiva=1)
"""

from pathlib import Path
import pandas as pd

from config import (
    rating_to_sentiment,
    BINARY_LABEL_MAP,
    DATA_PROCESSED,
    CSV_PROCESSED,
    CLEAN_RATINGS_CSV,
    EVENTS_CSV,
    EVENTS_TABLE_FILE,
)


def assign_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assegna l'etichetta di sentiment in base al rating:
      <= 2.5 stelle → negativa
      3.0 - 3.5     → neutrale
      >= 4.0 stelle → positiva
    """
    df["label"] = df["rating"].apply(rating_to_sentiment)
    counts = df["label"].value_counts()
    print("  Distribuzione etichette:")
    for lbl in ["negativa", "neutrale", "positiva"]:
        print(f"    {lbl:<10s}: {counts.get(lbl, 0):,}")
    return df


def build_events_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Conserva la tabella degli eventi con tutte le informazioni ordinate per timestamp.
    """
    cols = ["user_id", "movie_id", "rating", "label", "timestamp"]
    if "title" in df.columns:
        cols.append("title")
    if "genres" in df.columns:
        cols.append("genres")

    events = df[cols].copy()
    events = events.sort_values("timestamp").reset_index(drop=True)
    return events


def build_graph_edges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Costruisce la vista degli archi del grafo bipartito (src=user_id, dst=movie_id).
    """
    edges = df[["user_id", "movie_id", "rating", "label"]].copy()
    edges = edges.rename(columns={"user_id": "src", "movie_id": "dst"})

    n_users = edges["src"].nunique()
    n_movies = edges["dst"].nunique()
    n_edges = len(edges)

    print(f"\n  ── Grafo Bipartito Utente-Film ──")
    print(f"  Nodi utente: {n_users:,}")
    print(f"  Nodi film:   {n_movies:,}")
    print(f"  Archi totali:{n_edges:,}")

    return edges


def build_binary_subset(df: pd.DataFrame, binary_map: dict = BINARY_LABEL_MAP) -> pd.DataFrame:
    """
    Crea il subset per il classificatore binario:
      - Esclude valutazioni neutrali
      - Assegna etichetta binaria (0 = negativa, 1 = positiva)
    """
    binary_df = df[df["label"] != "neutrale"].copy()
    binary_df["binary_label"] = binary_df["label"].map(binary_map).astype(int)

    n_neg = (binary_df["binary_label"] == 0).sum()
    n_pos = (binary_df["binary_label"] == 1).sum()

    print(f"\n  ── Subset Binario (no neutrale) ──")
    print(f"  Totale istanze: {len(binary_df):,}")
    print(f"  Negativi (0):   {n_neg:,} ({n_neg / len(binary_df) * 100:.1f}%)")
    print(f"  Positivi (1):   {n_pos:,} ({n_pos / len(binary_df) * 100:.1f}%)")

    return binary_df


def save_outputs(events: pd.DataFrame) -> None:
    """Salva la tabella degli eventi in CSV e Parquet."""
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    CSV_PROCESSED.mkdir(parents=True, exist_ok=True)

    events.to_csv(EVENTS_CSV, index=False)
    print(f"  Tabella eventi CSV salvata:     {EVENTS_CSV}")

    events.to_parquet(EVENTS_TABLE_FILE, index=False)
    print(f"  Tabella eventi Parquet salvata: {EVENTS_TABLE_FILE}")


def run(input_path: Path = CLEAN_RATINGS_CSV):
    """Esegue lo Step 3 completo: etichette e grafo, salvando events.csv."""
    print("=" * 60)
    print("STEP 3 – Definizione del grafo bipartito e delle etichette")
    print("=" * 60)

    print(f"  Caricamento valutazioni pulite da: {input_path}")
    df = pd.read_csv(input_path, parse_dates=["timestamp"])

    df = assign_labels(df)
    events = build_events_table(df)
    edges = build_graph_edges(df)
    binary_df = build_binary_subset(df)

    save_outputs(events)
    print()
    return df, events, edges, binary_df


if __name__ == "__main__":
    df, events, edges, binary_df = run()
