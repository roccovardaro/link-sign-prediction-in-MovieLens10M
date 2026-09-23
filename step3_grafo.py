"""
Step 3 – Definizione del grafo bipartito e delle etichette.

Costruisce:
  1. Tabella degli eventi (tutte le interazioni con data, testo, rating)
  2. Grafo bipartito utente-prodotto con etichette di sentiment
  3. Subset binario per addestramento (esclude neutrale)
"""

import pandas as pd

from config import (
    LABEL_MAP,
    BINARY_LABEL_MAP,
    DATA_PROCESSED,
    CSV_PROCESSED,
    CLEAN_REVIEWS_CSV,
    EVENTS_CSV,
    EVENTS_TABLE_FILE,
)


def assign_labels(df, label_map=LABEL_MAP):
    """
    Assegna l'etichetta di sentiment in base al rating.

    Mapping:
      1-2 stelle → negativa
      3 stelle   → neutrale
      4-5 stelle → positiva
    """
    df["label"] = df["rating"].map(label_map)
    counts = df["label"].value_counts()
    print("  Distribuzione etichette:")
    for lbl in ["negativa", "neutrale", "positiva"]:
        print(f"    {lbl}: {counts.get(lbl, 0):,}")
    return df


def build_events_table(df):
    """
    Conserva la tabella degli eventi con tutte le informazioni:
    date, testo, rating, e eventuali recensioni ripetute
    (già gestite nello step 2, qui si salva il risultato).
    """
    events = df[["user_id", "parent_asin", "rating", "label",
                  "review_text", "has_text", "timestamp"]].copy()
    events = events.sort_values("timestamp").reset_index(drop=True)
    return events


def build_graph_edges(df):
    """
    Costruisce la vista degli archi del grafo bipartito (src, dst, rating, label).
    Include tutte le recensioni (anche neutrali).
    """
    edges = df[["user_id", "parent_asin", "rating", "label"]].copy()
    edges = edges.rename(columns={"user_id": "src", "parent_asin": "dst"})

    n_users = edges["src"].nunique()
    n_products = edges["dst"].nunique()
    n_edges = len(edges)

    print(f"\n  ── Grafo Bipartito ──")
    print(f"  Nodi utente:   {n_users:,}")
    print(f"  Nodi prodotto: {n_products:,}")
    print(f"  Archi totali:  {n_edges:,}")

    return edges


def build_binary_subset(df, binary_map=BINARY_LABEL_MAP):
    """
    Crea il subset per il classificatore binario del segno:
      - Esclude le recensioni neutrali
      - Assegna etichetta binaria (0 = negativa, 1 = positiva)
    """
    binary_df = df[df["label"] != "neutrale"].copy()
    binary_df["binary_label"] = binary_df["label"].map(binary_map)

    print(f"\n  ── Subset Binario (no neutrale) ──")
    print(f"  Totale: {len(binary_df):,}")
    print(f"  Negativi (0): {(binary_df['binary_label'] == 0).sum():,}")
    print(f"  Positivi (1): {(binary_df['binary_label'] == 1).sum():,}")

    return binary_df


def save_outputs(events: pd.DataFrame):
    """Salva la sola tabella degli eventi su disco in CSV e Parquet."""
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    CSV_PROCESSED.mkdir(parents=True, exist_ok=True)

    events.to_csv(EVENTS_CSV, index=False)
    print(f"  Tabella eventi CSV salvata:     {EVENTS_CSV}")

    events.to_parquet(EVENTS_TABLE_FILE, index=False)
    print(f"  Tabella eventi Parquet salvata: {EVENTS_TABLE_FILE}")


def run(input_path=CLEAN_REVIEWS_CSV):
    """Esegue lo step 3 completo: etichette e grafo, salvando events.csv."""
    print("=" * 60)
    print("STEP 3 – Definizione del grafo e delle etichette")
    print("=" * 60)

    print(f"  Caricamento recensioni pulite da: {input_path}")
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
