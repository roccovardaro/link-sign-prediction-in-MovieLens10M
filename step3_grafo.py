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
    EVENTS_TABLE_FILE,
    GRAPH_EDGES_FILE,
    CLEAN_REVIEWS_FILE, CSV_PROCESSED,
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
    Costruisce la lista di archi del grafo bipartito.
    Ogni arco collega un nodo utente a un nodo prodotto.
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


def save_outputs(events:pd.DataFrame, edges:pd.DataFrame, clean_df:pd.DataFrame):
    """Salva tutti gli output su disco in formato Parquet."""
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    CSV_PROCESSED.mkdir(parents=True, exist_ok=True)

    events.to_csv(CSV_PROCESSED / "events.csv")
    edges.to_csv(CSV_PROCESSED / "edges.csv")
    clean_df.to_csv(CSV_PROCESSED / "clean_reviews.csv")

    events.to_parquet(EVENTS_TABLE_FILE, index=False)
    print(f"  Tabella eventi salvata:  {EVENTS_TABLE_FILE}")

    edges.to_parquet(GRAPH_EDGES_FILE, index=False)
    print(f"  Archi del grafo salvati: {GRAPH_EDGES_FILE}")

    clean_df.to_parquet(CLEAN_REVIEWS_FILE, index=False)
    print(f"  Recensioni pulite:       {CLEAN_REVIEWS_FILE}")


def run(df):
    """Esegue lo step 3 completo."""
    print("=" * 60)
    print("STEP 3 – Definizione del grafo e delle etichette")
    print("=" * 60)

    df = assign_labels(df)

    events = build_events_table(df)
    edges = build_graph_edges(df)
    binary_df = build_binary_subset(df)

    save_outputs(events, edges, df)

    print()
    return df, events, edges, binary_df


if __name__ == "__main__":
    from step1_acquisizione import run as step1_run
    from step2_pulizia import run as step2_run

    df = step1_run()
    df, stats = step2_run(df)
    df, events, edges, binary_df = run(df)
