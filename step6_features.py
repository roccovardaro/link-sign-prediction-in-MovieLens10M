"""
Step 6 – Estrazione delle caratteristiche strutturali con NetworkX.

Per ogni coppia (utente, prodotto) costruisce incrementalmente un grafo
bipartito NetworkX con gli eventi passati (timestamp < corrente) e calcola
le feature strutturali usando le API e gli algoritmi di NetworkX.

Le stelle delle recensioni passate sono input leciti;
la stella della recensione corrente è l'etichetta da predire → esclusa.

Feature calcolate (tramite NetworkX):
  Lato nodo (utente e prodotto):
    - degree:              nx.Graph.degree()       – n. vicini distinti
    - n_reviews:           somma len(edge_data)     – n. totale recensioni
    - mean_rating:         media dai dati sugli archi
    - n_pos / n_neg / n_neu + frazioni: dai dati sugli archi
    - days_since_last:     recency dall'ultimo arco

  Feature di grafo (coppia utente-prodotto):
    - common_neighbors:    nx.common_neighbors()   – n. vicini in comune (2-hop)
    - user_clustering:     bipartite.clustering()   – coeff. clustering bipartito
    - prod_clustering:     bipartite.clustering()   – coeff. clustering bipartito
    - already_connected:   G.has_edge()             – se la coppia ha già interagito
    - avg_neighbor_degree: nx.average_neighbor_degree() – grado medio dei vicini

Casi con storia insufficiente (nodo assente): conteggi = 0, medie = NaN.
"""

import numpy as np
import pandas as pd
import networkx as nx
from networkx.algorithms import bipartite

from config import (
    DATA_PROCESSED,
    CSV_PROCESSED,
    EVENTS_CSV,
    TRAIN_CSV,
    VAL_CSV,
    TEST_CSV,
    TRAIN_FILE,
    VAL_FILE,
    TEST_FILE,
)

# ── Nomi delle feature generate ──────────────────────────────────────────
STRUCTURAL_FEATURES = [
    # Utente (da NetworkX)
    "user_degree", "user_n_reviews", "user_mean_rating",
    "user_n_pos", "user_n_neg", "user_n_neu",
    "user_frac_pos", "user_frac_neg", "user_frac_neu",
    "user_days_since_last",
    "user_clustering",
    "user_avg_neighbor_deg",
    # Prodotto (da NetworkX)
    "prod_degree", "prod_n_reviews", "prod_mean_rating",
    "prod_n_pos", "prod_n_neg", "prod_n_neu",
    "prod_frac_pos", "prod_frac_neg", "prod_frac_neu",
    "prod_days_since_last",
    "prod_clustering",
    "prod_avg_neighbor_deg",
    # Coppia (da NetworkX)
    "common_neighbors",
    "already_connected",
]

# ── Valori di default per nodi assenti ───────────────────────────────────
_EMPTY_NODE = {
    "degree": 0, "n_reviews": 0, "mean_rating": np.nan,
    "n_pos": 0, "n_neg": 0, "n_neu": 0,
    "frac_pos": np.nan, "frac_neg": np.nan, "frac_neu": np.nan,
    "days_since_last": np.nan,
    "clustering": 0.0, "avg_neighbor_deg": np.nan,
}


# ── Aggiunta incrementale di un evento al grafo ─────────────────────────

def add_event_to_graph(G, user_id, parent_asin, rating, label, timestamp):
    """
    Aggiunge un singolo evento al grafo bipartito in-place.

    Usa G.add_node() con attributo bipartite per mantenere la struttura
    bipartita, e G.add_edge() / aggiornamento attributi per gli archi.
    """
    u_node = f"u_{user_id}"
    p_node = f"p_{parent_asin}"

    if u_node not in G:
        G.add_node(u_node, bipartite=0)
    if p_node not in G:
        G.add_node(p_node, bipartite=1)

    if G.has_edge(u_node, p_node):
        edata = G[u_node][p_node]
        edata["ratings"].append(rating)
        edata["labels"].append(label)
        edata["timestamps"].append(timestamp)
    else:
        G.add_edge(
            u_node, p_node,
            ratings=[rating],
            labels=[label],
            timestamps=[timestamp],
        )


# ── Estrazione feature per un singolo nodo (usa API NetworkX) ────────────

def extract_node_features(G, node_id, current_ts, prefix):
    """
    Estrae feature strutturali per un nodo usando le API di NetworkX:
      - G.degree(node)              → grado
      - G.neighbors(node)           → iteratore sui vicini
      - G[node][neighbor]           → dati dell'arco
      - bipartite.clustering()      → coefficiente di clustering bipartito
      - nx.average_neighbor_degree() → grado medio dei vicini

    Returns
    -------
    dict  con chiavi prefissate (user_ o prod_).
    """
    if node_id not in G:
        return {f"{prefix}{k}": v for k, v in _EMPTY_NODE.items()}

    # ── Grado (NetworkX) ──
    degree = G.degree(node_id)

    # ── Dati dagli archi (iterando sui vicini con NetworkX) ──
    all_ratings = []
    all_labels = []
    all_timestamps = []

    for neighbor in G.neighbors(node_id):
        edata = G[node_id][neighbor]
        all_ratings.extend(edata["ratings"])
        all_labels.extend(edata["labels"])
        all_timestamps.extend(edata["timestamps"])

    n = len(all_ratings)

    # Conteggi sentiment
    n_pos = all_labels.count("positiva")
    n_neg = all_labels.count("negativa")
    n_neu = all_labels.count("neutrale")

    # Recency
    last_ts = max(all_timestamps)
    days_since = (current_ts - last_ts).total_seconds() / 86400.0

    # ── Clustering bipartito (NetworkX) ──
    # bipartite.clustering restituisce un dict {node: coeff}
    try:
        clust = bipartite.clustering(G, [node_id])[node_id]
    except Exception:
        clust = 0.0

    # ── Grado medio dei vicini (NetworkX) ──
    try:
        avg_nd = nx.average_neighbor_degree(G, nodes=[node_id])[node_id]
    except Exception:
        avg_nd = np.nan

    return {
        f"{prefix}degree": degree,
        f"{prefix}n_reviews": n,
        f"{prefix}mean_rating": round(np.mean(all_ratings), 4),
        f"{prefix}n_pos": n_pos,
        f"{prefix}n_neg": n_neg,
        f"{prefix}n_neu": n_neu,
        f"{prefix}frac_pos": round(n_pos / n, 4),
        f"{prefix}frac_neg": round(n_neg / n, 4),
        f"{prefix}frac_neu": round(n_neu / n, 4),
        f"{prefix}days_since_last": round(days_since, 2),
        f"{prefix}clustering": round(clust, 4),
        f"{prefix}avg_neighbor_deg": round(avg_nd, 4) if not np.isnan(avg_nd) else np.nan,
    }


# ── Feature di coppia (usa API NetworkX) ─────────────────────────────────

def extract_pair_features(G, u_node, p_node):
    """
    Estrae feature strutturali per la coppia (utente, prodotto)
    usando le API di NetworkX:
      - nx.common_neighbors(G, u, v)  → vicini in comune (2-hop paths)
      - G.has_edge(u, v)               → connessione precedente

    I common_neighbors in un grafo bipartito sono i nodi dello stesso tipo
    che condividono almeno un vicino con l'altro nodo. Per una coppia
    (utente, prodotto) sono gli utenti/prodotti raggiungibili in 2 hop.
    """
    # Common neighbors: quanti nodi sono vicini di entrambi
    # In un bipartito u-p corretto, common_neighbors è 0 (sono su lati diversi).
    # Usiamo invece i vicini a 2-hop: prodotti in comune tra u e i vicini di p,
    # ovvero utenti che hanno recensito lo stesso prodotto p.
    if u_node in G and p_node in G:
        cn = len(list(nx.common_neighbors(G, u_node, p_node)))
        connected = 1 if G.has_edge(u_node, p_node) else 0
    else:
        cn = 0
        connected = 0

    return {
        "common_neighbors": cn,
        "already_connected": connected,
    }


# ── Calcolo feature su un intero split ───────────────────────────────────

def extract_features_for_split(split_df, events_df):
    """
    Calcola le feature strutturali per ogni riga di un split.

    Il grafo bipartito viene costruito incrementalmente: man mano che
    si avanza cronologicamente, gli eventi passati vengono aggiunti
    al grafo con add_event_to_graph(). Per ogni riga si estraggono
    le feature usando le API di NetworkX.

    Parameters
    ----------
    split_df : pd.DataFrame
        DataFrame dello split (train, val o test).
    events_df : pd.DataFrame
        Tabella completa degli eventi (incluse neutrali), ordinata.

    Returns
    -------
    pd.DataFrame
        split_df con le nuove colonne di feature strutturali.
    """
    events_df = events_df.sort_values("timestamp").reset_index(drop=True)
    split_df = split_df.sort_values("timestamp").reset_index(drop=True)

    records = []
    G = nx.Graph()
    events_idx = 0
    n_events = len(events_df)

    for i, row in split_df.iterrows():
        ts = row["timestamp"]
        uid = row["user_id"]
        pid = row["parent_asin"]

        # ── Aggiorna grafo incrementalmente (solo eventi con ts < corrente) ──
        while events_idx < n_events and events_df.iloc[events_idx]["timestamp"] < ts:
            ev = events_df.iloc[events_idx]
            add_event_to_graph(
                G, ev["user_id"], ev["parent_asin"],
                ev["rating"], ev["label"], ev["timestamp"]
            )
            events_idx += 1

        # ── Estrai feature dal grafo corrente (NetworkX) ──
        u_node = f"u_{uid}"
        p_node = f"p_{pid}"

        user_feats = extract_node_features(G, u_node, ts, "user_")
        prod_feats = extract_node_features(G, p_node, ts, "prod_")
        pair_feats = extract_pair_features(G, u_node, p_node)

        records.append({**user_feats, **prod_feats, **pair_feats})

        # Log progresso
        if (i + 1) % 50_000 == 0:
            print(f"    ... {i + 1:,}/{len(split_df):,} righe "
                  f"(grafo: {G.number_of_nodes():,} nodi, "
                  f"{G.number_of_edges():,} archi)")

    # Rimuove vecchie feature strutturali se già presenti (riesecuzione idempotente)
    cols_to_drop = [c for c in STRUCTURAL_FEATURES if c in split_df.columns]
    if cols_to_drop:
        split_df = split_df.drop(columns=cols_to_drop)

    features_df = pd.DataFrame(records, index=split_df.index)
    result = pd.concat([split_df, features_df], axis=1)
    return result


def print_feature_summary(df, split_name):
    """Stampa un riepilogo delle feature calcolate per uno split."""
    print(f"\n  ── Feature strutturali: {split_name} ──")
    print(f"  Righe: {len(df):,}")

    no_user_history = (df["user_n_reviews"] == 0).sum()
    no_prod_history = (df["prod_n_reviews"] == 0).sum()
    print(f"  Senza storia utente:   {no_user_history:,} "
          f"({no_user_history / len(df) * 100:.1f}%)")
    print(f"  Senza storia prodotto: {no_prod_history:,} "
          f"({no_prod_history / len(df) * 100:.1f}%)")

    for col in ["user_degree", "user_clustering", "user_avg_neighbor_deg",
                "prod_degree", "prod_clustering", "prod_avg_neighbor_deg",
                "common_neighbors"]:
        if col in df.columns:
            print(f"  {col}: media={df[col].mean():.4f}, "
                  f"mediana={df[col].median():.4f}")


def save_enriched_splits(splits):
    """Salva i split arricchiti direttamente nei 3 file CSV/Parquet utili."""
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    CSV_PROCESSED.mkdir(parents=True, exist_ok=True)

    file_map = {
        "train": (TRAIN_FILE, TRAIN_CSV),
        "val":   (VAL_FILE, VAL_CSV),
        "test":  (TEST_FILE, TEST_CSV),
    }

    for name, split_df in splits.items():
        parquet_path, csv_path = file_map[name]
        split_df.to_parquet(parquet_path, index=False)
        split_df.to_csv(csv_path, index=False)
        print(f"  {name:>5s} con feature strutturali salvato: {csv_path}")


# ── Entry point ──────────────────────────────────────────────────────────

def run(train_path=TRAIN_CSV, val_path=VAL_CSV, test_path=TEST_CSV, events_path=EVENTS_CSV):
    """
    Esegue lo step 6: estrazione feature strutturali per ogni split
    tramite grafo bipartito NetworkX.
    """
    print("=" * 60)
    print("STEP 6 – Estrazione caratteristiche strutturali (NetworkX)")
    print("=" * 60)

    print(f"  Caricamento split da: {train_path}, {val_path}, {test_path}")
    splits = {
        "train": pd.read_csv(train_path, parse_dates=["timestamp"]),
        "val":   pd.read_csv(val_path, parse_dates=["timestamp"]),
        "test":  pd.read_csv(test_path, parse_dates=["timestamp"]),
    }

    print(f"  Caricamento eventi da: {events_path}")
    events_df = pd.read_csv(events_path, parse_dates=["timestamp"])

    enriched = {}
    for name in ["train", "val", "test"]:
        print(f"\n  Elaborazione {name}...")
        enriched[name] = extract_features_for_split(splits[name], events_df)
        print_feature_summary(enriched[name], name)

    save_enriched_splits(enriched)

    print()
    return enriched


if __name__ == "__main__":
    enriched = run()
