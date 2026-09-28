"""
Step 6 – Estrazione delle caratteristiche strutturali con NetworkX.

Per ogni coppia (utente, prodotto) costruisce incrementalmente un grafo
bipartito NetworkX con gli eventi passati (timestamp < corrente) e calcola
le feature strutturali usando le API e gli algoritmi di NetworkX.

Le stelle delle recensioni passate sono input leciti;
la stella della recensione corrente è l'etichetta da predire → esclusa.

Feature calcolate:
  Lato nodo (utente e prodotto):
    - degree:              G.degree()          – n. vicini distinti
    - n_reviews:           contatore incrementale – n. totale recensioni
    - mean_rating:         media incrementale dai contatori
    - n_pos / n_neg / n_neu + frazioni: contatori incrementali
    - days_since_last:     recency dall'ultimo evento

  Feature di coppia (utente-prodotto):
    - already_connected:   G.has_edge()           – se la coppia ha già interagito

Casi con storia insufficiente (nodo assente): conteggi = 0, medie = NaN.
"""

import numpy as np
import pandas as pd
import networkx as nx

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
    # Utente
    "user_degree", "user_n_reviews", "user_mean_rating",
    "user_n_pos", "user_n_neg", "user_n_neu",
    "user_frac_pos", "user_frac_neg", "user_frac_neu",
    "user_days_since_last",
    # Prodotto
    "prod_degree", "prod_n_reviews", "prod_mean_rating",
    "prod_n_pos", "prod_n_neg", "prod_n_neu",
    "prod_frac_pos", "prod_frac_neg", "prod_frac_neu",
    "prod_days_since_last",
    # Coppia
    "already_connected",
]


# ── Statistiche incrementali per nodo ────────────────────────────────────

class NodeStats:
    """Contatori accumulati per un nodo. Ogni update è O(1)."""
    __slots__ = ("n_reviews", "rating_sum", "n_pos", "n_neg", "n_neu", "last_ts")

    def __init__(self):
        self.n_reviews = 0
        self.rating_sum = 0.0
        self.n_pos = 0
        self.n_neg = 0
        self.n_neu = 0
        self.last_ts = None

    def update(self, rating, label, timestamp):
        self.n_reviews += 1
        self.rating_sum += rating
        if label == "positiva":
            self.n_pos += 1
        elif label == "negativa":
            self.n_neg += 1
        else:
            self.n_neu += 1
        if self.last_ts is None or timestamp > self.last_ts:
            self.last_ts = timestamp


# ── Aggiunta incrementale di un evento al grafo ─────────────────────────

def add_event(G, node_stats, user_id, parent_asin, rating, label, timestamp):
    """Aggiunge un evento al grafo bipartito e aggiorna le statistiche."""
    u = f"u_{user_id}"
    p = f"p_{parent_asin}"

    if u not in G:
        G.add_node(u, bipartite=0)
    if p not in G:
        G.add_node(p, bipartite=1)

    if not G.has_edge(u, p):
        G.add_edge(u, p)

    for nd in (u, p):
        if nd not in node_stats:
            node_stats[nd] = NodeStats()
        node_stats[nd].update(rating, label, timestamp)


# ── Estrazione feature per un nodo ───────────────────────────────────────

def _node_features(G, node_id, current_ts, prefix, node_stats):
    """Feature di un nodo: tutto O(1) grazie ai contatori incrementali."""
    if node_id not in node_stats:
        return {
            f"{prefix}degree": 0, f"{prefix}n_reviews": 0,
            f"{prefix}mean_rating": np.nan,
            f"{prefix}n_pos": 0, f"{prefix}n_neg": 0, f"{prefix}n_neu": 0,
            f"{prefix}frac_pos": np.nan, f"{prefix}frac_neg": np.nan,
            f"{prefix}frac_neu": np.nan,
            f"{prefix}days_since_last": np.nan,
        }

    s = node_stats[node_id]
    n = s.n_reviews
    return {
        f"{prefix}degree": G.degree(node_id),
        f"{prefix}n_reviews": n,
        f"{prefix}mean_rating": round(s.rating_sum / n, 4),
        f"{prefix}n_pos": s.n_pos,
        f"{prefix}n_neg": s.n_neg,
        f"{prefix}n_neu": s.n_neu,
        f"{prefix}frac_pos": round(s.n_pos / n, 4),
        f"{prefix}frac_neg": round(s.n_neg / n, 4),
        f"{prefix}frac_neu": round(s.n_neu / n, 4),
        f"{prefix}days_since_last": round(
            (current_ts - s.last_ts).total_seconds() / 86400.0, 2
        ),
    }


# ── Feature di coppia ────────────────────────────────────────────────────

def _pair_features(G, u_node, p_node):
    """Connessione precedente tra utente e prodotto (1-hop)."""
    connected = 1 if (u_node in G and p_node in G and G.has_edge(u_node, p_node)) else 0
    return {"already_connected": connected}


# ── Calcolo feature su un intero split ───────────────────────────────────

def extract_features_for_split(split_df, events_df):
    """
    Calcola le feature strutturali per ogni riga di un split.

    Il grafo viene costruito incrementalmente; le statistiche per nodo
    vengono mantenute con contatori O(1). Nessun calcolo costoso
    (clustering, avg_neighbor_degree) — solo operazioni veloci.
    """
    events_df = events_df.copy()
    split_df = split_df.copy()
    events_df["timestamp"] = pd.to_datetime(events_df["timestamp"], format="ISO8601")
    split_df["timestamp"] = pd.to_datetime(split_df["timestamp"], format="ISO8601")
    events_df = events_df.sort_values("timestamp").reset_index(drop=True)
    split_df = split_df.sort_values("timestamp").reset_index(drop=True)

    # Pre-conversione in array NumPy (evita overhead di iloc)
    ev_user = events_df["user_id"].values
    ev_asin = events_df["parent_asin"].values
    ev_rating = events_df["rating"].values
    ev_label = events_df["label"].values
    ev_ts = events_df["timestamp"].values

    G = nx.Graph()
    node_stats = {}
    records = []
    events_idx = 0
    n_events = len(events_df)
    n_split = len(split_df)

    for i, row in split_df.iterrows():
        ts = row["timestamp"]

        # ── Aggiorna grafo con eventi passati (ts < corrente) ──
        while events_idx < n_events:
            t = pd.Timestamp(ev_ts[events_idx])
            if t >= ts:
                break
            add_event(G, node_stats,
                      ev_user[events_idx], ev_asin[events_idx],
                      ev_rating[events_idx], ev_label[events_idx], t)
            events_idx += 1

        # ── Estrai feature ──
        u = f"u_{row['user_id']}"
        p = f"p_{row['parent_asin']}"

        feats = _node_features(G, u, ts, "user_", node_stats)
        feats.update(_node_features(G, p, ts, "prod_", node_stats))
        feats.update(_pair_features(G, u, p))
        records.append(feats)

        if (i + 1) % 50_000 == 0:
            print(f"    ... {i + 1:,}/{n_split:,} righe "
                  f"(grafo: {G.number_of_nodes():,} nodi, "
                  f"{G.number_of_edges():,} archi)")

    # Rimuove vecchie feature strutturali se presenti (inclusa common_neighbors se presente da run precedenti)
    cleanup_cols = list(set(STRUCTURAL_FEATURES + ["common_neighbors"]))
    cols_to_drop = [c for c in cleanup_cols if c in split_df.columns]
    if cols_to_drop:
        split_df = split_df.drop(columns=cols_to_drop)

    features_df = pd.DataFrame(records, index=split_df.index)
    return pd.concat([split_df, features_df], axis=1)


def print_feature_summary(df, split_name):
    """Stampa un riepilogo delle feature calcolate per uno split."""
    print(f"\n  ── Feature strutturali: {split_name} ──")
    print(f"  Righe: {len(df):,}")
    no_user = (df["user_n_reviews"] == 0).sum()
    no_prod = (df["prod_n_reviews"] == 0).sum()
    print(f"  Senza storia utente:   {no_user:,} ({no_user / len(df) * 100:.1f}%)")
    print(f"  Senza storia prodotto: {no_prod:,} ({no_prod / len(df) * 100:.1f}%)")
    for col in ["user_degree", "prod_degree", "already_connected"]:
        if col in df.columns:
            print(f"  {col}: media={df[col].mean():.4f}, "
                  f"mediana={df[col].median():.4f}")


def save_enriched_splits(splits):
    """Salva i split arricchiti nei file CSV e Parquet."""
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    CSV_PROCESSED.mkdir(parents=True, exist_ok=True)
    file_map = {
        "train": (TRAIN_FILE, TRAIN_CSV),
        "val":   (VAL_FILE, VAL_CSV),
        "test":  (TEST_FILE, TEST_CSV),
    }
    for name, split_df in splits.items():
        pq, csv = file_map[name]
        split_df.to_parquet(pq, index=False)
        split_df.to_csv(csv, index=False)
        print(f"  {name:>5s} salvato: {csv}")


# ── Entry point ──────────────────────────────────────────────────────────

def run(train_path=TRAIN_CSV, val_path=VAL_CSV, test_path=TEST_CSV,
        events_path=EVENTS_CSV):
    """Esegue lo step 6: estrazione feature strutturali per ogni split."""
    print("=" * 60)
    print("STEP 6 – Estrazione caratteristiche strutturali (NetworkX)")
    print("=" * 60)

    splits = {
        "train": pd.read_csv(train_path, parse_dates=["timestamp"]),
        "val":   pd.read_csv(val_path, parse_dates=["timestamp"]),
        "test":  pd.read_csv(test_path, parse_dates=["timestamp"]),
    }

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
