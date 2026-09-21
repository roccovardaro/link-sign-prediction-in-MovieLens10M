"""
Configurazione centralizzata del progetto.
Tutti i percorsi, nomi di colonne e mapping etichette sono definiti qui.
"""

from pathlib import Path

# ── Percorsi ────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent
DATA_RAW = ROOT_DIR / "data" / "raw"
DATA_PROCESSED = ROOT_DIR / "data" / "processed"
CSV_PROCESSED = ROOT_DIR / "data" / "processed" /"csv"
RESULTS_DIR = ROOT_DIR / "results"

RAW_FILE = DATA_RAW / "Handmade_Products.jsonl"

# ── Colonne da conservare dal file review ────────────────────────────────
KEEP_COLUMNS = ["user_id", "parent_asin", "rating", "title", "text", "timestamp"]

# ── Mapping etichette (rating → sentiment) ───────────────────────────────
LABEL_MAP = {
    1: "negativa",
    2: "negativa",
    3: "neutrale",
    4: "positiva",
    5: "positiva",
}

# Mapping binario per il classificatore (esclude neutrale)
BINARY_LABEL_MAP = {
    "negativa": 0,
    "positiva": 1,
}

# ── Split temporale (proporzioni) ────────────────────────────────────────
# Le proporzioni devono sommare a 1.0
SPLIT_RATIOS = {
    "train": 0.70,
    "val":   0.15,
    "test":  0.15,
}

# ── Nomi file di output ─────────────────────────────────────────────────
EVENTS_TABLE_FILE = DATA_PROCESSED / "events.parquet"
GRAPH_EDGES_FILE = DATA_PROCESSED / "graph_edges.parquet"
CLEAN_REVIEWS_FILE = DATA_PROCESSED / "reviews_clean.parquet"
STATS_FILE = RESULTS_DIR / "exploration_stats.json"
SPLIT_INFO_FILE = RESULTS_DIR / "split_info.json"

# File per ogni split
TRAIN_FILE = DATA_PROCESSED / "train.parquet"
VAL_FILE   = DATA_PROCESSED / "val.parquet"
TEST_FILE  = DATA_PROCESSED / "test.parquet"
