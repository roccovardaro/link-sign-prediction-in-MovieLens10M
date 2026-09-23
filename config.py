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

# ── File CSV utili (uno per ogni fase, nessuna ridondanza) ───────────────
RAW_REVIEWS_CSV   = CSV_PROCESSED / "raw_reviews.csv"
CLEAN_REVIEWS_CSV = CSV_PROCESSED / "clean_reviews.csv"
EVENTS_CSV        = CSV_PROCESSED / "events.csv"
TRAIN_CSV         = CSV_PROCESSED / "train.csv"
VAL_CSV           = CSV_PROCESSED / "val.csv"
TEST_CSV          = CSV_PROCESSED / "test.csv"

# ── File Parquet ────────────────────────────────────────────────────────
EVENTS_TABLE_FILE = DATA_PROCESSED / "events.parquet"
TRAIN_FILE        = DATA_PROCESSED / "train.parquet"
VAL_FILE          = DATA_PROCESSED / "val.parquet"
TEST_FILE         = DATA_PROCESSED / "test.parquet"

# ── File risultati e statistiche ────────────────────────────────────────
STATS_FILE      = RESULTS_DIR / "exploration_stats.json"
SPLIT_INFO_FILE = RESULTS_DIR / "split_info.json"

# ── TF-IDF (Step 9) ─────────────────────────────────────────────────────
TFIDF_MAX_FEATURES = 20_000
TFIDF_MIN_DF = 2
TFIDF_SUBLINEAR_TF = True
VECTORIZER_FILE = DATA_PROCESSED / "tfidf_vectorizer.joblib"
