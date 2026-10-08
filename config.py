"""
Configurazione centralizzata del progetto (Dataset: MovieLens 10M).
Tutti i percorsi, nomi di colonne, mapping etichette e parametri sono definiti qui.
"""

from pathlib import Path

# ── Percorsi ────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
MOVIELENS_DIR = DATA_DIR / "ml-10M100K"

# File sorgente MovieLens 10M
RATINGS_FILE = MOVIELENS_DIR / "ratings.dat"
MOVIES_FILE  = MOVIELENS_DIR / "movies.dat"
TAGS_FILE    = MOVIELENS_DIR / "tags.dat"

# Directory output elaborati
DATA_PROCESSED = DATA_DIR / "processed"
CSV_PROCESSED  = DATA_PROCESSED / "csv"
RESULTS_DIR    = ROOT_DIR / "results"

# ── Campionamento ───────────────────────────────────────────────────────
# MovieLens 10M contiene oltre 10 milioni di valutazioni.
# Impostare un intero (es. 100_000) per campionare le righe più recenti o casuali
# in modo da garantire tempi di calcolo rapidi e reattivi, oppure None per l'intero dataset.
SAMPLE_SIZE = 10_000_000

# ── Colonne dataset ─────────────────────────────────────────────────────
# Nomi colonne standardizzati per MovieLens
KEEP_COLUMNS = ["user_id", "movie_id", "rating", "timestamp", "title", "genres"]

# ── Mapping etichette (rating → sentiment/segno) ────────────────────────
# In MovieLens le valutazioni vanno da 0.5 a 5.0 con incrementi di 0.5 stelle:
#   0.5 - 2.5 stelle → negativa
#   3.0 - 3.5 stelle → neutrale
#   4.0 - 5.0 stelle → positiva
def rating_to_sentiment(r: float) -> str:
    if r <= 2.5:
        return "negativa"
    elif r >= 4.0:
        return "positiva"
    else:
        return "neutrale"

# Mapping binario per il classificatore di segno (esclude neutrale)
BINARY_LABEL_MAP = {
    "negativa": 0,
    "positiva": 1,
}

# ── Split temporale (proporzioni) ────────────────────────────────────────
SPLIT_RATIOS = {
    "train": 0.70,
    "val":   0.15,
    "test":  0.15,
}

# ── File CSV pipeline ───────────────────────────────────────────────────
RAW_RATINGS_CSV   = CSV_PROCESSED / "raw_ratings.csv"
CLEAN_RATINGS_CSV = CSV_PROCESSED / "clean_ratings.csv"
EVENTS_CSV        = CSV_PROCESSED / "events.csv"
TRAIN_CSV         = CSV_PROCESSED / "train.csv"
VAL_CSV           = CSV_PROCESSED / "val.csv"
TEST_CSV          = CSV_PROCESSED / "test.csv"

# Dataset dedicati a Link Prediction (coppie positive e negative con link_exists)
TRAIN_LINK_CSV    = CSV_PROCESSED / "train_link.csv"
VAL_LINK_CSV      = CSV_PROCESSED / "val_link.csv"
TEST_LINK_CSV     = CSV_PROCESSED / "test_link.csv"

# Alias per retrocompatibilità
RAW_REVIEWS_CSV   = RAW_RATINGS_CSV
CLEAN_REVIEWS_CSV = CLEAN_RATINGS_CSV

# File Parquet (opzionali per salvataggio compatto)
EVENTS_TABLE_FILE = DATA_PROCESSED / "events.parquet"
TRAIN_FILE        = DATA_PROCESSED / "train.parquet"
VAL_FILE          = DATA_PROCESSED / "val.parquet"
TEST_FILE         = DATA_PROCESSED / "test.parquet"

TRAIN_LINK_FILE   = DATA_PROCESSED / "train_link.parquet"
VAL_LINK_FILE     = DATA_PROCESSED / "val_link.parquet"
TEST_LINK_FILE    = DATA_PROCESSED / "test_link.parquet"

# ── File risultati e statistiche ────────────────────────────────────────
STATS_FILE      = RESULTS_DIR / "exploration_stats.json"
SPLIT_INFO_FILE = RESULTS_DIR / "split_info.json"

# ── Modelli Two-Stage Pipeline ───────────────────────────────────────────
LINK_MODEL_FILE = DATA_PROCESSED / "link_model.joblib"
SIGN_MODEL_FILE = DATA_PROCESSED / "sign_model.joblib"


