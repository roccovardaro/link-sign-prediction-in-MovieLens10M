"""
Step 6.1 – Pipeline Estrazione Feature per Sign Prediction (MovieLens 10M).

Costruisce il dataset dedicato per lo Stadio 2 (Sign Prediction):
  - Modella la polarità dell'interazione (gradimento vs scontento) su link esistenti:
      1 = Valutazione Positiva (rating 4.0 - 5.0)
      0 = Valutazione Negativa (rating 0.5 - 2.5)
  - Allena Signed SVD sui rating centrati del set di addestramento.
  - Estrae online le 15 feature valutative e collaborative per il segno.
  - Salva train_sign.csv, val_sign.csv, test_sign.csv in data/processed/csv/.

Uso:
  python step6_1_sign_features.py
"""

from pathlib import Path
import time

from config import (
    EVENTS_CSV,
    TRAIN_CSV,
    VAL_CSV,
    TEST_CSV,
    TRAIN_SIGN_CSV,
    VAL_SIGN_CSV,
    TEST_SIGN_CSV,
)
from features.common import load_dataset, save_split_csv
from features.sign_features import (
    SIGN_FEATURE_COLS,
    TARGET_COL,
    extract_sign_features,
    train_signed_svd_embeddings,
    compute_signed_svd_similarity,
    build_sign_dataset,
    load_sign_data,
)

# Alias per retrocompatibilità
build_sign_split_datasets = build_sign_dataset

__all__ = [
    "SIGN_FEATURE_COLS",
    "TARGET_COL",
    "extract_sign_features",
    "train_signed_svd_embeddings",
    "compute_signed_svd_similarity",
    "build_sign_split_datasets",
    "build_sign_dataset",
    "load_sign_data",
    "load_dataset",
    "run",
]


def run(
    train_path: Path = TRAIN_CSV,
    val_path: Path = VAL_CSV,
    test_path: Path = TEST_CSV,
    events_path: Path = EVENTS_CSV,
):
    """Esegue la pipeline completa di Step 6.1: creazione dataset per Sign Prediction."""
    t_start = time.time()
    print("=" * 60)
    print("STEP 6.1 – Creazione Dataset Specifico per Sign Prediction")
    print("=" * 60)
    print(f"  Feature specifiche per il segno ({len(SIGN_FEATURE_COLS)}):")
    for f in SIGN_FEATURE_COLS:
        print(f"    - {f}")

    print("\n1. Caricamento split di base ed eventi...")
    splits = {
        "train": load_dataset(train_path),
        "val":   load_dataset(val_path),
        "test":  load_dataset(test_path),
    }
    events_df = load_dataset(events_path)

    print("\n2. Calcolo fattori latenti Signed SVD su train...")
    user_vecs, movie_vecs = train_signed_svd_embeddings(splits["train"])

    print("\n3. Estrazione causale feature di segno su train, val e test...")
    sign_datasets = {}
    for name in ["train", "val", "test"]:
        sign_datasets[name] = build_sign_dataset(
            name=name,
            split_df=splits[name],
            events_df=events_df,
            user_vecs=user_vecs,
            movie_vecs=movie_vecs,
        )

    print("\n4. Salvataggio su disco (CSV)...")
    save_map = {
        "train": TRAIN_SIGN_CSV,
        "val":   VAL_SIGN_CSV,
        "test":  TEST_SIGN_CSV,
    }
    for name, df in sign_datasets.items():
        save_split_csv(df, save_map[name], "Sign Stage Dedicato", name)

    print(f"\nStep 6.1 completato con successo in {time.time() - t_start:.1f}s!\n")
    return sign_datasets


if __name__ == "__main__":
    sign_datasets = run()
