"""
Step 6 – Pipeline Estrazione Feature per Link Prediction (MovieLens 10M).

Modulo compatto e modulare per:
  1. Negative sampling e creazione dataset Link Prediction (train_link, val_link, test_link).
  2. Estrazione causale online delle 7 feature essenziali per lo Stadio 1 (Link Prediction).
  3. Salvataggio su disco dei dataset pronti in data/processed/csv/.

Uso:
  python step6_features.py
"""

from pathlib import Path
import time

from config import (
    EVENTS_CSV,
    TRAIN_CSV,
    VAL_CSV,
    TEST_CSV,
    TRAIN_LINK_CSV,
    VAL_LINK_CSV,
    TEST_LINK_CSV,
)
from features.common import load_dataset, save_split_csv
from features.link_features import (
    LINK_FEATURE_COLS,
    TARGET_COL,
    extract_link_features,
    sample_negative_pairs,
    build_link_dataset,
    load_link_data,
)

# Alias per retrocompatibilità
ALL_FEATURE_COLS = LINK_FEATURE_COLS
build_split_datasets = build_link_dataset

__all__ = [
    "LINK_FEATURE_COLS",
    "ALL_FEATURE_COLS",
    "TARGET_COL",
    "extract_link_features",
    "sample_negative_pairs",
    "build_split_datasets",
    "build_link_dataset",
    "load_link_data",
    "load_dataset",
    "run",
]


def run(
    train_path: Path = TRAIN_CSV,
    val_path: Path = VAL_CSV,
    test_path: Path = TEST_CSV,
    events_path: Path = EVENTS_CSV,
    ratio: float = 1.0,
):
    """Esegue la pipeline completa di Step 6: creazione dataset per Link Prediction."""
    t_start = time.time()
    print("=" * 60)
    print("STEP 6 – Estrazione Feature e Creazione Dataset (Link Prediction)")
    print("=" * 60)
    print(f"  Feature impiegate ({len(LINK_FEATURE_COLS)}):")
    for f in LINK_FEATURE_COLS:
        print(f"    - {f}")

    print("\n1. Caricamento split di base ed eventi...")
    splits = {
        "train": load_dataset(train_path),
        "val":   load_dataset(val_path),
        "test":  load_dataset(test_path),
    }
    events_df = load_dataset(events_path)

    print("\n2. Costruzione dataset Link Prediction...")
    link_datasets = {}
    seeds = {"train": 42, "val": 43, "test": 44}

    for name in ["train", "val", "test"]:
        # Train usa negative sampling; Val e Test usano esclusivamente dati veri
        curr_ratio = ratio if name == "train" else 0.0
        link_datasets[name] = build_link_dataset(
            name=name,
            split_df=splits[name],
            events_df=events_df,
            ratio=curr_ratio,
            seed=seeds[name],
        )

    print("\n3. Salvataggio su disco (CSV)...")
    save_map = {
        "train": TRAIN_LINK_CSV,
        "val":   VAL_LINK_CSV,
        "test":  TEST_LINK_CSV,
    }
    for name, df in link_datasets.items():
        save_split_csv(df, save_map[name], "Link Stage", name)

    print(f"\nStep 6 completato con successo in {time.time() - t_start:.1f}s!\n")
    return link_datasets


if __name__ == "__main__":
    link_datasets = run()
