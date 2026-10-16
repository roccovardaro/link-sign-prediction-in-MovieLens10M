"""
Step 5 – Separazione temporale tra addestramento, validazione e test (MovieLens 10M).

Suddivide cronologicamente il dataset binario (esclusi i neutrali) in tre finestre:
  - Training (70%):    apprendimento dei modelli
  - Validation (15%):  scelta iperparametri e soglie
  - Test (15%):        valutazione finale out-of-time

Lo split è strettamente temporale: nessuna informazione futura filtra nei set precedenti.
"""

import json
from pathlib import Path
import pandas as pd

from config import (
    SPLIT_RATIOS,
    SPLIT_INFO_FILE,
    TRAIN_CSV,
    VAL_CSV,
    TEST_CSV,
    EVENTS_CSV,
    BINARY_LABEL_MAP,
    DATA_PROCESSED,
    CSV_PROCESSED,
    RESULTS_DIR,
)


def temporal_split(df: pd.DataFrame, ratios: dict = SPLIT_RATIOS) -> tuple[dict, dict]:
    """
    Divide il DataFrame in train / val / test in base al timestamp ordinato.
    """
    assert abs(sum(ratios.values()) - 1.0) < 1e-9, (
        f"Le proporzioni devono sommare a 1.0, ottenuto {sum(ratios.values())}"
    )

    df = df.sort_values("timestamp").reset_index(drop=True)
    n = len(df)

    n_train = int(n * ratios["train"])
    n_val = int(n * ratios["val"])

    splits = {
        "train": df.iloc[:n_train].copy(),
        "val":   df.iloc[n_train:n_train + n_val].copy(),
        "test":  df.iloc[n_train + n_val:].copy(),
    }

    split_info = {}
    for name, split_df in splits.items():
        split_info[name] = {
            "n_samples": len(split_df),
            "percentuale": round(len(split_df) / n * 100, 2),
            "timestamp_inizio": str(split_df["timestamp"].min()),
            "timestamp_fine": str(split_df["timestamp"].max()),
        }
        if "binary_label" in split_df.columns:
            split_info[name]["n_negativi"] = int((split_df["binary_label"] == 0).sum())
            split_info[name]["n_positivi"] = int((split_df["binary_label"] == 1).sum())

    return splits, split_info


def verify_no_leakage(splits: dict) -> None:
    """Verifica l'assenza di sovrapposizione temporale tra train, val e test."""
    order = ["train", "val", "test"]
    for i in range(len(order) - 1):
        curr_set = order[i]
        next_set = order[i + 1]
        max_curr = splits[curr_set]["timestamp"].max()
        min_next = splits[next_set]["timestamp"].min()
        if max_curr > min_next:
            raise ValueError(
                f"Data leakage temporale! {curr_set} max ({max_curr}) > {next_set} min ({min_next})"
            )
    print("  ✓ Nessun data leakage temporale rilevato tra i set")


def save_splits(splits: dict, split_info: dict) -> None:
    """Salva i tre set su disco in CSV."""
    CSV_PROCESSED.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    file_map = {
        "train": TRAIN_CSV,
        "val":   VAL_CSV,
        "test":  TEST_CSV,
    }

    for name, split_df in splits.items():
        csv_path = file_map[name]
        split_df.to_csv(csv_path, index=False)
        print(f"  {name:>5s} salvato: {csv_path}  ({len(split_df):,} righe)")

    with open(SPLIT_INFO_FILE, "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2, ensure_ascii=False)
    print(f"  Info split salvate: {SPLIT_INFO_FILE}")


def print_split_summary(split_info: dict) -> None:
    """Stampa a video il prospetto temporale dei tre set."""
    print("\n  ── Riepilogo Split Temporale MovieLens ──")
    print(f"  {'Set':<8} {'Righe':>8}  {'%':>6}  {'Da':<22} {'A':<22}  {'Neg (0)':>8}  {'Pos (1)':>8}")
    print("  " + "-" * 90)
    for name in ["train", "val", "test"]:
        info = split_info[name]
        neg = info.get("n_negativi", "—")
        pos = info.get("n_positivi", "—")
        print(
            f"  {name:<8} {info['n_samples']:>8,}  "
            f"{info['percentuale']:>5.1f}%  "
            f"{info['timestamp_inizio'][:19]:<22} "
            f"{info['timestamp_fine'][:19]:<22}  "
            f"{neg:>8,}  {pos:>8,}"
        )
    print()


def run(events_path: Path = EVENTS_CSV):
    """Esegue lo Step 5: split temporale del subset binario."""
    print("=" * 60)
    print("STEP 5 – Separazione temporale train / validation / test")
    print("=" * 60)

    print(f"  Caricamento eventi da: {events_path}")
    df = pd.read_csv(events_path, parse_dates=["timestamp"])

    binary_df = df[df["label"] != "neutrale"].copy()
    if "binary_label" not in binary_df.columns:
        binary_df["binary_label"] = binary_df["label"].map(BINARY_LABEL_MAP).astype(int)

    print(f"  Subset binario estratto da events: {len(binary_df):,} righe")

    splits, split_info = temporal_split(binary_df)
    verify_no_leakage(splits)
    print_split_summary(split_info)
    save_splits(splits, split_info)

    print()
    return splits


if __name__ == "__main__":
    splits = run()
