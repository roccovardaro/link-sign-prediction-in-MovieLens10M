"""
Step 5 – Separazione temporale tra addestramento, validazione e test.

Suddivide cronologicamente il dataset binario (no neutrale) in tre finestre:
  - Training:    apprendimento dei modelli
  - Validation:  scelta iperparametri e soglie
  - Test:        valutazione finale (mai usato per scegliere il modello)

Le proporzioni sono definite in config.SPLIT_RATIOS.
Lo split è puramente temporale: nessuna informazione futura filtra nei set precedenti.
Tutte le configurazioni sperimentali devono usare le stesse coppie e gli stessi intervalli.
"""

import json

import pandas as pd

from config import (
    SPLIT_RATIOS,
    SPLIT_INFO_FILE,
    TRAIN_FILE,
    VAL_FILE,
    TEST_FILE,
    DATA_PROCESSED,
    CSV_PROCESSED,
    RESULTS_DIR,
)


def temporal_split(df, ratios=SPLIT_RATIOS):
    """
    Divide il DataFrame in train / val / test in base al timestamp.

    Il dataset deve essere già ordinato cronologicamente.
    I confini temporali vengono calcolati sulle proporzioni indicate.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame ordinato per timestamp, con colonna 'timestamp'.
    ratios : dict
        Proporzioni {'train': float, 'val': float, 'test': float}.

    Returns
    -------
    dict[str, pd.DataFrame]
        Dizionario con chiavi 'train', 'val', 'test'.
    dict
        Informazioni sugli intervalli temporali (per riproducibilità).
    """
    assert abs(sum(ratios.values()) - 1.0) < 1e-9, \
        f"Le proporzioni devono sommare a 1.0, ottenuto {sum(ratios.values())}"

    # Assicura ordinamento cronologico
    df = df.sort_values("timestamp").reset_index(drop=True)
    n = len(df)

    # Calcola i confini (indici)
    n_train = int(n * ratios["train"])
    n_val = int(n * ratios["val"])
    # Il test prende tutto il resto per evitare perdite da arrotondamento
    # n_test = n - n_train - n_val

    splits = {
        "train": df.iloc[:n_train].copy(),
        "val":   df.iloc[n_train:n_train + n_val].copy(),
        "test":  df.iloc[n_train + n_val:].copy(),
    }

    # Calcola informazioni sugli intervalli per riproducibilità
    split_info = {}
    for name, split_df in splits.items():
        split_info[name] = {
            "n_samples": len(split_df),
            "percentuale": round(len(split_df) / n * 100, 2),
            "timestamp_inizio": str(split_df["timestamp"].min()),
            "timestamp_fine": str(split_df["timestamp"].max()),
        }
        # Distribuzione etichette nel subset binario
        if "binary_label" in split_df.columns:
            split_info[name]["n_negativi"] = int(
                (split_df["binary_label"] == 0).sum()
            )
            split_info[name]["n_positivi"] = int(
                (split_df["binary_label"] == 1).sum()
            )

    return splits, split_info


def verify_no_leakage(splits):
    """
    Verifica che non ci sia sovrapposizione temporale tra i set.
    Il timestamp massimo di ogni set deve essere ≤ al minimo del successivo.
    """
    order = ["train", "val", "test"]
    for i in range(len(order) - 1):
        current = order[i]
        next_set = order[i + 1]
        max_curr = splits[current]["timestamp"].max()
        min_next = splits[next_set]["timestamp"].min()
        if max_curr > min_next:
            raise ValueError(
                f"Data leakage! {current} max ({max_curr}) > "
                f"{next_set} min ({min_next})"
            )
    print("  ✓ Nessun data leakage temporale rilevato")


def save_splits(splits, split_info):
    """Salva i tre set e le informazioni di split su disco."""
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    CSV_PROCESSED.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    file_map = {"train": TRAIN_FILE, "val": VAL_FILE, "test": TEST_FILE}

    for name, split_df in splits.items():
        path = file_map[name]
        split_df.to_parquet(path, index=False)
        split_df.to_csv(CSV_PROCESSED / f"{name}.csv", index=False)
        print(f"  {name:>5s} salvato: {path}  ({len(split_df):,} righe)")

    # Salva info per riproducibilità
    with open(SPLIT_INFO_FILE, "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2, ensure_ascii=False)
    print(f"  Info split salvate: {SPLIT_INFO_FILE}")


def print_split_summary(split_info):
    """Stampa un report leggibile dello split."""
    print("\n  ── Riepilogo Split Temporale ──")
    print(f"  {'Set':<10} {'Righe':>8}  {'%':>6}  {'Da':<26} {'A':<26}  {'Neg':>6}  {'Pos':>6}")
    print("  " + "-" * 100)
    for name in ["train", "val", "test"]:
        info = split_info[name]
        neg = info.get("n_negativi", "—")
        pos = info.get("n_positivi", "—")
        print(
            f"  {name:<10} {info['n_samples']:>8,}  "
            f"{info['percentuale']:>5.1f}%  "
            f"{info['timestamp_inizio']:<26} "
            f"{info['timestamp_fine']:<26}  "
            f"{neg:>6}  {pos:>6}"
        )
    print()


# ── Entry point ──────────────────────────────────────────────────────────

def run(binary_df):
    """
    Esegue lo step 5: split temporale del subset binario.

    Parameters
    ----------
    binary_df : pd.DataFrame
        Subset binario (senza recensioni neutrali) dallo step 3.

    Returns
    -------
    dict[str, pd.DataFrame]
        Dizionario con i tre split.
    """
    print("=" * 60)
    print("STEP 5 – Separazione train / validation / test")
    print("=" * 60)

    splits, split_info = temporal_split(binary_df)
    verify_no_leakage(splits)
    print_split_summary(split_info)
    save_splits(splits, split_info)

    print()
    return splits


if __name__ == "__main__":
    from step1_acquisizione import run as step1_run
    from step2_pulizia import run as step2_run
    from step3_grafo import run as step3_run

    df = step1_run()
    df, stats = step2_run(df)
    df, events, edges, binary_df = step3_run(df)
    splits = run(binary_df)
