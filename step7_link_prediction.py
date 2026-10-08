"""
Step 7 – Stadio 1: Link Prediction (Esistenza dell'Interazione nel Grafo).

Obiettivo:
  Dato un utente 'u' e un film 'm', predire se si formerà un'interazione (arco nel grafo bipartito),
  indipendentemente dal gradimento positivo o negativo.

Target binario:
  1 = Link reale (l'interazione si verifica effettivamente nel dataset)
  0 = Link assente (coppia utente-film generata con Negative Sampling)

I dataset con campionamento ed estrazione feature sono generati direttamente nello Step 6.
Questo step carica i dataset pronti, addestra il modello, valuta le metriche e salva l'artefatto.
"""

from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

from config import (
    TRAIN_LINK_CSV,
    VAL_LINK_CSV,
    TEST_LINK_CSV,
    TRAIN_LINK_FILE,
    VAL_LINK_FILE,
    TEST_LINK_FILE,
    LINK_MODEL_FILE,
    EVENTS_CSV,
)

from step6_features import (
    LINK_FEATURE_COLS,
    load_dataset,
    sample_negative_pairs,
    extract_link_features,
)

TARGET_COL = "link_exists"


def generate_negative_samples(df: pd.DataFrame, ratio: float = 0.5, seed: int = 42, events_df: pd.DataFrame = None) -> pd.DataFrame:
    """Helper di utilità per generare campioni negativi con feature causali estratte."""
    events = events_df if events_df is not None else pd.read_csv(EVENTS_CSV)
    negatives = sample_negative_pairs(df, events, ratio=ratio, seed=seed)
    return extract_link_features(negatives, events)


def load_link_data(parquet_path: Path, csv_path: Path):
    """Carica il dataset preparato da Step 6 ed estrae matrice X e target binario."""
    df = load_dataset(parquet_path, csv_path)
    return df[LINK_FEATURE_COLS].fillna(0), df[TARGET_COL].astype(int)


def evaluate(model, X, y, split_name: str, scaler=None) -> dict:
    """Valuta il modello di link prediction calcolando le metriche prestazionali."""
    X_eval = scaler.transform(X) if scaler is not None else X
    y_pred = model.predict(X_eval)
    y_proba = model.predict_proba(X_eval)[:, 1]

    is_binary = len(set(y)) > 1

    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "recall": recall_score(y, y_pred, zero_division=0),
    }

    if is_binary:
        metrics.update({
            "balanced_acc": balanced_accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred, zero_division=0),
            "f1": f1_score(y, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y, y_proba),
            "pr_auc": average_precision_score(y, y_proba),
        })
    else:
        # Metriche sui dati veri senza negativi (tutte istanze reali)
        metrics["detection_rate"] = float((y_pred == 1).mean())
        metrics["mean_link_proba"] = float(y_proba.mean())

    print(f"\n  ── Metriche Link Prediction [{split_name}] ──")
    for k, v in metrics.items():
        print(f"  {k:<16}: {v:.4f}")

    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    print("  Confusion Matrix:")
    if is_binary:
        print(f"    Negativi reali: {cm[0][0]:<7} corretti (TN) | {cm[0][1]:<7} falsi allarmi (FP)")
    print(f"    Positivi reali: {cm[1][0]:<7} mancati (FN)  | {cm[1][1]:<7} intercettati (TP)")
    if not is_binary:
        print(f"    --> Rilevati con successo: {cm[1][1]:,} / {len(y):,} link reali ({cm[1][1] * 100 / max(len(y), 1):.2f}%)")

    return metrics


def run(
    save_model: bool = True,
    model_type: str = "gbm",
):
    """
    Addestra e valuta il modello di Link Prediction:
    - Train: con negative sampling per apprendere a distinguere presenza/assenza di interazione.
    - Val & Test: valutati direttamente sui dati veri (solo link reali, no negative sampling).
    """
    print("=" * 60)
    print("STADIO 1: LINK PREDICTION (Training & Valutazione)")
    print("=" * 60)
    print(f"  Feature impiegate ({len(LINK_FEATURE_COLS)}):")
    for f in LINK_FEATURE_COLS:
        print(f"    - {f}")

    print("\n  Caricamento dataset Link Prediction...")
    X_train, y_train = load_link_data(TRAIN_LINK_FILE, TRAIN_LINK_CSV)
    X_val, y_val     = load_link_data(VAL_LINK_FILE, VAL_LINK_CSV)
    X_test, y_test   = load_link_data(TEST_LINK_FILE, TEST_LINK_CSV)

    print(f"  Train: {len(X_train):,} coppie ({(y_train == 1).sum():,} reali, {(y_train == 0).sum():,} negative)")
    print(f"  Val:   {len(X_val):,} link reali (dati veri, no negative sampling)")
    print(f"  Test:  {len(X_test):,} link reali (dati veri, no negative sampling)")

    scaler = None
    if model_type == "lr":
        print("\n  Addestramento Logistic Regression...")
        scaler = StandardScaler()
        X_train_fit = scaler.fit_transform(X_train)
        model = LogisticRegression(max_iter=500, random_state=42, class_weight="balanced")
        model.fit(X_train_fit, y_train)
    else:
        print("\n  Addestramento HistGradientBoostingClassifier...")
        model = HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.08,
            max_depth=6,
            class_weight="balanced",
            random_state=42,
        )
        model.fit(X_train, y_train)

    evaluate(model, X_val, y_val, "Validation (Dati Veri)", scaler=scaler)
    test_metrics = evaluate(model, X_test, y_test, "Test (Dati Veri)", scaler=scaler)

    if save_model:
        LINK_MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model": model,
            "scaler": scaler,
            "features": LINK_FEATURE_COLS,
            "model_type": model_type,
        }, LINK_MODEL_FILE)
        print(f"\n  Modello Link Prediction salvato in: {LINK_MODEL_FILE}")

    print("\nStadio 1 completato con successo.\n")
    return model, scaler, test_metrics


if __name__ == "__main__":
    model, scaler, metrics = run()
