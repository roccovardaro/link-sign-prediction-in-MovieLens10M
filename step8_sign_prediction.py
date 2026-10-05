"""
Step 8 – Stadio 2: Sign Prediction (Classificazione del Segno dell'Interazione).

Obiettivo:
  Dato un link esistente (o predetto come tale dallo Stadio 1), determinare
  se l'interazione sarà di gradimento o scontento:
    1 = Valutazione Positiva (4 o 5 stelle)
    0 = Valutazione Negativa (0.5 - 2.5 stelle)

Feature impiegate:
  - Storico sentiment utente:   user_mean_rating, user_frac_pos, user_frac_neg
  - Storico sentiment film:     movie_mean_rating, movie_frac_pos, movie_frac_neg
  - Attività e recency:         user_n_reviews, movie_n_reviews,
                                user_days_since_last, movie_days_since_last
  - Affinità e coppie:          genre_match_score, svd_similarity, already_connected
"""

from pathlib import Path
import joblib
import numpy as np
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
    confusion_matrix,
    classification_report,
)

from config import (
    TRAIN_CSV,
    VAL_CSV,
    TEST_CSV,
    SIGN_MODEL_FILE,
)

# ── Feature per la predizione del Segno ─────────────────────────────────
SIGN_FEATURE_COLS = [
    "user_mean_rating",
    "user_frac_pos",
    "user_frac_neg",
    "user_n_reviews",
    "user_days_since_last",
    "movie_mean_rating",
    "movie_frac_pos",
    "movie_frac_neg",
    "movie_n_reviews",
    "movie_days_since_last",
    "genre_match_score",
    "svd_similarity",
    "already_connected",
]

TARGET_COL = "binary_label"


def load_and_prepare_sign_dataset(csv_path: Path):
    """Carica lo split ed estrae la matrice X e il target binario del segno."""
    df = pd.read_csv(csv_path)

    for col in SIGN_FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0

    X = df[SIGN_FEATURE_COLS].copy()
    # Imputazione valori mancanti: medie rating con default neutro (3.5)
    X["user_mean_rating"] = X["user_mean_rating"].fillna(3.5)
    X["movie_mean_rating"] = X["movie_mean_rating"].fillna(3.5)
    X = X.fillna(0.0)

    y = df[TARGET_COL].astype(int)
    return X, y


def evaluate(model, X, y, split_name: str, scaler=None) -> dict:
    """Calcola e stampa le metriche essenziali per la classificazione del segno."""
    X_eval = scaler.transform(X) if scaler is not None else X

    y_pred = model.predict(X_eval)
    y_proba = model.predict_proba(X_eval)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "balanced_acc": balanced_accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1_macro": f1_score(y, y_pred, average="macro", zero_division=0),
        "f1_pos": f1_score(y, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y, y_proba),
    }

    print(f"\n  ── Metriche Sign Prediction [{split_name}] ──")
    for k, v in metrics.items():
        print(f"  {k:<14}: {v:.4f}")

    cm = confusion_matrix(y, y_pred)
    print(f"  Confusion Matrix:")
    print(f"    Negativi reali: {cm[0][0]:<7} | Falsi Positivi: {cm[0][1]:<7}")
    print(f"    Falsi Negativi: {cm[1][0]:<7} | Positivi reali: {cm[1][1]:<7}")

    print("\n  Report di classificazione:")
    print(classification_report(y, y_pred, target_names=["negativa (0)", "positiva (1)"], digits=4))

    return metrics


def run(
    train_path: Path = TRAIN_CSV,
    val_path: Path = VAL_CSV,
    test_path: Path = TEST_CSV,
    save_model: bool = True,
    model_type: str = "gbm",
):
    """Addestramento e test del modello di Sign Prediction (Stadio 2)."""
    print("=" * 60)
    print("STADIO 2: SIGN PREDICTION (Segno dell'Interazione)")
    print("=" * 60)
    print(f"  Feature impiegate ({len(SIGN_FEATURE_COLS)}):")
    for f in SIGN_FEATURE_COLS:
        print(f"    - {f}")

    print("\n  Caricamento split...")
    X_train, y_train = load_and_prepare_sign_dataset(train_path)
    X_val, y_val = load_and_prepare_sign_dataset(val_path)
    X_test, y_test = load_and_prepare_sign_dataset(test_path)

    print(f"  Train: {len(X_train):,} righe | Val: {len(X_val):,} | Test: {len(X_test):,}")

    scaler = None
    if model_type == "lr":
        print("\n  Addestramento Logistic Regression (class_weight='balanced')...")
        scaler = StandardScaler()
        X_train_fit = scaler.fit_transform(X_train)
        model = LogisticRegression(max_iter=500, random_state=42, class_weight="balanced")
        model.fit(X_train_fit, y_train)
    else:
        print("\n  Addestramento HistGradientBoostingClassifier (class_weight='balanced')...")
        model = HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.08,
            max_depth=6,
            class_weight="balanced",
            random_state=42,
        )
        model.fit(X_train, y_train)

    evaluate(model, X_val, y_val, "Validation", scaler=scaler)
    test_metrics = evaluate(model, X_test, y_test, "Test", scaler=scaler)

    if save_model:
        SIGN_MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model": model,
            "scaler": scaler,
            "features": SIGN_FEATURE_COLS,
            "model_type": model_type,
        }, SIGN_MODEL_FILE)
        print(f"\n  Modello Sign Prediction salvato in: {SIGN_MODEL_FILE}")

    print("\nStadio 2 completato con successo.\n")
    return model, scaler, test_metrics


if __name__ == "__main__":
    model, scaler, metrics = run()
