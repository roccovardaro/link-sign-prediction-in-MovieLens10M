"""
Step 8 – Stadio 2: Sign Prediction (Classificazione del Segno dell'Interazione).

Obiettivo:
  Dato un link esistente (o predetto come tale dallo Stadio 1), determinare
  se l'interazione sarà di gradimento o scontento:
    1 = Valutazione Positiva (4 o 5 stelle)
    0 = Valutazione Negativa (0.5 - 2.5 stelle)

Feature impiegate (15 feature da step6_1_sign_features.py):
  - Storico e attitudine utente: user_mean_rating, user_rating_std, user_frac_pos, user_frac_neg, user_review_count
  - Storico e consenso film:     movie_mean_rating, movie_rating_std, movie_frac_pos, movie_frac_neg, movie_review_count
  - Baseline e ottimismo:        rating_baseline_score, user_movie_rating_diff, user_movie_optimism
  - Affinità di genere firmata:  user_genre_mean_rating
  - Filtro collaborativo latente:signed_svd_similarity
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
    TRAIN_SIGN_CSV,
    VAL_SIGN_CSV,
    TEST_SIGN_CSV,
    SIGN_MODEL_FILE,
)
from features.sign_features import (
    SIGN_FEATURE_COLS,
    TARGET_COL,
    load_sign_data,
)

SIGN_THRESHOLD = 0.4


def load_and_prepare_sign_dataset(csv_path: Path = TRAIN_SIGN_CSV):
    """Carica lo split ed estrae la matrice X e il target binario del segno."""
    path = Path(csv_path) if csv_path is not None else TRAIN_SIGN_CSV
    if not path.exists():
        path = TRAIN_CSV
    return load_sign_data(path)


def evaluate(model, X, y, split_name: str, scaler=None, threshold: float = SIGN_THRESHOLD) -> dict:
    """Calcola e stampa le metriche essenziali per la classificazione del segno."""
    X_eval = scaler.transform(X) if scaler is not None else X

    y_proba = model.predict_proba(X_eval)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "balanced_acc": balanced_accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1_macro": f1_score(y, y_pred, average="macro", zero_division=0),
        "f1_pos": f1_score(y, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y, y_proba),
    }

    print(f"\n  ── Metriche Sign Prediction [{split_name}] (Soglia = {threshold:.2f}) ──")
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
    train_csv: Path = TRAIN_SIGN_CSV,
    val_csv: Path = VAL_SIGN_CSV,
    test_csv: Path = TEST_SIGN_CSV,
    save_model: bool = True,
    model_type: str = "gbm",
    threshold: float = SIGN_THRESHOLD,
    **kwargs,
):
    """Addestramento e test del modello di Sign Prediction (Stadio 2)."""
    print("=" * 60)
    print("STADIO 2: SIGN PREDICTION (Segno dell'Interazione)")
    print("=" * 60)
    print(f"  Feature impiegate ({len(SIGN_FEATURE_COLS)}):")
    for f in SIGN_FEATURE_COLS:
        print(f"    - {f}")

    print("\n  Caricamento split...")
    t_c = train_csv if Path(train_csv).exists() else TRAIN_CSV
    v_c = val_csv if Path(val_csv).exists() else VAL_CSV
    te_c = test_csv if Path(test_csv).exists() else TEST_CSV

    X_train, y_train = load_and_prepare_sign_dataset(t_c)
    X_val, y_val = load_and_prepare_sign_dataset(v_c)
    X_test, y_test = load_and_prepare_sign_dataset(te_c)

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
            max_iter=300,
            learning_rate=0.08,
            max_depth=6,
            class_weight="balanced",
            random_state=42,
        )
        model.fit(X_train, y_train)

    evaluate(model, X_val, y_val, "Validation", scaler=scaler, threshold=threshold)
    test_metrics = evaluate(model, X_test, y_test, "Test", scaler=scaler, threshold=threshold)

    if save_model:
        SIGN_MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model": model,
            "scaler": scaler,
            "features": SIGN_FEATURE_COLS,
            "threshold": threshold,
            "model_type": model_type,
        }, SIGN_MODEL_FILE)
        print(f"\n  Modello Sign Prediction salvato in: {SIGN_MODEL_FILE} (Soglia: {threshold:.2f})")

    print("\nStadio 2 completato con successo.\n")
    return model, scaler, test_metrics, threshold


if __name__ == "__main__":
    model, scaler, metrics, threshold = run()
