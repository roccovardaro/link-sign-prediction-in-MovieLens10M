"""
Link Prediction – Classificazione con feature da train.csv.

Utilizza tutte le feature strutturali estratte in Step 6 tranne i conteggi
e le frazioni di sentiment delle stelle (n_pos, n_neg, frac_pos, ecc.),
mantenendo come unico segnale sui voti il rating medio.

Feature utilizzate:
  - Utente:   user_degree, user_n_reviews, user_mean_rating, user_days_since_last
  - Prodotto: prod_degree, prod_n_reviews, prod_mean_rating, prod_days_since_last
  - Coppia:   already_connected
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from config import TRAIN_CSV, VAL_CSV, TEST_CSV

# ── Feature selezionate ──────────────────────────────────────────────────
# Tutte le feature di train, escluse quelle di dettaglio sulle stelle (pos/neg/neu)
FEATURE_COLS = [
    # Utente
    "user_degree",
    "user_n_reviews",
    "user_mean_rating",
    "user_days_since_last",
    # Prodotto
    "prod_degree",
    "prod_n_reviews",
    "prod_mean_rating",
    "prod_days_since_last",
    # Coppia
    "already_connected",
]

TARGET_COL = "binary_label"


def load_and_prepare(csv_path):
    """Carica uno split CSV ed estrae matrice X (imputando i NaN con 0) e target y."""
    df = pd.read_csv(csv_path)

    # Verifica presenza delle colonne (fallback a 0 se manca already_connected)
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0

    X = df[FEATURE_COLS].fillna(0)
    y = df[TARGET_COL]
    return X, y


def evaluate(model, X_scaled, y, split_name):
    """Calcola e stampa le metriche essenziali di classificazione."""
    y_pred = model.predict(X_scaled)
    y_proba = model.predict_proba(X_scaled)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y, y_proba),
    }

    print(f"\n  ── Metriche [{split_name}] ──")
    for k, v in metrics.items():
        print(f"  {k:<12}: {v:.4f}")

    cm = confusion_matrix(y, y_pred)
    print(f"  Confusion Matrix:\n    Neg predetto: {cm[0][0]:<6} | Falsi Positivi: {cm[0][1]}")
    print(f"    Falsi Neg:    {cm[1][0]:<6} | Pos predetto:   {cm[1][1]}")

    return metrics


def run(train_path=TRAIN_CSV, val_path=VAL_CSV, test_path=TEST_CSV):
    """Esecuzione semplificata e modulare del modello."""
    print("=" * 60)
    print("LINK PREDICTION – Solo Rating Medio e Topologia")
    print("=" * 60)
    print(f"  Feature impiegate ({len(FEATURE_COLS)}):")
    for f in FEATURE_COLS:
        print(f"    - {f}")

    # 1. Caricamento dati
    print("\n  Caricamento split...")
    X_train, y_train = load_and_prepare(train_path)
    X_val, y_val = load_and_prepare(val_path)
    X_test, y_test = load_and_prepare(test_path)

    print(f"  Train: {len(X_train):,} righe | Val: {len(X_val):,} | Test: {len(X_test):,}")

    # 2. Standardizzazione
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # 3. Addestramento
    print("\n  Addestramento Logistic Regression...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_s, y_train)

    # 4. Valutazione
    evaluate(model, X_val_s, y_val, "Validation")
    test_metrics = evaluate(model, X_test_s, y_test, "Test")

    # 5. Pesi delle feature (coefficienti)
    print("\n  ── Coefficienti Feature (Importanza) ──")
    coef_df = pd.DataFrame({
        "feature": FEATURE_COLS,
        "coefficiente": model.coef_[0],
        "importanza": np.abs(model.coef_[0]),
    }).sort_values("importanza", ascending=False).reset_index(drop=True)

    for _, row in coef_df.iterrows():
        print(f"  {row['feature']:<25} coeff: {row['coefficiente']:+.4f} (|coeff|={row['importanza']:.4f})")

    print("\nEsecuzione completata con successo.\n")
    return model, test_metrics


if __name__ == "__main__":
    model, metrics = run()
