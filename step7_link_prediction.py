"""
Step 7 – Stadio 1: Link Prediction (Esistenza dell'Interazione nel Grafo).

Obiettivo:
  Dato un utente 'u' e un film 'm', predire se si formerà un'interazione (arco nel grafo bipartito),
  indipendentemente dal gradimento positivo o negativo.

Target binario:
  1 = Link reale (l'interazione si verifica effettivamente nel dataset)
  0 = Link assente (coppia utente-film generata con Negative Sampling)

Caratteristiche utilizzate:
  - Topologiche & Popolarità: user_degree, movie_degree, preferential_attachment,
                              log_user_degree, log_movie_degree
  - Attività & Recency:       user_n_reviews, movie_n_reviews,
                              user_days_since_last, movie_days_since_last
  - Affinità avanzata:        genre_match_score, svd_similarity, already_connected

Modello:
  HistGradientBoostingClassifier / LogisticRegression per eccellenti prestazioni su metriche ROC-AUC,
  PR-AUC, F1 e Balanced Accuracy.
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
    average_precision_score,
    confusion_matrix,
    classification_report,
)

from config import (
    TRAIN_CSV,
    VAL_CSV,
    TEST_CSV,
    LINK_MODEL_FILE,
)

# ── Feature per Link Prediction ─────────────────────────────────────────
LINK_FEATURE_COLS = [
    "user_degree",
    "movie_degree",
    "preferential_attachment",
    "log_user_degree",
    "log_movie_degree",
    "user_n_reviews",
    "movie_n_reviews",
    "user_days_since_last",
    "movie_days_since_last",
    "genre_match_score",
    "svd_similarity",
    "already_connected",
]

TARGET_COL = "link_exists"


def generate_negative_samples(df: pd.DataFrame, ratio: float = 0.5, seed: int = 42) -> pd.DataFrame:
    """
    Genera coppie negative (utente, film) che NON hanno interagito nello split.
    Associa correttamente le feature del nodo utente e del nodo film campionato.
    """
    rng = np.random.default_rng(seed)
    n_neg = int(len(df) * ratio)

    existing_pairs = set(zip(df["user_id"].values, df["movie_id"].values))
    unique_movies = df.drop_duplicates(subset=["movie_id"]).reset_index(drop=True)

    valid_pairs = []
    while len(valid_pairs) < n_neg:
        batch_size = max((n_neg - len(valid_pairs)) * 2, 200)
        u_idx = rng.integers(0, len(df), size=batch_size)
        m_idx = rng.integers(0, len(unique_movies), size=batch_size)
        users = df["user_id"].values[u_idx]
        movies = unique_movies["movie_id"].values[m_idx]

        for idx_u, idx_m, u, m in zip(u_idx, m_idx, users, movies):
            if (u, m) not in existing_pairs:
                valid_pairs.append((idx_u, idx_m))
                if len(valid_pairs) >= n_neg:
                    break

    valid_u_idx = [p[0] for p in valid_pairs]
    valid_m_idx = [p[1] for p in valid_pairs]

    u_deg = df["user_degree"].values[valid_u_idx]
    m_deg = unique_movies["movie_degree"].values[valid_m_idx]

    # Per genre_match e svd_sim nei negativi, un'approssimazione realistica è un match debole/nullo
    # oppure ricalcolato se disponibile
    neg_genre = np.zeros(len(valid_pairs), dtype=np.float32)
    neg_svd = np.zeros(len(valid_pairs), dtype=np.float32)

    neg_df = pd.DataFrame({
        "user_id": df["user_id"].values[valid_u_idx],
        "movie_id": unique_movies["movie_id"].values[valid_m_idx],
        "user_degree": u_deg,
        "movie_degree": m_deg,
        "preferential_attachment": u_deg * m_deg,
        "log_user_degree": np.log1p(u_deg),
        "log_movie_degree": np.log1p(m_deg),
        "user_n_reviews": df["user_n_reviews"].values[valid_u_idx],
        "movie_n_reviews": unique_movies["movie_n_reviews"].values[valid_m_idx],
        "user_days_since_last": df["user_days_since_last"].values[valid_u_idx],
        "movie_days_since_last": unique_movies["movie_days_since_last"].values[valid_m_idx],
        "genre_match_score": neg_genre,
        "svd_similarity": neg_svd,
        "already_connected": 0,
        "link_exists": 0,
    })

    return neg_df


def load_and_prepare_link_dataset(csv_path: Path, ratio: float = 0.5, seed: int = 42):
    """
    Carica lo split CSV, crea gli esempi positivi reali (link_exists=1)
    e gli esempi negativi campionati (link_exists=0), estraendo X e y.
    """
    df = pd.read_csv(csv_path)

    pos_df = df.copy()
    if "already_connected" not in pos_df.columns:
        pos_df["already_connected"] = 0
    if "preferential_attachment" not in pos_df.columns:
        pos_df["preferential_attachment"] = pos_df["user_degree"].fillna(0) * pos_df["movie_degree"].fillna(0)
    if "log_user_degree" not in pos_df.columns:
        pos_df["log_user_degree"] = np.log1p(pos_df["user_degree"].fillna(0))
    if "log_movie_degree" not in pos_df.columns:
        pos_df["log_movie_degree"] = np.log1p(pos_df["movie_degree"].fillna(0))
    if "genre_match_score" not in pos_df.columns:
        pos_df["genre_match_score"] = 0.0
    if "svd_similarity" not in pos_df.columns:
        pos_df["svd_similarity"] = 0.0

    pos_df["link_exists"] = 1

    neg_df = generate_negative_samples(df, ratio=ratio, seed=seed)

    cols = LINK_FEATURE_COLS + ["link_exists"]
    combined = pd.concat([pos_df[cols], neg_df[cols]], ignore_index=True)

    X = combined[LINK_FEATURE_COLS].fillna(0)
    y = combined["link_exists"]
    return X, y


def evaluate(model, X, y, split_name: str, scaler=None) -> dict:
    """Valuta il modello di link prediction calcolando le metriche prestazionali."""
    X_eval = scaler.transform(X) if scaler is not None else X

    y_pred = model.predict(X_eval)
    y_proba = model.predict_proba(X_eval)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "balanced_acc": balanced_accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y, y_proba),
        "pr_auc": average_precision_score(y, y_proba),
    }

    print(f"\n  ── Metriche Link Prediction [{split_name}] ──")
    for k, v in metrics.items():
        print(f"  {k:<14}: {v:.4f}")

    cm = confusion_matrix(y, y_pred)
    print(f"  Confusion Matrix:")
    print(f"    Negativi reali: {cm[0][0]:<7} corretti (TN) | {cm[0][1]:<7} falsi allarmi (FP)")
    print(f"    Positivi reali: {cm[1][0]:<7} mancati (FN)  | {cm[1][1]:<7} intercettati (TP)")

    return metrics


def run(
    train_path: Path = TRAIN_CSV,
    val_path: Path = VAL_CSV,
    test_path: Path = TEST_CSV,
    ratio: float = 0.5,
    save_model: bool = True,
    model_type: str = "gbm",
):
    """
    Addestra e valuta il modello di Link Prediction ad alte prestazioni.
    model_type: 'gbm' (HistGradientBoosting, raccomandato) oppure 'lr' (LogisticRegression).
    """
    print("=" * 60)
    print("STADIO 1: LINK PREDICTION (MovieLens 10M)")
    print("=" * 60)
    print(f"  Feature impiegate ({len(LINK_FEATURE_COLS)}):")
    for f in LINK_FEATURE_COLS:
        print(f"    - {f}")

    print(f"\n  Costruzione dataset con Negative Sampling (ratio={ratio})...")
    X_train, y_train = load_and_prepare_link_dataset(train_path, ratio=ratio, seed=42)
    X_val, y_val = load_and_prepare_link_dataset(val_path, ratio=ratio, seed=43)
    X_test, y_test = load_and_prepare_link_dataset(test_path, ratio=ratio, seed=44)

    print(f"  Train: {len(X_train):,} coppie ({(y_train == 1).sum():,} reali, {(y_train == 0).sum():,} negative)")
    print(f"  Val:   {len(X_val):,} coppie")
    print(f"  Test:  {len(X_test):,} coppie")

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

    evaluate(model, X_val, y_val, "Validation", scaler=scaler)
    test_metrics = evaluate(model, X_test, y_test, "Test", scaler=scaler)

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
