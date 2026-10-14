"""
pipeline_two_stage.py – Pipeline Integrata a Due Stadi (MovieLens 10M).

Architettura:
  1. Stadio 1 (Link Prediction):
     - Data una coppia (utente, film), predice se si formerà un'interazione (arco).
     - Se predice NO LINK  --> Il processo termina ("Nessuna interazione").
     - Se predice LINK     --> Si passa allo Stadio 2.

  2. Stadio 2 (Sign Prediction):
     - Valuta il segno della valutazione attesa per il link confermato:
       --> "Positiva" (4-5 stelle) oppure "Negativa" (0.5 - 2.5 stelle).

Uso:
  python pipeline_two_stage.py
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from config import (
    TEST_CSV,
    EVENTS_CSV,
    LINK_MODEL_FILE,
    SIGN_MODEL_FILE,
    TEST_LINK_FILE,
    TEST_LINK_CSV,
)
from step6_features import (
    LINK_FEATURE_COLS,
    extract_link_features,
    load_dataset,
    sample_negative_pairs,
)
from step7_link_prediction import (
    run as run_link_stage,
)
from step8_sign_prediction import (
    run as run_sign_stage,
    SIGN_FEATURE_COLS,
    SIGN_THRESHOLD,
)


def load_or_train_models():
    """Carica i modelli salvati su disco oppure li addestra se non presenti."""
    if LINK_MODEL_FILE.exists() and SIGN_MODEL_FILE.exists():
        print("  Caricamento modelli pre-addestrati da disco...")
        link_bundle = joblib.load(LINK_MODEL_FILE)
        sign_bundle = joblib.load(SIGN_MODEL_FILE)
        sign_bundle.setdefault("threshold", SIGN_THRESHOLD)
        if link_bundle.get("features") != LINK_FEATURE_COLS:
            print("  Schema feature aggiornato: riaddestramento modello link...")
            model, scaler, _ = run_link_stage()
            link_bundle = {"model": model, "scaler": scaler, "features": LINK_FEATURE_COLS}
        if sign_bundle.get("features") != SIGN_FEATURE_COLS:
            print("  Schema feature aggiornato: riaddestramento modello sign...")
            sign_res = run_sign_stage()
            sign_model, sign_scaler = sign_res[0], sign_res[1]
            sign_threshold = sign_res[3] if len(sign_res) > 3 else SIGN_THRESHOLD
            sign_bundle = {"model": sign_model, "scaler": sign_scaler, "features": SIGN_FEATURE_COLS, "threshold": sign_threshold}
    else:
        print("  Modelli non trovati, avvio addestramento dei due stadi...")
        link_model, link_scaler, _ = run_link_stage()
        sign_res = run_sign_stage()
        sign_model, sign_scaler = sign_res[0], sign_res[1]
        sign_threshold = sign_res[3] if len(sign_res) > 3 else SIGN_THRESHOLD
        link_bundle = {"model": link_model, "scaler": link_scaler}
        sign_bundle = {"model": sign_model, "scaler": sign_scaler, "threshold": sign_threshold}

    return link_bundle, sign_bundle


def prepare_joint_test_set(test_path: Path = TEST_CSV, ratio: float = 0.5, seed: int = 42) -> pd.DataFrame:
    """
    Costruisce un test set congiunto contenente sia link reali (con rispettivo segno)
    sia link negativi (coppie senza interazione).
    """
    df = pd.read_csv(test_path)

    # 1. Positivi reali: link_exists=1, sign_true=binary_label (0 o 1)
    events = pd.read_csv(EVENTS_CSV, parse_dates=["timestamp"])
    pos = extract_link_features(df, events)
    if "genre_match_score" not in pos.columns:
        pos["genre_match_score"] = 0.0
    if "svd_similarity" not in pos.columns:
        pos["svd_similarity"] = 0.0

    pos["link_true"] = 1
    pos["sign_true"] = pos["binary_label"]

    # 2. Negativi campionati: link_exists=0, sign_true=-1 (nessun segno)
    neg_pairs = sample_negative_pairs(df, events, ratio=ratio, seed=seed)
    neg = extract_link_features(neg_pairs, events)
    neg["link_true"] = 0
    neg["sign_true"] = -1

    for col in SIGN_FEATURE_COLS:
        if col not in neg.columns:
            neg[col] = 0.0

    all_cols = list(set(LINK_FEATURE_COLS + SIGN_FEATURE_COLS + ["link_true", "sign_true"]))
    for c in all_cols:
        if c not in pos.columns:
            pos[c] = 0.0
        if c not in neg.columns:
            neg[c] = 0.0

    combined = pd.concat([pos[all_cols], neg[all_cols]], ignore_index=True)
    return combined


def run_cascade_evaluation(link_bundle: dict, sign_bundle: dict, test_df: pd.DataFrame):
    """Esegue la valutazione end-to-end della cascata a due stadi sul test set."""
    link_model, link_scaler = link_bundle["model"], link_bundle.get("scaler")
    sign_model, sign_scaler = sign_bundle["model"], sign_bundle.get("scaler")
    sign_threshold = sign_bundle.get("threshold", SIGN_THRESHOLD)

    print("\n" + "=" * 60)
    print("VALUTAZIONE PIPELINE A DUE STADI (END-TO-END)")
    print("=" * 60)
    print(f"  Totale istanze di test valutate: {len(test_df):,}")
    n_real_links = (test_df["link_true"] == 1).sum()
    n_no_links = (test_df["link_true"] == 0).sum()
    print(f"    - Link reali presenti:        {n_real_links:,}")
    print(f"    - Coppie senza interazione:   {n_no_links:,}")

    # ── STADIO 1: Predizione Esistenza Link ──
    X_link = test_df[LINK_FEATURE_COLS].fillna(0)
    X_link_eval = link_scaler.transform(X_link) if link_scaler is not None else X_link
    pred_link = link_model.predict(X_link_eval)

    link_cm = confusion_matrix(test_df["link_true"], pred_link)
    link_accuracy = (pred_link == test_df["link_true"]).mean()
    link_recall = link_cm[1][1] / (link_cm[1][0] + link_cm[1][1])

    print("\n  ── Risultati Stadio 1 (Link Prediction) ──")
    print(f"  Accuracy Link       : {link_accuracy:.4f}")
    print(f"  Recall Link Reali   : {link_recall:.4f} ({link_cm[1][1]:,} intercettati su {n_real_links:,})")
    print(f"  Falsi Allarmi (FP)  : {link_cm[0][1]:,} coppie senza link scambiate per link")

    # ── STADIO 2: Predizione Segno sui soli Link Previsti ──
    mask_pred_link = pred_link == 1
    n_pred_links = mask_pred_link.sum()

    print(f"\n  ── Risultati Stadio 2 (Sign Prediction su {n_pred_links:,} link previsti | Soglia = {sign_threshold:.2f}) ──")

    # Classi finali a cascata:
    # 0 = Nessun link predetto
    # 1 = Link predetto con Segno Negativo (0.5 - 2.5 stelle)
    # 2 = Link predetto con Segno Positivo (4 - 5 stelle)
    y_pred_cascade = np.zeros(len(test_df), dtype=int)

    if n_pred_links > 0:
        X_sign_sub = test_df.loc[mask_pred_link, SIGN_FEATURE_COLS].copy()
        X_sign_sub["user_mean_rating"] = X_sign_sub["user_mean_rating"].fillna(3.5)
        X_sign_sub["movie_mean_rating"] = X_sign_sub["movie_mean_rating"].fillna(3.5)
        X_sign_sub = X_sign_sub.fillna(0.0)

        X_sign_eval = sign_scaler.transform(X_sign_sub) if sign_scaler is not None else X_sign_sub
        sign_proba = sign_model.predict_proba(X_sign_eval)[:, 1]
        pred_sign = (sign_proba >= sign_threshold).astype(int)

        # segno 0 -> classe 1 (negativa); segno 1 -> classe 2 (positiva)
        y_pred_cascade[mask_pred_link] = np.where(pred_sign == 1, 2, 1)

    # Ground truth a 3 classi:
    # 0 = Nessun link reale
    # 1 = Link reale con segno Negativo (sign_true == 0)
    # 2 = Link reale con segno Positivo (sign_true == 1)
    y_true_cascade = np.zeros(len(test_df), dtype=int)
    y_true_cascade[test_df["sign_true"] == 0] = 1
    y_true_cascade[test_df["sign_true"] == 1] = 2

    target_names = ["Nessun Link (0)", "Link Negativo (1)", "Link Positivo (2)"]
    print("\n  ── Report Finale a Cascata (3 Classi) ──")
    print(classification_report(y_true_cascade, y_pred_cascade, target_names=target_names, digits=4))

    print("  Matrice di Confusione Globale (Righe=Reale, Colonne=Predetto):")
    cm_cascade = confusion_matrix(y_true_cascade, y_pred_cascade)
    header = f"  {'':<20} | {'Pred: No Link':<14} | {'Pred: Negativo':<15} | {'Pred: Positivo':<15}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for idx, name in enumerate(["Reale: No Link", "Reale: Negativo", "Reale: Positivo"]):
        print(f"  {name:<20} | {cm_cascade[idx][0]:<14} | {cm_cascade[idx][1]:<15} | {cm_cascade[idx][2]:<15}")


def predict_pair(user_dict: dict, movie_dict: dict, link_bundle: dict, sign_bundle: dict) -> dict:
    """
    Funzione di inferenza rapida per una singola coppia utente-film.
    """
    link_model, link_scaler = link_bundle["model"], link_bundle.get("scaler")
    sign_model, sign_scaler = sign_bundle["model"], sign_bundle.get("scaler")

    # Le feature di coppia devono provenire dallo stesso estrattore temporale.
    values = {**user_dict, **movie_dict}
    missing = [col for col in LINK_FEATURE_COLS if col not in values]
    if missing:
        raise ValueError(f"Feature mancanti: {missing}. Usare extract_link_features per la coppia.")
    link_row = pd.DataFrame([values])[LINK_FEATURE_COLS].fillna(0)

    if link_scaler is not None:
        link_row_eval = link_scaler.transform(link_row)
    else:
        link_row_eval = link_row

    link_proba = float(link_model.predict_proba(link_row_eval)[0, 1])
    link_pred = int(link_proba >= 0.5)

    if link_pred == 0:
        return {
            "link_exists": False,
            "link_probability": round(link_proba, 4),
            "predicted_sign": None,
            "sign_probability": None,
            "final_outcome": "Nessuna interazione prevista",
        }

    sign_row = pd.DataFrame([{
        "user_mean_rating": user_dict.get("user_mean_rating", 3.5),
        "user_frac_pos": user_dict.get("user_frac_pos", 0.0),
        "user_frac_neg": user_dict.get("user_frac_neg", 0.0),
        "user_n_reviews": user_dict.get("user_n_reviews", 0),
        "user_days_since_last": user_dict.get("user_days_since_last", 0),
        "movie_mean_rating": movie_dict.get("movie_mean_rating", 3.5),
        "movie_frac_pos": movie_dict.get("movie_frac_pos", 0.0),
        "movie_frac_neg": movie_dict.get("movie_frac_neg", 0.0),
        "movie_n_reviews": movie_dict.get("movie_n_reviews", 0),
        "movie_days_since_last": movie_dict.get("movie_days_since_last", 0),
        "genre_match_score": user_dict.get("genre_match_score", 0.0),
        "svd_similarity": user_dict.get("svd_similarity", 0.0),
        "already_connected": user_dict.get("already_connected", 0),
    }])[SIGN_FEATURE_COLS]

    if sign_scaler is not None:
        sign_row_eval = sign_scaler.transform(sign_row)
    else:
        sign_row_eval = sign_row

    sign_threshold = sign_bundle.get("threshold", SIGN_THRESHOLD)
    sign_proba = float(sign_model.predict_proba(sign_row_eval)[0, 1])
    sign_label = "positiva" if sign_proba >= sign_threshold else "negativa"

    return {
        "link_exists": True,
        "link_probability": round(link_proba, 4),
        "predicted_sign": sign_label,
        "sign_probability": round(sign_proba, 4),
        "final_outcome": f"Interazione {sign_label.upper()}",
    }


def main():
    link_bundle, sign_bundle = load_or_train_models()
    print("\n  Preparazione test set congiunto per la valutazione a cascata...")
    test_df = prepare_joint_test_set()
    run_cascade_evaluation(link_bundle, sign_bundle, test_df)


if __name__ == "__main__":
    main()
