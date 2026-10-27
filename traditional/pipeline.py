"""
traditional/pipeline.py – Macrofase Pipeline d'Inferenza a Due Stadi e Valutazione a Cascata.

Architettura a Due Stadi:
  1. Stadio 1 (Link Prediction):
     - Data una coppia (utente, film), predice se si formerà un'interazione (arco).
     - Se predice NO LINK --> Il processo termina ("Nessuna interazione").
     - Se predice LINK    --> Si passa allo Stadio 2.

  2. Stadio 2 (Sign Prediction):
     - Valuta il segno della valutazione attesa per il link confermato:
       --> "Positiva" (4-5 stelle) oppure "Negativa" (0.5 - 2.5 stelle).
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from config import (
    TEST_CSV,
    EVENTS_CSV,
    TEST_LINK_CSV,
    TEST_SIGN_CSV,
    CSV_PROCESSED,
    LINK_MODEL_FILE,
    SIGN_MODEL_FILE,
)
from traditional.features.link_features import (
    LINK_FEATURE_COLS,
    extract_link_features,
    sample_negative_pairs,
)
from traditional.features.sign_features import (
    SIGN_FEATURE_COLS,
    extract_sign_features,
)
from traditional.model import DEFAULT_LINK_THRESHOLD, DEFAULT_SIGN_THRESHOLD
from traditional.trainer import train_link_model, train_sign_model


class TraditionalTwoStagePipeline:
    """Pipeline integrata a due stadi per l'approccio tradizionale (Link + Sign)."""

    def __init__(
        self,
        link_bundle: dict,
        sign_bundle: dict,
        link_threshold: float = DEFAULT_LINK_THRESHOLD,
        sign_threshold: Optional[float] = None,
    ):
        self.link_bundle = link_bundle
        self.sign_bundle = sign_bundle
        self.link_model = link_bundle["model"]
        self.link_scaler = link_bundle.get("scaler")
        self.sign_model = sign_bundle["model"]
        self.sign_scaler = sign_bundle.get("scaler")
        self.link_threshold = link_threshold
        self.sign_threshold = (
            sign_threshold if sign_threshold is not None
            else sign_bundle.get("threshold", DEFAULT_SIGN_THRESHOLD)
        )
        self.link_bundle["threshold"] = self.link_threshold
        self.sign_bundle["threshold"] = self.sign_threshold

    @classmethod
    def load(
        cls,
        link_model_path: Path = LINK_MODEL_FILE,
        sign_model_path: Path = SIGN_MODEL_FILE,
        link_threshold: float = DEFAULT_LINK_THRESHOLD,
        sign_threshold: Optional[float] = None,
    ) -> "TraditionalTwoStagePipeline":
        """Carica i bundle salvati su disco e restituisce la pipeline pronta all'uso."""
        l_path = Path(link_model_path)
        s_path = Path(sign_model_path)
        if not l_path.exists():
            raise FileNotFoundError(f"Modello link non trovato in: {l_path}. Addestrare prima con train_link_model.")
        if not s_path.exists():
            raise FileNotFoundError(f"Modello sign non trovato in: {s_path}. Addestrare prima con train_sign_model.")

        link_bundle = joblib.load(l_path)
        sign_bundle = joblib.load(s_path)
        return cls(
            link_bundle=link_bundle,
            sign_bundle=sign_bundle,
            link_threshold=link_threshold,
            sign_threshold=sign_threshold,
        )

    def predict_pair(self, user_dict: dict, movie_dict: dict) -> dict:
        """Esegue l'inferenza per una singola coppia utente-film."""
        return predict_pair(user_dict, movie_dict, self.link_bundle, self.sign_bundle)

    def evaluate_cascade(self, test_df: pd.DataFrame) -> dict:
        """Esegue la valutazione end-to-end a cascata su un DataFrame di test."""
        return run_cascade_evaluation(self.link_bundle, self.sign_bundle, test_df)


def load_or_train_models() -> Tuple[dict, dict]:
    """Carica i modelli salvati su disco oppure li addestra se non presenti."""
    if LINK_MODEL_FILE.exists() and SIGN_MODEL_FILE.exists():
        print("  Caricamento modelli pre-addestrati da disco...")
        link_bundle = joblib.load(LINK_MODEL_FILE)
        sign_bundle = joblib.load(SIGN_MODEL_FILE)
        sign_bundle.setdefault("threshold", DEFAULT_SIGN_THRESHOLD)

        if link_bundle.get("features") != LINK_FEATURE_COLS:
            print("  Schema feature aggiornato: riaddestramento modello link...")
            model, scaler, _ = train_link_model()
            link_bundle = {"model": model, "scaler": scaler, "features": LINK_FEATURE_COLS}

        if sign_bundle.get("features") != SIGN_FEATURE_COLS:
            print("  Schema feature aggiornato: riaddestramento modello sign...")
            sign_res = train_sign_model()
            sign_model, sign_scaler = sign_res[0], sign_res[1]
            sign_threshold = sign_res[3] if len(sign_res) > 3 else DEFAULT_SIGN_THRESHOLD
            sign_bundle = {"model": sign_model, "scaler": sign_scaler, "features": SIGN_FEATURE_COLS, "threshold": sign_threshold}
    else:
        print("  Modelli non trovati, avvio addestramento dei due stadi...")
        link_model, link_scaler, _ = train_link_model()
        sign_res = train_sign_model()
        sign_model, sign_scaler = sign_res[0], sign_res[1]
        sign_threshold = sign_res[3] if len(sign_res) > 3 else DEFAULT_SIGN_THRESHOLD
        link_bundle = {"model": link_model, "scaler": link_scaler, "features": LINK_FEATURE_COLS}
        sign_bundle = {"model": sign_model, "scaler": sign_scaler, "features": SIGN_FEATURE_COLS, "threshold": sign_threshold}

    return link_bundle, sign_bundle


def get_joint_test_cache_path(ratio: float, seed: int) -> Path:
    """Restituisce il path della cache parquet per il test set congiunto."""
    return CSV_PROCESSED / f"joint_test_r{int(ratio * 100)}_s{seed}.parquet"


def prepare_joint_test_set(
    test_path: Path = TEST_CSV,
    ratio: float = 0.5,
    seed: int = 42,
    force_recompute: bool = False,
) -> pd.DataFrame:
    """
    Costruisce (o carica dalla cache) il test set congiunto per la valutazione a cascata,
    garantendo che sia i link reali sia le coppie negative abbiano a disposizione
    tutte le feature richieste da entrambi gli stadi (7 di link, 15 di segno) e gli ID utente/film.
    """
    cache_path = get_joint_test_cache_path(ratio, seed)
    if not force_recompute and cache_path.exists():
        print(f"  Caricamento test set congiunto dalla cache: {cache_path.name}...")
        return pd.read_parquet(cache_path)

    print("  Preparazione test set congiunto (positivi reali + negativi campionati)...")

    # 1. Positivi reali: caricamento da file pre-elaborati (istantaneo) o estrazione on-the-fly
    if TEST_LINK_CSV.exists() and TEST_SIGN_CSV.exists():
        print("  Caricamento feature link e segno per i positivi reali da file pre-elaborati...")
        pos_link = pd.read_csv(TEST_LINK_CSV)
        pos_sign = pd.read_csv(TEST_SIGN_CSV)
        pos = pos_link.copy()
        for col in SIGN_FEATURE_COLS:
            if col in pos_sign.columns:
                pos[col] = pos_sign[col]
            else:
                pos[col] = 0.0
    else:
        print("  Estrazione feature per i positivi reali...")
        df = pd.read_csv(test_path)
        events = pd.read_csv(EVENTS_CSV, parse_dates=["timestamp"])
        pos = extract_link_features(df, events)
        pos_sign = extract_sign_features(df, events)
        for col in SIGN_FEATURE_COLS:
            pos[col] = pos_sign[col] if col in pos_sign.columns else 0.0

    if "genre_match_score" not in pos.columns:
        pos["genre_match_score"] = 0.0

    pos["link_true"] = 1
    pos["sign_true"] = pos["binary_label"]

    # 2. Negativi campionati: coppie senza interazione
    print(f"  Campionamento coppie negative (ratio={ratio}, seed={seed})...")
    events = pd.read_csv(EVENTS_CSV, parse_dates=["timestamp"])
    df_source = pd.read_csv(test_path) if "user_id" not in pos.columns else pos
    neg_pairs = sample_negative_pairs(df_source, events, ratio=ratio, seed=seed)

    print(f"  Estrazione feature link per {len(neg_pairs):,} coppie negative...")
    neg = extract_link_features(neg_pairs, events)
    if "genre_match_score" not in neg.columns:
        neg["genre_match_score"] = 0.0

    print(f"  Estrazione feature segno per {len(neg_pairs):,} coppie negative...")
    neg_sign = extract_sign_features(neg_pairs, events)
    for col in SIGN_FEATURE_COLS:
        if col in neg_sign.columns:
            neg[col] = neg_sign[col]
        else:
            neg[col] = 0.0

    neg["link_true"] = 0
    neg["sign_true"] = -1

    id_cols = [c for c in ["user_id", "movie_id", "timestamp", "title", "genres"] if c in pos.columns]
    keep_cols = list(dict.fromkeys(id_cols + LINK_FEATURE_COLS + SIGN_FEATURE_COLS + ["link_true", "sign_true"]))

    for c in keep_cols:
        if c not in pos.columns:
            pos[c] = 0.0
        if c not in neg.columns:
            neg[c] = 0.0

    combined = pd.concat([pos[keep_cols], neg[keep_cols]], ignore_index=True)
    for c in ["timestamp", "title", "genres"]:
        if c in combined.columns:
            combined[c] = combined[c].fillna("").astype(str)

    # Salvataggio in cache parquet per esecuzioni successive istantanee
    try:
        combined.to_parquet(cache_path, index=False)
        print(f"  Test set congiunto salvato in cache: {cache_path.name}")
    except Exception as e:
        print(f"  [Warning] Impossibile salvare cache parquet: {e}")

    return combined


def run_cascade_evaluation(link_bundle: dict, sign_bundle: dict, test_df: pd.DataFrame) -> dict:
    """Esegue la valutazione end-to-end della cascata a due stadi sul test set."""
    link_model, link_scaler = link_bundle["model"], link_bundle.get("scaler")
    sign_model, sign_scaler = sign_bundle["model"], sign_bundle.get("scaler")
    sign_threshold = sign_bundle.get("threshold", DEFAULT_SIGN_THRESHOLD)

    print("\n" + "=" * 60)
    print("VALUTAZIONE PIPELINE A DUE STADI (END-TO-END)")
    print("=" * 60)
    print(f"  Totale istanze di test valutate: {len(test_df):,}")
    n_real_links = int((test_df["link_true"] == 1).sum())
    n_no_links = int((test_df["link_true"] == 0).sum())
    print(f"    - Link reali presenti:        {n_real_links:,}")
    print(f"    - Coppie senza interazione:   {n_no_links:,}")

    # ── STADIO 1: Predizione Esistenza Link ──
    X_link = test_df[LINK_FEATURE_COLS].fillna(0)
    X_link_eval = link_scaler.transform(X_link) if link_scaler is not None else X_link
    pred_link = link_model.predict(X_link_eval)

    link_cm = confusion_matrix(test_df["link_true"], pred_link)
    link_accuracy = float((pred_link == test_df["link_true"]).mean())
    link_recall = float(link_cm[1][1] / (link_cm[1][0] + link_cm[1][1]))

    print("\n  ── Risultati Stadio 1 (Link Prediction) ──")
    print(f"  Accuracy Link       : {link_accuracy:.4f}")
    print(f"  Recall Link Reali   : {link_recall:.4f} ({link_cm[1][1]:,} intercettati su {n_real_links:,})")
    print(f"  Falsi Allarmi (FP)  : {link_cm[0][1]:,} coppie senza link scambiate per link")

    # ── STADIO 2: Predizione Segno sui soli Link Previsti ──
    mask_pred_link = pred_link == 1
    n_pred_links = int(mask_pred_link.sum())

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

        n_pos_pred = int((pred_sign == 1).sum())
        n_neg_pred = int((pred_sign == 0).sum())
        print(f"    --> Previsti come Positivi: {n_pos_pred:,} ({n_pos_pred * 100 / n_pred_links:.2f}%)")
        print(f"    --> Previsti come Negativi: {n_neg_pred:,} ({n_neg_pred * 100 / n_pred_links:.2f}%)")

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
    print(classification_report(y_true_cascade, y_pred_cascade, target_names=target_names, digits=4, zero_division=0))

    print("  Matrice di Confusione Globale (Righe=Reale, Colonne=Predetto):")
    cm_cascade = confusion_matrix(y_true_cascade, y_pred_cascade, labels=[0, 1, 2])
    header = f"  {'':<20} | {'Pred: No Link':<14} | {'Pred: Negativo':<15} | {'Pred: Positivo':<15}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for idx, name in enumerate(["Reale: No Link", "Reale: Negativo", "Reale: Positivo"]):
        print(f"  {name:<20} | {cm_cascade[idx][0]:<14} | {cm_cascade[idx][1]:<15} | {cm_cascade[idx][2]:<15}")

    report_dict = classification_report(y_true_cascade, y_pred_cascade, target_names=target_names, output_dict=True, zero_division=0)
    return {
        "link_accuracy": link_accuracy,
        "link_recall": link_recall,
        "classification_report": report_dict,
        "confusion_matrix": cm_cascade.tolist(),
    }


def predict_pair(user_dict: dict, movie_dict: dict, link_bundle: dict, sign_bundle: dict) -> dict:
    """Funzione di inferenza rapida per una singola coppia utente-film."""
    link_model, link_scaler = link_bundle["model"], link_bundle.get("scaler")
    sign_model, sign_scaler = sign_bundle["model"], sign_bundle.get("scaler")

    values = {**user_dict, **movie_dict}
    missing = [col for col in LINK_FEATURE_COLS if col not in values]
    if missing:
        raise ValueError(f"Feature mancanti: {missing}. Usare extract_link_features per la coppia.")
    link_row = pd.DataFrame([values])[LINK_FEATURE_COLS].fillna(0)

    link_row_eval = link_scaler.transform(link_row) if link_scaler is not None else link_row
    link_proba = float(link_model.predict_proba(link_row_eval)[0, 1])
    link_pred = int(link_proba >= DEFAULT_LINK_THRESHOLD)

    if link_pred == 0:
        return {
            "link_exists": False,
            "link_probability": round(link_proba, 4),
            "predicted_sign": None,
            "sign_probability": None,
            "final_outcome": "Nessuna interazione prevista",
        }

    values_sign = {**user_dict, **movie_dict}
    sign_row_dict = {col: values_sign.get(col, 0.0) for col in SIGN_FEATURE_COLS}
    sign_row = pd.DataFrame([sign_row_dict])[SIGN_FEATURE_COLS]

    sign_row_eval = sign_scaler.transform(sign_row) if sign_scaler is not None else sign_row
    sign_threshold = sign_bundle.get("threshold", DEFAULT_SIGN_THRESHOLD)
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
