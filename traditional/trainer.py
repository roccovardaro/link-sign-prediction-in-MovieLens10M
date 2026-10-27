"""
traditional/trainer.py – Macrofase Addestramento, Ottimizzazione e Valutazione (MovieLens 10M).

Gestisce:
  1. Addestramento del modello di Link Prediction (Stadio 1).
  2. Addestramento del modello di Sign Prediction (Stadio 2).
  3. Calcolo delle metriche scientifiche rigorose (ROC-AUC, PR-AUC, F1, Balanced Acc, Confusion Matrix).
  4. Salvataggio su disco dei modelli addestrati (link_model.joblib, sign_model.joblib).
"""

from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import joblib
import pandas as pd
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
    TRAIN_LINK_CSV,
    VAL_LINK_CSV,
    TEST_LINK_CSV,
    TRAIN_SIGN_CSV,
    VAL_SIGN_CSV,
    TEST_SIGN_CSV,
    LINK_MODEL_FILE,
    SIGN_MODEL_FILE,
)
from traditional.features.link_features import LINK_FEATURE_COLS, load_link_data
from traditional.features.sign_features import SIGN_FEATURE_COLS, load_sign_data
from traditional.model import (
    create_link_model,
    create_sign_model,
    DEFAULT_LINK_THRESHOLD,
    DEFAULT_SIGN_THRESHOLD,
)


def evaluate_link(model: Any, X: pd.DataFrame, y: pd.Series, split_name: str, scaler: Optional[Any] = None) -> dict:
    """Valuta il modello di link prediction calcolando le metriche prestazionali."""
    X_eval = scaler.transform(X) if scaler is not None else X
    y_pred = model.predict(X_eval)
    y_proba = model.predict_proba(X_eval)[:, 1]

    is_binary = len(set(y)) > 1

    metrics = {
        "accuracy": float(accuracy_score(y, y_pred)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
    }

    if is_binary:
        metrics.update({
            "balanced_acc": float(balanced_accuracy_score(y, y_pred)),
            "precision": float(precision_score(y, y_pred, zero_division=0)),
            "f1": float(f1_score(y, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, y_proba)),
            "pr_auc": float(average_precision_score(y, y_proba)),
        })
    else:
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


def evaluate_sign(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    split_name: str,
    scaler: Optional[Any] = None,
    threshold: float = DEFAULT_SIGN_THRESHOLD,
) -> dict:
    """Calcola e stampa le metriche essenziali per la classificazione del segno."""
    X_eval = scaler.transform(X) if scaler is not None else X

    y_proba = model.predict_proba(X_eval)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y, y_pred)),
        "balanced_acc": float(balanced_accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1_macro": float(f1_score(y, y_pred, average="macro", zero_division=0)),
        "f1_pos": float(f1_score(y, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, y_proba)),
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


def train_link_model(
    train_csv: Path = TRAIN_LINK_CSV,
    val_csv: Path = VAL_LINK_CSV,
    test_csv: Path = TEST_LINK_CSV,
    save_model: bool = True,
    model_type: str = "gbm",
    random_state: int = 42,
    save_path: Path = LINK_MODEL_FILE,
) -> Tuple[Any, Optional[Any], dict]:
    """
    Addestra e valuta il modello di Link Prediction:
      - Train con negative sampling per distinguere presenza/assenza di interazione.
      - Val e Test valutati sui dati reali.
    """
    print("=" * 60)
    print("STADIO 1: LINK PREDICTION (Training & Valutazione)")
    print("=" * 60)
    print(f"  Feature impiegate ({len(LINK_FEATURE_COLS)}):")
    for f in LINK_FEATURE_COLS:
        print(f"    - {f}")

    print("\n  Caricamento dataset Link Prediction...")
    X_train, y_train = load_link_data(train_csv)
    X_val, y_val     = load_link_data(val_csv)
    X_test, y_test   = load_link_data(test_csv)

    print(f"  Train: {len(X_train):,} coppie ({(y_train == 1).sum():,} reali, {(y_train == 0).sum():,} negative)")
    print(f"  Val:   {len(X_val):,} link reali (dati veri, no negative sampling)")
    print(f"  Test:  {len(X_test):,} link reali (dati veri, no negative sampling)")

    model, scaler = create_link_model(model_type=model_type, random_state=random_state)
    print(f"\n  Addestramento modello Link Prediction ({model_type.upper()})...")

    if scaler is not None:
        X_train_fit = scaler.fit_transform(X_train)
        model.fit(X_train_fit, y_train)
    else:
        model.fit(X_train, y_train)

    evaluate_link(model, X_val, y_val, "Validation (Dati Veri)", scaler=scaler)
    test_metrics = evaluate_link(model, X_test, y_test, "Test (Dati Veri)", scaler=scaler)

    if save_model:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model": model,
            "scaler": scaler,
            "features": LINK_FEATURE_COLS,
            "model_type": model_type,
        }, save_path)
        print(f"\n  Modello Link Prediction salvato in: {save_path}")

    print("\nStadio 1 completato con successo.\n")
    return model, scaler, test_metrics


def train_sign_model(
    train_csv: Path = TRAIN_SIGN_CSV,
    val_csv: Path = VAL_SIGN_CSV,
    test_csv: Path = TEST_SIGN_CSV,
    save_model: bool = True,
    model_type: str = "gbm",
    threshold: float = DEFAULT_SIGN_THRESHOLD,
    random_state: int = 42,
    save_path: Path = SIGN_MODEL_FILE,
) -> Tuple[Any, Optional[Any], dict, float]:
    """Addestramento e test del modello di Sign Prediction (Stadio 2)."""
    print("=" * 60)
    print("STADIO 2: SIGN PREDICTION (Segno dell'Interazione)")
    print("=" * 60)
    print(f"  Feature impiegate ({len(SIGN_FEATURE_COLS)}):")
    for f in SIGN_FEATURE_COLS:
        print(f"    - {f}")

    print("\n  Caricamento dataset Sign Prediction...")
    X_train, y_train = load_sign_data(train_csv)
    X_val, y_val     = load_sign_data(val_csv)
    X_test, y_test   = load_sign_data(test_csv)

    print(f"  Train: {len(X_train):,} righe | Val: {len(X_val):,} | Test: {len(X_test):,}")

    model, scaler = create_sign_model(model_type=model_type, random_state=random_state)
    print(f"\n  Addestramento modello Sign Prediction ({model_type.upper()})...")

    if scaler is not None:
        X_train_fit = scaler.fit_transform(X_train)
        model.fit(X_train_fit, y_train)
    else:
        model.fit(X_train, y_train)

    evaluate_sign(model, X_val, y_val, "Validation", scaler=scaler, threshold=threshold)
    test_metrics = evaluate_sign(model, X_test, y_test, "Test", scaler=scaler, threshold=threshold)

    if save_model:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model": model,
            "scaler": scaler,
            "features": SIGN_FEATURE_COLS,
            "threshold": threshold,
            "model_type": model_type,
        }, save_path)
        print(f"\n  Modello Sign Prediction salvato in: {save_path} (Soglia: {threshold:.2f})")

    print("\nStadio 2 completato con successo.\n")
    return model, scaler, test_metrics, threshold


class TraditionalTrainer:
    """Trainer specializzato per i modelli dell'approccio tradizionale."""

    def __init__(
        self,
        link_model_type: str = "gbm",
        sign_model_type: str = "gbm",
        sign_threshold: float = DEFAULT_SIGN_THRESHOLD,
        link_threshold: float = DEFAULT_LINK_THRESHOLD,
        random_state: int = 42,
    ):
        self.link_model_type = link_model_type
        self.sign_model_type = sign_model_type
        self.sign_threshold = sign_threshold
        self.link_threshold = link_threshold
        self.random_state = random_state

    def train_link(self, save_model: bool = True, save_path: Path = LINK_MODEL_FILE):
        return train_link_model(
            save_model=save_model,
            model_type=self.link_model_type,
            random_state=self.random_state,
            save_path=save_path,
        )

    def train_sign(self, save_model: bool = True, save_path: Path = SIGN_MODEL_FILE):
        return train_sign_model(
            save_model=save_model,
            model_type=self.sign_model_type,
            threshold=self.sign_threshold,
            random_state=self.random_state,
            save_path=save_path,
        )

    def train_all(self, save_models: bool = True) -> dict:
        """Addestra congiuntamente entrambi gli stadi e restituisce bundle e metriche."""
        link_model, link_scaler, link_metrics = self.train_link(save_model=save_models)
        sign_model, sign_scaler, sign_metrics, threshold = self.train_sign(save_model=save_models)

        link_bundle = {"model": link_model, "scaler": link_scaler, "features": LINK_FEATURE_COLS}
        sign_bundle = {"model": sign_model, "scaler": sign_scaler, "features": SIGN_FEATURE_COLS, "threshold": threshold}

        return {
            "link_bundle": link_bundle,
            "sign_bundle": sign_bundle,
            "link_metrics": link_metrics,
            "sign_metrics": sign_metrics,
        }
