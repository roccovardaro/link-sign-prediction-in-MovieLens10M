"""
traditional/model.py – Macrofase Definizione e Configurazione dei Modelli (MovieLens 10M).

Definisce i classificatori, i pre-processori, le feature e le soglie operative per i due stadi:
  - Stadio 1 (Link Prediction): HistGradientBoostingClassifier (default) o LogisticRegression + StandardScaler
  - Stadio 2 (Sign Prediction): HistGradientBoostingClassifier (default) o LogisticRegression + StandardScaler
"""

from typing import Tuple, Optional, Any, List
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from traditional.features.link_features import LINK_FEATURE_COLS, TARGET_COL as LINK_TARGET_COL
from traditional.features.sign_features import SIGN_FEATURE_COLS, TARGET_COL as SIGN_TARGET_COL

# Soglie decisionali di default
DEFAULT_LINK_THRESHOLD = 0.5
DEFAULT_SIGN_THRESHOLD = 0.4


class TraditionalModelBundle:
    """Contenitore per un modello addestrato, relativo scaler, feature e iperparametri."""

    def __init__(
        self,
        model: Any,
        scaler: Optional[StandardScaler] = None,
        features: Optional[List[str]] = None,
        threshold: float = 0.5,
        model_type: str = "gbm",
    ):
        self.model = model
        self.scaler = scaler
        self.features = features or []
        self.threshold = threshold
        self.model_type = model_type

    def to_dict(self) -> dict:
        """Serializza il bundle in un dizionario compatibile con joblib."""
        data = {
            "model": self.model,
            "scaler": self.scaler,
            "features": self.features,
            "model_type": self.model_type,
        }
        if self.threshold is not None:
            data["threshold"] = self.threshold
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "TraditionalModelBundle":
        """Ricostruisce il bundle da un dizionario serializzato."""
        return cls(
            model=data["model"],
            scaler=data.get("scaler"),
            features=data.get("features", []),
            threshold=data.get("threshold", 0.5),
            model_type=data.get("model_type", "gbm"),
        )


def create_link_model(
    model_type: str = "gbm",
    random_state: int = 42,
) -> Tuple[Any, Optional[StandardScaler]]:
    """
    Inizializza il modello per lo Stadio 1 (Link Prediction).
    - 'gbm': HistGradientBoostingClassifier con pesi bilanciati (raccomandato per dati non lineari).
    - 'lr': LogisticRegression standardizzata con pesi bilanciati.
    """
    if model_type == "lr":
        scaler = StandardScaler()
        model = LogisticRegression(max_iter=500, random_state=random_state, class_weight="balanced")
    elif model_type == "gbm":
        scaler = None
        model = HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.08,
            max_depth=6,
            class_weight="balanced",
            random_state=random_state,
        )
    else:
        raise ValueError(f"Tipo modello non riconosciuto: '{model_type}'. Scegliere tra 'gbm' e 'lr'.")

    return model, scaler


def create_sign_model(
    model_type: str = "gbm",
    random_state: int = 42,
) -> Tuple[Any, Optional[StandardScaler]]:
    """
    Inizializza il modello per lo Stadio 2 (Sign Prediction).
    - 'gbm': HistGradientBoostingClassifier con 300 alberi e pesi bilanciati.
    - 'lr': LogisticRegression standardizzata con pesi bilanciati.
    """
    if model_type == "lr":
        scaler = StandardScaler()
        model = LogisticRegression(max_iter=500, random_state=random_state, class_weight="balanced")
    elif model_type == "gbm":
        scaler = None
        model = HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.08,
            max_depth=6,
            class_weight="balanced",
            random_state=random_state,
        )
    else:
        raise ValueError(f"Tipo modello non riconosciuto: '{model_type}'. Scegliere tra 'gbm' e 'lr'.")

    return model, scaler
