"""
Package traditional: Implementazione Modulare dell'Approccio Tradizionale a Due Stadi
per Link Prediction e Sign Prediction (MovieLens 10M).
"""

from traditional.dataset import TraditionalDataset, build_all_datasets
from traditional.model import (
    create_link_model,
    create_sign_model,
    TraditionalModelBundle,
    LINK_FEATURE_COLS,
    SIGN_FEATURE_COLS,
    DEFAULT_LINK_THRESHOLD,
    DEFAULT_SIGN_THRESHOLD,
)
from traditional.trainer import (
    TraditionalTrainer,
    train_link_model,
    train_sign_model,
    evaluate_link,
    evaluate_sign,
)
from traditional.pipeline import (
    TraditionalTwoStagePipeline,
    load_or_train_models,
    prepare_joint_test_set,
    run_cascade_evaluation,
    predict_pair,
)

__all__ = [
    "TraditionalDataset",
    "build_all_datasets",
    "create_link_model",
    "create_sign_model",
    "TraditionalModelBundle",
    "LINK_FEATURE_COLS",
    "SIGN_FEATURE_COLS",
    "DEFAULT_LINK_THRESHOLD",
    "DEFAULT_SIGN_THRESHOLD",
    "TraditionalTrainer",
    "train_link_model",
    "train_sign_model",
    "evaluate_link",
    "evaluate_sign",
    "TraditionalTwoStagePipeline",
    "load_or_train_models",
    "prepare_joint_test_set",
    "run_cascade_evaluation",
    "predict_pair",
]
