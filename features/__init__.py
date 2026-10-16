"""
features – Modulo centralizzato per l'estrazione e gestione delle feature (MovieLens 10M).
Include:
  - common: utilità di I/O, parsing dei generi e costanti condivise
  - link_features: topologia, recency e attività a 30gg per Link Prediction (7 feature)
  - sign_features: attitudine, consenso film, baseline Koren e Signed SVD per Sign Prediction (15 feature)
"""

from features.common import (
    GLOBAL_MEAN_RATING,
    SECONDS_PER_DAY,
    WINDOW_30D_SEC,
    GENRE_MAP,
    load_dataset,
    save_split_csv,
    parse_genres_tuple,
    parse_genres_to_ids,
    compute_genre_match,
)

from features.link_features import (
    LINK_FEATURE_COLS,
    TARGET_COL as LINK_TARGET_COL,
    LinkNodeState,
    extract_link_features,
    sample_negative_pairs,
    build_link_dataset,
    load_link_data,
)

from features.sign_features import (
    SIGN_FEATURE_COLS,
    TARGET_COL as SIGN_TARGET_COL,
    UserSignState,
    MovieSignState,
    extract_sign_features,
    train_signed_svd_embeddings,
    compute_signed_svd_similarity,
    build_sign_dataset,
    load_sign_data,
)

__all__ = [
    "GLOBAL_MEAN_RATING",
    "SECONDS_PER_DAY",
    "WINDOW_30D_SEC",
    "GENRE_MAP",
    "load_dataset",
    "save_split_csv",
    "parse_genres_tuple",
    "parse_genres_to_ids",
    "compute_genre_match",
    "LINK_FEATURE_COLS",
    "LINK_TARGET_COL",
    "LinkNodeState",
    "extract_link_features",
    "sample_negative_pairs",
    "build_link_dataset",
    "load_link_data",
    "SIGN_FEATURE_COLS",
    "SIGN_TARGET_COL",
    "UserSignState",
    "MovieSignState",
    "extract_sign_features",
    "train_signed_svd_embeddings",
    "compute_signed_svd_similarity",
    "build_sign_dataset",
    "load_sign_data",
]
