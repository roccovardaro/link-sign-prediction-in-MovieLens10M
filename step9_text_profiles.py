"""
Step 9 – Costruzione dei profili testuali con TF-IDF.

Per ogni data t:
  1. Trasforma i testi delle recensioni passate con TF-IDF
  2. Calcola il vettore medio delle recensioni scritte da ciascun utente
  3. Calcola il vettore medio delle recensioni ricevute da ciascun prodotto
  4. Recupera i due vettori per ogni coppia candidata

Il vettorizzatore viene addestrato SOLO sulla storia iniziale
(precedente al primo esempio predittivo) per evitare leakage.

Aggiunge anche:
  - user_n_texts / prod_n_texts: n. di testi disponibili nel profilo
  - user_empty_profile / prod_empty_profile: flag profilo vuoto (0/1)

Da salvare: vettorizzatore e profili testuali storici.
"""

import numpy as np
import pandas as pd
from scipy.sparse import vstack as sparse_vstack
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

from config import (
    TFIDF_MAX_FEATURES,
    TFIDF_MIN_DF,
    TFIDF_SUBLINEAR_TF,
    VECTORIZER_FILE,
    DATA_PROCESSED,
    CSV_PROCESSED,
    EVENTS_CSV,
    TRAIN_CSV,
    VAL_CSV,
    TEST_CSV,
    TRAIN_FILE,
    VAL_FILE,
    TEST_FILE,
)

TEXT_FEATURES = [
    "user_n_texts", "user_empty_profile", "user_tfidf_norm", "user_tfidf_mean", "user_tfidf_max", "user_tfidf_nonzero",
    "prod_n_texts", "prod_empty_profile", "prod_tfidf_norm", "prod_tfidf_mean", "prod_tfidf_max", "prod_tfidf_nonzero",
]


# ── Addestramento del vettorizzatore ─────────────────────────────────────

def fit_vectorizer(events_df, first_prediction_ts):
    """
    Addestra il TfidfVectorizer sulla storia iniziale: tutti i testi
    con timestamp < first_prediction_ts.

    Parameters
    ----------
    events_df : pd.DataFrame
        Tabella eventi completa (incluse neutrali) con colonna 'review_text'.
    first_prediction_ts : pd.Timestamp
        Timestamp del primo esempio predittivo (min timestamp del train).

    Returns
    -------
    TfidfVectorizer
        Vettorizzatore addestrato.
    """
    # Solo testi precedenti al primo esempio predittivo
    initial_history = events_df[events_df["timestamp"] < first_prediction_ts].copy()

    # Filtra righe senza testo
    texts = initial_history["review_text"].fillna("").astype(str)
    texts = texts[texts.str.strip() != ""]

    print(f"  Testi per fit del vectorizer: {len(texts):,}")
    print(f"  Periodo: fino a {first_prediction_ts}")

    vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        min_df=TFIDF_MIN_DF,
        sublinear_tf=TFIDF_SUBLINEAR_TF,
    )
    vectorizer.fit(texts)

    print(f"  Vocabolario: {len(vectorizer.vocabulary_):,} termini")
    return vectorizer


def save_vectorizer(vectorizer, path=VECTORIZER_FILE):
    """Salva il vettorizzatore su disco con joblib."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, path)
    print(f"  Vectorizer salvato: {path}")


# ── Calcolo profili testuali per uno split ───────────────────────────────

def build_text_profiles(split_df, events_df, vectorizer):
    """
    Per ogni riga dello split, costruisce i profili testuali TF-IDF
    di utente e prodotto usando SOLO i testi passati.

    Approccio incrementale:
      - Ordina split e eventi per timestamp
      - Mantiene profili cumulativi per utente e prodotto
      - Per ogni riga, aggiorna i profili con i nuovi testi passati

    Parameters
    ----------
    split_df : pd.DataFrame
        Uno split (train/val/test) con colonne user_id, parent_asin, timestamp.
    events_df : pd.DataFrame
        Tabella eventi completa ordinata per timestamp.
    vectorizer : TfidfVectorizer
        Vettorizzatore già addestrato (fit su storia iniziale).

    Returns
    -------
    pd.DataFrame
        split_df con le feature testuali aggiunte.
    """
    events_df = events_df.sort_values("timestamp").reset_index(drop=True)
    split_df = split_df.sort_values("timestamp").reset_index(drop=True)

    n_features = len(vectorizer.vocabulary_)

    # Profili cumulativi: somma dei vettori TF-IDF e conteggio testi
    user_sum = {}    # user_id → np.array (somma vettori)
    user_count = {}  # user_id → int (n. testi)
    prod_sum = {}    # parent_asin → np.array
    prod_count = {}  # parent_asin → int

    events_idx = 0
    n_events = len(events_df)
    records = []

    for i, row in split_df.iterrows():
        ts = row["timestamp"]
        uid = row["user_id"]
        pid = row["parent_asin"]

        # ── Aggiorna profili con eventi passati (timestamp < corrente) ──
        while events_idx < n_events and events_df.iloc[events_idx]["timestamp"] < ts:
            ev = events_df.iloc[events_idx]
            text = str(ev.get("review_text", "")).strip()

            if text:
                # Trasforma il testo con il vectorizer → vettore sparse
                vec = vectorizer.transform([text]).toarray().flatten()

                # Aggiorna profilo utente
                ev_uid = ev["user_id"]
                if ev_uid not in user_sum:
                    user_sum[ev_uid] = np.zeros(n_features)
                    user_count[ev_uid] = 0
                user_sum[ev_uid] += vec
                user_count[ev_uid] += 1

                # Aggiorna profilo prodotto
                ev_pid = ev["parent_asin"]
                if ev_pid not in prod_sum:
                    prod_sum[ev_pid] = np.zeros(n_features)
                    prod_count[ev_pid] = 0
                prod_sum[ev_pid] += vec
                prod_count[ev_pid] += 1

            events_idx += 1

        # ── Recupera profili per la coppia corrente ──
        feat_row = _get_profile_features(
            uid, pid, user_sum, user_count, prod_sum, prod_count, n_features
        )
        records.append(feat_row)

        # Log progresso
        if (i + 1) % 50_000 == 0:
            print(f"    ... {i + 1:,}/{len(split_df):,} righe elaborate")

    # Rimuove vecchie feature testuali se già presenti (riesecuzione idempotente)
    cols_to_drop = [c for c in TEXT_FEATURES if c in split_df.columns]
    if cols_to_drop:
        split_df = split_df.drop(columns=cols_to_drop)

    features_df = pd.DataFrame(records, index=split_df.index)
    result = pd.concat([split_df, features_df], axis=1)
    return result


def _get_profile_features(uid, pid, user_sum, user_count, prod_sum, prod_count, n_features):
    """
    Recupera le feature del profilo testuale per una coppia (utente, prodotto).

    Restituisce il vettore medio TF-IDF per utente e prodotto,
    più metadati (n_texts, empty_profile).

    Per mantenere il DataFrame gestibile, salva solo statistiche aggregate
    del vettore TF-IDF (norma, media, max) anziché tutti i 20K valori.
    """
    feat = {}

    # ── Profilo utente ──
    if uid in user_sum and user_count[uid] > 0:
        user_avg = user_sum[uid] / user_count[uid]
        feat["user_n_texts"] = user_count[uid]
        feat["user_empty_profile"] = 0
        feat["user_tfidf_norm"] = round(float(np.linalg.norm(user_avg)), 6)
        feat["user_tfidf_mean"] = round(float(np.mean(user_avg)), 6)
        feat["user_tfidf_max"] = round(float(np.max(user_avg)), 6)
        feat["user_tfidf_nonzero"] = int(np.count_nonzero(user_avg))
    else:
        feat["user_n_texts"] = 0
        feat["user_empty_profile"] = 1
        feat["user_tfidf_norm"] = 0.0
        feat["user_tfidf_mean"] = 0.0
        feat["user_tfidf_max"] = 0.0
        feat["user_tfidf_nonzero"] = 0

    # ── Profilo prodotto ──
    if pid in prod_sum and prod_count[pid] > 0:
        prod_avg = prod_sum[pid] / prod_count[pid]
        feat["prod_n_texts"] = prod_count[pid]
        feat["prod_empty_profile"] = 0
        feat["prod_tfidf_norm"] = round(float(np.linalg.norm(prod_avg)), 6)
        feat["prod_tfidf_mean"] = round(float(np.mean(prod_avg)), 6)
        feat["prod_tfidf_max"] = round(float(np.max(prod_avg)), 6)
        feat["prod_tfidf_nonzero"] = int(np.count_nonzero(prod_avg))
    else:
        feat["prod_n_texts"] = 0
        feat["prod_empty_profile"] = 1
        feat["prod_tfidf_norm"] = 0.0
        feat["prod_tfidf_mean"] = 0.0
        feat["prod_tfidf_max"] = 0.0
        feat["prod_tfidf_nonzero"] = 0

    return feat


# ── Report e salvataggio ─────────────────────────────────────────────────

def print_text_summary(df, split_name):
    """Stampa un riepilogo dei profili testuali."""
    print(f"\n  ── Profili testuali: {split_name} ──")
    print(f"  Righe: {len(df):,}")

    empty_user = (df["user_empty_profile"] == 1).sum()
    empty_prod = (df["prod_empty_profile"] == 1).sum()
    print(f"  Profilo utente vuoto:   {empty_user:,} "
          f"({empty_user / len(df) * 100:.1f}%)")
    print(f"  Profilo prodotto vuoto: {empty_prod:,} "
          f"({empty_prod / len(df) * 100:.1f}%)")

    for col in ["user_n_texts", "user_tfidf_norm",
                "prod_n_texts", "prod_tfidf_norm"]:
        if col in df.columns:
            print(f"  {col}: media={df[col].mean():.4f}, "
                  f"mediana={df[col].median():.4f}")


def save_text_splits(splits):
    """Salva i split finali arricchiti direttamente nei 3 file CSV/Parquet utili."""
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    CSV_PROCESSED.mkdir(parents=True, exist_ok=True)

    file_map = {
        "train": (TRAIN_FILE, TRAIN_CSV),
        "val":   (VAL_FILE, VAL_CSV),
        "test":  (TEST_FILE, TEST_CSV),
    }

    for name, split_df in splits.items():
        parquet_path, csv_path = file_map[name]
        split_df.to_parquet(parquet_path, index=False)
        split_df.to_csv(csv_path, index=False)
        print(f"  {name:>5s} finale con profili salvato: {csv_path}")


# ── Entry point ──────────────────────────────────────────────────────────

def run(train_path=TRAIN_CSV, val_path=VAL_CSV, test_path=TEST_CSV, events_path=EVENTS_CSV):
    """
    Esegue lo step 9: costruzione profili testuali TF-IDF.
    """
    print("=" * 60)
    print("STEP 9 – Costruzione profili testuali (TF-IDF)")
    print("=" * 60)

    print(f"  Caricamento split da: {train_path}, {val_path}, {test_path}")
    splits = {
        "train": pd.read_csv(train_path, parse_dates=["timestamp"]),
        "val":   pd.read_csv(val_path, parse_dates=["timestamp"]),
        "test":  pd.read_csv(test_path, parse_dates=["timestamp"]),
    }

    print(f"  Caricamento eventi da: {events_path}")
    events_df = pd.read_csv(events_path, parse_dates=["timestamp"])

    # Il primo esempio predittivo è il min timestamp del train
    first_ts = splits["train"]["timestamp"].min()

    # 1. Addestra il vectorizer sulla storia iniziale
    vectorizer = fit_vectorizer(events_df, first_ts)
    save_vectorizer(vectorizer)

    # 2. Calcola profili per ogni split
    text_enriched = {}
    for name in ["train", "val", "test"]:
        print(f"\n  Elaborazione {name}...")
        text_enriched[name] = build_text_profiles(
            splits[name], events_df, vectorizer
        )
        print_text_summary(text_enriched[name], name)

    save_text_splits(text_enriched)

    print()
    return text_enriched, vectorizer


if __name__ == "__main__":
    text_enriched, vectorizer = run()
