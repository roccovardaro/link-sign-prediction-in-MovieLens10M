"""
Step 2 – Pulizia ed esplorazione.

Operazioni:
  - Validazione tipi (identificativi, rating, timestamp)
  - Gestione valori mancanti
  - Rimozione duplicati effettivi
  - Gestione recensioni ripetute della stessa coppia (user, product)
  - Pulizia HTML e spazi superflui dal testo
  - Ordinamento cronologico
  - Flag per recensioni senza testo
  - Calcolo statistiche esplorative
"""

import json
import re

import pandas as pd

from config import RESULTS_DIR, STATS_FILE


# ── Pulizia ──────────────────────────────────────────────────────────────

def validate_types(df):
    """Assicura i tipi corretti per le colonne chiave."""
    df["user_id"] = df["user_id"].astype(str)
    df["parent_asin"] = df["parent_asin"].astype(str)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


def handle_missing(df):
    """
    Gestisce i valori mancanti:
      - Rimuove righe senza user_id, parent_asin o rating validi.
      - Le recensioni senza testo sono mantenute e segnalate con flag.
    """
    before = len(df)
    df = df.dropna(subset=["user_id", "parent_asin", "rating"]).copy()
    # Rating deve essere intero 1-5
    df = df[df["rating"].between(1, 5)].copy()
    df["rating"] = df["rating"].astype(int)

    # Flag: testo assente
    df["has_text"] = df["review_text"].astype(str).str.strip().ne("")

    after = len(df)
    print(f"  Valori mancanti/invalidi rimossi: {before - after:,}")
    return df


def remove_exact_duplicates(df):
    """Rimuove duplicati completamente identici su tutte le colonne."""
    before = len(df)
    df = df.drop_duplicates().copy()
    after = len(df)
    print(f"  Duplicati esatti rimossi: {before - after:,}")
    return df


def handle_repeated_reviews(df):
    """
    Se la stessa coppia (user_id, parent_asin) ha più recensioni,
    mantiene solo l'ultima in ordine cronologico.
    """
    before = len(df)
    df = (
        df.sort_values("timestamp")
        .drop_duplicates(subset=["user_id", "parent_asin"], keep="last")
        .copy()
    )
    after = len(df)
    print(f"  Recensioni ripetute (stessa coppia) rimosse: {before - after:,}")
    return df


def clean_html_and_whitespace(text):
    """Rimuove tag HTML e spazi/newline superflui da una stringa."""
    if not isinstance(text, str) or text.strip() == "":
        return text
    # Rimuove tag HTML
    text = re.sub(r"<[^>]+>", " ", text)
    # Rimuove entità HTML comuni
    text = re.sub(r"&[a-zA-Z]+;", " ", text)
    # Normalizza spazi/newline multipli
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_text_column(df):
    """Applica la pulizia HTML/spazi alla colonna review_text."""
    df["review_text"] = df["review_text"].apply(clean_html_and_whitespace)
    print("  Pulizia HTML e spazi completata")
    return df


def sort_chronologically(df):
    """Ordina il DataFrame per timestamp crescente."""
    df = df.sort_values("timestamp").reset_index(drop=True)
    print("  Ordinamento cronologico completato")
    return df


# ── Esplorazione ─────────────────────────────────────────────────────────

def compute_stats(df):
    """
    Calcola statistiche esplorative:
      - Numerosità totale
      - Distribuzione delle stelle
      - Attività utenti (recensioni per utente)
      - Popolarità prodotti (recensioni per prodotto)
      - Disponibilità testi
      - Range temporale
    """
    stats = {
        "totale_recensioni": int(len(df)),
        "utenti_unici": int(df["user_id"].nunique()),
        "prodotti_unici": int(df["parent_asin"].nunique()),
        "distribuzione_stelle": df["rating"].value_counts().sort_index().to_dict(),
        "percentuale_con_testo": round(df["has_text"].mean() * 100, 2),
        "recensioni_per_utente": {
            "media": round(df.groupby("user_id").size().mean(), 2),
            "mediana": float(df.groupby("user_id").size().median()),
            "max": int(df.groupby("user_id").size().max()),
        },
        "recensioni_per_prodotto": {
            "media": round(df.groupby("parent_asin").size().mean(), 2),
            "mediana": float(df.groupby("parent_asin").size().median()),
            "max": int(df.groupby("parent_asin").size().max()),
        },
        "range_temporale": {
            "inizio": str(df["timestamp"].min()),
            "fine": str(df["timestamp"].max()),
        },
    }

    # Converte le chiavi int a str per JSON
    stats["distribuzione_stelle"] = {
        str(k): int(v) for k, v in stats["distribuzione_stelle"].items()
    }

    return stats


def save_stats(stats, path=STATS_FILE):
    """Salva le statistiche in formato JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"  Statistiche salvate in: {path}")


def print_stats(stats):
    """Stampa un report esplorativo leggibile."""
    print("\n  ── Report Esplorativo ──")
    print(f"  Totale recensioni:    {stats['totale_recensioni']:,}")
    print(f"  Utenti unici:         {stats['utenti_unici']:,}")
    print(f"  Prodotti unici:       {stats['prodotti_unici']:,}")
    print(f"  Con testo:            {stats['percentuale_con_testo']}%")
    print(f"  Range temporale:      {stats['range_temporale']['inizio']} → {stats['range_temporale']['fine']}")
    print("  Distribuzione stelle:")
    for star, count in sorted(stats["distribuzione_stelle"].items()):
        print(f"    {star} stelle: {count:,}")
    print()


# ── Entry point ──────────────────────────────────────────────────────────

def run(df):
    """Esegue lo step 2 completo: pulizia + esplorazione."""
    print("=" * 60)
    print("STEP 2 – Pulizia ed esplorazione")
    print("=" * 60)

    df = validate_types(df)
    df = handle_missing(df)
    df = remove_exact_duplicates(df)
    df = handle_repeated_reviews(df)
    df = clean_text_column(df)
    df = sort_chronologically(df)

    stats = compute_stats(df)
    print_stats(stats)
    save_stats(stats)

    return df, stats


if __name__ == "__main__":
    from step1_acquisizione import run as step1_run

    df = step1_run()
    df, stats = run(df)
    print(df.head())
