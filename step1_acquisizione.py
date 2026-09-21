"""
Step 1 – Acquisizione dei dati.

Carica il file review JSONL, conserva solo le colonne richieste,
unisce titolo e testo in un unico campo testuale.
"""

import pandas as pd

from config import RAW_FILE, KEEP_COLUMNS


def load_reviews(path=RAW_FILE, columns=KEEP_COLUMNS):
    """
    Carica il file JSONL delle recensioni e restituisce un DataFrame
    con le sole colonne di interesse.

    Parameters
    ----------
    path : Path
        Percorso del file JSONL grezzo.
    columns : list[str]
        Colonne da conservare.

    Returns
    -------
    pd.DataFrame
    """
    if not path.exists():
        raise FileNotFoundError(f"File non trovato: {path.resolve()}")

    df = pd.read_json(path, lines=True)
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"Colonne mancanti nel file: {missing}")

    df = df[columns].copy()
    return df


def merge_title_text(df):
    """
    Unisce title e text in un unico campo 'review_text'.
    Il titolo viene anteposto al testo, separato da un punto e spazio.
    Se uno dei due è mancante, viene usato solo l'altro.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con colonne 'title' e 'text'.

    Returns
    -------
    pd.DataFrame
        DataFrame con la nuova colonna 'review_text';
        le colonne 'title' e 'text' originali vengono rimosse.
    """
    title = df["title"].fillna("").astype(str).str.strip()
    text = df["text"].fillna("").astype(str).str.strip()

    # Concatena: "titolo. testo" – evita ". " iniziale se titolo vuoto
    df["review_text"] = (title + ". " + text).str.strip(". ").str.strip()

    # Se entrambi sono vuoti, il campo resta stringa vuota
    df.drop(columns=["title", "text"], inplace=True)
    return df


def run(path=RAW_FILE):
    """Esegue lo step 1 completo e restituisce il DataFrame."""
    print("=" * 60)
    print("STEP 1 – Acquisizione dei dati")
    print("=" * 60)

    df = load_reviews(path)
    print(f"  Righe caricate: {len(df):,}")
    print(f"  Colonne: {list(df.columns)}")

    df = merge_title_text(df)
    print(f"  Campo 'review_text' creato (title + text uniti)")
    print(f"  Colonne finali: {list(df.columns)}")
    print()

    return df


if __name__ == "__main__":
    df = run()
    print(df.head())
