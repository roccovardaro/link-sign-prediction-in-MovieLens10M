"""
crea_dataset.py – Pipeline completa per la creazione del dataset.

Esegue in sequenza:
  Step 1: Acquisizione dei dati
  Step 2: Pulizia ed esplorazione
  Step 3: Definizione del grafo e delle etichette
  Step 5: Separazione temporale train / validation / test
  Step 6: Estrazione caratteristiche strutturali

Uso:
  python crea_dataset.py
"""

from step1_acquisizione import run as step1
from step2_pulizia import run as step2
from step3_grafo import run as step3
from step5_split import run as step5
from step6_features import run as step6


def main():
    # Step 1 – Acquisizione
    df = step1()

    # Step 2 – Pulizia ed esplorazione
    df, stats = step2(df)

    # Step 3 – Grafo ed etichette
    df, events, edges, binary_df = step3(df)

    # Step 5 – Split temporale
    splits = step5(binary_df)

    # Step 6 – Feature strutturali
    enriched = step6(splits, events)

    print("=" * 60)
    print("PIPELINE COMPLETATA")
    print("=" * 60)
    print(f"  Recensioni finali:  {len(df):,}")
    print(f"  Archi del grafo:    {len(edges):,}")
    print(f"  Subset binario:     {len(binary_df):,}")
    print(f"  Eventi salvati:     {len(events):,}")
    print(f"  Train (+ feat):     {len(enriched['train']):,}")
    print(f"  Validation (+ feat):{len(enriched['val']):,}")
    print(f"  Test (+ feat):      {len(enriched['test']):,}")


if __name__ == "__main__":
    main()

