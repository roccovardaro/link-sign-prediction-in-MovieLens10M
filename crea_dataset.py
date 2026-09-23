"""
crea_dataset.py – Pipeline completa per la creazione del dataset.

Ogni step è indipendente e comunica tramite i file CSV utili salvati su disco:
  Step 1: Acquisizione dei dati
  Step 2: Pulizia ed esplorazione
  Step 3: Definizione del grafo ed etichette (salva events.csv)
  Step 5: Split temporale train / validation / test (legge events.csv, salva train.csv, val.csv, test.csv)
  Step 6: Estrazione caratteristiche strutturali (legge train/val/test.csv ed events.csv, aggiorna train/val/test.csv)
  Step 9: Costruzione profili testuali TF-IDF (legge train/val/test.csv ed events.csv, finalizza train/val/test.csv)

Uso:
  python crea_dataset.py
"""

from step1_acquisizione import run as step1
from step2_pulizia import run as step2
from step3_grafo import run as step3
from step5_split import run as step5
from step6_features import run as step6
from step9_text_profiles import run as step9


def main():
    # Step 1 – Acquisizione (legge jsonl, salva raw_reviews.csv)
    step1()

    # Step 2 – Pulizia ed esplorazione (legge raw_reviews.csv, salva clean_reviews.csv)
    step2()

    # Step 3 – Grafo ed etichette (legge clean_reviews.csv, salva events.csv)
    step3()

    # Step 5 – Split temporale (legge events.csv, salva train.csv, val.csv, test.csv)
    step5()

    # Step 6 – Feature strutturali (legge train/val/test.csv ed events.csv, arricchisce train/val/test.csv)
    step6()

    # Step 9 – Profili testuali TF-IDF (legge train/val/test.csv ed events.csv, finalizza train/val/test.csv)
    step9()

    print("=" * 60)
    print("PIPELINE COMPLETATA CON SUCCESSO")
    print("=" * 60)
    print("\nTutti gli step hanno comunicato esclusivamente tramite file CSV su disco.")
    print("File CSV nella cartella data/processed/csv/:")
    print("  1. raw_reviews.csv    (da Step 1)")
    print("  2. clean_reviews.csv  (da Step 2)")
    print("  3. events.csv         (da Step 3)")
    print("  4. train.csv          (da Step 5, arricchito da Step 6 e 9)")
    print("  5. val.csv            (da Step 5, arricchito da Step 6 e 9)")
    print("  6. test.csv           (da Step 5, arricchito da Step 6 e 9)")


if __name__ == "__main__":
    main()
