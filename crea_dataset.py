"""
crea_dataset.py – Pipeline completa per la creazione e preparazione del dataset MovieLens 10M.

Ogni step è indipendente e comunica tramite i file CSV utili salvati su disco:
  Step 1: Acquisizione dei dati MovieLens 10M (salva raw_ratings.csv)
  Step 2: Pulizia ed esplorazione (salva clean_ratings.csv e exploration_stats.json)
  Step 3: Definizione del grafo bipartito ed etichette (salva events.csv)
  Step 5: Split temporale train / validation / test (salva train.csv, val.csv, test.csv)
  Step 6: Estrazione caratteristiche strutturali e collaborative (arricchisce train/val/test.csv)

Uso:
  python crea_dataset.py
"""

import time
from step1_acquisizione import run as step1
from step2_pulizia import run as step2
from step3_grafo import run as step3
from step5_split import run as step5
from step6_features import run as step6
from step6_1_sign_features import run as step6_1


def main():
    start_total = time.time()
    print("=" * 60)
    print("AVVIO PIPELINE COMPLETA CREAZIONE DATASET (MovieLens 10M)")
    print("=" * 60)

    # Step 1 – Acquisizione
    t0 = time.time()
    step1()
    print(f"--> Step 1 completato in {time.time() - t0:.1f}s\n")

    # Step 2 – Pulizia ed esplorazione
    t0 = time.time()
    step2()
    print(f"--> Step 2 completato in {time.time() - t0:.1f}s\n")

    # Step 3 – Grafo ed etichette
    t0 = time.time()
    step3()
    print(f"--> Step 3 completato in {time.time() - t0:.1f}s\n")

    # Step 5 – Split temporale
    t0 = time.time()
    step5()
    print(f"--> Step 5 completato in {time.time() - t0:.1f}s\n")

    # Step 6 – Feature strutturali e collaborative per Link Prediction
    t0 = time.time()
    step6()
    print(f"--> Step 6 completato in {time.time() - t0:.1f}s\n")

    # Step 6.1 – Feature specifiche di gradimento per Sign Prediction
    t0 = time.time()
    step6_1()
    print(f"--> Step 6.1 completato in {time.time() - t0:.1f}s\n")

    elapsed = time.time() - start_total
    print("=" * 60)
    print(f"PIPELINE DATASET COMPLETATA CON SUCCESSO in {elapsed:.1f}s")
    print("=" * 60)
    print("\nFile generati in data/processed/csv/:")
    print("  1. raw_ratings.csv   (da Step 1)")
    print("  2. clean_ratings.csv (da Step 2)")
    print("  3. events.csv        (da Step 3)")
    print("  4. train_link.csv    (da Step 6 per Link Prediction)")
    print("  5. val_link.csv      (da Step 6 per Link Prediction)")
    print("  6. test_link.csv     (da Step 6 per Link Prediction)")
    print("  7. train_sign.csv    (da Step 6.1 per Sign Prediction)")
    print("  8. val_sign.csv      (da Step 6.1 per Sign Prediction)")
    print("  9. test_sign.csv     (da Step 6.1 per Sign Prediction)")
    print("\nPer addestrare e valutare i modelli:")
    print("  - Stadio 1 (Link Prediction):  python step7_link_prediction.py")
    print("  - Stadio 2 (Sign Prediction):  python step8_sign_prediction.py")
    print("  - Pipeline a due stadi:        python pipeline_two_stage.py\n")


if __name__ == "__main__":
    main()
