"""
run_traditional.py – Script Eseguibile Principale per l'Approccio Tradizionale a Due Stadi.

Esegue:
  1. Verifica e caricamento dei dataset per Link e Sign Prediction.
  2. Inizializzazione e addestramento del Modello Link Prediction (Stadio 1).
  3. Inizializzazione e addestramento del Modello Sign Prediction (Stadio 2).
  4. Valutazione end-to-end a cascata (Pipeline a Due Stadi su test set congiunto).
  5. Salvataggio delle metriche in results/traditional_metrics.json.
  6. Dimostrazione dell'inferenza della pipeline su coppie d'esempio.

Uso:
  python run_traditional.py
  python run_traditional.py --model-type gbm --sign-threshold 0.4
  python run_traditional.py --skip-train
  python run_traditional.py --prepare-dataset
"""

import argparse
import json
from pathlib import Path

from config import RESULTS_DIR, LINK_MODEL_FILE, SIGN_MODEL_FILE
from traditional.dataset import TraditionalDataset, build_all_datasets
from traditional.trainer import TraditionalTrainer
from traditional.pipeline import (
    TraditionalTwoStagePipeline,
    prepare_joint_test_set,
    run_cascade_evaluation,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Addestramento e Valutazione Approccio Tradizionale a Due Stadi")
    parser.add_argument(
        "--model-type",
        type=str,
        default="gbm",
        choices=["gbm", "lr"],
        help="Tipo di classificatore: 'gbm' (HistGradientBoosting) o 'lr' (LogisticRegression)",
    )
    parser.add_argument(
        "--sign-threshold",
        type=float,
        default=0.4,
        help="Soglia decisionale per la classe positiva nel Sign Prediction (default: 0.4)",
    )
    parser.add_argument(
        "--link-threshold",
        type=float,
        default=0.5,
        help="Soglia decisionale per l'esistenza del link (default: 0.5)",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Se specificato, usa i modelli pre-addestrati su disco invece di riaddestrarli",
    )
    parser.add_argument(
        "--prepare-dataset",
        action="store_true",
        help="Se specificato, rigenera interamente tutti i file CSV della pipeline prima del training",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed per riproducibilità",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("PIPELINE TRADIZIONALE A DUE STADI: LINK & SIGN PREDICTION (MovieLens 10M)")
    print("=" * 70)
    print(f"  Parametri:")
    print(f"    - Modello:         {args.model_type.upper()}")
    print(f"    - Soglia Link:     {args.link_threshold}")
    print(f"    - Soglia Segno:    {args.sign_threshold}")
    print(f"    - Random Seed:     {args.seed}")
    print(f"    - Skip Train:      {args.skip_train}")

    dataset = TraditionalDataset()

    # 1. Preparazione Dataset se richiesta o se mancano file
    if args.prepare_dataset or not dataset.is_prepared():
        print("\n[Dataset] Generazione file elaborati per l'approccio tradizionale...")
        build_all_datasets()

    # 2. Addestramento o Caricamento Modelli
    if args.skip_train and LINK_MODEL_FILE.exists() and SIGN_MODEL_FILE.exists():
        print("\n[Training] Skip train attivo: caricamento modelli pre-addestrati da disco...")
        pipeline = TraditionalTwoStagePipeline.load(
            link_model_path=LINK_MODEL_FILE,
            sign_model_path=SIGN_MODEL_FILE,
            link_threshold=args.link_threshold,
            sign_threshold=args.sign_threshold,
        )
        link_metrics = {}
        sign_metrics = {}
    else:
        trainer = TraditionalTrainer(
            link_model_type=args.model_type,
            sign_model_type=args.model_type,
            sign_threshold=args.sign_threshold,
            link_threshold=args.link_threshold,
            random_state=args.seed,
        )
        results = trainer.train_all(save_models=True)
        link_metrics = results["link_metrics"]
        sign_metrics = results["sign_metrics"]

        pipeline = TraditionalTwoStagePipeline(
            link_bundle=results["link_bundle"],
            sign_bundle=results["sign_bundle"],
            link_threshold=args.link_threshold,
            sign_threshold=args.sign_threshold,
        )

    # 3. Valutazione a Cascata sul Test Set
    print("\n[Valutazione] Preparazione test set congiunto (reali + negativi)...")
    test_df = prepare_joint_test_set(seed=args.seed)
    cascade_metrics = pipeline.evaluate_cascade(test_df)

    # 4. Salvataggio Metriche
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_file = RESULTS_DIR / "traditional_metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump({
            "config": vars(args),
            "link_stage_metrics": link_metrics,
            "sign_stage_metrics": sign_metrics,
            "cascade_metrics": cascade_metrics,
        }, f, indent=2)
    print(f"\n[Risultati] Metriche salvate in: {metrics_file}")

    # 5. Dimostrazione Inferenza su Coppie d'Esempio
    print("\n" + "=" * 70)
    print("DIMOSTRAZIONE INFERENZA: PIPELINE TRADIZIONALE A DUE STADI")
    print("=" * 70)
    sample_pairs = test_df.head(5)
    for idx, row in sample_pairs.iterrows():
        u_dict = {col: row[col] for col in test_df.columns if col.startswith("user_") or col == "user_id"}
        m_dict = {col: row[col] for col in test_df.columns if col.startswith("movie_") or col in ["movie_id", "genres", "title", "rating_baseline_score", "user_movie_rating_diff", "user_movie_optimism", "user_genre_mean_rating", "signed_svd_similarity", "genre_match_score"]}
        
        pred = pipeline.predict_pair(u_dict, m_dict)
        u_id = int(row.get("user_id", -1))
        m_id = int(row.get("movie_id", -1))
        print(f"  Coppia (User: {u_id}, Movie: {m_id}):")
        print(f"    --> Probabilità Link: {pred['link_probability']:.4f} | Link Previsto: {pred['link_exists']}")
        if pred["link_exists"]:
            print(f"    --> Probabilità Segno Positivo: {pred['sign_probability']:.4f} | Classe: {pred['predicted_sign']}")
        print(f"    --> Verdetto Finale: {pred['final_outcome']}\n")


if __name__ == "__main__":
    main()
