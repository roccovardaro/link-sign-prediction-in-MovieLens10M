"""
run_gnn.py – Script Eseguibile Principale per il Modello GNN Multi-Task.

Esegue:
  1. Costruzione del grafo bipartito e campionamento dei negativi.
  2. Inizializzazione della Multi-Task Bipartite GNN.
  3. Addestramento congiunto oppure caricamento con --skip-train.
  4. Valutazione rigorosa su Test Set (metriche Link, Sign e Cascata a 3 classi).
  5. Salvataggio delle metriche in results/gnn_metrics.json.
  6. Dimostrazione della pipeline a due stadi su coppie d'esempio.

Uso:
  python run_gnn.py
  python run_gnn.py --skip-train
  python run_gnn.py --skip-train --link-threshold 0.50 --sign-threshold 0.50
  python run_gnn.py --sample-size 50000 --epochs 5
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any
import torch

from config import RESULTS_DIR, DATA_PROCESSED
from gnn.dataset import BipartiteGraphDataset
from gnn.model import MultiTaskBipartiteGNN
from gnn.trainer import GNNTrainer
from gnn.pipeline import GNNTwoStagePipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Addestramento e Valutazione GNN Multi-Task per Link e Sign Prediction")
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Se specificato, usa il modello pre-addestrato su disco (gnn_multitask_model.pt) invece di riaddestrarlo",
    )
    parser.add_argument(
        "--link-threshold",
        type=float,
        default=None,
        help="Soglia decisionale personalizzata per l'esistenza del link (default: valore ottimale calibrato)",
    )
    parser.add_argument(
        "--sign-threshold",
        type=float,
        default=None,
        help="Soglia decisionale personalizzata per il segno positivo (default: valore ottimale calibrato)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=1_000_000,
        help="Numero di righe di train da campionare per calcoli rapidi (0 per usare l'intero dataset)",
    )
    parser.add_argument("--epochs", type=int, default=15, help="Numero di epoche di addestramento")
    parser.add_argument("--dim", type=int, default=64, help="Dimensione degli embedding dei nodi")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Dimensione nascosta dei layer GNN e MLP")
    parser.add_argument("--num-layers", type=int, default=2, help="Numero di layer di convoluzione GNN")
    parser.add_argument("--lr", type=float, default=0.003, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=16384, help="Batch size delle coppie di supervisione")
    parser.add_argument("--lambda-sign", type=float, default=1.0, help="Peso lambda per la Sign Loss")
    parser.add_argument("--seed", type=int, default=42, help="Random seed per la riproducibilità")
    parser.add_argument(
        "--split-mode",
        type=str,
        default="transductive",
        choices=["transductive", "temporal"],
        help="Modalità di split: 'transductive' (standard per GNN edge split) o 'temporal' (split temporale rigido da CSV)",
    )
    parser.add_argument("--device", type=str, default=None, help="Dispositivo (cpu, mps, cuda)")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    checkpoint_file = DATA_PROCESSED / "gnn_multitask_model.pt"

    print("=" * 70)
    print("PIPELINE GNN MULTI-TASK: LINK & SIGN PREDICTION (Bipartite Graph)")
    print("=" * 70)
    print(f"  Parametri:")
    print(f"    - Skip Train:    {args.skip_train}")
    print(f"    - Split Mode:    {args.split_mode.upper()} {'(Standard GNN Edge-Holdout)' if args.split_mode == 'transductive' else '(Split Temporale CSV)'}")
    print(f"    - Sample Size:   {args.sample_size if args.sample_size > 0 else 'Intero Dataset'}")
    if not args.skip_train:
        print(f"    - Epoche:        {args.epochs}")
        print(f"    - Embedding Dim: {args.dim}")
        print(f"    - Hidden Dim:    {args.hidden_dim}")
        print(f"    - GNN Layers:    {args.num_layers}")
        print(f"    - Lambda Segno:  {args.lambda_sign}")
        print(f"    - Batch Size:    {args.batch_size}")

    sample_size = args.sample_size if args.sample_size > 0 else None

    # 1. Costruzione Grafo
    dataset_builder = BipartiteGraphDataset(
        sample_size=sample_size,
        negative_ratio=1.0,
        split_mode=args.split_mode,
        seed=args.seed,
    )
    bundle = dataset_builder.load_data()

    # 2. Inizializzazione Modello GNN
    model = MultiTaskBipartiteGNN(
        num_users=bundle.num_users,
        num_movies=bundle.num_movies,
        embedding_dim=args.dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    )

    # 3. Inizializzazione Trainer
    trainer = GNNTrainer(
        model=model,
        bundle=bundle,
        lr=args.lr,
        lambda_sign=args.lambda_sign,
        batch_size=args.batch_size,
        device=args.device,
    )

    # 4. Addestramento o Caricamento Checkpoint (Skip Train)
    if args.skip_train and checkpoint_file.exists():
        print(f"\n[Training] Skip train attivo: caricamento checkpoint pre-addestrato da disco...")
        checkpoint = torch.load(checkpoint_file, map_location=trainer.device)
        trainer.model.load_state_dict(checkpoint["model_state_dict"])

        calibrated_link_th = args.link_threshold if args.link_threshold is not None else checkpoint.get("optimal_link_threshold", 0.52)
        calibrated_sign_th = args.sign_threshold if args.sign_threshold is not None else checkpoint.get("optimal_sign_threshold", 0.5)

        print(f"  Soglie decisionali applicate -> Link: {calibrated_link_th:.2f}, Segno: {calibrated_sign_th:.2f}")

        # Valutazione su Val e Test
        val_metrics = trainer.evaluate(
            bundle.val_data,
            split_name="Val",
            link_threshold=calibrated_link_th,
            sign_threshold=calibrated_sign_th,
        )
        test_metrics = trainer.evaluate(
            bundle.test_data,
            split_name="Test",
            link_threshold=calibrated_link_th,
            sign_threshold=calibrated_sign_th,
        )
        best_val_metrics = checkpoint.get("val_metrics", val_metrics)
    elif args.skip_train:
        raise FileNotFoundError(
            f"Checkpoint GNN non trovato in: {checkpoint_file}. "
            "Eseguire prima 'python run_gnn.py' senza --skip-train per addestrare il modello."
        )
    else:
        results = trainer.train(
            epochs=args.epochs,
            save_path=checkpoint_file,
        )
        test_metrics = results["test_metrics"]
        best_val_metrics = results["best_val_metrics"]
        calibrated_link_th = args.link_threshold if args.link_threshold is not None else best_val_metrics.get("link_optimal_threshold", 0.5)
        calibrated_sign_th = args.sign_threshold if args.sign_threshold is not None else best_val_metrics.get("sign_optimal_threshold", 0.5)

    # 5. Stampa Metriche Test Singoli Stadi
    print("\n  ── Metriche su TEST SET (GNN Multi-Task) ──")
    for k, v in test_metrics.items():
        if isinstance(v, float):
            print(f"    {k:<24}: {v:.4f}")

    # 6. Valutazione a Cascata sul Test Set (3 Classi end-to-end)
    print("\n[Valutazione] Valutazione a cascata end-to-end sul Test Set...")
    cascade_metrics = trainer.evaluate_cascade(
        bundle.test_data,
        link_threshold=calibrated_link_th,
        sign_threshold=calibrated_sign_th,
        verbose=True,
    )

    # 7. Salvataggio Metriche
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_file = RESULTS_DIR / "gnn_metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump({
            "config": vars(args),
            "test_metrics": test_metrics,
            "best_val_metrics": best_val_metrics,
            "cascade_metrics": cascade_metrics,
        }, f, indent=2)
    print(f"\n[Risultati] Metriche salvate in: {metrics_file}")

    # 8. Dimostrazione Pipeline a Due Stadi su Esempi
    print("\n" + "=" * 70)
    print("DIMOSTRAZIONE INFERENZA: PIPELINE GNN A DUE STADI")
    print("=" * 70)
    pipeline = GNNTwoStagePipeline.load(checkpoint_file, device=trainer.device.type)

    # Seleziona 5 coppie di test per testare l'inferenza
    test_u = bundle.test_data.pairs[0][:5].tolist()
    test_m = bundle.test_data.pairs[1][:5].tolist()

    for u_idx, m_idx in zip(test_u, test_m):
        orig_u = bundle.idx2user[u_idx]
        orig_m = bundle.idx2movie[m_idx]
        pred = pipeline.predict(orig_u, orig_m)
        print(f"  Coppia (User: {orig_u}, Movie: {orig_m}):")
        print(f"    --> Probabilità Link: {pred['link_probability']:.4f} | Esito: {pred['link_predicted']}")
        if pred['link_predicted']:
            print(f"    --> Probabilità Segno Positivo: {pred['sign_probability']:.4f} | Classe: {pred['sign_predicted']}")
        print(f"    --> Verdetto: {pred['decision']}\n")


if __name__ == "__main__":
    main()
