"""
run_gnn.py – Script Eseguibile Principale per il Modello GNN Multi-Task.

Esegue:
  1. Costruzione del grafo bipartito e campionamento dei negativi.
  2. Inizializzazione della Multi-Task Bipartite GNN.
  3. Addestramento congiunto (Loss Link + Loss Segno Mascherata).
  4. Valutazione rigorosa su Test Set out-of-time.
  5. Salvataggio delle metriche in results/gnn_metrics.json.
  6. Dimostrazione della pipeline a due stadi su coppie d'esempio.

Uso:
  python run_gnn.py
  python run_gnn.py --sample-size 50000 --epochs 5
"""

import argparse
import json
from pathlib import Path
import torch

from config import RESULTS_DIR, DATA_PROCESSED
from gnn.dataset import BipartiteGraphDataset
from gnn.model import MultiTaskBipartiteGNN
from gnn.trainer import GNNTrainer
from gnn.pipeline import GNNTwoStagePipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Addestramento GNN Multi-Task per Link e Sign Prediction")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=500_000,
        help="Numero di righe di train da campionare per calcoli rapidi (0 per usare l'intero dataset)",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Numero di epoche di addestramento")
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

    print("=" * 70)
    print("PIPELINE GNN MULTI-TASK: LINK & SIGN PREDICTION (Bipartite Graph)")
    print("=" * 70)
    print(f"  Parametri:")
    print(f"    - Split Mode:    {args.split_mode.upper()} {'(Standard GNN Edge-Holdout)' if args.split_mode == 'transductive' else '(Split Temporale CSV)'}")
    print(f"    - Sample Size:   {args.sample_size if args.sample_size > 0 else 'Intero Dataset'}")
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
    checkpoint_file = DATA_PROCESSED / "gnn_multitask_model.pt"
    trainer = GNNTrainer(
        model=model,
        bundle=bundle,
        lr=args.lr,
        lambda_sign=args.lambda_sign,
        batch_size=args.batch_size,
        device=args.device,
    )

    # 4. Addestramento & Valutazione
    results = trainer.train(
        epochs=args.epochs,
        save_path=checkpoint_file,
    )

    # 5. Salvataggio Metriche
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_file = RESULTS_DIR / "gnn_metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump({
            "config": vars(args),
            "test_metrics": results["test_metrics"],
            "best_val_metrics": results["best_val_metrics"],
        }, f, indent=2)
    print(f"\n[Risultati] Metriche salvate in: {metrics_file}")

    # 6. Dimostrazione Pipeline a Due Stadi
    print("\n" + "=" * 70)
    print("DIMOSTRAZIONE INFERENZA: PIPELINE GNN A DUE STADI")
    print("=" * 70)
    pipeline = GNNTwoStagePipeline.load(checkpoint_file, device=trainer.device.type)

    # Seleziona 3 coppie di test per testare l'inferenza
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
