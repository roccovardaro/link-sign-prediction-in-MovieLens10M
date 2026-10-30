"""
gnn/trainer.py – Modulo di Training, Ottimizzazione e Valutazione per GNN Multi-Task.

Gestisce:
  1. Addestramento con Loss Multi-Task congiunta (Link BCE + lambda * Sign BCE).
  2. Mascheramento condizionato: la Sign Loss viene calcolata SOLO sulle interazioni reali.
  3. Mini-batching delle coppie di supervisione per scalabilità e stabilità numerica.
  4. Calcolo delle metriche scientifiche (ROC-AUC, PR-AUC, F1, Balanced Accuracy) per entrambi i task.
  5. Salvataggio del checkpoint del modello migliore.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from gnn.dataset import BipartiteGraphBundle, GraphSplitData
from gnn.model import MultiTaskBipartiteGNN
from config import DATA_PROCESSED, RESULTS_DIR


class GNNTrainer:
    """Trainer specializzato per la Bipartite Multi-Task GNN."""

    def __init__(
        self,
        model: MultiTaskBipartiteGNN,
        bundle: BipartiteGraphBundle,
        lr: float = 0.003,
        weight_decay: float = 1e-4,
        lambda_sign: float = 1.0,
        batch_size: int = 16384,
        device: Optional[str] = None,
    ):
        self.model = model
        self.bundle = bundle
        self.lambda_sign = lambda_sign
        self.batch_size = batch_size

        # Rilevamento automatico device (priorità: MPS su Mac Apple Silicon -> CUDA -> CPU)
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        print(f"[GNNTrainer] Dispositivo selezionato: {self.device}")

        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        # Criteri di perdita
        self.criterion_link = nn.BCEWithLogitsLoss()
        self.criterion_sign = nn.BCEWithLogitsLoss()

        # Peso per bilanciare la classe negativa minoritaria (Pos-Weight/Sample-Weight)
        self.neg_class_weight = getattr(self.bundle, "neg_class_weight", 1.0)

        # Spostamento degli archi di message passing (Signed relations) sul device
        if hasattr(self.bundle, "edge_dict") and self.bundle.edge_dict is not None:
            self.edge_dict = {k: v.to(self.device) for k, v in self.bundle.edge_dict.items()}
        else:
            self.edge_dict = None

        self.edge_index_u2m = self.bundle.edge_index_u2m.to(self.device)
        self.edge_index_m2u = self.bundle.edge_index_m2u.to(self.device)

    def train_epoch(self) -> Dict[str, float]:
        """Esegue una singola epoca di addestramento su tutti i batch."""
        self.model.train()
        train_data = self.bundle.train_data
        n_pairs = train_data.pairs.shape[1]

        # Shuffle degli indici delle coppie per l'epoca
        perm = torch.randperm(n_pairs)
        pairs_shuffled = train_data.pairs[:, perm]
        link_labels_shuffled = train_data.link_labels[perm]
        sign_labels_shuffled = train_data.sign_labels[perm]
        real_mask_shuffled = train_data.real_mask[perm]

        total_loss = 0.0
        total_link_loss = 0.0
        total_sign_loss = 0.0
        n_batches = 0

        # Calcolo degli embedding dei nodi tramite il Signed GNN Encoder
        z_u, z_m = self.model.encode(edge_dict=self.edge_dict, edge_index_u2m=self.edge_index_u2m, edge_index_m2u=self.edge_index_m2u)

        self.optimizer.zero_grad()

        # Iterazione a mini-batch sulle coppie di supervisione
        for start_idx in range(0, n_pairs, self.batch_size):
            end_idx = min(start_idx + self.batch_size, n_pairs)
            batch_count = end_idx - start_idx
            weight = batch_count / n_pairs
            
            b_pairs = pairs_shuffled[:, start_idx:end_idx].to(self.device)
            b_link_labels = link_labels_shuffled[start_idx:end_idx].to(self.device)
            b_sign_labels = sign_labels_shuffled[start_idx:end_idx].to(self.device)
            b_real_mask = real_mask_shuffled[start_idx:end_idx].to(self.device)

            # 1. Predizione Link
            link_logits = self.model.link_head(z_u, z_m, b_pairs)
            loss_link = self.criterion_link(link_logits, b_link_labels)

            # 2. Predizione Sign (Mascherata SOLO sui link reali + bilanciata per classe minoritaria)
            loss_sign = torch.tensor(0.0, device=self.device)
            if b_real_mask.any():
                real_pairs = b_pairs[:, b_real_mask]
                real_targets = b_sign_labels[b_real_mask]
                sign_logits = self.model.predict_sign_logits(z_u, z_m, real_pairs)

                # Feature 1: Ponderazione della classe negativa per impedire il collasso sulla classe maggioritaria
                weights = torch.where(real_targets == 0.0, self.neg_class_weight, 1.0)
                loss_sign = torch.nn.functional.binary_cross_entropy_with_logits(
                    sign_logits, real_targets, weight=weights
                )

            # 3. Loss Totale Multi-Task ponderata per il batch
            batch_loss = loss_link + self.lambda_sign * loss_sign
            weighted_loss = batch_loss * weight

            is_last = (end_idx >= n_pairs)
            weighted_loss.backward(retain_graph=not is_last)

            total_loss += batch_loss.item() * weight
            total_link_loss += loss_link.item() * weight
            total_sign_loss += loss_sign.item() * weight
            n_batches += 1

        self.optimizer.step()

        return {
            "loss": total_loss,
            "loss_link": total_link_loss,
            "loss_sign": total_sign_loss,
        }

    @torch.no_grad()
    def evaluate(
        self,
        split_data: GraphSplitData,
        split_name: str = "Val",
        link_threshold: Optional[float] = None,
        sign_threshold: Optional[float] = None,
    ) -> Dict[str, float]:
        """Valuta il modello sullo split specificato calcolando tutte le metriche."""
        self.model.eval()
        z_u, z_m = self.model.encode(edge_dict=self.edge_dict, edge_index_u2m=self.edge_index_u2m, edge_index_m2u=self.edge_index_m2u)

        pairs = split_data.pairs.to(self.device)
        link_targets = split_data.link_labels.cpu().numpy()
        sign_targets = split_data.sign_labels.cpu().numpy()
        real_mask = split_data.real_mask.cpu().numpy()

        # Probabilità predette
        link_probs = self.model.predict_link_proba(z_u, z_m, pairs).cpu().numpy()
        sign_probs = self.model.predict_sign_proba(z_u, z_m, pairs).cpu().numpy()

        metrics = {}

        # ── Metriche Link Prediction ──
        if len(np.unique(link_targets)) > 1:
            if link_threshold is None:
                best_link_th = 0.5
                best_link_f1 = 0.0
                for th in np.linspace(0.3, 0.7, 21):
                    preds = (link_probs >= th).astype(int)
                    f1 = f1_score(link_targets, preds, zero_division=0)
                    if f1 > best_link_f1:
                        best_link_f1 = f1
                        best_link_th = float(th)
                chosen_link_th = best_link_th
            else:
                chosen_link_th = link_threshold

            link_preds = (link_probs >= chosen_link_th).astype(int)
            metrics["link_optimal_threshold"] = chosen_link_th
            metrics["link_roc_auc"] = float(roc_auc_score(link_targets, link_probs))
            metrics["link_pr_auc"] = float(average_precision_score(link_targets, link_probs))
            metrics["link_acc"] = float(accuracy_score(link_targets, link_preds))
            metrics["link_f1"] = float(f1_score(link_targets, link_preds, zero_division=0))
        else:
            metrics["link_acc"] = float(accuracy_score(link_targets, (link_probs >= 0.5).astype(int)))
            metrics["link_f1"] = 1.0

        # ── Metriche Sign Prediction (solo sugli archi reali) ──
        real_sign_targets = sign_targets[real_mask]
        real_sign_probs = sign_probs[real_mask]

        if len(real_sign_targets) > 0 and len(np.unique(real_sign_targets)) > 1:
            # Calibrazione soglia ottimale per balanced accuracy se richiesta
            if sign_threshold is None:
                best_th = 0.5
                best_bacc = 0.0
                for th in np.linspace(0.2, 0.8, 25):
                    bacc = balanced_accuracy_score(real_sign_targets, (real_sign_probs >= th).astype(int))
                    if bacc > best_bacc:
                        best_bacc = bacc
                        best_th = float(th)
                chosen_th = best_th
            else:
                chosen_th = sign_threshold

            real_sign_preds = (real_sign_probs >= chosen_th).astype(int)

            metrics["sign_optimal_threshold"] = chosen_th
            metrics["sign_roc_auc"] = float(roc_auc_score(real_sign_targets, real_sign_probs))
            metrics["sign_acc"] = float(accuracy_score(real_sign_targets, real_sign_preds))
            metrics["sign_balanced_acc"] = float(balanced_accuracy_score(real_sign_targets, real_sign_preds))
            metrics["sign_precision"] = float(precision_score(real_sign_targets, real_sign_preds, zero_division=0))
            metrics["sign_recall"] = float(recall_score(real_sign_targets, real_sign_preds, zero_division=0))
            metrics["sign_f1"] = float(f1_score(real_sign_targets, real_sign_preds, zero_division=0))
        elif len(real_sign_targets) > 0:
            metrics["sign_acc"] = float(accuracy_score(real_sign_targets, (real_sign_probs >= 0.5).astype(int)))

        return metrics

    @torch.no_grad()
    def evaluate_cascade(
        self,
        split_data: GraphSplitData,
        link_threshold: float = 0.5,
        sign_threshold: float = 0.5,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Valuta il modello in modalità a cascata end-to-end (3 classi):
          0 = Nessun Link
          1 = Link Negativo
          2 = Link Positivo
        Rende le metriche direttamente confrontabili con run_traditional.py.
        """
        self.model.eval()
        z_u, z_m = self.model.encode(
            edge_dict=self.edge_dict,
            edge_index_u2m=self.edge_index_u2m,
            edge_index_m2u=self.edge_index_m2u,
        )

        pairs = split_data.pairs.to(self.device)
        link_targets = split_data.link_labels.cpu().numpy()
        sign_targets = split_data.sign_labels.cpu().numpy()

        link_probs = self.model.predict_link_proba(z_u, z_m, pairs).cpu().numpy()
        sign_probs = self.model.predict_sign_proba(z_u, z_m, pairs).cpu().numpy()

        # Ground truth a 3 classi:
        # 0 = Nessun link reale
        # 1 = Link reale negativo (sign == 0)
        # 2 = Link reale positivo (sign == 1)
        y_true = np.zeros(len(link_targets), dtype=int)
        y_true[(link_targets == 1) & (sign_targets == 0)] = 1
        y_true[(link_targets == 1) & (sign_targets == 1)] = 2

        # Predizione a cascata a 3 classi
        pred_link = link_probs >= link_threshold
        pred_sign = sign_probs >= sign_threshold

        y_pred = np.zeros(len(link_targets), dtype=int)
        y_pred[pred_link & (~pred_sign)] = 1
        y_pred[pred_link & pred_sign] = 2

        target_names = ["Nessun Link (0)", "Link Negativo (1)", "Link Positivo (2)"]
        report_dict = classification_report(
            y_true, y_pred, target_names=target_names, digits=4, zero_division=0, output_dict=True
        )
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist()
        acc = float(accuracy_score(y_true, y_pred))
        balanced_acc = float(balanced_accuracy_score(y_true, y_pred))

        link_cm = confusion_matrix(link_targets, (link_probs >= link_threshold).astype(int))
        link_accuracy = float(accuracy_score(link_targets, (link_probs >= link_threshold).astype(int)))
        link_recall = float(link_cm[1][1] / max(link_cm[1][0] + link_cm[1][1], 1))

        if verbose:
            n_real_links = int((link_targets == 1).sum())
            print(f"\n  ── Risultati GNN Stadio 1 (Link Prediction | Soglia = {link_threshold:.2f}) ──")
            print(f"  Accuracy Link       : {link_accuracy:.4f}")
            print(f"  Recall Link Reali   : {link_recall:.4f} ({link_cm[1][1]:,} intercettati su {n_real_links:,})")
            print(f"  Falsi Allarmi (FP)  : {link_cm[0][1]:,} coppie senza link scambiate per link")

            mask_pred_link = pred_link == 1
            n_pred = int(mask_pred_link.sum())
            n_pos_pred = int((pred_link & pred_sign).sum())
            n_neg_pred = int((pred_link & (~pred_sign)).sum())
            print(f"\n  ── Risultati GNN Stadio 2 (Sign Prediction su {n_pred:,} link previsti | Soglia = {sign_threshold:.2f}) ──")
            print(f"    --> Previsti come Positivi: {n_pos_pred:,} ({n_pos_pred * 100 / max(n_pred, 1):.2f}%)")
            print(f"    --> Previsti come Negativi: {n_neg_pred:,} ({n_neg_pred * 100 / max(n_pred, 1):.2f}%)")

            print("\n  ── Report Finale a Cascata GNN (3 Classi) ──")
            print(classification_report(y_true, y_pred, target_names=target_names, digits=4, zero_division=0))

            print("  Matrice di Confusione Globale (Righe=Reale, Colonne=Predetto):")
            header = f"  {'':<20} | {'Pred: No Link':<14} | {'Pred: Negativo':<15} | {'Pred: Positivo':<15}"
            print(header)
            print("  " + "-" * (len(header) - 2))
            for i, name in enumerate(target_names):
                print(f"  {name:<20} | {cm[i][0]:<14,d} | {cm[i][1]:<15,d} | {cm[i][2]:<15,d}")

        return {
            "accuracy": acc,
            "balanced_accuracy": balanced_acc,
            "link_accuracy": link_accuracy,
            "link_recall": link_recall,
            "classification_report": report_dict,
            "confusion_matrix": cm,
        }

    def train(
        self,
        epochs: int = 15,
        save_path: Optional[Path] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Esegue il ciclo completo di addestramento e salva il miglior modello."""
        if save_path is None:
            save_path = DATA_PROCESSED / "gnn_multitask_model.pt"

        best_score = -1.0
        best_metrics = {}
        history = []

        print(f"\n[GNNTrainer] Inizio addestramento per {epochs} epoche...")
        t0 = time.time()

        for epoch in range(1, epochs + 1):
            t_ep = time.time()
            train_losses = self.train_epoch()
            val_metrics = self.evaluate(self.bundle.val_data, split_name="Val")

            # Score composito per il checkpoint: media tra Link ROC-AUC e Sign Balanced Accuracy
            link_auc = val_metrics.get("link_roc_auc", val_metrics.get("link_acc", 0.0))
            sign_bacc = val_metrics.get("sign_balanced_acc", val_metrics.get("sign_acc", 0.0))
            composite_score = 0.5 * (link_auc + sign_bacc)

            history.append({
                "epoch": epoch,
                **train_losses,
                **{f"val_{k}": v for k, v in val_metrics.items()},
            })

            if verbose:
                print(
                    f"  Epoca {epoch:02d}/{epochs:02d} [{time.time() - t_ep:.1f}s] "
                    f"Loss: {train_losses['loss']:.4f} (Link: {train_losses['loss_link']:.4f}, Sign: {train_losses['loss_sign']:.4f}) | "
                    f"Val Link AUC: {val_metrics.get('link_roc_auc', 0.0):.4f} | "
                    f"Val Sign BalAcc: {val_metrics.get('sign_balanced_acc', 0.0):.4f}"
                )

            # Salvataggio del checkpoint migliore
            if composite_score > best_score:
                best_score = composite_score
                best_metrics = val_metrics
                optimal_sign_th = val_metrics.get("sign_optimal_threshold", 0.5)
                optimal_link_th = val_metrics.get("link_optimal_threshold", 0.5)
                self.save_checkpoint(
                    save_path,
                    optimal_link_threshold=optimal_link_th,
                    optimal_sign_threshold=optimal_sign_th,
                    val_metrics=best_metrics,
                )

        print(f"[GNNTrainer] Addestramento completato in {time.time() - t0:.1f}s!")
        print(f"  Miglior Modello salvato in: {save_path}")

        # Valutazione finale su Test Set con le soglie calibrate su Val
        self.load_checkpoint(save_path)
        calibrated_sign_th = best_metrics.get("sign_optimal_threshold", 0.5)
        calibrated_link_th = best_metrics.get("link_optimal_threshold", 0.5)
        test_metrics = self.evaluate(
            self.bundle.test_data,
            split_name="Test",
            link_threshold=calibrated_link_th,
            sign_threshold=calibrated_sign_th,
        )
        print("\n  ── Metriche Finali su TEST SET ──")
        for k, v in test_metrics.items():
            print(f"    {k:<22}: {v:.4f}")

        return {
            "best_val_metrics": best_metrics,
            "test_metrics": test_metrics,
            "history": history,
        }

    def save_checkpoint(
        self,
        path: Path,
        optimal_link_threshold: float = 0.5,
        optimal_sign_threshold: float = 0.5,
        val_metrics: Optional[Dict[str, float]] = None,
    ):
        """Salva lo stato completo del modello e i metadati dei nodi."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "num_users": self.bundle.num_users,
            "num_movies": self.bundle.num_movies,
            "user2idx": self.bundle.user2idx,
            "movie2idx": self.bundle.movie2idx,
            "idx2user": self.bundle.idx2user,
            "idx2movie": self.bundle.idx2movie,
            "edge_dict": {k: v.cpu() for k, v in self.edge_dict.items()} if self.edge_dict is not None else None,
            "edge_index_u2m": self.bundle.edge_index_u2m,
            "edge_index_m2u": self.bundle.edge_index_m2u,
            "neg_class_weight": self.neg_class_weight,
            "optimal_link_threshold": optimal_link_threshold,
            "optimal_sign_threshold": optimal_sign_threshold,
            "val_metrics": val_metrics,
        }, path)

    def load_checkpoint(self, path: Path):
        """Carica i pesi del checkpoint salvato."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
