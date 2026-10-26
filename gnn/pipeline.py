"""
gnn/pipeline.py – Pipeline d'Inferenza a Due Stadi con GNN.

Permette l'inferenza real-time o a batch simulando il flusso a due stadi:
  1. Stadio 1: Predizione dell'esistenza dell'arco (Link Prediction).
     Se probabilità < soglia_link -> Esito: "Nessuna interazione".
  2. Stadio 2: Se l'arco è previsto come esistente, stima il gradimento (Sign Prediction).
     Se probabilità_segno >= soglia_segno -> "Positiva", altrimenti "Negativa".
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import torch

from gnn.model import MultiTaskBipartiteGNN
from config import DATA_PROCESSED


class GNNTwoStagePipeline:
    """Pipeline integrata a due stadi basata sul modello GNN Multi-Task."""

    def __init__(
        self,
        model: MultiTaskBipartiteGNN,
        user2idx: Dict[int, int],
        movie2idx: Dict[int, int],
        edge_dict: Optional[Dict[Tuple[str, str, str], torch.Tensor]] = None,
        edge_index_u2m: Optional[torch.Tensor] = None,
        edge_index_m2u: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        link_threshold: float = 0.5,
        sign_threshold: float = 0.5,
    ):
        self.model = model
        self.user2idx = user2idx
        self.movie2idx = movie2idx
        self.link_threshold = link_threshold
        self.sign_threshold = sign_threshold

        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device

        self.model.to(self.device)
        self.model.eval()

        self.edge_dict = (
            {k: v.to(self.device) for k, v in edge_dict.items()}
            if edge_dict is not None else None
        )
        self.edge_index_u2m = edge_index_u2m.to(self.device) if edge_index_u2m is not None else None
        self.edge_index_m2u = edge_index_m2u.to(self.device) if edge_index_m2u is not None else None

        # Cache degli embedding calcolati per inferenza ad alta velocità
        self._cached_zu = None
        self._cached_zm = None
        self._update_embeddings_cache()

    @torch.no_grad()
    def _update_embeddings_cache(self):
        """Pre-calcola gli embedding dei nodi per evitare di rieseguire la GNN ad ogni singola query."""
        self.model.eval()
        zu, zm = self.model.encode(
            edge_dict=self.edge_dict,
            edge_index_u2m=self.edge_index_u2m,
            edge_index_m2u=self.edge_index_m2u,
        )
        self._cached_zu = zu
        self._cached_zm = zm

    @classmethod
    def load(
        cls,
        model_path: Optional[Path] = None,
        link_threshold: float = 0.5,
        sign_threshold: float = 0.5,
        device: Optional[str] = None,
    ) -> "GNNTwoStagePipeline":
        """Carica il modello salvato su disco e restituisce la pipeline pronta all'uso."""
        if model_path is None:
            model_path = DATA_PROCESSED / "gnn_multitask_model.pt"

        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint non trovato in: {path}. Addestrare prima il modello con run_gnn.py")

        target_device = torch.device(device) if device else (
            torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        )

        checkpoint = torch.load(path, map_location=target_device)
        num_users = checkpoint["num_users"]
        num_movies = checkpoint["num_movies"]

        model = MultiTaskBipartiteGNN(num_users=num_users, num_movies=num_movies)
        model.load_state_dict(checkpoint["model_state_dict"])

        calibrated_sign_th = checkpoint.get("optimal_sign_threshold", sign_threshold) if sign_threshold == 0.5 else sign_threshold
        calibrated_link_th = checkpoint.get("optimal_link_threshold", link_threshold) if link_threshold == 0.5 else link_threshold

        return cls(
            model=model,
            user2idx=checkpoint["user2idx"],
            movie2idx=checkpoint["movie2idx"],
            edge_dict=checkpoint.get("edge_dict"),
            edge_index_u2m=checkpoint.get("edge_index_u2m"),
            edge_index_m2u=checkpoint.get("edge_index_m2u"),
            device=target_device,
            link_threshold=calibrated_link_th,
            sign_threshold=calibrated_sign_th,
        )

    @torch.no_grad()
    def predict(self, user_id: int, movie_id: int) -> Dict[str, Any]:
        """
        Predice l'esito a due stadi per una singola coppia (user_id, movie_id).
        """
        if user_id not in self.user2idx or movie_id not in self.movie2idx:
            return {
                "user_id": user_id,
                "movie_id": movie_id,
                "status": "unseen_node",
                "link_predicted": False,
                "link_probability": 0.0,
                "sign_predicted": None,
                "sign_probability": None,
                "decision": "Nessuna interazione (Nodo non presente nel grafo)",
            }

        u_idx = self.user2idx[user_id]
        m_idx = self.movie2idx[movie_id]
        pair = torch.tensor([[u_idx], [m_idx]], dtype=torch.long, device=self.device)

        # ── Stadio 1: Link Prediction ──
        p_link = self.model.predict_link_proba(self._cached_zu, self._cached_zm, pair).item()
        link_exists = p_link >= self.link_threshold

        if not link_exists:
            return {
                "user_id": user_id,
                "movie_id": movie_id,
                "status": "ok",
                "link_predicted": False,
                "link_probability": round(p_link, 4),
                "sign_predicted": None,
                "sign_probability": None,
                "decision": "Nessuna interazione prevista (Stadio 1 esclude la formazione dell'arco)",
            }

        # ── Stadio 2: Sign Prediction ──
        p_sign = self.model.predict_sign_proba(self._cached_zu, self._cached_zm, pair).item()
        is_positive = p_sign >= self.sign_threshold
        sign_label = "positiva" if is_positive else "negativa"

        return {
            "user_id": user_id,
            "movie_id": movie_id,
            "status": "ok",
            "link_predicted": True,
            "link_probability": round(p_link, 4),
            "sign_predicted": sign_label,
            "sign_probability": round(p_sign, 4),
            "decision": f"Interazione confermata -> Valutazione prevista: {sign_label.upper()} ({p_sign * 100:.1f}%)",
        }
