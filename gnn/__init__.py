"""
Package GNN: Implementazione Modulare Multi-Task Bipartite GNN
per Link Prediction e Sign Prediction (MovieLens 10M / Amazon Reviews).
"""

from gnn.model import MultiTaskBipartiteGNN
from gnn.dataset import BipartiteGraphDataset
from gnn.trainer import GNNTrainer
from gnn.pipeline import GNNTwoStagePipeline

__all__ = [
    "MultiTaskBipartiteGNN",
    "BipartiteGraphDataset",
    "GNNTrainer",
    "GNNTwoStagePipeline",
]
