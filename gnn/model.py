"""
gnn/model.py – Architettura Modulare Multi-Task Bipartite GNN.

Comprende:
  1. BipartiteGNNEncoder: Encoder a grafo eterogeneo (Utente - Film) con HeteroConv (GraphSAGE).
  2. LinkPredictionHead: Rete MLP per predire l'esistenza dell'arco (Link Prediction).
  3. SignPredictionHead: Rete MLP per predire la polarità del voto/recensione (Sign Prediction).
  4. MultiTaskBipartiteGNN: Modello unificato end-to-end con embedding condivisi.
"""

from typing import Tuple, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv


class BipartiteGNNEncoder(nn.Module):
    """
    Encoder GNN per il grafo bipartito utente-film.
    Propaga le informazioni sui vicini a 2 hop, consentendo il filtraggio collaborativo implicito.
    """

    def __init__(
        self,
        num_users: int,
        num_movies: int,
        embedding_dim: int = 64,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # Embedding ID iniziali apprendibili per utenti e film
        self.user_emb = nn.Embedding(num_users, embedding_dim)
        self.movie_emb = nn.Embedding(num_movies, embedding_dim)

        # Inizializzazione Xavier/Glorot
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.movie_emb.weight)

        # Costruzione dei layer HeteroConv con Signed Message Passing
        # I messaggi di gradimento (likes) e dissenso (dislikes) vengono propagati e appresi separatamente
        self.convs = nn.ModuleList()
        in_dim = embedding_dim
        for _ in range(num_layers):
            conv = HeteroConv({
                ('user', 'likes', 'movie'): SAGEConv((in_dim, in_dim), hidden_dim, aggr='mean'),
                ('movie', 'rev_likes', 'user'): SAGEConv((in_dim, in_dim), hidden_dim, aggr='mean'),
                ('user', 'dislikes', 'movie'): SAGEConv((in_dim, in_dim), hidden_dim, aggr='mean'),
                ('movie', 'rev_dislikes', 'user'): SAGEConv((in_dim, in_dim), hidden_dim, aggr='mean'),
            }, aggr='sum')
            self.convs.append(conv)
            in_dim = hidden_dim

    def forward(
        self,
        edge_dict: Optional[Dict[Tuple[str, str, str], torch.Tensor]] = None,
        edge_index_u2m: Optional[torch.Tensor] = None,
        edge_index_m2u: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Esegue il forward pass sui nodi del grafo bipartito firmato.
        Ritorna gli embedding finali (z_user, z_movie).
        """
        x_dict = {
            'user': self.user_emb.weight,
            'movie': self.movie_emb.weight,
        }

        if edge_dict is None:
            # Fallback per compatibilità con grafo non firmato
            u2m = edge_index_u2m if edge_index_u2m is not None else torch.empty((2, 0), dtype=torch.long)
            m2u = edge_index_m2u if edge_index_m2u is not None else torch.empty((2, 0), dtype=torch.long)
            empty_edge = torch.empty((2, 0), dtype=torch.long, device=u2m.device)
            edge_dict = {
                ('user', 'likes', 'movie'): u2m,
                ('movie', 'rev_likes', 'user'): m2u,
                ('user', 'dislikes', 'movie'): empty_edge,
                ('movie', 'rev_dislikes', 'user'): empty_edge,
            }

        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_dict)
            if i < self.num_layers - 1:
                x_dict = {key: F.relu(x) for key, x in x_dict.items()}
                x_dict = {key: F.dropout(x, p=self.dropout, training=self.training) for key, x in x_dict.items()}

        return x_dict['user'], x_dict['movie']


class EdgePairMLPHead(nn.Module):
    """
    Testa di decodifica MLP su coppie di nodi (u, m).
    Combina:
      - Concatenazione [z_u || z_m]
      - Element-wise product [z_u * z_m] per catturare interazioni bilaterali dirette.
    """

    def __init__(self, node_dim: int, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        # Input dim: z_u (d) + z_m (d) + (z_u * z_m) (d) = 3 * d
        in_dim = node_dim * 3
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)  # Restituisce logit
        )

    def forward(self, z_u: torch.Tensor, z_m: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
        """
        pairs: [2, N] dove [0] = user_indices, [1] = movie_indices.
        Ritorna: logit 1D di dimensione [N].
        """
        u_emb = z_u[pairs[0]]
        m_emb = z_m[pairs[1]]
        feat = torch.cat([u_emb, m_emb, u_emb * m_emb], dim=-1)
        return self.net(feat).squeeze(-1)


class MultiTaskBipartiteGNN(nn.Module):
    """
    Modello Multi-Task completo per Link Prediction e Sign Prediction.
    Condivide l'encoder e possiede due teste specializzate.
    """

    def __init__(
        self,
        num_users: int,
        num_movies: int,
        embedding_dim: int = 64,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout_gnn: float = 0.1,
        dropout_head: float = 0.2,
    ):
        super().__init__()
        self.num_users = num_users
        self.num_movies = num_movies

        # 1. Shared Encoder Backbone
        self.encoder = BipartiteGNNEncoder(
            num_users=num_users,
            num_movies=num_movies,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout_gnn,
        )

        # 2. Testa 1: Link Prediction (Arco esiste o no?)
        self.link_head = EdgePairMLPHead(
            node_dim=hidden_dim,
            hidden_dim=hidden_dim,
            dropout=dropout_head,
        )

        # 3. Testa 2: Sign Prediction (Valutazione Positiva o Negativa?)
        self.sign_head = EdgePairMLPHead(
            node_dim=hidden_dim,
            hidden_dim=hidden_dim,
            dropout=dropout_head,
        )

        # 4. Bias Espliciti di Segno (Attitudine Generosità Utente + Consenso/Qualità Film)
        self.user_sign_bias = nn.Embedding(num_users, 1)
        self.movie_sign_bias = nn.Embedding(num_movies, 1)
        self.global_sign_bias = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.user_sign_bias.weight)
        nn.init.zeros_(self.movie_sign_bias.weight)

    def encode(
        self,
        edge_dict: Optional[Dict[Tuple[str, str, str], torch.Tensor]] = None,
        edge_index_u2m: Optional[torch.Tensor] = None,
        edge_index_m2u: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calcola e restituisce gli embedding di utenti e film propagando sui grafi firmati."""
        return self.encoder(edge_dict=edge_dict, edge_index_u2m=edge_index_u2m, edge_index_m2u=edge_index_m2u)

    def predict_sign_logits(
        self,
        z_u: torch.Tensor,
        z_m: torch.Tensor,
        pairs: torch.Tensor,
    ) -> torch.Tensor:
        """Calcola i logit di gradimento combinando l'MLP di interazione con i bias del nodo."""
        mlp_logits = self.sign_head(z_u, z_m, pairs)
        u_bias = self.user_sign_bias(pairs[0]).squeeze(-1)
        m_bias = self.movie_sign_bias(pairs[1]).squeeze(-1)
        return mlp_logits + u_bias + m_bias + self.global_sign_bias

    def forward(
        self,
        pairs: torch.Tensor,
        edge_dict: Optional[Dict[Tuple[str, str, str], torch.Tensor]] = None,
        edge_index_u2m: Optional[torch.Tensor] = None,
        edge_index_m2u: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward completo:
          1. Propaga i messaggi con l'encoder firmato
          2. Emette i logit di link e sign per le coppie fornite
        """
        z_u, z_m = self.encode(edge_dict=edge_dict, edge_index_u2m=edge_index_u2m, edge_index_m2u=edge_index_m2u)
        link_logits = self.link_head(z_u, z_m, pairs)
        sign_logits = self.predict_sign_logits(z_u, z_m, pairs)
        return link_logits, sign_logits

    def predict_link_proba(
        self,
        z_u: torch.Tensor,
        z_m: torch.Tensor,
        pairs: torch.Tensor,
    ) -> torch.Tensor:
        """Restituisce le probabilità sigmoidee per l'esistenza del link."""
        logits = self.link_head(z_u, z_m, pairs)
        return torch.sigmoid(logits)

    def predict_sign_proba(
        self,
        z_u: torch.Tensor,
        z_m: torch.Tensor,
        pairs: torch.Tensor,
    ) -> torch.Tensor:
        """Restituisce le probabilità sigmoidee per il segno positivo (>= 4 stelle)."""
        logits = self.predict_sign_logits(z_u, z_m, pairs)
        return torch.sigmoid(logits)
