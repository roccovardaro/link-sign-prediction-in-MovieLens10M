"""
gnn/dataset.py – Costruzione del dataset e del grafo bipartito per GNN.

Gestisce:
  1. Caricamento dei dati dai file CSV del progetto (train, val, test).
  2. Mappatura degli ID sparsi (user_id, movie_id) in indici continui [0, N-1].
  3. Costruzione degli archi di Message Passing (esclusivamente dal set di Train per evitare Data Leakage).
  4. Generazione degli archi di supervisione con Negative Sampling per il Link e etichette per il Segno.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict
import numpy as np
import pandas as pd
import torch

from config import (
    TRAIN_CSV,
    VAL_CSV,
    TEST_CSV,
    EVENTS_CSV,
)


@dataclass
class GraphSplitData:
    """Contenitore per le coppie di supervisione e relative etichette di uno split."""
    pairs: torch.Tensor          # Shape [2, N_pairs]: [0] = user_idx, [1] = movie_idx
    link_labels: torch.Tensor    # Shape [N_pairs]: 1.0 per link reale, 0.0 per link negativo
    sign_labels: torch.Tensor    # Shape [N_pairs]: 1.0 per positiva, 0.0 per negativa, -1 per non applicabile
    real_mask: torch.Tensor      # Shape [N_pairs]: True se l'arco è reale (valido per la Sign Loss)


@dataclass
class BipartiteGraphBundle:
    """Bundle completo contenente struttura del grafo e split pronti per l'addestramento."""
    num_users: int
    num_movies: int
    user2idx: Dict[int, int]
    movie2idx: Dict[int, int]
    idx2user: Dict[int, int]
    idx2movie: Dict[int, int]
    # Archi di Signed Message Passing (relazioni eterogenee likes / dislikes)
    edge_dict: Dict[Tuple[str, str, str], torch.Tensor]
    # Archi di message passing generici (retrocompatibilità)
    edge_index_u2m: torch.Tensor
    edge_index_m2u: torch.Tensor
    # Peso ponderato per bilanciare la classe negativa minoritaria nella Sign Loss
    neg_class_weight: float
    # Split di supervisione
    train_data: GraphSplitData
    val_data: GraphSplitData
    test_data: GraphSplitData


class BipartiteGraphDataset:
    """
    Costruttore del dataset a grafo bipartito Utente-Film per approccio Multi-Task GNN.
    """

    def __init__(
        self,
        train_path: Path = TRAIN_CSV,
        val_path: Path = VAL_CSV,
        test_path: Path = TEST_CSV,
        sample_size: Optional[int] = None,
        negative_ratio: float = 1.0,
        split_mode: str = "transductive",  # "transductive" (standard GNN edge split) oppure "temporal" (file CSV pre-split)
        seed: int = 42,
    ):
        self.train_path = Path(train_path)
        self.val_path = Path(val_path)
        self.test_path = Path(test_path)
        self.sample_size = sample_size
        self.negative_ratio = negative_ratio
        self.split_mode = split_mode
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def load_data(self) -> BipartiteGraphBundle:
        """Carica i dati e costruisce il bundle per la GNN."""
        print(f"\n[BipartiteGraphDataset] Caricamento dataset (Modalità split: {self.split_mode})...")
        cols = ["user_id", "movie_id", "binary_label"]

        if self.split_mode == "transductive":
            # Nella Link Prediction standard per GNN, si estraggono le interazioni
            # e si suddividono gli ARCHI (70% train message-passing, 15% val, 15% test).
            # Così la GNN impara gli embedding dei nodi dal train e predice i legami futuri/nascosti.
            nrows = self.sample_size if (self.sample_size is not None and self.sample_size > 0) else None
            df_full = pd.read_csv(self.train_path, usecols=cols, nrows=nrows)
            
            # Filtra nodi con almeno 2 interazioni per garantire presenza sia in train che in val/test
            u_counts = df_full["user_id"].value_counts()
            valid_users = set(u_counts[u_counts >= 2].index)
            df_full = df_full[df_full["user_id"].isin(valid_users)].reset_index(drop=True)

            n_total = len(df_full)
            n_train = int(n_total * 0.70)
            n_val = int(n_total * 0.15)

            # Shuffle deterministico degli archi
            indices = self.rng.permutation(n_total)
            train_idx = indices[:n_train]
            val_idx = indices[n_train:n_train + n_val]
            test_idx = indices[n_train + n_val:]

            df_train = df_full.iloc[train_idx].copy()
            df_val = df_full.iloc[val_idx].copy()
            df_test = df_full.iloc[test_idx].copy()
        else:
            # Modalità temporale stretta dai file preesistenti
            nrows_train = self.sample_size if (self.sample_size is not None and self.sample_size > 0) else None
            nrows_val = int(self.sample_size * 0.2) if (self.sample_size is not None and self.sample_size > 0) else None
            nrows_test = int(self.sample_size * 0.2) if (self.sample_size is not None and self.sample_size > 0) else None

            df_train = pd.read_csv(self.train_path, usecols=cols, nrows=nrows_train)
            df_val = pd.read_csv(self.val_path, usecols=cols, nrows=nrows_val)
            df_test = pd.read_csv(self.test_path, usecols=cols, nrows=nrows_test)

        print(f"  Righe caricate: Train={len(df_train):,}, Val={len(df_val):,}, Test={len(df_test):,}")

        # 2. Costruzione del vocabolario globale nodi
        unique_users = sorted(set(df_train["user_id"]).union(df_val["user_id"]).union(df_test["user_id"]))
        unique_movies = sorted(set(df_train["movie_id"]).union(df_val["movie_id"]).union(df_test["movie_id"]))

        user2idx = {uid: i for i, uid in enumerate(unique_users)}
        movie2idx = {mid: i for i, mid in enumerate(unique_movies)}
        idx2user = {i: uid for uid, i in user2idx.items()}
        idx2movie = {i: mid for mid, i in movie2idx.items()}

        num_users = len(user2idx)
        num_movies = len(movie2idx)
        print(f"  Nodi univoci rilevati: {num_users:,} Utenti, {num_movies:,} Film")

        # 3. Mappatura degli indici
        train_u = df_train["user_id"].map(user2idx).to_numpy(dtype=np.int64)
        train_m = df_train["movie_id"].map(movie2idx).to_numpy(dtype=np.int64)
        train_sign = df_train["binary_label"].to_numpy(dtype=np.float32)

        val_u = df_val["user_id"].map(user2idx).to_numpy(dtype=np.int64)
        val_m = df_val["movie_id"].map(movie2idx).to_numpy(dtype=np.int64)
        val_sign = df_val["binary_label"].to_numpy(dtype=np.float32)

        test_u = df_test["user_id"].map(user2idx).to_numpy(dtype=np.int64)
        test_m = df_test["movie_id"].map(movie2idx).to_numpy(dtype=np.int64)
        test_sign = df_test["binary_label"].to_numpy(dtype=np.float32)

        # 4. Message Passing Edges (SOLO DAGLI ARCHI DI TRAIN REALI!)
        edge_index_u2m = torch.tensor(np.stack([train_u, train_m]), dtype=torch.long)
        edge_index_m2u = torch.tensor(np.stack([train_m, train_u]), dtype=torch.long)

        # Signed Message Passing (Relazioni distinte: likes vs dislikes)
        pos_mask = (train_sign == 1.0)
        neg_mask = (train_sign == 0.0)

        edge_index_pos_u2m = torch.tensor(np.stack([train_u[pos_mask], train_m[pos_mask]]), dtype=torch.long)
        edge_index_pos_m2u = torch.tensor(np.stack([train_m[pos_mask], train_u[pos_mask]]), dtype=torch.long)
        edge_index_neg_u2m = torch.tensor(np.stack([train_u[neg_mask], train_m[neg_mask]]), dtype=torch.long)
        edge_index_neg_m2u = torch.tensor(np.stack([train_m[neg_mask], train_u[neg_mask]]), dtype=torch.long)

        edge_dict = {
            ('user', 'likes', 'movie'): edge_index_pos_u2m,
            ('movie', 'rev_likes', 'user'): edge_index_pos_m2u,
            ('user', 'dislikes', 'movie'): edge_index_neg_u2m,
            ('movie', 'rev_dislikes', 'user'): edge_index_neg_m2u,
        }

        # Calcolo peso di bilanciamento per la classe negativa minoritaria (Pos-Weight/Sample-Weight)
        n_pos_train = int(pos_mask.sum())
        n_neg_train = int(neg_mask.sum())
        neg_class_weight = float(n_pos_train / max(n_neg_train, 1))

        print(f"  Archi Message Passing: {edge_index_u2m.shape[1]:,} totali "
              f"({n_pos_train:,} likes, {n_neg_train:,} dislikes)")
        print(f"  Ponderazione classe negativa (Pos-Weight/Class-Weight): {neg_class_weight:.2f}x")

        # 5. Generazione split con Negative Sampling per Link Prediction
        user_seen_movies = {}
        for u, m in zip(train_u, train_m):
            if u not in user_seen_movies:
                user_seen_movies[u] = set()
            user_seen_movies[u].add(m)

        train_split = self._create_split_data(train_u, train_m, train_sign, num_movies, user_seen_movies, add_negatives=True)
        val_split = self._create_split_data(val_u, val_m, val_sign, num_movies, user_seen_movies, add_negatives=True)
        test_split = self._create_split_data(test_u, test_m, test_sign, num_movies, user_seen_movies, add_negatives=True)

        return BipartiteGraphBundle(
            num_users=num_users,
            num_movies=num_movies,
            user2idx=user2idx,
            movie2idx=movie2idx,
            idx2user=idx2user,
            idx2movie=idx2movie,
            edge_dict=edge_dict,
            edge_index_u2m=edge_index_u2m,
            edge_index_m2u=edge_index_m2u,
            neg_class_weight=neg_class_weight,
            train_data=train_split,
            val_data=val_split,
            test_data=test_split,
        )

    def _create_split_data(
        self,
        pos_u: np.ndarray,
        pos_m: np.ndarray,
        pos_sign: np.ndarray,
        num_movies: int,
        user_seen_movies: dict,
        add_negatives: bool = True,
    ) -> GraphSplitData:
        """Crea il contenitore GraphSplitData campionando archi negativi per il link."""
        n_pos = len(pos_u)
        
        if add_negatives and self.negative_ratio > 0.0:
            n_neg = int(n_pos * self.negative_ratio)
            neg_u = self.rng.choice(pos_u, size=n_neg, replace=True)
            candidates = self.rng.integers(0, num_movies, size=n_neg)

            # Controllo collisioni rapido: il tasso di collisione medio è < 1%
            for i in range(n_neg):
                u = neg_u[i]
                seen = user_seen_movies.get(u, set())
                if candidates[i] in seen:
                    while candidates[i] in seen:
                        candidates[i] = int(self.rng.integers(0, num_movies))

            neg_m = candidates

            all_u = np.concatenate([pos_u, neg_u])
            all_m = np.concatenate([pos_m, neg_m])
            link_labels = np.concatenate([np.ones(n_pos, dtype=np.float32), np.zeros(n_neg, dtype=np.float32)])
            sign_labels = np.concatenate([pos_sign, np.full(n_neg, -1.0, dtype=np.float32)])
            real_mask = np.concatenate([np.ones(n_pos, dtype=bool), np.zeros(n_neg, dtype=bool)])
        else:
            all_u = pos_u
            all_m = pos_m
            link_labels = np.ones(n_pos, dtype=np.float32)
            sign_labels = pos_sign
            real_mask = np.ones(n_pos, dtype=bool)

        # Shuffle delle coppie
        perm = self.rng.permutation(len(all_u))
        pairs = torch.tensor(np.stack([all_u[perm], all_m[perm]]), dtype=torch.long)
        link_t = torch.tensor(link_labels[perm], dtype=torch.float32)
        sign_t = torch.tensor(sign_labels[perm], dtype=torch.float32)
        mask_t = torch.tensor(real_mask[perm], dtype=torch.bool)

        return GraphSplitData(
            pairs=pairs,
            link_labels=link_t,
            sign_labels=sign_t,
            real_mask=mask_t,
        )
