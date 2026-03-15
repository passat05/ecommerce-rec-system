from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# Shared MLP block

def _mlp_block(in_dim: int, hidden_dims: List[int], out_dim: int, dropout: float) -> nn.Sequential:
    layers = []
    dim = in_dim
    for h in hidden_dims:
        layers += [nn.Linear(dim, h), nn.ReLU(), nn.Dropout(dropout)]
        dim = h
    layers.append(nn.Linear(dim, out_dim))
    return nn.Sequential(*layers)


# Two-Tower model

class TwoTowerModel(nn.Module):
    """
    Parameters
    ----------
    n_items      : int   Vocabulary size for item IDs
    n_categories : int   Vocabulary size for category IDs
    n_price_buckets : int  Price quantile buckets (default 11: 0-9 + unknown)
    embedding_dim : int  Output dimension of both towers
    hidden_dims   : list Intermediate MLP dimensions (shared)
    dropout       : float
    """

    def __init__(
        self,
        n_items: int,
        n_categories: int,
        n_price_buckets: int = 11,
        embedding_dim: int = 64,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [256, 128]

        # Shared item embedding (used by BOTH towers)
        self.item_emb = nn.Embedding(n_items + 10, embedding_dim, padding_idx=0)

        # Item Tower extra features
        cat_emb_dim   = min(50, n_categories // 2 + 1)
        price_emb_dim = 8
        self.cat_emb   = nn.Embedding(n_categories + 10, cat_emb_dim, padding_idx=0)
        self.price_emb = nn.Embedding(n_price_buckets, price_emb_dim, padding_idx=0)

        # Item tower input = item_emb + cat_emb + price_emb
        item_in_dim = embedding_dim + cat_emb_dim + price_emb_dim
        self.item_tower = _mlp_block(item_in_dim, hidden_dims, embedding_dim, dropout)

        # User tower input = weighted-pooled item_emb (same dim as item_emb)
        self.user_tower = _mlp_block(embedding_dim, hidden_dims, embedding_dim, dropout)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.01)

    def encode_items(
        self,
        item_ids: torch.Tensor,       # (B,)
        category_ids: torch.Tensor,   # (B,)
        price_buckets: torch.Tensor,  # (B,)
    ) -> torch.Tensor:
        """Encode a batch of items → L2-normalised (B, emb_dim)."""
        i_emb = self.item_emb(item_ids)                   # (B, emb_dim)
        c_emb = self.cat_emb(category_ids)                # (B, cat_emb_dim)
        p_emb = self.price_emb(price_buckets)             # (B, price_emb_dim)
        x     = torch.cat([i_emb, c_emb, p_emb], dim=-1) # (B, item_in_dim)
        out   = self.item_tower(x)                        # (B, emb_dim)
        return F.normalize(out, p=2, dim=-1)

    def encode_users(
        self,
        history_ids: torch.Tensor,     # (B, K)  — item ID history, 0=padding
        history_weights: torch.Tensor, # (B, K)  — float event-type weights
    ) -> torch.Tensor:
        """Encode a batch of users via history → L2-normalised (B, emb_dim)."""
        # (B, K, emb_dim)
        h_emb = self.item_emb(history_ids)

        # Mask padding tokens
        mask = (history_ids != 0).float().unsqueeze(-1)   # (B, K, 1)
        w    = history_weights.unsqueeze(-1) * mask        # (B, K, 1)
        w_sum = w.sum(dim=1).clamp(min=1e-8)              # (B, 1)

        pooled = (h_emb * w).sum(dim=1) / w_sum           # (B, emb_dim)
        out    = self.user_tower(pooled)
        return F.normalize(out, p=2, dim=-1)

    def forward(
        self,
        # User inputs
        history_ids: torch.Tensor,
        history_weights: torch.Tensor,
        # Positive item inputs
        pos_item_ids: torch.Tensor,
        pos_cat_ids: torch.Tensor,
        pos_price: torch.Tensor,
        # Negative item inputs (optional — used during training)
        neg_item_ids: Optional[torch.Tensor] = None,
        neg_cat_ids: Optional[torch.Tensor] = None,
        neg_price: Optional[torch.Tensor] = None,
    ):
        """
        Returns (user_emb, pos_item_emb) at inference,
        or (user_emb, pos_item_emb, neg_item_embs) during training.
        """
        user_emb  = self.encode_users(history_ids, history_weights)
        pos_emb   = self.encode_items(pos_item_ids, pos_cat_ids, pos_price)

        if neg_item_ids is not None:
            neg_emb = self.encode_items(neg_item_ids, neg_cat_ids, neg_price)
            return user_emb, pos_emb, neg_emb

        return user_emb, pos_emb


# InfoNCE loss (in-batch negatives)

def infonce_loss(
    user_emb: torch.Tensor,   # (B, D)  — L2-normalised
    item_emb: torch.Tensor,   # (B, D)  — L2-normalised (positive items)
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    In-batch InfoNCE: for each user, all OTHER items in the batch
    are treated as negatives.

    Returns scalar loss.
    """
    # Similarity matrix (B, B)
    logits = (user_emb @ item_emb.T) / temperature

    # Diagonal = positive pairs
    labels = torch.arange(len(user_emb), device=user_emb.device)
    loss   = F.cross_entropy(logits, labels)
    return loss
