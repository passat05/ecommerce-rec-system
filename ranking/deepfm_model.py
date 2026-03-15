from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FieldEmbedder(nn.Module):
    """
    Embeds one categorical field to a fixed dimension d.
    Numeric scalars are first bucketed (done in dataset.py) then embedded.
    """

    def __init__(self, vocab_size: int, embed_dim: int, padding_idx: int = 0):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        nn.init.normal_(self.emb.weight, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.emb(x)


class DeepFM(nn.Module):

    def __init__(
        self,
        n_categories: int, # Category vocab size
        n_event_types: int = 4, # Event type vocab (0=view,1=addtocart,2=transaction)
        n_price_buckets: int = 11,
        n_recency_buckets: int = 11,
        n_popularity_buckets: int = 11,
        retrieval_emb_dim: int = 64, # Dim of Two-Tower embeddings (user + item)
        field_emb_dim: int = 16, # d — each field projected to this dim
        hidden_dims: Optional[List[int]] = None, # Deep MLP hidden layer sizes
        dropout: float = 0.3,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [256, 128, 64]
        self.field_emb_dim = field_emb_dim

        # Categorical field embedders
        self.cat_emb = FieldEmbedder(n_categories + 1, field_emb_dim)
        self.event_emb = FieldEmbedder(n_event_types + 1, field_emb_dim)
        self.price_emb = FieldEmbedder(n_price_buckets, field_emb_dim)
        self.recency_emb = FieldEmbedder(n_recency_buckets, field_emb_dim)
        self.pop_emb = FieldEmbedder(n_popularity_buckets, field_emb_dim)

        # Dense Two-Tower embeddings → projected to field_emb_dim
        self.user_proj = nn.Linear(retrieval_emb_dim, field_emb_dim, bias=False)
        self.item_proj = nn.Linear(retrieval_emb_dim, field_emb_dim, bias=False)

        # Total fields: 5 categorical + user_emb + item_emb = 7
        self.n_fields = 7
        flat_dim = self.n_fields * field_emb_dim

        # Wide: first-order linear terms
        self.wide_linear = nn.Linear(flat_dim, 1, bias=True)

        # Deep: MLP

        deep_layers: List[nn.Module] = []
        in_dim = flat_dim
        for h in hidden_dims:
            deep_layers += [
                nn.Linear(in_dim, h),
                nn.LayerNorm(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = h
        deep_layers.append(nn.Linear(in_dim, 1))
        self.deep = nn.Sequential(*deep_layers)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _fm_second_order(self, field_embs: torch.Tensor) -> torch.Tensor:
        """
        Efficient FM order-2 interaction.
        """
        sum_then_sq = field_embs.sum(dim=1) ** 2           # (B, d)
        sq_then_sum = (field_embs ** 2).sum(dim=1)         # (B, d)
        fm_out = 0.5 * (sum_then_sq - sq_then_sum).sum(dim=-1, keepdim=True)  # (B, 1)
        return fm_out

    def forward(
        self,
        category_id: torch.Tensor,
        event_type: torch.Tensor,
        price_bucket: torch.Tensor,
        recency_bucket: torch.Tensor,
        pop_bucket: torch.Tensor,
        user_emb: torch.Tensor,
        item_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Returns relevance score in (0, 1). Shape: (B,)"""
        d = self.field_emb_dim

        # Embed each field → (B, d)
        f_cat    = self.cat_emb(category_id)
        f_event  = self.event_emb(event_type)
        f_price  = self.price_emb(price_bucket)
        f_rec    = self.recency_emb(recency_bucket)
        f_pop    = self.pop_emb(pop_bucket)
        f_user   = self.user_proj(user_emb)
        f_item   = self.item_proj(item_emb)

        # Stack → (B, n_fields=7, d)
        field_embs = torch.stack(
            [f_cat, f_event, f_price, f_rec, f_pop, f_user, f_item], dim=1
        )

        # Flatten for wide + deep
        flat = field_embs.view(field_embs.size(0), -1) # (B, n_fields*d)

        # FM second-order
        fm_out = self._fm_second_order(field_embs) # (B, 1)
        wide_out = self.wide_linear(flat) # (B, 1)
        deep_out = self.deep(flat) # (B, 1)

        logit = fm_out + wide_out + deep_out
        return torch.sigmoid(logit).squeeze(-1)  # (B,)