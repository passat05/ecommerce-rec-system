from typing import Dict, Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


N_RECENCY_BUCKETS  = 11
N_POP_BUCKETS      = 11


def _bucketize(values: np.ndarray, n_buckets: int) -> np.ndarray:
    """Quantile-bucket a 1D float array into [0, n_buckets-1]."""
    quantiles = np.linspace(0, 100, n_buckets + 1)
    edges = np.percentile(values[np.isfinite(values)], quantiles[1:-1])
    return np.searchsorted(edges, values).astype(np.int8)


class RankingDataset(Dataset):

    def __init__(
        self,
        df: pd.DataFrame, # interaction DataFrame (train/val/test)
        user_emb_path: str, # path to user_embeddings.npy  — loaded as memmap
        item_emb: np.ndarray, # np.ndarray (n_items, emb_dim) — pre-loaded fully
        positive_only: bool = False,
    ):
        df = df.copy()
        if positive_only:
            df = df[df["label"] == 1]

        self.labels = df["label"].values.astype(np.float32)
        self.user_ids = df["user_id"].values.astype(np.int32)
        self.item_ids = df["item_id"].values.astype(np.int32)
        self.cat_ids = df["category_id"].values.astype(np.int32)
        self.event_types = df["event_type"].values.astype(np.int8)
        self.price_buckets = df["price_bucket"].fillna(0).values.astype(np.int8)

        # Bucket recency and popularity
        rec = df["recency_days"].fillna(df["recency_days"].median()).values
        pop = df["n_views"].fillna(0).values.astype(np.float32)
        self.recency_buckets = _bucketize(rec, N_RECENCY_BUCKETS)
        self.pop_buckets = _bucketize(pop, N_POP_BUCKETS)

        # User embeddings: memmap — only paged rows loaded per batch
        self.user_emb_mm = np.load(user_emb_path, mmap_mode="r")
        self.item_emb = item_emb.astype(np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        uid = int(self.user_ids[idx])
        iid = int(self.item_ids[idx])

        # Memmap slice — only reads the one row needed
        u_emb = torch.tensor(self.user_emb_mm[uid].copy(),dtype=torch.float32)
        i_emb = torch.tensor(self.item_emb[iid],dtype=torch.float32)

        return {
            "label": torch.tensor(self.labels[idx],dtype=torch.float),
            "category_id": torch.tensor(self.cat_ids[idx],dtype=torch.long),
            "event_type": torch.tensor(int(self.event_types[idx]),dtype=torch.long),
            "price_bucket": torch.tensor(int(self.price_buckets[idx]),dtype=torch.long),
            "recency_bucket": torch.tensor(int(self.recency_buckets[idx]),dtype=torch.long),
            "pop_bucket": torch.tensor(int(self.pop_buckets[idx]),dtype=torch.long),
            "user_emb": u_emb,
            "item_emb": i_emb,
        }
