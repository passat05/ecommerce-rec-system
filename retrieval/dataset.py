from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class TwoTowerDataset(Dataset):
    """
    Parameters
    ----------
    interactions  : pd.DataFrame  columns: user_id, item_id, label, ...
    user_history  : pd.DataFrame  columns: user_id, history (list[int])
    item_meta_idx : dict          item_id → (category_id, price_bucket)
    history_len   : int           Max history length K (pad with 0)
    positive_only : bool          If True, only keep label=1 rows
    """

    EVENT_WEIGHT = {0: 1.0, 1: 2.0, 2: 3.0}   # view, addtocart, transaction

    def __init__(
        self,
        interactions: pd.DataFrame,
        user_history: pd.DataFrame,
        item_meta_idx: Dict[int, tuple],
        history_len: int = 20,
        positive_only: bool = True,
    ):
        if positive_only:
            interactions = interactions[interactions["label"] == 1].copy()

        self.user_ids  = interactions["user_id"].values.astype(np.int32)
        self.item_ids  = interactions["item_id"].values.astype(np.int32)
        self.labels    = interactions["label"].values.astype(np.float32)
        self.event_types = interactions["event_type"].values.astype(np.int8)

        self.history_len   = history_len
        self.item_meta_idx = item_meta_idx

        # Build user_id → history lookup
        self.user_history: Dict[int, List[int]] = {}
        for _, row in user_history.iterrows():
            self.user_history[int(row["user_id"])] = row["history"]

    def _get_history(self, user_id: int):
        """Return padded (K,) history and corresponding weights."""
        hist = self.user_history.get(user_id, [])[-self.history_len:]
        K    = self.history_len
        ids     = np.zeros(K, dtype=np.int32)
        weights = np.zeros(K, dtype=np.float32)

        for i, iid in enumerate(hist):
            ids[i]     = iid + 1   # shift: 0 = padding
            weights[i] = 1.0       # uniform weight (event type unknown in history)

        return ids, weights

    def _get_item_feats(self, item_id: int):
        cat_id, price = self.item_meta_idx.get(item_id, (0, 0))
        return cat_id, price

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        uid = int(self.user_ids[idx])
        iid = int(self.item_ids[idx])

        hist_ids, hist_w = self._get_history(uid)
        cat_id, price    = self._get_item_feats(iid)

        return {
            "history_ids":    torch.tensor(hist_ids,    dtype=torch.long),
            "history_weights":torch.tensor(hist_w,      dtype=torch.float),
            "item_id":        torch.tensor(iid + 1,     dtype=torch.long),  # 0=pad
            "category_id":    torch.tensor(cat_id,      dtype=torch.long),
            "price_bucket":   torch.tensor(price,       dtype=torch.long),
            "label":          torch.tensor(self.labels[idx], dtype=torch.float),
        }


def build_item_meta_lookup(
    item_meta: pd.DataFrame,
    cat2idx: Dict[int, int] = None,
) -> Dict[int, tuple]:
    """
    Build {item_id → (category_id_encoded, price_bucket)} for fast __getitem__.
    """
    lookup: Dict[int, tuple] = {}
    for idx, row in item_meta.iterrows():
        raw_cat    = int(row.get("categoryid", -1))
        price      = int(row.get("price_bucket", 0))

        if cat2idx is not None:
            # Map raw categoryid → encoded sequential index. Unknown → 0.
            cat_enc = cat2idx.get(raw_cat, cat2idx.get(str(raw_cat), 0))
        else:
            # Assume item_meta already contains encoded IDs (e.g. DuckDB pipeline)
            cat_enc = max(raw_cat, 0)   # guard against -1 sentinel

        lookup[int(idx)] = (cat_enc, price)
    return lookup