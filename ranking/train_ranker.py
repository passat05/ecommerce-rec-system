import json
import logging
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ranking.deepfm_model import DeepFM
from retrieval.dataset import build_item_meta_lookup
from retrieval.two_tower import TwoTowerModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)

def compute_bucket_edges(values: np.ndarray, n_buckets: int) -> np.ndarray:
    quantiles = np.linspace(0, 100, n_buckets + 1)
    finite    = values[np.isfinite(values)]
    return np.percentile(finite, quantiles[1:-1])


def apply_bucket_edges(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.searchsorted(edges, values).astype(np.int8)


# LambdaRank loss
def lambda_rank_loss(
    scores: torch.Tensor,   # (n_items,)  — model scores for all items in list
    labels: torch.Tensor,   # (n_items,)  — 1=positive, 0=negative
    k: int = 10,
) -> torch.Tensor:
    """
    LambdaRank loss for a single user's item list.
    """
    n = len(scores)
    if n == 0 or labels.sum() == 0:
        return torch.tensor(0.0, requires_grad=True, device=scores.device)

    # Current ranking by model score (descending)
    # rank[i] = position of item i (1-indexed)
    _, order = scores.sort(descending=True)
    rank = torch.zeros(n, device=scores.device)
    rank[order] = torch.arange(1, n + 1, dtype=torch.float, device=scores.device)

    # Discount: 1 / log2(rank + 1)
    discount = 1.0 / torch.log2(rank + 1.0)  # (n,)

    # All pos-neg pairs
    pos_idx = (labels == 1).nonzero(as_tuple=True)[0]  # indices of positives
    neg_idx = (labels == 0).nonzero(as_tuple=True)[0]  # indices of negatives

    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return torch.tensor(0.0, requires_grad=True, device=scores.device)

    # Broadcast: pos_i × neg_j matrices
    pi = pos_idx.unsqueeze(1).expand(-1, len(neg_idx))  # (n_pos, n_neg)
    nj = neg_idx.unsqueeze(0).expand(len(pos_idx), -1)  # (n_pos, n_neg)

    s_pos = scores[pi]       # (n_pos, n_neg) — positive scores
    s_neg = scores[nj]       # (n_pos, n_neg) — negative scores

    # |ΔNDCG| for each pair — only count pairs where at least one item is in top-K
    d_pos = discount[pi]     # (n_pos, n_neg)
    d_neg = discount[nj]     # (n_pos, n_neg)

    # Mask: only weight pairs where rank_pos or rank_neg <= k
    in_topk = (rank[pi] <= k) | (rank[nj] <= k)

    delta_ndcg = (d_pos - d_neg).abs() * in_topk.float()  # (n_pos, n_neg)

    # BPR-style loss, weighted by |ΔNDCG|
    pair_loss = -delta_ndcg * F.logsigmoid(s_pos - s_neg)

    # Normalise by number of relevant pairs to keep loss scale stable
    n_pairs = in_topk.float().sum().clamp(min=1)
    return pair_loss.sum() / n_pairs


# LambdaRank Dataset

class LambdaRankDataset(Dataset):
    """
    Each sample = one user + ALL their items (pos + neg) in training data.
    The LambdaRank loss is computed over the full list per user.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        user_emb_path: str,
        item_emb: np.ndarray,
        recency_edges: np.ndarray,
        pop_edges: np.ndarray,
        max_items_per_user: int = 50,
    ):
        """
        max_items_per_user : cap item list length per user.
        Retail users rarely have >50 unique item interactions in the
        training window; capping keeps batches a fixed size.
        """
        self.max_items = max_items_per_user

        # Build per-user item list
        item_feats = (
            df.sort_values("item_id")
            .drop_duplicates("item_id")
            .set_index("item_id")
        )
        rec = item_feats["recency_days"].fillna(item_feats["recency_days"].median()).values
        pop = item_feats["n_views"].fillna(0).values.astype(np.float32)
        item_feats = item_feats.copy()
        item_feats["recency_bucket"] = apply_bucket_edges(rec, recency_edges)
        item_feats["pop_bucket"]     = apply_bucket_edges(pop, pop_edges)

        self.item_cat = item_feats["category_id"].fillna(0).astype(int).to_dict()
        self.item_price = item_feats["price_bucket"].fillna(0).astype(int).to_dict()
        self.item_event = item_feats["event_type"].fillna(0).astype(int).to_dict()
        self.item_recency = item_feats["recency_bucket"].to_dict()
        self.item_pop = item_feats["pop_bucket"].to_dict()

        # Group by user
        user_groups = df.groupby("user_id")
        self.users   = []
        self.items   = []
        self.labels  = []

        for uid, grp in user_groups:
            # Sort: positives first (helps sampling in short lists)
            grp = grp.sort_values("label", ascending=False)
            iids   = grp["item_id"].values[:max_items_per_user].astype(np.int32)
            lbls   = grp["label"].values[:max_items_per_user].astype(np.float32)
            # Skip users with no positives (nothing to rank)
            if lbls.sum() == 0:
                continue
            self.users.append(int(uid))
            self.items.append(iids)
            self.labels.append(lbls)

        log.info("LambdaRankDataset: %d users, avg %.1f items/user",
                 len(self.users),
                 np.mean([len(x) for x in self.items]))

        self.user_emb_mm = np.load(user_emb_path, mmap_mode="r")
        self.item_emb = item_emb.astype(np.float32)

    def _item_feats(self, iid: int) -> tuple:
        return (
            self.item_cat.get(iid, 0),
            self.item_price.get(iid, 0),
            self.item_event.get(iid, 0),
            self.item_recency.get(iid, 0),
            self.item_pop.get(iid, 0),
        )

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx: int):
        uid = self.users[idx]
        iids = self.items[idx]
        labels = self.labels[idx]
        n = len(iids)

        u_emb = torch.tensor(self.user_emb_mm[uid].copy(), dtype=torch.float32)
        i_embs = torch.tensor(self.item_emb[iids], dtype=torch.float32)

        cats = torch.tensor([self.item_cat.get(i, 0) for i in iids], dtype=torch.long)
        prices = torch.tensor([self.item_price.get(i, 0) for i in iids], dtype=torch.long)
        events = torch.tensor([self.item_event.get(i, 0) for i in iids], dtype=torch.long)
        recencys = torch.tensor([self.item_recency.get(i, 0) for i in iids], dtype=torch.long)
        pops = torch.tensor([self.item_pop.get(i, 0) for i in iids], dtype=torch.long)
        lbls = torch.tensor(labels, dtype=torch.float32)

        return {
            "uid": uid,
            "user_emb": u_emb,      # (emb_dim,)
            "item_emb": i_embs,     # (n, emb_dim)
            "cat": cats,            # (n,)
            "price": prices,        # (n,)
            "event": events,        # (n,)
            "recency": recencys,    # (n,)
            "pop": pops,            # (n,)
            "labels": lbls,         # (n,)
            "n_items": n,
        }


def lambda_rank_collate(batch):
    """
    Custom collate: items per user vary in length → keep as list,
    process each user separately in the training loop.
    """
    return batch   # list of dicts; training loop iterates over users


# Step 1: Generate user embeddings

def generate_user_embeddings(cfg: dict, device: torch.device) -> None:
    dcfg = cfg["data"]
    rcfg = cfg["retrieval"]

    log.info("Loading Two-Tower model for user embedding generation …")

    with open(dcfg["item_id_map_path"]) as f:
        item2idx = json.load(f)
    with open(dcfg["category_map_path"]) as f:
        cat2idx: dict = {int(k): int(v) for k, v in json.load(f).items()}

    item_meta = pd.read_parquet(dcfg["item_meta_path"])
    user_hist = pd.read_parquet(dcfg["user_history_path"])
    item_lookup = build_item_meta_lookup(item_meta, cat2idx=cat2idx)

    ckpt = torch.load(rcfg["model_path"], map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        n_items, n_categories, n_price_buckets = (
            ckpt["n_items"], ckpt["n_categories"], ckpt["n_price_buckets"]
        )
        state_dict = ckpt["state_dict"]
    else:
        n_items = len(item2idx)
        n_categories = len(cat2idx)
        n_price_buckets = rcfg.get("n_price_buckets", 11)
        state_dict = ckpt

    model = TwoTowerModel(
        n_items=n_items, n_categories=n_categories,
        n_price_buckets=n_price_buckets,
        embedding_dim=rcfg["embedding_dim"],
        hidden_dims=rcfg["hidden_dims"], dropout=0.0,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    n_users = int(user_hist["user_id"].max()) + 1
    emb_dim = rcfg["embedding_dim"]
    out_path = dcfg["user_embeddings_path"]
    K = dcfg["history_len"]

    log.info("Generating embeddings for %d users → %s", n_users, out_path)
    user_emb_mm = np.lib.format.open_memmap(
        out_path, mode="w+", dtype=np.float32, shape=(n_users, emb_dim)
    )

    hist_ids_all = np.zeros((n_users, K), dtype=np.int32)
    hist_w_all = np.zeros((n_users, K), dtype=np.float32)
    for _, row in user_hist.iterrows():
        uid = int(row["user_id"])
        hist = row["history"][-K:]
        for i, iid in enumerate(hist):
            hist_ids_all[uid, i] = iid + 1
            hist_w_all[uid, i] = 1.0

    chunk = 2048
    with torch.no_grad():
        for start in tqdm(range(0, n_users, chunk), desc="User embeddings"):
            end = min(start + chunk, n_users)
            h_ids = torch.tensor(hist_ids_all[start:end], dtype=torch.long).to(device)
            h_w = torch.tensor(hist_w_all[start:end],   dtype=torch.float).to(device)
            emb = model.encode_users(h_ids, h_w).cpu().numpy()
            user_emb_mm[start:end] = emb

    user_emb_mm.flush()
    log.info("User embeddings written → %s  shape=(%d, %d)", out_path, n_users, emb_dim)


# Step 2: Train DeepFM with LambdaRank

def train_deepfm(cfg: dict, device: torch.device) -> None:
    dcfg = cfg["data"]
    kcfg = cfg["ranking"]

    log.info("Loading data for DeepFM (LambdaRank) …")
    train_df = pd.read_parquet(dcfg["train_path"])
    val_df = pd.read_parquet(dcfg["val_path"])
    item_emb = np.load(dcfg["item_embeddings_path"])

    with open(dcfg["category_map_path"]) as f:
        n_categories = len(json.load(f))

    # Save bucket edges (used by evaluate.py)
    train_rec = train_df["recency_days"].fillna(train_df["recency_days"].median()).values
    train_pop = train_df["n_views"].fillna(0).values.astype(np.float32)
    recency_edges = compute_bucket_edges(train_rec, 11)
    pop_edges = compute_bucket_edges(train_pop, 11)

    edges_path = Path(dcfg["processed_path"]) / "bucket_edges.npz"
    np.savez(str(edges_path), recency_edges=recency_edges, pop_edges=pop_edges)
    log.info("Bucket edges saved → %s", edges_path)

    # Save per-item popularity
    item_pop = train_df.groupby("item_id").size().rename("n_views").reset_index()
    item_pop["pop_bucket"] = apply_bucket_edges(
        item_pop["n_views"].values.astype(np.float32), pop_edges
    )
    item_pop_path = Path(dcfg["processed_path"]) / "item_popularity.parquet"
    item_pop.set_index("item_id").to_parquet(str(item_pop_path))
    log.info("Item popularity saved → %s", item_pop_path)

    # Datasets
    max_items = kcfg.get("max_items_per_user", 50)
    train_ds = LambdaRankDataset(
        train_df, dcfg["user_embeddings_path"], item_emb,
        recency_edges, pop_edges, max_items_per_user=max_items,
    )
    val_ds = LambdaRankDataset(
        val_df, dcfg["user_embeddings_path"], item_emb,
        recency_edges, pop_edges, max_items_per_user=max_items,
    )

    # No padding collate — each item in batch is one full user list
    train_loader = DataLoader(train_ds, batch_size=kcfg["batch_size"],
                              shuffle=True, num_workers=0,
                              collate_fn=lambda_rank_collate)
    val_loader = DataLoader(val_ds,   batch_size=kcfg["batch_size"],
                              shuffle=False, num_workers=0,
                              collate_fn=lambda_rank_collate)

    model = DeepFM(
        n_categories = n_categories,
        retrieval_emb_dim = cfg["retrieval"]["embedding_dim"],
        field_emb_dim = kcfg["embedding_dim"],
        hidden_dims = kcfg["hidden_dims"],
        dropout = kcfg["dropout"],
    ).to(device)
    log.info("DeepFM params: %d", sum(p.numel() for p in model.parameters()))

    optimizer = torch.optim.Adam(model.parameters(), lr=kcfg["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=2, factor=0.5
    )

    ndcg_k    = cfg["evaluation"]["k_values"][1]   # e.g. k=10
    best_val  = float("inf")
    model_path = kcfg["model_path"]
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)

    def run_epoch(loader, train: bool):
        model.train() if train else model.eval()
        total_loss, n_batches = 0.0, 0

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for user_batch in tqdm(loader,
                                   desc="train" if train else "val",
                                   leave=False):
                batch_loss = torch.tensor(0.0, device=device)
                n_users_in_batch = 0

                for sample in user_batch:
                    n = sample["n_items"]
                    if n == 0:
                        continue

                    u_emb  = sample["user_emb"].unsqueeze(0).expand(n, -1).to(device)
                    i_emb  = sample["item_emb"].to(device)
                    cat    = sample["cat"].to(device)
                    price  = sample["price"].to(device)
                    event  = sample["event"].to(device)
                    recenc = sample["recency"].to(device)
                    pop    = sample["pop"].to(device)
                    labels = sample["labels"].to(device)

                    scores = model(
                        category_id    = cat,
                        event_type     = event,
                        price_bucket   = price,
                        recency_bucket = recenc,
                        pop_bucket     = pop,
                        user_emb       = u_emb,
                        item_emb       = i_emb,
                    )  # (n,)

                    loss = lambda_rank_loss(scores, labels, k=ndcg_k)
                    batch_loss = batch_loss + loss
                    n_users_in_batch += 1

                if n_users_in_batch == 0:
                    continue

                batch_loss = batch_loss / n_users_in_batch

                if train:
                    optimizer.zero_grad()
                    batch_loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                total_loss += batch_loss.item()
                n_batches  += 1

        return total_loss / max(n_batches, 1)

    for epoch in range(1, kcfg["epochs"] + 1):
        avg_train = run_epoch(train_loader, train=True)
        avg_val   = run_epoch(val_loader,   train=False)
        scheduler.step(avg_val)

        log.info("Epoch %d/%d | train_loss=%.4f | val_loss=%.4f",
                 epoch, kcfg["epochs"], avg_train, avg_val)

        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(), model_path)
            log.info("  ✓ val_loss=%.4f — DeepFM saved.", best_val)

    log.info("LambdaRank training complete. Best val_loss=%.4f", best_val)


def main():
    cfg    = load_config()
    device = torch.device(
        "mps"  if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available()          else
        "cpu"
    )
    log.info("Device: %s", device)
    generate_user_embeddings(cfg, device)
    train_deepfm(cfg, device)


if __name__ == "__main__":
    main()