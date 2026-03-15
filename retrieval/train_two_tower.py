import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retrieval.dataset import TwoTowerDataset, build_item_meta_lookup
from retrieval.two_tower import TwoTowerModel, infonce_loss

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def evaluate_recall(model, val_loader, device, k=10):
    model.eval()
    hits, total = 0, 0
    with torch.no_grad():
        for batch in val_loader:
            u_emb = model.encode_users(
                batch["history_ids"].to(device),
                batch["history_weights"].to(device),
            )
            i_emb = model.encode_items(
                batch["item_id"].to(device),
                batch["category_id"].to(device),
                batch["price_bucket"].to(device),
            )
            sims = u_emb @ i_emb.T
            topk = sims.topk(k, dim=1).indices
            diag = torch.arange(len(u_emb), device=device)
            hits  += (topk == diag.unsqueeze(1)).any(dim=1).sum().item()
            total += len(u_emb)
    return hits / max(total, 1)


def main():
    cfg   = load_config()
    dcfg  = cfg["data"]
    rcfg  = cfg["retrieval"]

    device = torch.device(
        "mps"  if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available()          else
        "cpu"
    )
    log.info("Device: %s", device)

    # Load data
    log.info("Loading processed data …")
    train_df  = pd.read_parquet(dcfg["train_path"])
    val_df    = pd.read_parquet(dcfg["val_path"])
    item_meta = pd.read_parquet(dcfg["item_meta_path"])
    user_hist = pd.read_parquet(dcfg["user_history_path"])

    with open(dcfg["item_id_map_path"]) as f:
        item2idx = json.load(f)

    with open(dcfg["category_map_path"]) as f:
        cat2idx: dict = {int(k): int(v) for k, v in json.load(f).items()}

    n_items      = len(item2idx)
    n_categories = len(cat2idx)

    n_price_buckets = int(train_df["price_bucket"].max()) + 1

    log.info(
        "n_items=%d  n_categories=%d  n_price_buckets=%d",
        n_items, n_categories, n_price_buckets,
    )

    item_lookup = build_item_meta_lookup(item_meta, cat2idx=cat2idx)

    train_ds = TwoTowerDataset(train_df, user_hist, item_lookup,
                               history_len=dcfg["history_len"])
    val_ds   = TwoTowerDataset(val_df,   user_hist, item_lookup,
                               history_len=dcfg["history_len"])

    train_loader = DataLoader(train_ds, batch_size=rcfg["batch_size"],
                              shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=rcfg["batch_size"] * 2,
                              shuffle=False, num_workers=2)

    # Build model
    model = TwoTowerModel(
        n_items         = n_items,
        n_categories    = n_categories,
        n_price_buckets = n_price_buckets,
        embedding_dim   = rcfg["embedding_dim"],
        hidden_dims     = rcfg["hidden_dims"],
        dropout         = rcfg["dropout"],
    ).to(device)

    log.info("Two-Tower params: %d", sum(p.numel() for p in model.parameters()))

    optimizer = torch.optim.Adam(model.parameters(), lr=rcfg["learning_rate"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=rcfg["epochs"]
    )

    best_recall = 0.0
    model_path  = rcfg["model_path"]
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)

    # Training loop
    for epoch in range(1, rcfg["epochs"] + 1):
        model.train()
        total_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            h_ids = batch["history_ids"].to(device)
            h_w   = batch["history_weights"].to(device)
            i_ids = batch["item_id"].to(device)
            c_ids = batch["category_id"].to(device)
            p_ids = batch["price_bucket"].to(device)

            user_emb = model.encode_users(h_ids, h_w)
            item_emb = model.encode_items(i_ids, c_ids, p_ids)

            loss = infonce_loss(user_emb, item_emb, temperature=rcfg["temperature"])

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        recall   = evaluate_recall(model, val_loader, device, k=10)
        log.info("Epoch %d/%d | loss=%.4f | val Recall@10=%.4f",
                 epoch, rcfg["epochs"], avg_loss, recall)

        if recall > best_recall:
            best_recall = recall
            torch.save({
                "state_dict":      model.state_dict(),
                "n_items":         n_items,
                "n_categories":    n_categories,
                "n_price_buckets": n_price_buckets,
                "embedding_dim":   rcfg["embedding_dim"],
                "hidden_dims":     rcfg["hidden_dims"],
            }, model_path)
            log.info("  ✓ New best Recall@10=%.4f — model saved.", best_recall)

    # Export item embeddings
    log.info("Exporting item embeddings …")
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    cat_arr   = np.zeros(n_items + 1, dtype=np.int32)
    price_arr = np.zeros(n_items + 1, dtype=np.int32)
    for iid, (cat, price) in item_lookup.items():
        if iid < n_items:
            cat_arr[iid + 1]   = cat
            price_arr[iid + 1] = price

    all_ids    = torch.arange(1, n_items + 1, dtype=torch.long)
    all_cats   = torch.tensor(cat_arr[1:],   dtype=torch.long)
    all_prices = torch.tensor(price_arr[1:], dtype=torch.long)

    chunk, embs = 4096, []
    with torch.no_grad():
        for start in range(0, n_items, chunk):
            end = min(start + chunk, n_items)
            e = model.encode_items(
                all_ids[start:end].to(device),
                all_cats[start:end].to(device),
                all_prices[start:end].to(device),
            ).cpu().numpy()
            embs.append(e)

    item_emb_arr = np.vstack(embs).astype(np.float32)
    np.save(dcfg["item_embeddings_path"], item_emb_arr)
    log.info("Item embeddings saved → %s  shape=%s",
             dcfg["item_embeddings_path"], item_emb_arr.shape)
    log.info("Training complete. Best val Recall@10=%.4f", best_recall)


if __name__ == "__main__":
    main()