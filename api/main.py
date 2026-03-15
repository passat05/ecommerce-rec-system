"""
api/main.py
-----------
FastAPI serving layer for the E-commerce Recommendation System.

Inference flow:
    POST /recommend  {user_id}
      → memmap user embedding lookup (or on-the-fly for cold users)
      → FAISS candidate retrieval  (top-N)
      → DeepFM re-ranking
      → return top-K results

IndexError fix applied
───────────────────────
    TwoTowerModel is loaded from the checkpoint dict saved by
    train_two_tower.py, which contains the exact vocab sizes
    (n_items, n_categories, n_price_buckets) used during training.
    This prevents IndexError when rebuilding the model at startup.

Run:
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
"""

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schema import HealthResponse, RecommendRequest, RecommendResponse, RecommendedItem

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

CONFIG_PATH = os.getenv("CONFIG_PATH", "configs/config.yaml")


class AppState:
    cfg           = None
    device        = None
    user_emb_mm   = None
    item_emb      = None
    item_idx_map  : dict = None
    cat2idx       : dict = None   # raw categoryid → encoded sequential index
    n_categories  : int  = 0      # DeepFM cat_emb vocab size
    retriever     = None
    two_tower     = None
    deepfm        = None
    item_meta     = None
    user_history  = None
    top_n  : int  = 50
    top_k  : int  = 10
    history_len: int = 20


state = AppState()


def load_config(path: str = CONFIG_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Loading artefacts …")
    state.cfg = load_config()
    dcfg = state.cfg["data"]

    state.device = torch.device(
        "mps"  if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available()          else
        "cpu"
    )
    log.info("Device: %s", state.device)

    # Embeddings
    state.user_emb_mm = np.load(dcfg["user_embeddings_path"], mmap_mode="r")
    state.item_emb    = np.load(dcfg["item_embeddings_path"]).astype(np.float32)
    log.info("Embeddings — users: %s (memmap)  items: %s",
             state.user_emb_mm.shape, state.item_emb.shape)

    # Item reverse map
    with open(dcfg["item_id_map_path"]) as f:
        raw = json.load(f)
    state.item_idx_map = {v: k for k, v in raw.items()}

    # FAISS
    from faiss_index.search import FAISSRetriever
    state.retriever   = FAISSRetriever(state.cfg["faiss"]["index_path"])
    state.top_n       = state.cfg["faiss"]["top_n"]
    state.top_k       = state.cfg["api"]["top_k"]
    state.history_len = dcfg["history_len"]

    # Item meta (for DeepFM features)
    state.item_meta = pd.read_parquet(dcfg["item_meta_path"])

    # cat2idx: raw categoryid → encoded sequential index (needed by DeepFM)
    with open(dcfg["category_map_path"]) as f:
        raw_cat = json.load(f)
    state.cat2idx      = {int(k): int(v) for k, v in raw_cat.items()}
    state.n_categories = len(state.cat2idx)

    # User history (for on-the-fly cold-user encoding)
    hist_df = pd.read_parquet(dcfg["user_history_path"])
    state.user_history = {
        int(r.user_id): r.history for _, r in hist_df.iterrows()
    }

    # Two-Tower — FIX: load checkpoint dict for exact vocab sizes
    rcfg = state.cfg["retrieval"]
    ckpt = torch.load(rcfg["model_path"], map_location=state.device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        n_items         = ckpt["n_items"]
        n_cats          = ckpt["n_categories"]
        n_price_buckets = ckpt["n_price_buckets"]
        tt_state        = ckpt["state_dict"]
    else:
        # Fallback for old-style plain state_dict
        with open(dcfg["item_id_map_path"]) as f:
            n_items = len(json.load(f))
        with open(dcfg["category_map_path"]) as f:
            n_cats = len(json.load(f))
        n_price_buckets = 11
        tt_state        = ckpt

    from retrieval.two_tower import TwoTowerModel
    tt = TwoTowerModel(
        n_items         = n_items,
        n_categories    = n_cats,
        n_price_buckets = n_price_buckets,
        embedding_dim   = rcfg["embedding_dim"],
        hidden_dims     = rcfg["hidden_dims"],
        dropout         = 0.0,
    ).to(state.device)
    tt.load_state_dict(tt_state)
    tt.eval()
    state.two_tower = tt

    # DeepFM
    kcfg        = state.cfg["ranking"]
    deepfm_path = kcfg["model_path"]
    if Path(deepfm_path).exists():
        from ranking.deepfm_model import DeepFM
        dfm = DeepFM(
            n_categories      = n_cats,
            retrieval_emb_dim = rcfg["embedding_dim"],
            field_emb_dim     = kcfg["embedding_dim"],
            hidden_dims       = kcfg["hidden_dims"],
            dropout           = 0.0,
        ).to(state.device)
        dfm.load_state_dict(torch.load(deepfm_path, map_location=state.device))
        dfm.eval()
        state.deepfm = dfm
        log.info("DeepFM loaded.")
    else:
        log.warning("DeepFM model not found — using retrieval scores only.")

    log.info("API ready.")
    yield
    log.info("Shutdown.")


app = FastAPI(
    title="E-commerce Recommendation API",
    description="Two-Tower + FAISS + DeepFM recommendation system.",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


def _get_user_embedding(uid: int) -> np.ndarray:
    """Return user embedding from memmap, or compute on-the-fly for cold users."""
    if uid < len(state.user_emb_mm):
        emb = state.user_emb_mm[uid].copy()
        if not np.allclose(emb, 0):
            return emb

    # Cold user: encode from history
    hist = state.user_history.get(uid, [])[-state.history_len:]
    K    = state.history_len
    ids  = np.zeros(K, dtype=np.int32)
    w    = np.zeros(K, dtype=np.float32)
    for i, iid in enumerate(hist):
        ids[i] = iid + 1
        w[i]   = 1.0

    with torch.no_grad():
        emb = state.two_tower.encode_users(
            torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(state.device),
            torch.tensor(w,   dtype=torch.float).unsqueeze(0).to(state.device),
        ).cpu().numpy()[0]
    return emb


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status       = "ok",
        n_users      = int(state.user_emb_mm.shape[0]) if state.user_emb_mm is not None else 0,
        n_items      = int(state.item_emb.shape[0])    if state.item_emb    is not None else 0,
        deepfm_loaded= state.deepfm is not None,
    )


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    t0    = time.perf_counter()
    uid   = req.user_id
    top_k = req.top_k or state.top_k

    if state.user_emb_mm is None:
        raise HTTPException(503, "Embeddings not loaded.")

    user_emb = _get_user_embedding(uid)
    cand_ids, ret_scores = state.retriever.retrieve_top_n(user_emb, n=state.top_n)

    # DeepFM re-ranking
    if state.deepfm is not None:
        n_c   = len(cand_ids)
        u_emb = torch.tensor(
            np.tile(user_emb, (n_c, 1)), dtype=torch.float32
        ).to(state.device)
        i_emb = torch.tensor(
            state.item_emb[cand_ids], dtype=torch.float32
        ).to(state.device)

        def _feat(col: str, default: int = 0, max_val: int = None) -> torch.Tensor:
            vals = []
            for iid in cand_ids:
                try:
                    raw = int(state.item_meta.loc[iid, col])
                    # Encode raw categoryid → sequential index via cat2idx
                    if col == "categoryid":
                        raw = state.cat2idx.get(raw, state.cat2idx.get(str(raw), 0))
                except Exception:
                    raw = default
                vals.append(raw)
            t = torch.tensor(vals, dtype=torch.long).to(state.device)
            if max_val is not None:
                t = t.clamp(0, max_val)
            return t

        with torch.no_grad():
            scores = state.deepfm(
                category_id    = _feat("categoryid", max_val=state.n_categories),
                event_type     = torch.zeros(n_c, dtype=torch.long).to(state.device),
                price_bucket   = _feat("price_bucket", max_val=10),
                recency_bucket = torch.zeros(n_c, dtype=torch.long).to(state.device),
                pop_bucket     = torch.zeros(n_c, dtype=torch.long).to(state.device),
                user_emb       = u_emb,
                item_emb       = i_emb,
            ).cpu().numpy()
        order      = np.argsort(scores)[::-1]
        final_ids  = cand_ids[order]
        final_scrs = scores[order]
    else:
        final_ids  = cand_ids
        final_scrs = ret_scores

    recs = [
        RecommendedItem(
            item_id     = int(iid),
            original_id = str(state.item_idx_map.get(int(iid), iid)),
            score       = round(float(s), 4),
        )
        for iid, s in zip(final_ids[:top_k], final_scrs[:top_k])
    ]

    log.info("user=%d top_k=%d latency=%.1fms",
             uid, top_k, (time.perf_counter() - t0) * 1000)
    return RecommendResponse(user_id=uid, recommendations=recs)