"""
faiss_index/search.py
---------------------
Runtime FAISS retriever singleton.
"""
import logging
from pathlib import Path
from typing import Tuple
import faiss
import numpy as np
import yaml

log = logging.getLogger(__name__)

def load_config(path="configs/config.yaml"):
    with open(path) as f: return yaml.safe_load(f)

class FAISSRetriever:
    def __init__(self, index_path: str):
        if not Path(index_path).exists():
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        self.index = faiss.read_index(index_path)
        if hasattr(self.index, "nprobe"):
            self.index.nprobe = 16
        log.info("FAISS index loaded: %d items", self.index.ntotal)

    def retrieve_top_n(self, user_emb: np.ndarray, n: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        q = np.atleast_2d(user_emb).astype(np.float32)
        scores, ids = self.index.search(q, n)
        return ids[0], scores[0]

    def batch_retrieve(self, user_embs: np.ndarray, n: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        scores, ids = self.index.search(user_embs.astype(np.float32), n)
        return ids, scores

_retriever: FAISSRetriever = None

def get_retriever(config_path="configs/config.yaml") -> FAISSRetriever:
    global _retriever
    if _retriever is None:
        cfg = load_config(config_path)
        _retriever = FAISSRetriever(cfg["faiss"]["index_path"])
    return _retriever

def retrieve_top_n(user_emb: np.ndarray, n: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    return get_retriever().retrieve_top_n(user_emb, n=n)
