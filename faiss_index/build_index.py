import logging
from pathlib import Path
import faiss
import numpy as np
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

def load_config(path="configs/config.yaml"):
    with open(path) as f: return yaml.safe_load(f)

def main():
    cfg = load_config()
    item_emb = np.load(cfg["data"]["item_embeddings_path"]).astype(np.float32)
    index_path = cfg["faiss"]["index_path"]
    n, dim = item_emb.shape
    log.info("Building FAISS index: %d items, dim=%d", n, dim)

    if n < 50_000:
        base  = faiss.IndexFlatIP(dim)
        index = faiss.IndexIDMap(base)
        index.add_with_ids(item_emb, np.arange(n, dtype=np.int64))
        log.info("IndexFlatIP (exact)")
    else:
        nlist = min(512, n // 39)
        quant = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quant, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(item_emb)
        index.add(item_emb)
        log.info("IndexIVFFlat (nlist=%d)", nlist)

    Path(index_path).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, index_path)
    log.info("Saved → %s  (ntotal=%d)", index_path, index.ntotal)

if __name__ == "__main__":
    main()
