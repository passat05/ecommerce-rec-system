import argparse
import logging
from pathlib import Path

import numpy as np
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def normalize_l2(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(norms == 0, 1.0, norms)


def export(normalize: bool = False) -> None:
    cfg      = load_config()
    dcfg     = cfg["data"]
    emb_path = dcfg["item_embeddings_path"]

    item_emb = np.load(emb_path)
    log.info("Loaded item_embeddings: shape=%s  dtype=%s", item_emb.shape, item_emb.dtype)

    if normalize:
        log.info("Applying L2 normalisation …")
        item_emb = normalize_l2(item_emb)

    out = item_emb.astype(np.float32)
    np.save(emb_path, out)
    log.info("Saved float32 embeddings → %s  (normalize=%s)", emb_path, normalize)

    # Quick sanity check
    norms = np.linalg.norm(out, axis=1)
    log.info("Norm stats — mean=%.3f  min=%.3f  max=%.3f", norms.mean(), norms.min(), norms.max())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--normalize", action="store_true",
                        help="L2-normalise embeddings (dot product → cosine similarity)")
    args = parser.parse_args()
    export(normalize=args.normalize)
