import argparse, json, logging, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def load_config(path="configs/config.yaml"):
    with open(path) as f: return yaml.safe_load(f)


def run_inference(user_ids, top_k=10):
    cfg  = load_config()
    dcfg = cfg["data"]
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")

    user_emb_mm = np.load(dcfg["user_embeddings_path"], mmap_mode="r")
    item_emb    = np.load(dcfg["item_embeddings_path"]).astype(np.float32)

    with open(dcfg["item_id_map_path"]) as f:
        item_idx_map = {v: k for k, v in json.load(f).items()}
    with open(dcfg["category_map_path"]) as f:
        n_cats = len(json.load(f))

    from faiss_index.search import FAISSRetriever
    retriever = FAISSRetriever(cfg["faiss"]["index_path"])
    top_n = cfg["faiss"]["top_n"]

    # DeepFM
    deepfm = None
    deepfm_path = cfg["ranking"]["model_path"]
    if Path(deepfm_path).exists():
        from ranking.deepfm_model import DeepFM
        rcfg = cfg["retrieval"]; kcfg = cfg["ranking"]
        deepfm = DeepFM(n_categories=n_cats,
                        retrieval_emb_dim=rcfg["embedding_dim"],
                        field_emb_dim=kcfg["embedding_dim"],
                        hidden_dims=kcfg["hidden_dims"], dropout=0.0).to(device)
        deepfm.load_state_dict(torch.load(deepfm_path, map_location=device))
        deepfm.eval()
        log.info("DeepFM loaded.")

    results = []
    for uid in user_ids:
        if uid < 0 or uid >= len(user_emb_mm):
            log.warning("user_id %d out of range — skipping.", uid); continue

        user_emb = user_emb_mm[uid].copy()
        cand_ids, ret_scores = retriever.retrieve_top_n(user_emb, n=top_n)

        if deepfm is not None:
            n_c   = len(cand_ids)
            u_t   = torch.tensor(np.tile(user_emb, (n_c, 1)), dtype=torch.float32).to(device)
            i_t   = torch.tensor(item_emb[cand_ids], dtype=torch.float32).to(device)
            with torch.no_grad():
                scores = deepfm(
                    category_id    = torch.zeros(n_c, dtype=torch.long).to(device),
                    event_type     = torch.zeros(n_c, dtype=torch.long).to(device),
                    price_bucket   = torch.zeros(n_c, dtype=torch.long).to(device),
                    recency_bucket = torch.zeros(n_c, dtype=torch.long).to(device),
                    pop_bucket     = torch.zeros(n_c, dtype=torch.long).to(device),
                    user_emb=u_t, item_emb=i_t,
                ).cpu().numpy()
            order = np.argsort(scores)[::-1]
            final_ids, final_scores = cand_ids[order], scores[order]
        else:
            final_ids, final_scores = cand_ids, ret_scores

        recs = [
            {"item_id": int(iid), "original_id": item_idx_map.get(int(iid), str(iid)),
             "score": round(float(s), 4)}
            for iid, s in zip(final_ids[:top_k], final_scores[:top_k])
        ]
        results.append({"user_id": uid, "recommendations": recs})
        print(f"\nUser {uid}:")
        for r in recs:
            print(f"  [{r['score']:+.4f}]  item={r['original_id']}  (idx={r['item_id']})")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user-ids", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--top-k",    type=int,  default=10)
    parser.add_argument("--output",   type=str,  default=None)
    args = parser.parse_args()

    results = run_inference(args.user_ids, top_k=args.top_k)
    if args.output:
        with open(args.output, "w") as f: json.dump(results, f, indent=2)
        log.info("Saved → %s", args.output)


if __name__ == "__main__":
    main()
