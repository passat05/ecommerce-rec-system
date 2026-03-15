import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.metrics import average_metrics
from faiss_index.search import FAISSRetriever
from ranking.deepfm_model import DeepFM

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)

def ground_truth(test_df: pd.DataFrame) -> dict:
    return (
        test_df[test_df["label"] == 1]
        .groupby("user_id")["item_id"]
        .apply(list)
        .to_dict()
    )

def exp1_retrieval_only(user_emb_mm, retriever, gt, top_n, k_values) -> dict:
    all_pred, all_rel = [], []
    for uid, rel in gt.items():
        if uid >= len(user_emb_mm):
            continue
        ids, _ = retriever.retrieve_top_n(user_emb_mm[uid], n=top_n)
        all_pred.append(ids.tolist())
        all_rel.append(rel)
    return average_metrics(all_pred, all_rel, k_values)

def build_item_feature_index(
    item_meta: pd.DataFrame,
    item_pop: pd.DataFrame,
    test_df: pd.DataFrame,
    recency_edges: np.ndarray,
    pop_edges: np.ndarray,
    cat2idx: dict,
    item2idx: dict,
) -> dict:
    """
    Build {encoded_item_id → {category_id, price_bucket, pop_bucket, recency_bucket}}
    for ALL 235k items — including the ~49k that appear in events.csv but
    have NO entry in item_properties (category=0, price=0, but real pop_bucket).

    item2idx : {str(raw_itemid): encoded_item_id}
    cat2idx  : {int(raw_cat):    encoded_cat_id}
    """
    raw_to_enc = {int(k): int(v) for k, v in item2idx.items()}
    pop_lookup = item_pop["pop_bucket"].to_dict() if not item_pop.empty else {}

    test_recency = (
        test_df.drop_duplicates("item_id")
        .set_index("item_id")[["recency_days"]]
        .copy()
    )
    rec_vals = test_recency["recency_days"].fillna(0).values
    test_recency["recency_bucket"] = np.searchsorted(recency_edges, rec_vals).astype(int)
    recency_lookup = test_recency["recency_bucket"].to_dict()

    # Step 1: items WITH properties (~185k)
    result = {}
    for raw_itemid, row in item_meta.iterrows():
        enc_id = raw_to_enc.get(int(raw_itemid))
        if enc_id is None:
            continue
        raw_cat = int(row.get("categoryid", -1)) if "categoryid" in item_meta.columns \
                  else int(row.get("category_id", -1))
        enc_cat = cat2idx.get(raw_cat, cat2idx.get(str(raw_cat), 0))
        result[enc_id] = {
            "category_id": enc_cat,
            "price_bucket": int(row.get("price_bucket", 0) or 0),
            "pop_bucket": int(pop_lookup.get(enc_id, 0)),
            "recency_bucket": int(recency_lookup.get(enc_id, 0)),
        }

    # Step 2: items WITHOUT properties (~49k) — cat=0, price=0, real pop
    filled_before = len(result)
    for enc_id in raw_to_enc.values():
        if enc_id not in result:
            result[enc_id] = {
                "category_id":    0,
                "price_bucket":   0,
                "pop_bucket":     int(pop_lookup.get(enc_id, 0)),
                "recency_bucket": int(recency_lookup.get(enc_id, 0)),
            }

    log.info(
        "Item feature index: %d total (%d with full features, %d pop-only)",
        len(result), filled_before, len(result) - filled_before,
    )
    return result


def exp2_retrieval_ranking(
    user_emb_mm, item_emb, retriever, deepfm,
    item_feature_index, gt, top_n, k_values, device,
    n_categories: int,
) -> dict:
    deepfm.eval()
    all_pred, all_rel = [], []

    for uid, rel in gt.items():
        if uid >= len(user_emb_mm):
            continue

        cand_ids, _ = retriever.retrieve_top_n(user_emb_mm[uid], n=top_n)
        n_c = len(cand_ids)

        u_emb = torch.tensor(
            np.tile(user_emb_mm[uid], (n_c, 1)), dtype=torch.float32
        ).to(device)
        i_emb = torch.tensor(item_emb[cand_ids], dtype=torch.float32).to(device)

        def feat_tensor(feat_name: str, default: int = 0,
                        max_val: int = None) -> torch.Tensor:
            vals = [
                item_feature_index.get(int(iid), {}).get(feat_name, default)
                for iid in cand_ids
            ]
            t = torch.tensor(vals, dtype=torch.long).to(device)
            if max_val is not None:
                t = t.clamp(0, max_val)
            return t

        with torch.no_grad():
            scores = deepfm(
                category_id = feat_tensor("category_id", max_val=n_categories),
                event_type = torch.zeros(n_c, dtype=torch.long).to(device),
                price_bucket = feat_tensor("price_bucket", max_val=10),
                recency_bucket = feat_tensor("recency_bucket", max_val=10),
                pop_bucket = feat_tensor("pop_bucket", max_val=10),
                user_emb = u_emb,
                item_emb = i_emb,
            ).cpu().numpy()

        ranked = cand_ids[np.argsort(scores)[::-1]]
        all_pred.append(ranked.tolist())
        all_rel.append(rel)

    return average_metrics(all_pred, all_rel, k_values)

def main():
    cfg      = load_config()
    dcfg     = cfg["data"]
    k_values = cfg["evaluation"]["k_values"]
    top_n    = cfg["faiss"]["top_n"]
    out_dir  = Path(cfg["evaluation"]["results_path"])
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "mps"  if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else
        "cpu"
    )

    log.info("Loading test artefacts …")
    test_df = pd.read_parquet(dcfg["test_path"])
    item_emb = np.load(dcfg["item_embeddings_path"]).astype(np.float32)
    user_emb_mm = np.load(dcfg["user_embeddings_path"], mmap_mode="r")
    retriever = FAISSRetriever(cfg["faiss"]["index_path"])
    gt = ground_truth(test_df)
    log.info("Test users with positives: %d", len(gt))

    all_results = {}

    # Experiment 1: Two-Tower + FAISS
    log.info("=== Exp 1: Two-Tower + FAISS ===")
    r1 = exp1_retrieval_only(user_emb_mm, retriever, gt, top_n, k_values)
    all_results["retrieval_only"] = r1
    for k, v in r1.items():
        log.info("  %s = %.4f", k, v)

    # Experiment 2: Two-Tower + FAISS + DeepFM (LambdaRank)
    deepfm_path = cfg["ranking"]["model_path"]
    if Path(deepfm_path).exists():
        with open(dcfg["category_map_path"]) as f:
            n_cat = len(json.load(f))

        deepfm = DeepFM(
            n_categories      = n_cat,
            retrieval_emb_dim = cfg["retrieval"]["embedding_dim"],
            field_emb_dim     = cfg["ranking"]["embedding_dim"],
            hidden_dims       = cfg["ranking"]["hidden_dims"],
            dropout           = 0.0,
        ).to(device)
        deepfm.load_state_dict(torch.load(deepfm_path, map_location=device))

        # Load item features from full item_meta, not test_df
        item_meta = pd.read_parquet(dcfg["item_meta_path"])

        # Load cat2idx to encode raw categoryid → sequential 0..N
        with open(dcfg["category_map_path"]) as f:
            cat2idx_raw = json.load(f)
        cat2idx = {int(k): int(v) for k, v in cat2idx_raw.items()}

        # Load bucket edges saved during training
        edges_path = Path(dcfg["processed_path"]) / "bucket_edges.npz"
        if edges_path.exists():
            edges = np.load(str(edges_path))
            recency_edges = edges["recency_edges"]
            pop_edges = edges["pop_edges"]
            log.info("Bucket edges loaded from %s", edges_path)
        else:
            log.warning("bucket_edges.npz not found — using test-set percentiles (suboptimal)")
            rec = test_df["recency_days"].fillna(0).values
            pop = test_df["n_views"].fillna(0).values.astype(np.float32)
            recency_edges = np.percentile(rec[np.isfinite(rec)],
                                          np.linspace(0, 100, 12)[1:-1])
            pop_edges     = np.percentile(pop[np.isfinite(pop)],
                                          np.linspace(0, 100, 12)[1:-1])

        item_pop_path = Path(dcfg["processed_path"]) / "item_popularity.parquet"
        if item_pop_path.exists():
            item_pop = pd.read_parquet(str(item_pop_path))
        else:
            log.warning("item_popularity.parquet not found — pop_bucket=0 for all items")
            item_pop = pd.DataFrame({"pop_bucket": []})
            item_pop.index.name = "item_id"

        with open(dcfg["item_id_map_path"]) as f:
            item2idx_raw = json.load(f)

        item_feature_index = build_item_feature_index(
            item_meta, item_pop, test_df, recency_edges, pop_edges,
            cat2idx=cat2idx,
            item2idx=item2idx_raw,
        )
        log.info("Item feature index built: %d items", len(item_feature_index))

        log.info("=== Exp 2: Two-Tower + FAISS + DeepFM (LambdaRank) ===")
        r2 = exp2_retrieval_ranking(
            user_emb_mm, item_emb, retriever, deepfm,
            item_feature_index, gt, top_n, k_values, device,
            n_categories=n_cat,
        )
        all_results["retrieval_deepfm"] = r2
        for k, v in r2.items():
            log.info(" %s = %.4f", k, v)
    else:
        log.warning("DeepFM model not found — skipping Exp 2.")

    # Save & print
    import json as _json
    results_path = out_dir / "evaluation_results.json"
    with open(results_path, "w") as f:
        _json.dump(all_results, f, indent=2)
    log.info("Results saved → %s", results_path)

    print("\n" + "=" * 64)
    print(f"{'Metric':<20} {'Two-Tower+FAISS':>20} {'+ DeepFM':>20}")
    print("=" * 64)
    for k in k_values:
        for metric in [f"Recall@{k}", f"NDCG@{k}"]:
            r1v = all_results.get("retrieval_only",  {}).get(metric, float("nan"))
            r2v = all_results.get("retrieval_deepfm",{}).get(metric, float("nan"))
            delta = r2v - r1v if not (np.isnan(r1v) or np.isnan(r2v)) else float("nan")
            sign  = "+" if delta > 0 else ""
            print(f"{metric:<20} {r1v:>20.4f} {r2v:>18.4f} ({sign}{delta:.4f})")
    print("=" * 64)


if __name__ == "__main__":
    main()