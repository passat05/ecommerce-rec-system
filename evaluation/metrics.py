from typing import List
import numpy as np


def recall_at_k(predicted: List[int], relevant: List[int], k: int) -> float:
    if not relevant: return 0.0
    return len(set(predicted[:k]) & set(relevant)) / len(relevant)

def precision_at_k(predicted: List[int], relevant: List[int], k: int) -> float:
    if k == 0: return 0.0
    return len(set(predicted[:k]) & set(relevant)) / k

def ndcg_at_k(predicted: List[int], relevant: List[int], k: int) -> float:
    if not relevant: return 0.0
    rel_set = set(relevant)
    dcg  = sum(1.0 / np.log2(i + 2) for i, p in enumerate(predicted[:k]) if p in rel_set)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0.0

def mrr(predicted: List[int], relevant: List[int]) -> float:
    rel_set = set(relevant)
    for i, p in enumerate(predicted):
        if p in rel_set: return 1.0 / (i + 1)
    return 0.0

def average_metrics(all_pred: List[List[int]], all_rel: List[List[int]], k_values: List[int]) -> dict:
    results = {}
    for k in k_values:
        results[f"Recall@{k}"] = float(np.mean([recall_at_k(p, r, k) for p, r in zip(all_pred, all_rel)]))
        results[f"NDCG@{k}"] = float(np.mean([ndcg_at_k(p, r, k) for p, r in zip(all_pred, all_rel)]))
        results[f"Precision@{k}"] = float(np.mean([precision_at_k(p, r, k) for p, r in zip(all_pred, all_rel)]))
    results["MRR"] = float(np.mean([mrr(p, r) for p, r in zip(all_pred, all_rel)]))
    return results
