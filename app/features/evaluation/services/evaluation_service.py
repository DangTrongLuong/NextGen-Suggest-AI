"""Evaluation Service - Tính Recall@K, Precision@K, NDCG@K, MRR, MAP."""
import numpy as np
import time
from typing import List, Dict, Callable


def precision_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    hits = sum(1 for r in retrieved[:k] if r in set(relevant))
    return hits / k if k > 0 else 0.0


def recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for r in retrieved[:k] if r in set(relevant))
    return hits / len(relevant)


def reciprocal_rank(retrieved: List[str], relevant: List[str]) -> float:
    rel_set = set(relevant)
    for i, r in enumerate(retrieved, 1):
        if r in rel_set:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    rel_set = set(relevant)
    dcg = sum(
        1.0 / np.log2(i + 1)
        for i, r in enumerate(retrieved[:k], 1)
        if r in rel_set
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def average_precision(retrieved: List[str], relevant: List[str]) -> float:
    rel_set = set(relevant)
    hits = 0
    ap = 0.0
    for i, r in enumerate(retrieved, 1):
        if r in rel_set:
            hits += 1
            ap += hits / i
    return ap / len(relevant) if relevant else 0.0


def run_evaluation(
    test_queries: List[Dict],
    search_fn: Callable[[str, int], List[str]],
    k_values: List[int] = None,
    method_name: str = "unknown",
) -> Dict:
    """
    Chạy đánh giá trên tập test queries.

    Args:
        test_queries: [{"query": "...", "relevant_ids": [...]}]
        search_fn: function(query, top_k) → List[product_id]
        k_values: danh sách K để đánh giá
        method_name: tên phương pháp

    Returns:
        Dict chứa tất cả metrics
    """
    if k_values is None:
        k_values = [5, 10, 20]

    max_k = max(k_values)

    metrics: Dict[str, List[float]] = {f"P@{k}": [] for k in k_values}
    metrics.update({f"R@{k}": [] for k in k_values})
    metrics.update({f"NDCG@{k}": [] for k in k_values})
    mrr_list, map_list, latency_list = [], [], []

    for item in test_queries:
        query = item["query"]
        relevant = item["relevant_ids"]

        t0 = time.perf_counter()
        retrieved_ids = search_fn(query, max_k)
        latency_ms = (time.perf_counter() - t0) * 1000
        latency_list.append(latency_ms)

        for k in k_values:
            metrics[f"P@{k}"].append(precision_at_k(retrieved_ids, relevant, k))
            metrics[f"R@{k}"].append(recall_at_k(retrieved_ids, relevant, k))
            metrics[f"NDCG@{k}"].append(ndcg_at_k(retrieved_ids, relevant, k))

        mrr_list.append(reciprocal_rank(retrieved_ids, relevant))
        map_list.append(average_precision(retrieved_ids, relevant))

    return {
        "method": method_name,
        "num_queries": len(test_queries),
        "k_values": k_values,
        "metrics": {k: round(float(np.mean(v)), 4) for k, v in metrics.items()},
        "MRR": round(float(np.mean(mrr_list)), 4),
        "MAP": round(float(np.mean(map_list)), 4),
        "avg_latency_ms": round(float(np.mean(latency_list)), 2),
    }
