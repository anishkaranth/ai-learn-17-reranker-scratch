"""Ranking metrics: recall@k, MRR@k, nDCG@k (graded relevance)."""
from __future__ import annotations

import math
from typing import Dict, List


def recall_at_k(ranked: List[str], rel: Dict[str, int], k: int) -> float:
    """Fraction of relevant docs (grade>0) found in the top k."""
    return sum(1 for d in ranked[:k] if rel.get(d, 0) > 0) / len(rel)


def mrr_at_k(ranked: List[str], rel: Dict[str, int], k: int = 10) -> float:
    for i, d in enumerate(ranked[:k], start=1):
        if rel.get(d, 0) > 0:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: List[str], rel: Dict[str, int], k: int) -> float:
    dcg = sum((2 ** rel.get(d, 0) - 1) / math.log2(i + 1) for i, d in enumerate(ranked[:k], start=1))
    ideal = sorted(rel.values(), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / math.log2(i + 1) for i, g in enumerate(ideal, start=1))
    return dcg / idcg if idcg else 0.0


def evaluate(ranked: List[str], rel: Dict[str, int]) -> Dict[str, float]:
    return {"recall@1": recall_at_k(ranked, rel, 1), "recall@3": recall_at_k(ranked, rel, 3),
            "recall@5": recall_at_k(ranked, rel, 5), "mrr@10": mrr_at_k(ranked, rel, 10),
            "ndcg@5": ndcg_at_k(ranked, rel, 5)}
