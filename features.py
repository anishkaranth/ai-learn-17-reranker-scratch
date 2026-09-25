"""First-stage retrieval (BM25 top-N) and query-document features for the reranker."""
from __future__ import annotations

import math
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np

from bm25 import BM25
from dense import LSADense
from text import tokenize

FEATURES = ["bm25", "bm25_rel_max", "dense_cos", "coverage", "title_overlap", "max_qtf", "log_len", "first_stage_rr"]


class FirstStage:
    def __init__(self, docs: Dict[str, Dict[str, str]], background: List[str], dense_dim: int = 32):
        self.docs = docs
        self.text = {k: v["title"] + " " + v["body"] for k, v in docs.items()}
        self.bm25 = BM25(self.text)
        self.dense = LSADense(self.text, background, dim=dense_dim)
        self.toks = {k: tokenize(t) for k, t in self.text.items()}
        self.title_toks = {k: set(tokenize(v["title"])) for k, v in docs.items()}

    def retrieve(self, query: str, depth: int) -> Tuple[List[str], np.ndarray]:
        s = self.bm25.scores(query)
        order = np.argsort(-s, kind="stable")[:depth]
        return [self.bm25.ids[i] for i in order], s[order]

    def features(self, query: str, cands: List[str], bm25_scores: np.ndarray) -> np.ndarray:
        q = tokenize(query)
        qset = set(q)
        qv = self.dense.embed(query)
        top = float(bm25_scores.max()) if len(bm25_scores) and bm25_scores.max() > 0 else 1.0
        rows = []
        for rank, (d, s) in enumerate(zip(cands, bm25_scores), start=1):
            tf = Counter(self.toks[d])
            rows.append([
                float(s),
                float(s) / top,
                float(self.dense.D[self.dense.ids.index(d)] @ qv),
                sum(1 for t in qset if tf[t]) / max(len(qset), 1),
                len(qset & self.title_toks[d]) / max(len(qset), 1),
                float(max((tf[t] for t in qset), default=0)),
                math.log(1 + len(self.toks[d])),
                1.0 / rank,
            ])
        return np.array(rows)


def build_pairs(fs: FirstStage, queries, depth: int):
    """-> list of dicts per query: candidate ids, feature matrix X, graded labels y."""
    out = []
    for qid, topic, q, rel in queries:
        cands, s = fs.retrieve(q, depth)
        X = fs.features(q, cands, s)
        y = np.array([rel.get(d, 0) for d in cands], dtype=float)
        out.append({"id": qid, "query": q, "cands": cands, "X": X, "y": y, "rel": rel})
    return out
