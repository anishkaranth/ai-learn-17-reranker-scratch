"""Okapi BM25 from scratch."""
from __future__ import annotations

import math
from collections import Counter
from typing import Dict, List

import numpy as np

from text import tokenize


class BM25:
    def __init__(self, docs: Dict[str, str], k1: float = 1.5, b: float = 0.75):
        self.ids = list(docs)
        self.k1, self.b = k1, b
        self.tfs = [Counter(tokenize(docs[i])) for i in self.ids]
        self.lens = np.array([sum(tf.values()) for tf in self.tfs], dtype=float)
        self.avgdl = float(self.lens.mean())
        df = Counter(t for tf in self.tfs for t in tf)
        N = len(self.ids)
        # BM25+ style non-negative idf: log(1 + (N - df + 0.5) / (df + 0.5))
        self.idf = {t: math.log(1 + (N - n + 0.5) / (n + 0.5)) for t, n in df.items()}

    def scores(self, query: str) -> np.ndarray:
        q = tokenize(query)
        s = np.zeros(len(self.ids))
        for j, tf in enumerate(self.tfs):
            norm = self.k1 * (1 - self.b + self.b * self.lens[j] / self.avgdl)
            for t in q:
                f = tf.get(t, 0)
                if f:
                    s[j] += self.idf[t] * f * (self.k1 + 1) / (f + norm)
        return s

    def search(self, query: str, k: int = 10) -> List[str]:
        s = self.scores(query)
        return [self.ids[i] for i in np.argsort(-s, kind="stable")[:k]]
