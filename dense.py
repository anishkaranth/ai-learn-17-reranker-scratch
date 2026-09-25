"""Dense retrieval: TF-IDF -> truncated SVD (LSA) -> L2-normalised vectors, cosine similarity.

The vocabulary, IDF and SVD projection are fitted on a background corpus only (a stand-in for an
embedding model's pre-training data), then the docs are embedded with that frozen model. Words that
co-occur in the background (e.g. 'refund' and 'money back') land close together, but tokens the model
never saw (error codes, product names) are out-of-vocabulary, just like rare strings for a real encoder.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Dict, List, Sequence

import numpy as np

from text import tokenize


class LSADense:
    def __init__(self, docs: Dict[str, str], background: Sequence[str], dim: int = 24):
        self.ids = list(docs)
        toks = [tokenize(t) for t in background]
        self.vocab = {w: i for i, w in enumerate(sorted({w for t in toks for w in t}))}
        df = Counter(w for t in toks for w in set(t))
        N = len(toks)
        self.idf = np.zeros(len(self.vocab))
        for w, i in self.vocab.items():
            self.idf[i] = math.log((1 + N) / (1 + df[w])) + 1
        X = np.stack([self._tfidf(t) for t in toks])
        # SVD of the term-document matrix; keep the top `dim` term directions
        _, S, Vt = np.linalg.svd(X, full_matrices=False)
        self.dim = min(dim, len(S))
        self.P = Vt[: self.dim].T  # (V, dim)
        self.explained = float((S[: self.dim] ** 2).sum() / (S ** 2).sum())
        self.D = np.stack([self.embed(docs[i]) for i in self.ids])
        doc_toks = [w for i in self.ids for w in tokenize(docs[i])]
        self.doc_oov_rate = float(np.mean([w not in self.vocab for w in doc_toks]))

    def _tfidf(self, toks: List[str]) -> np.ndarray:
        v = np.zeros(len(self.vocab))
        for w, c in Counter(toks).items():
            if w in self.vocab:
                v[self.vocab[w]] = (1 + math.log(c)) * self.idf[self.vocab[w]]
        n = np.linalg.norm(v)
        return v / n if n else v

    def _embed_tfidf(self, X: np.ndarray) -> np.ndarray:
        Z = X @ self.P
        n = np.linalg.norm(Z, axis=-1, keepdims=True)
        return Z / np.where(n == 0, 1, n)

    def embed(self, text: str) -> np.ndarray:
        return self._embed_tfidf(self._tfidf(tokenize(text))[None])[0]

    def scores(self, query: str) -> np.ndarray:
        return self.D @ self.embed(query)

    def search(self, query: str, k: int = 10) -> List[str]:
        s = self.scores(query)
        return [self.ids[i] for i in np.argsort(-s, kind="stable")[:k]]
