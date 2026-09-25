"""Learned rerankers trained with plain NumPy gradient descent.

  * PointwiseLR : logistic regression on (q, d) features, target = relevant (grade > 0)
  * PairwiseLR  : RankNet-style linear scorer; for pairs with y_i > y_j minimise log(1 + exp(-(s_i - s_j)))
  * PointwiseMLP: one hidden layer (tanh), regression onto the graded label with MSE
All models standardise features with train-set mean/std.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class _Base:
    def fit_scaler(self, X):
        self.mu = X.mean(0)
        self.sd = X.std(0) + 1e-8

    def _z(self, X):
        return (X - self.mu) / self.sd


class PointwiseLR(_Base):
    name = "pointwise_lr"

    def __init__(self, lr=0.1, epochs=500, l2=1e-3, seed=42):
        self.lr, self.epochs, self.l2, self.seed = lr, epochs, l2, seed

    def fit(self, groups: List[Dict]):
        X = np.vstack([g["X"] for g in groups]); y = (np.concatenate([g["y"] for g in groups]) > 0).astype(float)
        self.fit_scaler(X); Z = self._z(X)
        rng = np.random.default_rng(self.seed)
        self.w = rng.normal(0, 0.01, Z.shape[1]); self.b = 0.0
        self.loss = []
        for _ in range(self.epochs):
            p = _sigmoid(Z @ self.w + self.b)
            self.loss.append(float(-np.mean(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))))
            g = p - y
            self.w -= self.lr * (Z.T @ g / len(y) + self.l2 * self.w)
            self.b -= self.lr * g.mean()
        return self

    def score(self, X):
        return self._z(X) @ self.w + self.b


class PairwiseLR(_Base):
    name = "pairwise_lr"

    def __init__(self, lr=0.1, epochs=500, l2=1e-3, seed=42):
        self.lr, self.epochs, self.l2, self.seed = lr, epochs, l2, seed

    def fit(self, groups: List[Dict]):
        self.fit_scaler(np.vstack([g["X"] for g in groups]))
        diffs = []
        for g in groups:
            Z = self._z(g["X"]); y = g["y"]
            for i in range(len(y)):
                for j in range(len(y)):
                    if y[i] > y[j]:
                        diffs.append(Z[i] - Z[j])
        D = np.array(diffs)
        self.n_pairs = len(D)
        rng = np.random.default_rng(self.seed)
        self.w = rng.normal(0, 0.01, D.shape[1])
        self.loss = []
        for _ in range(self.epochs):
            m = D @ self.w
            self.loss.append(float(np.mean(np.log1p(np.exp(-np.clip(m, -30, 30))))))
            g = -_sigmoid(-m)
            self.w -= self.lr * (D.T @ g / len(D) + self.l2 * self.w)
        return self

    def score(self, X):
        return self._z(X) @ self.w


class PointwiseMLP(_Base):
    name = "pointwise_mlp"

    def __init__(self, hidden=16, lr=0.05, epochs=1500, l2=1e-3, seed=42):
        self.h, self.lr, self.epochs, self.l2, self.seed = hidden, lr, epochs, l2, seed

    def fit(self, groups: List[Dict]):
        X = np.vstack([g["X"] for g in groups]); y = np.concatenate([g["y"] for g in groups])
        self.fit_scaler(X); Z = self._z(X)
        rng = np.random.default_rng(self.seed)
        self.W1 = rng.normal(0, 1 / np.sqrt(Z.shape[1]), (Z.shape[1], self.h)); self.b1 = np.zeros(self.h)
        self.W2 = rng.normal(0, 1 / np.sqrt(self.h), self.h); self.b2 = 0.0
        self.loss = []
        n = len(y)
        for _ in range(self.epochs):
            H = np.tanh(Z @ self.W1 + self.b1)
            out = H @ self.W2 + self.b2
            err = out - y
            self.loss.append(float(np.mean(err ** 2)))
            gW2 = H.T @ err * 2 / n + self.l2 * self.W2
            gb2 = err.mean() * 2
            dH = np.outer(err, self.W2) * (1 - H ** 2) * 2 / n
            gW1 = Z.T @ dH + self.l2 * self.W1
            gb1 = dH.sum(0)
            self.W1 -= self.lr * gW1; self.b1 -= self.lr * gb1
            self.W2 -= self.lr * gW2; self.b2 -= self.lr * gb2
        return self

    def score(self, X):
        return np.tanh(self._z(X) @ self.W1 + self.b1) @ self.W2 + self.b2


def rerank(model, group) -> List[str]:
    s = model.score(group["X"])
    return [group["cands"][i] for i in np.argsort(-s, kind="stable")]
