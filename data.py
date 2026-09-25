"""Synthetic, seeded retrieval dataset with graded labels and a topic-level train/test split.

Each topic has 4 concepts; each concept has a canonical word and a synonym (pseudo-words, so no
prior knowledge leaks in). Per topic we write:
  * an answer doc   (grade 2): title + body cover all 4 concepts
  * a partial doc   (grade 1): covers 2 concepts
  * a stuffed doc   (grade 0): repeats one concept word many times (keyword stuffing) inside off-topic text
Queries use 2-3 concepts of a topic; each word is swapped for its synonym with probability p_syn.
A background corpus of canonical/synonym co-occurrences is used to fit the dense (LSA) feature.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

_CONS = "bdfgklmnprstvz"
_VOW = "aeiou"


def _pseudo_words(rng, n: int) -> List[str]:
    out, seen = [], set()
    while len(out) < n:
        w = "".join(rng.choice(list(_CONS)) + rng.choice(list(_VOW)) for _ in range(rng.integers(2, 4)))
        if w not in seen:
            seen.add(w)
            out.append(w)
    return out


def make_dataset(n_topics: int = 40, queries_per_topic: int = 3, p_syn: float = 0.5, test_frac: float = 0.3, seed: int = 42):
    rng = np.random.default_rng(seed)
    words = _pseudo_words(rng, n_topics * 8 + 60)
    filler = words[n_topics * 8:]
    topics = []
    for t in range(n_topics):
        w = words[t * 8:(t + 1) * 8]
        topics.append([(w[2 * i], w[2 * i + 1]) for i in range(4)])  # (canonical, synonym) x 4

    def fill(k):
        return list(rng.choice(filler, size=k))

    docs: Dict[str, Dict[str, str]] = {}
    for t, concepts in enumerate(topics):
        canon = [c for c, _ in concepts]
        body = canon + fill(10)
        rng.shuffle(body)
        docs[f"t{t:02d}_answer"] = {"title": " ".join(canon[:2]), "body": " ".join(body)}
        part = list(rng.choice(canon, size=2, replace=False))
        body = part + fill(12)
        rng.shuffle(body)
        docs[f"t{t:02d}_partial"] = {"title": " ".join(fill(2)), "body": " ".join(body)}
        other = topics[(t + 7) % n_topics]
        stuffed = [canon[int(rng.integers(0, 4))]] * 4 + [c for c, _ in other][:2] + fill(8)
        rng.shuffle(stuffed)
        docs[f"t{t:02d}_stuffed"] = {"title": " ".join(fill(2)), "body": " ".join(stuffed)}

    background = []
    for concepts in topics:
        for c, s in concepts:
            background.append(" ".join([c, s] + fill(3)))
        background.append(" ".join([c for c, _ in concepts] + [s for _, s in concepts]))

    queries: List[Tuple[str, int, str, Dict[str, int]]] = []
    for t, concepts in enumerate(topics):
        for j in range(queries_per_topic):
            k = int(rng.integers(2, 4))
            idx = rng.choice(4, size=k, replace=False)
            q = [concepts[i][1] if rng.random() < p_syn else concepts[i][0] for i in idx]
            rel = {f"t{t:02d}_answer": 2, f"t{t:02d}_partial": 1}
            queries.append((f"q{t:02d}_{j}", t, " ".join(q), rel))

    perm = rng.permutation(n_topics)
    test_topics = set(int(x) for x in perm[: int(round(test_frac * n_topics))])
    split = {"train": [q for q in queries if q[1] not in test_topics], "test": [q for q in queries if q[1] in test_topics]}
    return {"docs": docs, "background": background, "queries": queries, "split": split,
            "topics": topics, "test_topics": sorted(test_topics)}
