#!/usr/bin/env python3
"""Reranker smoke: BM25 first stage -> pointwise LR / pairwise LR / tiny MLP rerankers, train vs held-out test -> results/."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np

from data import make_dataset
from features import FEATURES, FirstStage, build_pairs
from metrics import evaluate
from rerankers import PairwiseLR, PointwiseLR, PointwiseMLP, rerank
from smoke_plots import make_plots, write_results_md

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
SEED = 42
DEPTH = 20
METRIC_KEYS = ["ndcg@5", "mrr@10", "recall@1", "recall@5"]


def _compact(js: str) -> str:
    return re.sub(r"\[\s+([^\[\]{}]*?)\s+\]", lambda m: "[" + re.sub(r"\s+", " ", m.group(1)) + "]", js)


def _eval(groups, order_fn):
    rows = [evaluate(order_fn(g), g["rel"]) for g in groups]
    return {k: round(float(np.mean([r[k] for r in rows])), 4) for k in METRIC_KEYS}, [r["ndcg@5"] for r in rows]


def _oracle(g):
    return [g["cands"][i] for i in np.argsort(-g["y"], kind="stable")]


def main() -> None:
    t0 = time.perf_counter()
    ds = make_dataset(seed=SEED)
    fs = FirstStage(ds["docs"], ds["background"])
    groups = {s: build_pairs(fs, ds["split"][s], DEPTH) for s in ("train", "test")}
    models = [PointwiseLR(seed=SEED), PairwiseLR(seed=SEED), PointwiseMLP(seed=SEED)]
    for mdl in models:
        mdl.fit(groups["train"])
    res = {"train": {}, "test": {}}
    per_q = {}
    for s in ("train", "test"):
        res[s]["bm25_first_stage"], per_q[(s, "bm25")] = _eval(groups[s], lambda g: g["cands"])
        for mdl in models:
            res[s][mdl.name], per_q[(s, mdl.name)] = _eval(groups[s], lambda g, mdl=mdl: rerank(mdl, g))
        res[s]["oracle"], _ = _eval(groups[s], _oracle)
    flips = {}
    for mdl in models:
        a, b = np.array(per_q[("test", "bm25")]), np.array(per_q[("test", mdl.name)])
        flips[mdl.name] = {"improved": int((b > a + 1e-9).sum()), "unchanged": int((abs(b - a) <= 1e-9).sum()), "worse": int((b < a - 1e-9).sum())}
    runtime = time.perf_counter() - t0

    m = {
        "project": "ai-learn-17-reranker-scratch", "seed": SEED,
        "config": {"depth": DEPTH, "n_topics": 40, "queries_per_topic": 3, "p_syn": 0.5, "test_frac": 0.3, "dense_dim": 32,
                   "lr_epochs": 500, "mlp_hidden": 16, "mlp_epochs": 1500},
        "dataset": {"n_docs": len(ds["docs"]), "n_queries": len(ds["queries"]), "n_train": len(ds["split"]["train"]),
                    "n_test": len(ds["split"]["test"]), "test_topics": ds["test_topics"]},
        "features": FEATURES, "metric_keys": METRIC_KEYS,
        "train_rows": int(sum(len(g["y"]) for g in groups["train"])), "n_pairs": models[1].n_pairs,
        "first_stage_recall": {s: round(float(np.mean([(g["y"] > 0).sum() / len(g["rel"]) for g in groups[s]])), 4) for s in groups},
        "train": res["train"], "test": res["test"], "test_flips": flips,
        "weights": {"pointwise_lr": [round(float(v), 4) for v in models[0].w], "pairwise_lr": [round(float(v), 4) for v in models[1].w]},
        "loss": {mdl.name: [round(mdl.loss[0], 4), round(mdl.loss[-1], 4)] for mdl in models},
        "runtime_s": round(runtime, 3),
    }
    RESULTS.mkdir(exist_ok=True)
    plots = make_plots(RESULTS, m)
    m["plots"] = plots
    (RESULTS / "metrics.json").write_text(_compact(json.dumps(m, indent=1)), encoding="utf-8")
    best = max((mdl.name for mdl in models), key=lambda n: res["test"][n]["ndcg@5"])
    shot = {"project": m["project"], "seed": SEED, "config": m["config"], "n_train": m["dataset"]["n_train"], "n_test": m["dataset"]["n_test"],
            "test_ndcg@5": {s: a["ndcg@5"] for s, a in res["test"].items()},
            "test_mrr@10": {s: a["mrr@10"] for s, a in res["test"].items()},
            "first_stage_recall@20_test": m["first_stage_recall"]["test"], "best_reranker": best,
            "test_ndcg@5_gain": round(res["test"][best]["ndcg@5"] - res["test"]["bm25_first_stage"]["ndcg@5"], 4),
            "runtime_s": m["runtime_s"],
            "pass": bool(res["test"][best]["ndcg@5"] > res["test"]["bm25_first_stage"]["ndcg@5"])}
    (RESULTS / "JSON.shot").write_text(json.dumps(shot, indent=2), encoding="utf-8")
    write_results_md(RESULTS, m, plots)
    for s in ("train", "test"):
        for n, a in res[s].items():
            print(f"  {s:5s} {n:18s} ndcg@5={a['ndcg@5']:.3f} mrr@10={a['mrr@10']:.3f}")
    print(f"best={best} gain={shot['test_ndcg@5_gain']:+.3f} pass={shot['pass']} runtime {runtime:.2f}s | wrote {RESULTS}")


if __name__ == "__main__":
    main()
