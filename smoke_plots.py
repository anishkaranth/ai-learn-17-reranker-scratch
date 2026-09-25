"""Matplotlib SVG plots + RESULTS.md writer for the reranker smoke run."""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from svg_utils import minify_svg  # noqa: E402

plt.rcParams["svg.hashsalt"] = "ai-learn-17"
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
_META = {"Date": None}
_COLORS = ["#adb5bd", "#2a9d8f", "#e9c46a", "#e76f51", "#264653"]


def _save(fig, path: Path) -> str:
    fig.tight_layout()
    buf = io.StringIO()
    fig.savefig(buf, format="svg", metadata=_META)
    plt.close(fig)
    path.write_text(minify_svg(buf.getvalue()), encoding="utf-8")
    return path.name


def make_plots(out: Path, m: Dict[str, Any]) -> List[str]:
    out.mkdir(exist_ok=True)
    names = []
    systems = list(m["test"])
    keys = ["ndcg@5", "mrr@10"]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    w = 0.8 / len(systems)
    x = np.arange(len(keys))
    for i, s in enumerate(systems):
        ax.bar(x + i * w - 0.4 + w / 2, [m["test"][s][k] for k in keys], w, label=s, color=_COLORS[i % 5])
    ax.set_xticks(x, keys)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_title("Held-out test queries: before vs after reranking")
    ax.legend(fontsize=8, loc="lower right")
    names.append(_save(fig, out / "rerank_before_after.svg"))

    fig, ax = plt.subplots(figsize=(7, 3.2))
    feats = m["features"]
    y = np.arange(len(feats))
    ax.barh(y + 0.2, m["weights"]["pointwise_lr"], 0.4, label="pointwise_lr", color=_COLORS[1])
    ax.barh(y - 0.2, m["weights"]["pairwise_lr"], 0.4, label="pairwise_lr", color=_COLORS[2])
    ax.set_yticks(y, feats)
    ax.axvline(0, color="#333", lw=0.8)
    ax.set_xlabel("weight on standardised feature")
    ax.set_title("Linear reranker feature weights")
    ax.legend(fontsize=8)
    names.append(_save(fig, out / "feature_weights.svg"))
    return names


def write_results_md(out: Path, m: Dict[str, Any], plots: List[str]) -> None:
    c = m["config"]; ds = m["dataset"]
    keys = m["metric_keys"]
    L = ["# Results -- ai-learn-17-reranker-scratch", "",
         f"**Seed:** `{m['seed']}` | docs={ds['n_docs']} | queries={ds['n_queries']} (train {ds['n_train']}, test {ds['n_test']}, split by topic) | "
         f"first stage = BM25 top-{c['depth']} | training pairs: pointwise rows={m['train_rows']}, pairwise pairs={m['n_pairs']} | features={len(m['features'])}", "",
         f"First-stage recall@{c['depth']}: train {m['first_stage_recall']['train']:.3f}, test {m['first_stage_recall']['test']:.3f}. A reranker can only reorder these candidates, so `oracle` (ideal order of the candidates) is the ceiling.", ""]
    for split in ("test", "train"):
        L += [f"## {split} split (real smoke run)", "", "| system | " + " | ".join(keys) + " |", "|---|" + "---:|" * len(keys)]
        for s, agg in m[split].items():
            L.append(f"| `{s}` | " + " | ".join(f"{agg[k]:.3f}" for k in keys) + " |")
        L.append("")
    L += ["## Test-query flips vs BM25 order (nDCG@5)", "", "| model | improved | unchanged | worse |", "|---|---:|---:|---:|"]
    for s, f in m["test_flips"].items():
        L.append(f"| `{s}` | {f['improved']} | {f['unchanged']} | {f['worse']} |")
    L += ["", "## Final training loss", "", "| model | first-epoch loss | final loss |", "|---|---:|---:|"]
    for s, l in m["loss"].items():
        L.append(f"| `{s}` | {l[0]:.4f} | {l[-1]:.4f} |")
    L += ["", "## Plots", ""] + [f"![{p}]({p})" for p in plots] + ["", f"Wall time: {m['runtime_s']:.2f}s on CPU.", ""]
    (out / "RESULTS.md").write_text("\n".join(L), encoding="utf-8")
