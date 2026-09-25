# ai-learn-17-reranker-scratch

**Phase D, day 17** of the AI learning track. Retrieval systems usually work in two stages. A cheap **first stage** (BM25 here) pulls the top-N candidates, then a **learned reranker** reorders them using richer query-document features. This repo builds both from scratch. It covers the feature extraction and three rerankers trained with NumPy gradient descent (**pointwise logistic regression**, **pairwise RankNet-style logistic regression** and a **tiny pointwise MLP**), and it measures nDCG@5 and MRR before and after reranking on a **held-out split of topics**.

It follows hybrid search (`ai-learn-16`) and reuses its BM25, LSA and metric code. NumPy + matplotlib only, no network, seed 42, under a second on CPU.

## What you'll learn

- How to turn retrieval into supervised learning: (query, candidate, graded label) rows for pointwise training, or (better, worse) pairs for pairwise training.
- Which features help a reranker beat BM25. Query-term **coverage** and **dense similarity** push the answer up. **Max query-term frequency** gets a negative weight because it catches keyword-stuffed docs.
- Why you evaluate on held-out *topics*, and why first-stage recall caps what any reranker can do (the `oracle` row).

## Architecture

```mermaid
flowchart LR
  subgraph Data["data.py (seed 42)"]
    T[40 topics x 4 concepts<br/>canonical + synonym] --> D[answer / partial / stuffed docs]
    T --> Q[120 queries, 50% synonyms]
    Q --> SP{topic split}
    SP --> TR[train 84]
    SP --> TE[test 36]
  end
  D --> FS[BM25 first stage: top-20]
  TR --> FS
  TE --> FS
  FS --> F["features: bm25, bm25/max, dense cos,<br/>coverage, title overlap, max qtf, log len, 1/rank"]
  F --> PL[pointwise LR]
  F --> PW[pairwise LR on feature diffs]
  F --> MLP[pointwise MLP 16 tanh]
  PL --> RR[rerank candidates]
  PW --> RR
  MLP --> RR
  RR --> EV[nDCG@5, MRR@10, recall@k: before vs after, train vs test]
  EV --> OUT[results/]
```

## Layout

| path | purpose |
|---|---|
| `data.py` | seeded synthetic corpus: pseudo-word topics, answer/partial/stuffed docs, synonym queries, background text, topic split |
| `bm25.py`, `dense.py`, `text.py`, `metrics.py` | BM25, LSA dense encoder, tokenizer and ranking metrics (same as `ai-learn-16`) |
| `features.py` | `FirstStage` (BM25 top-N) and the 8 query-document features; `build_pairs` |
| `rerankers.py` | `PointwiseLR`, `PairwiseLR`, `PointwiseMLP`, `rerank` |
| `run_smoke.py` / `smoke_plots.py` | train, evaluate before/after on train and test, SVG plots, `RESULTS.md` |
| `notebooks/reranker_walkthrough.ipynb` | walkthrough |
| `results/` | committed `RESULTS.md`, `metrics.json`, `JSON.shot`, `*.svg` |

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_smoke.py
```

## Headline results (held-out test topics, from `results/metrics.json`)

| system | nDCG@5 | MRR@10 | recall@1 |
|---|---:|---:|---:|
| BM25 first stage | 0.692 | 0.744 | 0.347 |
| pointwise LR | 0.754 | 0.801 | 0.389 |
| **pairwise LR** | **0.764** | **0.815** | **0.403** |
| pointwise MLP | 0.755 | 0.801 | 0.389 |
| oracle (ideal order of the top-20) | 0.804 | 0.833 | 0.417 |

- The best reranker (pairwise) adds **+0.072 nDCG@5** on unseen topics. It improves 11 of 36 test queries, and none get worse.
- First-stage recall@20 is only 0.75 on test, because BM25 cannot match queries written entirely in synonyms. That is why even the oracle stops at 0.804. Closing the gap needs a better first stage (hybrid, `ai-learn-16`), not a better reranker.
- The tiny MLP does no better than the linear models on 8 features. With this little data, simpler is enough.

## Limitations and next steps

The data is synthetic and the features are hand-built. A production reranker is a cross-encoder that reads the query and document together. The splits are small (36 test queries), so differences of about 0.01 are noise. Next is chunking strategies for RAG (`ai-learn-18`).
