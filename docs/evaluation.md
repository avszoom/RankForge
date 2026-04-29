# Phase 5 — Evaluation

> Honest measurement: how good is the system actually? Run `scripts/evaluate.py` and you get the numbers below.

---

## Quickstart

```bash
source .venv/bin/activate
python scripts/evaluate.py
```

Runs all 4 setups on the **129 held-out test queries** (the ones marked `split=test` in `queries.jsonl`) and writes:

- Overall comparison table (printed)
- Per-`query_type` breakdown (printed)
- `data/eval_results.parquet` — one row per (query × setup), 516 rows total
- `data/eval_summary.json` — aggregate tables, easy to grep / diff

Wall-clock: ~2.5 min (the LightGBM ranker runs the embedding model per query).

---

## What gets evaluated

Each test query is run through 4 setups, each producing a top-50 ranked list of doc IDs. The lists are scored against the labels in `relevance.jsonl`.

| Setup | What it is |
|---|---|
| **bm25** | `BM25Retriever().search()` — keyword retrieval only |
| **faiss** | `VectorRetriever().search()` — semantic retrieval only |
| **hybrid_rrf** | `HybridRetriever().search()` — BM25 + FAISS merged via Reciprocal Rank Fusion (no learning) |
| **ranker** | `LightGBMRanker().rank()` — full pipeline: hybrid candidates → features → trained LightGBM → reordered |

---

## Metrics

| Metric | Formula | Captures |
|---|---|---|
| **NDCG@K** | `DCG@K / IDCG@K`, with `gain = 2^rel − 1` and `discount = 1/log₂(i+1)` | Are highly relevant docs near the top? Range 0–1. |
| **MRR** | `1 / rank_of_first_relevant`, averaged across queries | How quickly does the first relevant doc appear? |
| **Precision@K** | `# relevant in top-K / K` | What fraction of top-K is relevant (label ≥ 2)? |
| **Recall@K** | `# relevant in top-K / # all relevant` | What fraction of relevant docs landed in top-K? |

"Relevant" for the binary metrics (MRR, P, R) means **label ≥ 2** by default (i.e. label 2 or 3). NDCG uses graded labels directly.

---

## Headline results (overall, n=129 test queries)

|            | NDCG@5  | NDCG@10 | NDCG@20 | MRR    | P@5    | P@10   | R@10   | R@50   |
|------------|---------|---------|---------|--------|--------|--------|--------|--------|
| bm25       | 0.654   | 0.642   | 0.604   | 0.720  | 0.673  | 0.649  | 0.260  | 0.406  |
| faiss      | 0.764   | 0.743   | 0.698   | 0.834  | 0.771  | 0.739  | 0.295  | 0.466  |
| hybrid_rrf | 0.762   | 0.741   | 0.685   | 0.821  | 0.772  | 0.738  | 0.295  | 0.460  |
| **ranker** | **0.910** | **0.883** | **0.764** | **0.950** | **0.907** | **0.871** | **0.349** | 0.463 |
| ranker_ce  | 0.824   | 0.806   | 0.733   | 0.875  | 0.834  | 0.804  | 0.322  | 0.463 |

**Headline:** **NDCG@10 = 0.883** for the trained LightGBM ranker. That's a **+14.2 NDCG points** lift over the best non-learned baseline (RRF at 0.741) and **+24 NDCG points** over BM25 alone.

**The cross-encoder rerank (`ranker_ce`) regressed by 7.7 NDCG points** vs LightGBM alone. See [§ "Why the cross-encoder didn't help"](#why-the-cross-encoder-didnt-help) below.

### Two non-obvious findings

1. **RRF barely helps over FAISS alone** (0.741 vs 0.743) on this data. The simple rank-fusion scheme isn't smart enough to combine BM25 and FAISS productively when their score scales differ wildly. The learned ranker is what makes the combination valuable.
2. **Recall@50 is similar across setups** (~0.46). All four setups draw from roughly the same candidate pool — reordering changes precision (top-K), not recall (set membership). The lift is in *ordering*, not *finding*.

---

## Per-`query_type` breakdown — where does the lift come from?

The 129 test queries are split across three types (synthesized in Phase 0):

```
exact          — keyword-heavy, e.g. "LLM inference cost optimization"
paraphrase     — semantic match needed, e.g. "ways to make AI API cheaper"
noisy          — short, telegraphic, no articles, e.g. "llm cost fix"
```

|            | NDCG@10 (exact) | NDCG@10 (paraphrase) | NDCG@10 (noisy) |
|------------|-----------------|----------------------|-----------------|
| bm25       | 0.792           | 0.515                | 0.622           |
| faiss      | 0.801           | 0.696                | 0.730           |
| hybrid_rrf | 0.846           | 0.647                | 0.740           |
| **ranker** | **0.954**       | **0.825**            | **0.865**       |
| ranker_ce  | 0.895           | 0.729                | 0.796           |
| **ranker lift over RRF** | **+10.8** | **+17.8** | **+12.5** |
| ranker_ce delta vs ranker | −5.9 | **−9.5** | −6.9 |

### Why the cross-encoder didn't help

Phase 7 was supposed to nudge NDCG@10 from 0.883 up toward 0.90+. Instead it regressed to 0.806 — a 7.7 point drop. The CE hurt **across every query type**, with the biggest damage on paraphrase queries (−9.5 NDCG@10). Honest analysis of why:

1. **The LightGBM stage is already very strong on this corpus.** When stage 1 already gets the right doc to rank 1–3 in 90%+ of cases, there's little headroom for stage 2 to improve. The CE has more opportunities to *introduce* errors than to fix them — and it took some.

2. **Domain mismatch with MS MARCO.** `cross-encoder/ms-marco-MiniLM-L-6-v2` was trained on real Bing queries against real web text. Our docs are GPT-generated synthetic blog posts in a uniform style. The model judges relevance using patterns it learned on a very different distribution.

3. **Synthetic-data signal mismatch.** On a sanity-check exact query (*"how to reduce LLM inference cost"*) the CE produces nicely-separated positive scores (5–8) and promotes the right docs. On a paraphrase query (*"ways to make AI API cheaper"*) all 20 candidates land in `[-8, -2]` — the CE is correctly saying *"none of these answer the question well"* but is then forced to pick a winner among bad options, and its ordering of "least bad" is noisier than LightGBM's tabular features predicted.

4. **Label scheme mismatch.** MS MARCO has binary relevance (clicked = relevant). Our labels are graded 0–3 with same-category-different-topic = label 2. The CE doesn't know about our label structure and may rate label-2 ≈ label-3 docs similarly, which pushes graded-NDCG down.

### What this finding actually means

It's an honest negative result: **on this specific dataset with a strong LightGBM baseline, the off-the-shelf MS MARCO CE doesn't add value.** That doesn't mean cross-encoders are useless — it means:

- **For real product data**, where LightGBM features may be weaker and queries are noisier, the CE often does help (this is well-documented in industry papers).
- **For synthetic data** like ours where the corpus is short, uniform, and well-aligned with the generated queries, the LightGBM ranker already extracts most of the available signal.
- **A domain-adapted CE** (fine-tuned on RankForge-style data, or just a different off-the-shelf model like `BAAI/bge-reranker-base`) might recover or beat the LightGBM. We didn't try this.

The RankForge code keeps both setups available — `--retriever ranker` for production-quality results, `--retriever ranker_ce` for the experiment. The cross-encoder explainer (`docs/cross-encoder-architecture.md`) is still pedagogically useful even though the eval result was unfavorable.

### What the per-type table tells us

- **Paraphrase queries get the biggest lift** (+17.8 NDCG points over RRF). This is exactly where heuristic combinations struggle — keyword overlap is misleading, and the rank-fusion can't tell when to trust which retriever. The learned ranker figures it out.
- **`hybrid_rrf` actually loses to `faiss` alone on paraphrase queries** (0.647 vs 0.696). RRF dilutes good semantic results with mediocre BM25 results. **The learning is what makes the fusion worth doing.**
- **Exact queries are easy.** Even BM25 alone hits 0.79 NDCG@10. The ranker still adds ~16 points, but the marginal value of learning is highest on the harder query types.
- **Noisy queries** (short telegraphic queries) show that the ranker recovers most of the ground that BM25 alone loses — its NDCG climbs from 0.62 (BM25) to 0.87 (ranker).

---

## A few interpretation notes

### Why is recall (R@10) only ~0.35?

Each test query has up to 25 docs labeled relevant (label ≥ 2): 10 same-topic (label 3) + 15 same-category-other-topic (label 2). With K=10, the *theoretical maximum* recall is 10/25 = 0.40. The ranker achieves 0.349 — about 87% of the best possible.

R@50 hits ~0.46 because there are ~25 relevant docs and only ~15 can be returned beyond the first 10 in the candidate pool (due to retriever caps). This is a property of the candidate pool, not the ranker.

### NDCG@5 vs NDCG@20

- The **ranker's NDCG@5 = 0.910** is higher than NDCG@10 = 0.883. The model is sharpest at the very top — exactly where users look.
- NDCG@20 = 0.764 — lower because positions 11–20 contain a lot of label-1 (weakly relevant) and label-0 docs that the model didn't push down hard enough. We're only training with `lambdarank_truncation_level=20` so that's expected.

### Why MRR = 0.950 is impressive

MRR counts **only the first relevant doc**. 0.95 means the first relevant doc is on average at position ~1.05 — roughly always at position 1. The user's first click should usually be useful.

---

## How to reproduce

```bash
# 1. Make sure data + indexes + model are built (done in earlier phases)
ls data/queries.jsonl data/relevance.jsonl
ls models/bm25.pkl models/faiss.index models/ranker.pkl

# 2. Run
python scripts/evaluate.py

# 3. Inspect raw rows (one per query × setup)
python -c "import pandas as pd; print(pd.read_parquet('data/eval_results.parquet').head())"

# 4. Or grep the JSON summary
cat data/eval_summary.json | python -m json.tool | head -40
```

Default args: `--top-k 50 --split test`. Use `--split all` to evaluate on the train queries too (sanity check).

---

## What this enables

The system is now **end-to-end honest-measured**. We can:

- Quote a defensible NDCG@10 on a held-out test set: **0.883**.
- Show per-query-type breakdowns to argue *where* the learning helps.
- Re-run after any change (more features, different LightGBM params, different retriever configs) and see the delta.
- Add baselines to the comparison without changing the evaluation harness — just plug them into `src/evaluation/evaluate.py`.

Phase 6 (Streamlit UI) is the natural next step — this is the project's "demo-able" form. Phase 7 (MiniBERT cross-encoder reranker) is optional but would tighten precision on the top-20 further.
