# Ranker — Feature Building + Learning-to-Rank

> Forward-looking design doc for **Phase 3 (feature builder)** and **Phase 4 (LightGBM ranker training)**. Not built yet; describes how the next milestone will work so we have a shared mental model.

---

## Why a ranker at all?

Phases 1–2 give us a **candidate set** per query — typically 60–100 docs out of 2520. Inside that set, BM25 and FAISS each have an opinion (a score, a rank), and the RRF fusion in `HybridRetriever` gives a hand-coded compromise.

But neither retriever alone — nor RRF — knows things like:
- "Does the doc's title overlap with the query?"
- "Is the doc fresh or 18 months old?"
- "Does the doc's category match what the query is implicitly about?"
- "Is BM25 score more trustworthy when query length is short?"

A ranker **learns** to combine all those signals into a single relevance score per `(query, doc)` pair, by training on the labels we already generated in `relevance.jsonl`.

---

## The two-phase pipeline

```
┌──────────────────────────────────────────────────────────────────┐
│                    PHASE 3 — FEATURE BUILDING                    │
│                                                                  │
│  data/queries.jsonl                                              │
│  data/corpus.jsonl                                               │
│  data/relevance.jsonl       ──→  per (query, candidate) pair:    │
│  HybridRetriever                                                 │
│                                                                  │
│                                  X = [bm25_score, vector_score,  │
│                                       title_overlap, ...]        │
│                                  y = relevance label (0–3)       │
│                                  groups = which rows belong to   │
│                                           the same query         │
│                                                                  │
│  output: data/features_train.parquet, data/features_test.parquet │
└────────────────────────────────┬─────────────────────────────────┘
                                 ↓
┌──────────────────────────────────────────────────────────────────┐
│                    PHASE 4 — RANKER TRAINING                     │
│                                                                  │
│  features_train  →  LightGBM LambdaRank  →  models/ranker.pkl    │
│                                                                  │
│  evaluation on features_test:                                    │
│    NDCG@10, MRR, Precision@K, Recall@K                           │
│                                                                  │
│  baselines compared: BM25 only, vector only, hybrid+RRF, ranker  │
└──────────────────────────────────────────────────────────────────┘
```

---

## Phase 3 — Feature Builder

### What "training data" means for a ranker

For supervised learning we need:

| | What it is | Source |
|---|---|---|
| **`X`** — feature matrix | one row per `(query, candidate_doc)` pair, ~12 numeric columns | hybrid retriever + corpus + query text |
| **`y`** — labels | the graded relevance for that pair (0, 1, 2, or 3) | `data/relevance.jsonl` |
| **`groups`** — query boundaries | how many rows in `X` belong to each query (rankers must know "which rows are mutually rankable") | derived from `query_id` |

### Feature list (full ~12 features)

| Feature | Where it comes from | What it captures |
|---|---|---|
| `bm25_score` | `Candidate.bm25_score` | raw keyword relevance |
| `bm25_rank` | `Candidate.bm25_rank` | rank position from BM25 (or `top_k+1` if missing) |
| `vector_score` | `Candidate.vector_score` | cosine similarity from FAISS |
| `vector_rank` | `Candidate.vector_rank` | rank position from FAISS |
| `retrieved_by_bm25` | `Candidate.retrieved_by_bm25` | binary flag — was this doc in BM25's top-50? |
| `retrieved_by_vector` | `Candidate.retrieved_by_vector` | binary flag — was this doc in FAISS's top-50? |
| `title_overlap` | `count(query_tokens ∩ title_tokens) / len(query_tokens)` | how much the title literally matches the query |
| `body_overlap` | same, on body tokens | how much the body literally matches |
| `doc_length` | token count of `title + body` | longer docs may be more authoritative — or just noisier |
| `query_length` | token count of the query | short queries behave differently from long ones |
| `freshness_days` | `(today − doc.created_at).days` | doc age — fresher often more relevant |
| `category_match` | 1 if the query's `target_category` equals the doc's `category` else 0 | leaks the train signal a bit; we'll measure with and without |

### How rows are constructed

For each query in `queries.jsonl`:

1. Run `HybridRetriever.search(query, top_k=100, per_retriever=50)` → ~60–100 candidates.
2. For each candidate, look up the relevance label from `relevance.jsonl`.
   - **If the candidate isn't in `relevance.jsonl` for this query**, default `y = 0` (not in our sample = not relevant).
3. Compute the 12 features above.
4. Emit one row per candidate.

A query that retrieved 80 candidates produces 80 rows. With 1344 queries, we end up with ~80–100k rows total.

### Train/test split

Already done at corpus generation time — `queries.jsonl` has a `split` field:
- 1215 train queries → ~100 k feature rows
- 129 test queries → ~10 k feature rows

**Critical: split by query, never by row.** If a query's rows leak across train/test, the ranker memorizes the query and metrics lie. The `groups` array enforces this naturally.

### Outputs

- `data/features_train.parquet` — `X` + `y` + `query_id` + `groups`
- `data/features_test.parquet` — same shape, held out for eval

Parquet because it's column-oriented (fast for LightGBM ingestion) and 5–10× smaller than JSONL.

---

## Phase 4 — Ranker Training

### Why LightGBM with LambdaRank

[**LightGBM**](https://lightgbm.readthedocs.io/) is a gradient-boosted decision tree library. For ranking we use the `lambdarank` objective.

| | Pointwise regression (simpler) | LambdaRank (proper LTR) |
|---|---|---|
| What it learns | Predict the label `y` for each (query, doc) | Predict an ordering that maximizes NDCG |
| Loss function | MSE between prediction and label | Pairwise loss weighted by ΔNDCG when swapping pairs |
| Knows about query groups? | No | Yes — the `group` array tells it which rows are mutually rankable |
| When it wins | Small data, simple problems | Whenever ranking quality matters more than absolute scores |

LambdaRank is the standard for learning-to-rank on graded labels (which we have). It's what Microsoft, LinkedIn, and Google have used for production search ranking for years, and it's surprisingly cheap to train — minutes on CPU at our scale.

### Training input shape

```python
import lightgbm as lgb

train_data = lgb.Dataset(
    X_train,                  # (N_train_rows, ~12)
    label=y_train,            # (N_train_rows,)  — values in {0, 1, 2, 3}
    group=group_sizes_train,  # list[int], one entry per query, summing to N_train_rows
)
```

The `group` array is what makes it a ranker, not a regressor. Example: if query A had 80 candidates and query B had 70, `group_sizes = [80, 70, ...]`.

### Training config (planned)

```python
params = {
    "objective": "lambdarank",
    "metric":    "ndcg",
    "ndcg_eval_at": [5, 10, 20],
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "verbose": -1,
}
model = lgb.train(params, train_data, num_boost_round=500,
                  valid_sets=[test_data], callbacks=[lgb.early_stopping(50)])
```

Output: `models/ranker.pkl` — pickled `lgb.Booster`, ~100 KB.

### What the trained model does at query time

```python
ranker = LightGBMRanker.load("models/ranker.pkl")
candidates = HybridRetriever().search(query, top_k=100)
features = build_features(query, candidates)        # (n_candidates, 12)
scores = ranker.predict(features)                   # (n_candidates,)
ranked = sorted(zip(candidates, scores), key=lambda x: -x[1])[:10]
```

The ranker's score is a real number with no fixed scale — only relative ordering matters. The top-10 is what we surface to the user.

---

## Metrics — How We Measure "Did the Ranker Help?"

For each test query, the ranker produces an ordered list. We compare that ordering to the ground-truth labels (0–3) from `relevance.jsonl`.

### NDCG@K — Normalized Discounted Cumulative Gain (the primary metric)

Captures: *"Are highly relevant docs ranked near the top?"*

**Formula:**

```
DCG@K  = Σ from i=1 to K of:   (2^rel_i − 1) / log₂(i + 1)
IDCG@K = DCG@K of the ideal (perfect) ranking
NDCG@K = DCG@K / IDCG@K       ∈ [0, 1]
```

The `(2^rel − 1)` term means a label-3 doc contributes far more than a label-2 (gain = 7 vs 3); the `log₂(i+1)` discounts later positions.

**Tiny worked example.** Query has docs with true labels `[3, 2, 0, 1]`.

```
Our ranking:   doc_A (rel=3), doc_B (rel=0), doc_C (rel=2), doc_D (rel=1)

Position 1: gain = (2^3 - 1) / log₂(2) = 7 / 1.000 = 7.000
Position 2: gain = (2^0 - 1) / log₂(3) = 0 / 1.585 = 0.000
Position 3: gain = (2^2 - 1) / log₂(4) = 3 / 2.000 = 1.500
Position 4: gain = (2^1 - 1) / log₂(5) = 1 / 2.322 = 0.431
DCG@4 = 8.931

Ideal ranking would be [3, 2, 1, 0]:
IDCG@4 = 7 + (3/log₂(3)) + (1/log₂(4)) + (0/log₂(5))
       = 7 + 1.893 + 0.500 + 0 = 9.393

NDCG@4 = 8.931 / 9.393 = 0.951
```

So the ranker scored 0.951 (close to perfect) for this query. The penalty came from putting the irrelevant doc at position 2 instead of the label-2 one.

We'll report **NDCG@5, NDCG@10, NDCG@20**. NDCG@10 is the headline number.

### MRR — Mean Reciprocal Rank

Captures: *"How quickly do we surface the first relevant result?"*

```
RR = 1 / rank_of_first_relevant_doc       (rel ≥ 2 considered relevant)
MRR = mean of RR across all queries
```

If the first relevant doc is at rank 1 → RR = 1.0. Rank 2 → 0.5. Rank 5 → 0.2. Rank 50 → 0.02.

### Precision@K — How many of the top-K are actually relevant?

```
P@K = |relevant docs in top K| / K
```

Threshold: relevant = label ≥ 2. So P@5 = "of the top 5 results, how many had label 2 or 3?".

### Recall@K — How many of all relevant docs did we get into the top-K?

```
R@K = |relevant docs in top K| / |all relevant docs for this query|
```

P@K and R@K trade off — you push K up, recall goes up, precision goes down.

---

## What we'll compare in Phase 5 evaluation

The interesting story is the lift from each stage:

| Setup | Expected NDCG@10 (rough) |
|---|---|
| BM25 only, top-10 | ~0.35–0.45 — strong on exact queries, weak on paraphrases |
| Vector only, top-10 | ~0.50–0.60 — better on paraphrases, weaker on rare-term queries |
| Hybrid + RRF (no learning) | ~0.55–0.65 — combines both |
| Hybrid + LightGBM ranker | **~0.70–0.80** — learns when to trust which signal, plus surface features |

These are eyeballed estimates; actuals come out of Phase 5. The `query_type` field lets us also report metrics **broken down by exact / paraphrase / noisy**, which is the most honest way to see what the ranker is actually fixing.

---

## Files we'll add (preview)

```
src/
  ranking/
    __init__.py
    feature_builder.py         # build features per (query, candidate) row
    train_ranker.py            # LightGBM training loop
    ranker.py                  # LightGBMRanker.score(features) wrapper
  evaluation/
    __init__.py
    metrics.py                 # ndcg_at_k, mrr, precision_at_k, recall_at_k
    evaluate.py                # run all baselines on test set, print comparison

scripts/
  build_features.py            # dump features_train.parquet + features_test.parquet
  train_ranker.py              # train & save models/ranker.pkl
  evaluate.py                  # eval all four setups on test set

models/
  ranker.pkl                   # trained LightGBM booster
```

CLI flow:

```bash
python scripts/build_features.py     # ~1 min
python scripts/train_ranker.py       # ~2-5 min on CPU
python scripts/evaluate.py           # ~1 min
```

---

## What this enables

Once Phase 4 lands, the system is end-to-end functional:

```
query → BM25 + FAISS → HybridRetriever → FeatureBuilder → LightGBMRanker → top-10
```

Phase 5 will benchmark and show the lift; Phase 6 will wrap it in a Streamlit demo; Phase 7 (optional) adds a MiniBERT cross-encoder to rerank only the top-20 from the ranker for the final cherry-on-top precision.

But the **ranker is the heart of the system** — it's where retrieval becomes ranking, and it's what learning-to-rank is named after.
