# Feature Builder — Phase 3

> Implementation reference for `src/ranking/feature_builder.py` and `scripts/build_features.py`. Run details + observed numbers from this corpus.
>
> The conceptual design (why, what, alternatives) lives in [`docs/ranker.md`](./ranker.md). This doc focuses on **what's actually built and how to use it**.

---

## Purpose

For every `(query, candidate_doc)` pair, produce a 14-column row containing:
- **2 IDs** for grouping/traceability,
- **11 features** the LightGBM ranker will learn from,
- **1 label** (graded relevance from `data/relevance.jsonl`, defaulting to 0).

The same builder runs at training time (with labels looked up) and at inference time (labels not needed). One code path → no train/serve skew.

---

## Quickstart

```bash
source .venv/bin/activate

# Full build (1344 queries -> ~105k rows, ~15s on Mac MPS)
python scripts/build_features.py

# Debug pass on the first 20 queries (smoke test)
python scripts/build_features.py --limit 20

# Tweak retriever sizes
python scripts/build_features.py --per-retriever 50 --top-k 100
```

Outputs land in `data/`:

```
data/features_train.parquet     1.3 MB    94,822 rows × 14 cols
data/features_test.parquet      0.2 MB    10,116 rows × 14 cols
data/features_meta.json         600 B     schema + run config
```

---

## Output schema

14 columns. Same schema in train and test parquet files.

### ID columns (carried through, not used as features)

| Column | Type | Meaning |
|---|---|---|
| `query_id` | string | foreign key to `queries.jsonl` |
| `doc_id` | string | foreign key to `corpus.jsonl` |

### Feature columns (11) — what the ranker sees

| Column | Type | Source / formula | Captures |
|---|---|---|---|
| `bm25_score` | float | from BM25 index | raw keyword relevance |
| `bm25_rank` | int | 1-based rank in BM25's top-50; **999** if BM25 didn't return this doc | rank position from BM25 |
| `vector_score` | float | cosine sim from FAISS (∈ [-1, 1]) | semantic relevance |
| `vector_rank` | int | 1-based rank in FAISS top-50; **999** if vector didn't return this doc | rank position from vector |
| `retrieved_by_bm25` | int (0/1) | flag | "did BM25 surface this at all?" |
| `retrieved_by_vector` | int (0/1) | flag | "did vector surface this at all?" |
| `title_overlap` | float ∈ [0, 1] | `|tokenize(query) ∩ tokenize(title)| / |tokenize(query)|` | literal title match |
| `body_overlap` | float ∈ [0, 1] | same on body tokens | literal body match |
| `doc_length` | int | token count of `title + body` (after stopword filter) | doc verbosity / authority |
| `query_length` | int | token count of query (after stopword filter) | short vs long queries behave differently |
| `freshness_days` | int | `corpus_max_date − doc.created_at` (in days) | doc age, anchored to corpus's most recent doc for reproducibility |

> **Removed feature: `category_match`.** An earlier version included a flag
> `int(doc.category == query.target_category)`. That **leaked the ground-truth
> category** of the query (a real ranker wouldn't know what the query is
> "supposed to" be about). A no-leak proxy (most-common category among the
> hybrid candidates) was tested but only contributed ~1% feature importance —
> redundant with `vector_rank` and `bm25_rank`. Dropped for the cleanest model.
> Honest NDCG@10 dropped from 0.934 (leaky) → **0.886** (this version).

### Label column (training only)

| Column | Type | Source |
|---|---|---|
| `relevance` | int ∈ {0, 1, 2, 3} | `relevance.jsonl` lookup; **defaults to 0** if the (query_id, doc_id) pair isn't in the labeled set |

### Row order

Stable sort: **`query_id` ASC, then `bm25_score` DESC, then `vector_score` DESC**.

This matters because LightGBM consumes a `groups` array — `[size_of_query_1, size_of_query_2, ...]` — and assumes consecutive rows belong to the same query. Sorting by `query_id` makes `groups` trivial to derive: `df.groupby('query_id', sort=False).size().tolist()`.

---

## Real numbers from this corpus

```
Total queries: 1344  (1215 train / 129 test, pre-split in queries.jsonl)
Total docs:    2520  (corpus + hard negatives)
Total labeled (query, doc) pairs in relevance.jsonl: 100,800
```

After running `build_features.py` with defaults (`per_retriever=50, top_k=100`):

| | Rows | Queries | Avg rows/query | File size |
|---|---|---|---|---|
| Train | **94,822** | 1215 | 78.0 | 1.4 MB |
| Test | **10,116** | 129 | 78.4 | 0.2 MB |
| **Total** | **104,938** | 1344 | 78.1 | |

Wall-clock: 15.2 s on Mac M-series with MPS acceleration.

### Label distribution

```
Train (94,822 rows):
  label 0:  75,999  (80.1%)
  label 1:   3,442  ( 3.6%)
  label 2:   3,532  ( 3.7%)
  label 3:  11,849  (12.5%)

Test (10,116 rows):
  label 0:   8,159  (80.7%)
  label 1:     354  ( 3.5%)
  label 2:     342  ( 3.4%)
  label 3:   1,261  (12.5%)
```

Train and test distributions are nearly identical → no obvious split skew.

### Coverage of explicit labels

```
Train: 19,335 / 94,822 rows had an explicit label in relevance.jsonl  (20.4%)
Test:   2,014 / 10,116 rows had an explicit label                     (19.9%)
```

The other ~80% used the default-0. **What this means:** for most candidates the retriever surfaced, we don't have an explicit label saying they're irrelevant — we're inferring it. This is the standard treatment in real systems (label coverage is always sparse), and the ranker still learns meaningful relevance gradations from the 20% that *are* labeled. If coverage drops much below ~15% we'd need to revisit the policy.

### Sample row (the actual first row of `features_train.parquet`)

```python
{
    'query_id': 'q_00001',                           # "how to improve LLM inference cost optimization"
    'doc_id': 'doc_00008',                           # "Cutting LLM Inference Costs: A Before-and-After Look"
    'bm25_score': 17.30,
    'bm25_rank': 1,
    'vector_score': 0.5696,
    'vector_rank': 6,
    'retrieved_by_bm25': 1,
    'retrieved_by_vector': 1,
    'title_overlap': 0.75,                           # 3 of 4 query tokens appear in title
    'body_overlap': 1.0,                             # all 4 query tokens appear in body
    'doc_length': 160,
    'query_length': 4,
    'freshness_days': 89,
    'relevance': 3                                   # highly relevant
}
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     scripts/build_features.py                       │
│                                                                     │
│   load_queries()           → list[query dict]                       │
│   FeatureBuilder()                                                  │
│     ._load()               → pre-tokenize 2520 docs (cached)        │
│                            → load 100,800 labels (in-memory dict)   │
│     .build_for_queries(queries)                                     │
│        for each query:                                              │
│          HybridRetriever.search(query, top_k=100, per_retriever=50) │
│          for each candidate:                                        │
│             compute 12 features                                     │
│             look up label (default 0)                               │
│             append row                                              │
│                                                                     │
│   split by `split` column → write 2 parquet files + meta.json       │
└─────────────────────────────────────────────────────────────────────┘
```

### Why pre-tokenize the corpus once

Naive: tokenize each doc fresh inside every `compute_features` call.
- 2520 docs × ~1344 queries × ~78 candidates per query = millions of redundant tokenizations.

`FeatureBuilder._load()` tokenizes each doc **once** and caches `(title_tokens, body_tokens, doc_length, category, created_at)` per doc_id. Memory cost: a few MB of Python sets. Speed: turns hours into seconds.

### Why `freshness_days` is anchored to the corpus, not `today()`

```python
self._reference_date = max(doc.created_at for doc in corpus)
freshness_days = (self._reference_date - doc.created_at).days
```

If we used `date.today()` the feature would drift with wall-clock time — the parquet file would be non-deterministic, the same model would score differently next week. Anchoring to the corpus's most recent doc keeps the dataset reproducible. At inference time on truly new queries, you'd want `today()` instead — that's a knob we'll add to `FeatureBuilder.__init__` when we wire up serving.

### Why `MISSING_RANK = 999`

When a doc is in BM25's top-50 but not FAISS's top-50 (or vice versa), the missing retriever's rank field could be `None`. LightGBM doesn't handle nulls gracefully in numeric columns. Options:

| Strategy | Trade-off |
|---|---|
| Use `top_k + 1` (= 51) | Implies "ranked just below the cutoff" — wrong; the doc could be at position 1000 |
| Use `top_k * 100` (= 5000) | Strong "way down the list" signal but creates a discontinuous distribution |
| Use a fixed sentinel like `999` | Clear "not retrieved by this stage" signal, well-separated from valid 1-50 ranks |

We use **999**. The `retrieved_by_*` flags carry the same information in a cleaner form, so the ranker should learn to use those primarily. The `_rank` columns are kept for cases where an explicit numeric position is useful.

---

## Caveats and design choices

### No category-based feature (label-leak avoidance)

An earlier version had `category_match = int(query.target_category == doc.category)`. That used the query's ground-truth target category as a feature, which a real ranker would never have at inference time. We measured the impact:

| Feature variant | Valid NDCG@10 | Notes |
|---|---|---|
| **`category_match` (leaky)** | 0.934 | Inflated — uses ground truth |
| `top_category_match` (no-leak proxy: most-common category among hybrid candidates) | 0.879 | Honest, but contributed only 1.1% feature importance |
| **No category feature (current)** | **0.886** | Honest, simplest, slightly better than the proxy |

The retriever's `vector_rank` (78% feature importance) and `bm25_rank` (10%) already encode "is this doc in the dominant category" implicitly — top-ranked docs almost always belong to the query's true category. An explicit category feature was redundant.

### Default-0 labeling for unlabeled candidates

About 80% of (query, doc) rows aren't in `relevance.jsonl`. We default them to `relevance=0`.

The alternative — train only on rows with explicit labels — is *cleaner* statistically but creates a serious train/serve mismatch: at inference, the ranker sees retriever candidates regardless of whether they're labeled. If the model never trained on those, it doesn't know what to do with them. Default-0 keeps train and serve aligned and lets the model learn "if the retriever surfaced something irrelevant, give it a low score."

### One-call-per-query overhead

The biggest cost is the embedding model encoding each query individually. We could batch queries through MiniLM (encode all 1344 in one call) and then per-query do FAISS lookups. Estimated speedup: 3-5×. Not worth doing yet at 15 seconds of total wall-clock — a Phase 5 optimization if needed.

---

## Validating the output

### Did the parquet file write correctly?

```bash
source .venv/bin/activate
python -c "
import pandas as pd
df = pd.read_parquet('data/features_train.parquet')
print('shape:', df.shape)
print('queries:', df['query_id'].nunique())
print('avg rows/query:', round(len(df) / df['query_id'].nunique(), 1))
print(df.dtypes)
"
```

### Build the LightGBM groups array

```python
import pandas as pd
df = pd.read_parquet('data/features_train.parquet')
groups = df.groupby('query_id', sort=False).size().tolist()
assert sum(groups) == len(df), "groups must sum to row count"
print(f"{len(groups)} queries, {len(df)} rows, "
      f"avg {len(df)/len(groups):.1f}, min {min(groups)}, max {max(groups)}")
```

This `groups` array is what Phase 4 will hand to LightGBM:

```python
import lightgbm as lgb
X = df[FEATURE_COLUMNS]
y = df['relevance']
train_data = lgb.Dataset(X, label=y, group=groups)
```

---

## What this enables

- **Phase 4 (`train_ranker.py`)** can now train LightGBM with LambdaRank.
- **Inference** uses the same `FeatureBuilder` (minus the relevance lookup) — the trained ranker maps a 12-feature vector to a relevance score per candidate.
- **Phase 5 evaluation** reads `features_test.parquet`, runs the trained ranker, computes NDCG@10/MRR/P@K/R@K against the `relevance` column. Baselines (BM25 only, vector only, RRF) score the same test rows for direct comparison.

The features are the contract between retrieval (Phase 1+2), training (Phase 4), and evaluation (Phase 5). Everything downstream depends on this schema staying stable.
