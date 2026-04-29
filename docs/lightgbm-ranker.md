# Phase 4 — Learning to Rank with LightGBM

> Foundations doc: explains LightGBM, LambdaRank, the `group` concept, NDCG-driven training, and why every hyperparameter is what it is. Read this before we code.

---

## 0. The 30-second version

We have **104,938 feature rows** from Phase 3 — one per `(query, candidate_doc)` pair, each with 12 features and a relevance label (0–3).

**Phase 4's job:** train a model that takes those 12 features and outputs a single number (predicted relevance score), so that within each query, sorting candidates by that score puts the most relevant ones on top.

The model is a **LightGBM gradient-boosted decision tree** trained with the **LambdaRank** objective. Once trained, it lives in `models/ranker.pkl` and replaces the RRF placeholder we wrote in Phase 2.

The rest of this doc unpacks what every word in that paragraph means.

---

## 1. Decision trees: the building block

A decision tree is a flowchart that takes a row of features and outputs a number.

```
                    bm25_score >= 8.5?
                  ┌────────┴────────┐
                no│                 │yes
        title_overlap >= 0.3?    vector_score >= 0.6?
        ┌──────┴──────┐          ┌──────┴──────┐
       0.2          1.4         2.7          3.1
```

For one row, you start at the root, answer the questions, and the leaf you land on is your predicted score. That's it — no neural network, no embeddings, just a sequence of branching `if` statements over numeric features.

**Why decision trees for ranking:**
- They handle messy, mixed-scale features natively. `bm25_score` ranges 0–30, `vector_score` ranges -1 to 1, `query_length` is an int — no normalization needed.
- They learn non-linear interactions automatically. "If `bm25_score` is high AND `title_overlap >= 0.5` AND `freshness_days < 90`" becomes a single leaf.
- They're interpretable. You can literally read the trees to see what the model relies on.

**Why one tree isn't enough:**
A single tree can only carve feature space into ~31 regions (with our `num_leaves=31` setting). Real signal is more nuanced. So we use lots of trees.

---

## 2. Gradient boosting: stacking trees additively

Gradient boosting trains trees **sequentially**, where each new tree fixes the mistakes of the ones before it.

```
prediction(row) = tree_1(row) + tree_2(row) + tree_3(row) + ... + tree_N(row)
```

The training loop:

```
1. Start with all predictions = 0
2. For round t = 1, 2, ..., N:
   a. Compute "gradient" = how wrong each prediction currently is, and in which direction
   b. Fit tree_t to predict that gradient
   c. Add tree_t to the ensemble (multiplied by learning_rate, e.g. 0.05)
3. Stop when validation metric stops improving
```

Concretely, after round 50 we have 50 trees. Each prediction is a sum of 50 small contributions. Each new tree only has to push predictions in the right direction — they don't have to be individually correct.

The "gradient" depends on what you're optimizing. For regression it's `prediction - target`. For ranking it's a different beast — that's where **LambdaRank** comes in.

### Why "Light" GBM specifically

There were earlier GBDT libraries (XGBoost, scikit-learn's GradientBoostingClassifier). LightGBM (from Microsoft, 2017) added two tricks that make it 5–20× faster:

| Trick | What it does |
|---|---|
| **Histogram-based splits** | Instead of trying every numeric value as a split candidate, bucket each feature into ~255 bins and only split on bin boundaries. Identical accuracy, dramatically less work. |
| **Leaf-wise growth** | Grow the leaf with the largest gradient first, instead of growing all branches one level at a time. Reaches better trees in fewer iterations. |

For our 100k-row dataset, training takes under a minute on CPU. XGBoost would take 3–5×. Both produce essentially identical models — LightGBM just gets there faster.

---

## 3. From regression to ranking — three problem framings

Before we get to LambdaRank, let's clarify what "training a ranker" even means. There are three common framings:

### 3.1 Pointwise (regression)

Treat each `(query, doc)` row independently. Predict the label as a real number.

```
loss = (prediction - true_label)^2
```

Pros: simple, any regression library works.
Cons: ignores that ranks matter, not absolute scores. A model that predicts 1.9 when the truth is 2 vs a model that predicts 2.5 are equally "wrong" by this loss — even if the second one's predictions sort docs in the right order and the first one's don't.

### 3.2 Pairwise

For each query, look at every pair of docs `(i, j)` where `label_i > label_j`. The model should output a higher score for `i` than `j`.

```
loss for pair (i, j) = log(1 + exp(-(score_i - score_j)))
```

Pros: directly optimizes for "right one ranked higher than wrong one."
Cons: treats all pairs equally — the pair "(rank-1, rank-2)" matters as much as "(rank-50, rank-51)." But users only see the top — pairs near the top should matter much more.

### 3.3 Listwise (LambdaRank's territory)

Look at the whole ranked list at once. Loss is a function of the entire order, not individual pairs.

LambdaRank is technically pairwise but with a twist that makes it behave listwise: **each pair's gradient is multiplied by how much swapping that pair would change NDCG**.

```
gradient for pair (i, j) = (pairwise gradient) × |ΔNDCG when we swap i and j|
```

Pairs near the top of the list have huge ΔNDCG when swapped. Pairs near the bottom have nearly zero. The model "spends" most of its capacity getting the top right, exactly where it matters.

This is what makes LambdaRank the standard for graded labels in industry search.

---

## 4. NDCG — the metric that drives everything

NDCG (Normalized Discounted Cumulative Gain) is the goal LambdaRank aims at. It captures: *"Are the highly relevant docs near the top of the list?"*

### 4.1 The formula

For a query with K shown results:

```
DCG@K  = Σ from i=1 to K of:   (2^rel_i − 1) / log₂(i + 1)
IDCG@K = DCG@K of the perfect ranking (sorted by true label)
NDCG@K = DCG@K / IDCG@K       ∈ [0, 1]
```

Two pieces:

- **`(2^rel − 1)` is the gain.** A label-3 doc contributes 7. A label-2 contributes 3. A label-1 contributes 1. A label-0 contributes 0. The exponential weighting means *one highly relevant doc is worth seven mildly relevant ones*. This is the empirically-validated heuristic that "users care way more about precision at the very top."

- **`log₂(i + 1)` is the discount.** Position 1 → log₂(2) = 1.0. Position 2 → log₂(3) ≈ 1.58. Position 10 → log₂(11) ≈ 3.46. Each later position contributes less. By position 50, each gain is divided by ~5.6 — most of NDCG is decided by the top 5-10.

### 4.2 Worked example with one of our queries

Query `q_00001` ("how to improve LLM inference cost optimization"). Suppose the ranker produces this top-5:

```
position    doc                              true relevance
   1        doc_00007 (LLM cost — relevant)         3
   2        doc_00012 (Networking unrelated)        0
   3        doc_00002 (LLM cost — relevant)         3
   4        doc_00131 (related ML topic)            1
   5        doc_00821 (Travel — irrelevant)         0
```

```
DCG@5 = (2^3 − 1)/log₂(2) + (2^0 − 1)/log₂(3) + (2^3 − 1)/log₂(4) + (2^1 − 1)/log₂(5) + (2^0 − 1)/log₂(6)
      =        7/1.000     +        0/1.585     +        7/2.000     +        1/2.322     +        0/2.585
      =        7.000       +        0           +        3.500       +        0.431       +        0
      =       10.931
```

The ideal ranking (sort by true label descending) would be `[3, 3, 1, 0, 0]`:

```
IDCG@5 = 7/1 + 7/1.585 + 1/2 + 0/2.322 + 0/2.585
       = 7 + 4.416 + 0.500 + 0 + 0
       = 11.916
```

```
NDCG@5 = 10.931 / 11.916 = 0.917
```

So the ranker scored 0.917 (very good, but not perfect). The penalty came from putting an irrelevant doc at position 2 instead of the second relevant doc at position 3.

### 4.3 NDCG is calculated per query, then averaged

```
overall NDCG@10 = (1 / num_queries) × Σ NDCG@10 for each query
```

This is critical: easy and hard queries get the same weight. Without per-query normalization, a single high-DCG query could dominate the average.

---

## 5. LambdaRank — connecting NDCG to gradients

Here's the magic. NDCG isn't differentiable — you can't take its derivative because it depends on ranks (integers), not scores (real numbers). So how can gradient boosting optimize it?

**LambdaRank's trick:** Make up a gradient that's directionally correct and weight it by ΔNDCG.

For each pair `(i, j)` in a query where `label_i > label_j`:

```
λ_ij = (∂σ/∂s) × |ΔNDCG_ij|
```

where:
- `(∂σ/∂s)` is the standard pairwise sigmoid derivative (says "push i's score up, j's score down").
- `|ΔNDCG_ij|` is "how much would NDCG change if I swapped docs i and j right now?"

The λ for each doc is the sum of pair contributions involving it:

```
λ_doc_i = Σ over j where label_i > label_j: + λ_ij
        + Σ over j where label_i < label_j: - λ_ij
```

That λ is what gets passed to the next decision tree as the gradient to fit.

### 5.1 Why this works in practice

Pairs near the top of the list have large ΔNDCG when swapped — moving a doc from rank 1 to rank 5 wrecks NDCG. So the trees learn most aggressively on the top of the list.

Pairs deep in the list have tiny ΔNDCG — moving rank 47 to rank 52 barely changes the metric. The trees mostly ignore those pairs.

Net effect: **the model's capacity goes where users actually look**, even though we never wrote down a closed-form NDCG loss.

### 5.2 LambdaMART = LambdaRank inside GBDT

LightGBM's `objective='lambdarank'` is technically **LambdaMART** — LambdaRank applied inside a Multiple Additive Regression Tree (i.e., GBDT) ensemble. The names are used interchangeably in practice.

It was the winner of the Yahoo Learning to Rank Challenge in 2010, and it's been the industry standard for tabular learning-to-rank ever since.

---

## 6. The `group` concept

This is the subtle bit that separates ranking from regression.

### 6.1 The problem

When LightGBM trains, it shuffles rows and computes losses across the whole dataset. For ranking, that's wrong — you can't compare a doc from query A against a doc from query B. They're not in the same list. Their scores aren't on the same scale (in the model's view).

### 6.2 The solution

LightGBM accepts a `group` array — an integer list whose values sum to the row count. The first `group[0]` rows are query 1's candidates. The next `group[1]` rows are query 2's. And so on.

```
features matrix (94,822 rows × 12 cols)        groups array (length 1215)
┌──────────────────────────────────────┐      ┌─────┐
│ row 0   (q_00001, doc_00008, ...)    │      │ 64  │ ← query 1 has 64 rows
│ row 1   (q_00001, doc_00007, ...)    │      │ 87  │ ← query 2 has 87 rows
│   ...                                 │      │ 62  │
│ row 63  (q_00001, doc_00321, ...)    │      │ 64  │
│ row 64  (q_00002, doc_00100, ...)    │      │ 78  │
│ row 65  (q_00002, doc_00101, ...)    │      │ ... │
│   ...                                 │      │     │
└──────────────────────────────────────┘      └─────┘
                                              sum = 94,822
```

Three guarantees the group array provides:
- **Pairs are only formed within a group.** Doc A from query 1 is never compared to Doc B from query 2.
- **NDCG is computed per group.** Then averaged across groups for the global metric.
- **Splits in trees still consider all rows.** A tree split like "bm25_score >= 8" applies globally — that's fine and what we want. Only the loss is per-query.

### 6.3 How we build groups (already done in Phase 3)

In `features_train.parquet`, rows are sorted by `query_id`. So:

```python
import pandas as pd
df = pd.read_parquet('data/features_train.parquet')
groups_train = df.groupby('query_id', sort=False).size().tolist()
# → [64, 87, 62, 64, 78, ..., 95]   length = 1215
assert sum(groups_train) == len(df)  # 94,822
```

That's all `lgb.Dataset(X, label=y, group=groups_train)` needs.

### 6.4 The cardinal sin: shuffling rows

If you accidentally shuffle the dataframe before computing groups, the group array becomes meaningless and training silently produces garbage. **Never shuffle** between feature-building and dataset construction. Keep `query_id` ASC ordering throughout.

---

## 7. The training loop in plain English

```
INITIAL STATE
- 1215 train queries, 94,822 rows, 12 features per row
- groups_train = [64, 87, 62, ...]
- 129 valid queries, 10,116 rows
- groups_valid = [...]
- predictions = zeros (94,822 of them)

FOR ROUND 1, 2, 3, ... 500:
  STEP A — score every row:
    s_i = sum of all trees built so far (round 1 = 0; round 2 = tree_1 only; ...)

  STEP B — compute LambdaRank gradients (per query group):
    for each query group:
      for each pair (i, j) in this group with label_i > label_j:
        compute ΔNDCG if i and j were swapped
        compute pairwise sigmoid gradient
        λ_ij = sigmoid_gradient × |ΔNDCG|
      accumulate into per-doc λ values

  STEP C — fit a new decision tree:
    tree_t splits on whichever feature most reduces summed-lambda residuals
    tree depth grows leaf-by-leaf until num_leaves=31 reached
    each leaf outputs the average lambda of rows landing there, scaled by learning_rate=0.05

  STEP D — append to ensemble:
    predictions[i] += learning_rate × tree_t(row_i)

  STEP E — every iteration, score the validation set:
    compute NDCG@10 on valid groups using current ensemble
    compare against best-so-far

  STEP F — early stopping check:
    if validation NDCG hasn't improved for 50 rounds:
      stop training, return the model from the best round
```

That's it. Maybe 200 rounds total before early stopping kicks in. ~30–60 seconds wall-clock.

---

## 8. Hyperparameters we'll use

```python
params = {
    'objective':       'lambdarank',
    'metric':          'ndcg',
    'ndcg_eval_at':    [5, 10, 20],
    'lambdarank_truncation_level': 20,
    'learning_rate':   0.05,
    'num_leaves':      31,
    'min_data_in_leaf': 20,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq':    5,
    'verbose':         -1,
}
```

| Param | Value | What it does | Knob to turn if... |
|---|---|---|---|
| `objective` | `lambdarank` | Use NDCG-weighted pairwise gradients | (don't change) |
| `metric` | `ndcg` | What gets reported each round + used for early stopping | (don't change) |
| `ndcg_eval_at` | `[5, 10, 20]` | Print NDCG@5, @10, @20 each round | (informational) |
| `lambdarank_truncation_level` | 20 | Only top-20 pairs contribute to gradients | Increase if you want the ranker to care about deeper positions |
| `learning_rate` | 0.05 | How much each new tree contributes | Lower (0.01) for finer convergence + more rounds |
| `num_leaves` | 31 | Tree complexity | Reduce to 15–20 if validation NDCG plateaus far below train NDCG (overfitting) |
| `min_data_in_leaf` | 20 | Don't make leaves with fewer than 20 rows | Increase if you're overfitting |
| `feature_fraction` | 0.9 | Each tree sees 90% of features | Lower for more regularization |
| `bagging_fraction` + `_freq` | 0.8, 5 | Each tree sees 80% of rows, refresh every 5 rounds | Stochastic regularization |

### Three knobs that actually matter

If training looks bad, change these in this order:
1. `num_leaves` — too high overfits, too low underfits.
2. `min_data_in_leaf` — too low overfits to query-specific patterns.
3. `learning_rate` — lower needs more rounds, but typically generalizes better.

The other knobs are best left at defaults.

---

## 9. What the trained model looks like

After training, `models/ranker.pkl` contains:

```python
booster: lgb.Booster
  ├── 200ish decision trees       (depending on early stopping)
  ├── for each tree:
  │     ├── feature splits
  │     ├── thresholds
  │     └── leaf values
  └── metadata (best_iteration, feature_names, parameters)
```

Total size: ~50–200 KB. Pickled, fast to load (<100 ms).

### Inference

```python
# Same FeatureBuilder from Phase 3, but we skip the relevance lookup
features = build_features_for_query(query, hybrid_candidates)   # (n_candidates, 12)
scores = booster.predict(features)                              # (n_candidates,)
ranked = sorted(zip(candidates, scores), key=lambda x: -x[1])[:10]
```

The output `scores` are real numbers with no fixed scale or meaning — only relative ordering matters within a query. Don't try to compare them across queries.

### Feature importance — what the model learned

LightGBM exposes per-feature importance:

```python
for feat, gain in zip(feature_names, booster.feature_importance(importance_type='gain')):
    print(f"  {feat:25s}  {gain:>10.0f}")
```

Output looks like:

```
bm25_score                   12,450
vector_score                 10,890
title_overlap                 6,210
title_overlap                 4,118
freshness_days                3,002
body_overlap                  2,701
retrieved_by_vector           1,886
vector_rank                   1,540
bm25_rank                     1,205
retrieved_by_bm25               904
doc_length                      632
query_length                    280
```

(Numbers illustrative.) This tells us exactly what the model relied on. In our actual trained model, `vector_rank` carries 77.5% of the gain and `bm25_rank` 10% — the rank-based features dominate. The `retrieved_by_*` flags ended up at 0% importance (redundant with the rank features), and we deliberately removed an earlier `category_match` feature that was a label leak. See [feature-builder.md](feature-builder.md#caveats-and-design-choices).

---

## 10. Sanity checks during/after training

### During — watch the train/valid gap

```
[50]   train ndcg@10 = 0.84   valid ndcg@10 = 0.76   ← gap = 0.08, normal
[100]  train ndcg@10 = 0.91   valid ndcg@10 = 0.79   ← gap = 0.12, still ok
[200]  train ndcg@10 = 0.96   valid ndcg@10 = 0.80   ← gap = 0.16, watch
[300]  train ndcg@10 = 0.99   valid ndcg@10 = 0.79   ← gap = 0.20, OVERFITTING
```

If the gap blows up, lower `num_leaves` or raise `min_data_in_leaf`. Healthy training has a small persistent gap, not an exploding one.

### After — break down NDCG by query type

```
            queries   NDCG@10
exact          53      0.86
paraphrase     47      0.74
noisy          29      0.71
overall       129      0.78
```

`exact` should be highest (BM25 alone solves most of those). `paraphrase` is where the learned ranker should win the most over BM25-only. `noisy` is the genuinely hard category. If `paraphrase` doesn't beat the BM25 baseline, the model isn't using vector_score effectively and we'd investigate.

### After — compare against baselines (this is Phase 5)

| Setup | Expected NDCG@10 |
|---|---|
| BM25 only | ~0.55 |
| Vector only | ~0.65 |
| Hybrid + RRF (no learning) | ~0.72 |
| **Hybrid + LightGBM ranker** | **~0.78–0.82** |

If the trained ranker doesn't beat RRF by at least 0.04, something is wrong with the features, the labels, or the training. Both of the previous baselines are also computed from the same parquet rows, so they're a strict apples-to-apples comparison.

---

## 11. Files Phase 4 will add

```
src/ranking/
  ranker.py                     # LightGBMRanker class for inference
  train.py                      # the training loop module
                                # (renamed from "train_ranker" to avoid clash with the script name)

scripts/
  train_ranker.py               # CLI: parse args, call src.ranking.train, save model

models/                         # gitignored except ranker_meta.json
  ranker.pkl
  ranker.txt
  ranker_meta.json              # committed; tiny; pins the schema and best params
```

---

## 12. After Phase 4 lands

The end-to-end search pipeline is finally complete:

```
   user query
       │
       ▼
   BM25 + FAISS  →  HybridRetriever  →  FeatureBuilder  →  LightGBMRanker  →  top-10
       (Phase 1)         (Phase 2)         (Phase 3)         (Phase 4)
```

Phase 5 measures how good it is (NDCG/MRR/P@K/R@K, baselines, per-query-type breakdown). Phase 6 wraps it in a Streamlit demo. Phase 7 (optional) bolts on a MiniBERT cross-encoder for the top-20 only.

But the **ranker is the brain**. It's where retrieval becomes ranking, and it's what learning-to-rank is named after.
