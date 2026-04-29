# LightGBM and LambdaRank — From Scratch (Beginner Friendly)

> A from-zero walkthrough of how LightGBM, NDCG, and LambdaRank work — with everyday analogies, tiny worked examples, and no jargon.
>
> If you only have time for one section, read **§4 (LambdaRank)**. That's the section everyone struggles with.

---

## 1. Decision trees — the "20 questions" game

Imagine I'm thinking of an animal and you have to guess by asking yes/no questions:

```
                "Does it have fur?"
                ┌─────yes────┴────no─────┐
                ▼                        ▼
       "Bigger than a cat?"      "Does it lay eggs?"
       ┌──yes─┴──no──┐          ┌─yes─┴──no─┐
       ▼             ▼          ▼           ▼
      Dog          Hamster    Snake      Fish
```

After a few yes/no questions, you land on an answer.

**A decision tree is exactly this game, but instead of guessing animals, it outputs a number.** Instead of asking about furry-ness, it asks about feature columns from our data.

For RankForge, the "animal" the tree is guessing is **"how relevant is this doc to this query?"** (a number, ideally close to the label 0/1/2/3). The features it can ask about are the 12 columns from Phase 3 (`bm25_score`, `vector_score`, `title_overlap`, etc.).

```
                  bm25_score >= 10?              ← did keyword search like it?
                ┌─────yes────┴────no─────┐
                ▼                        ▼
       title_overlap >= 0.5?     vector_score >= 0.4?
       ┌──yes─┴──no──┐           ┌──yes─┴──no──┐
       ▼             ▼           ▼             ▼
      2.8           1.4         1.1           0.2
                                              ↑
                          (the leaf "values" — the tree's guesses for relevance)
```

To use the tree on one row of features:
1. Take a row, e.g. `bm25_score=14, title_overlap=0.6, vector_score=0.3`.
2. Walk down. `bm25=14 → yes → title=0.6 → yes → leaf 2.8`.
3. The tree's prediction: **2.8**.

That's it. A tree is a flowchart of yes/no questions where the leaves output numbers.

### Why trees are good for our problem

- **No need to scale features.** `bm25_score` (0–30) and `vector_score` (-1 to 1) and `query_length` (1–10) all coexist happily — each split looks at one feature at a time.
- **Captures interactions automatically.** "If `bm25` is high AND `title_overlap >= 0.5` AND `vector` is also high" → a specific leaf.
- **Readable.** You can literally open the model, look at the trees, and see what the model relies on.

### The catch

A single tree only has a handful of leaves (we'll use 31). It can carve the world into 31 buckets max. Real signal is more nuanced. So we use **lots of trees, working together**.

---

## 2. Boosting — a team of dumb experts

Imagine grading a stack of essays. You hire 200 teachers — but each one is **bad at the job alone**:

- **Teacher 1** reads each essay and gives a rough guess. Wrong by 0.5 points on average.
- **Teacher 2** looks ONLY at where Teacher 1 was wrong, and tries to fix just those mistakes. Wrong by another 0.2.
- **Teacher 3** fixes Teacher 2's leftover mistakes. Wrong by 0.05.
- ... and so on.

The **final grade for an essay = Teacher 1's score + Teacher 2's adjustment + Teacher 3's adjustment + ...**

No single teacher is great. After 200 of them, the result is better than any one teacher could be alone.

### That's gradient boosting

In LightGBM:
- Each "teacher" is a small decision tree (max 31 leaves).
- Each new tree only has to learn **the residual mistakes** of all previous trees combined.
- The final prediction for any row = sum of all 200 tiny tree outputs.

```
final_prediction(query, doc) = tree_1(features) + tree_2(features) + ... + tree_200(features)
```

The word **"gradient"** just means "in which direction was the previous teacher wrong?" Tree 2 doesn't try to predict the label — it tries to predict "how much should I bump tree 1's prediction up or down to fix it?"

### Worked example

Say the true label for one row is `relevance = 3`.

```
Round 1:  tree_1 predicts 0.0       (everyone starts at 0)
          residual = 3 − 0 = 3.0    ← we need to push UP

Round 2:  tree_2 trained to predict 3.0    (it predicts 0.4)
          running prediction = 0.0 + 0.4 = 0.4
          residual = 3 − 0.4 = 2.6    ← still way off

Round 3:  tree_3 trained to predict 2.6    (it predicts 0.3)
          running prediction = 0.4 + 0.3 = 0.7
          ...

Round 50: running prediction = 2.85
          residual = 0.15    ← getting close

Round 200: running prediction = 2.97
          ← validation NDCG plateaus, training stops
```

Each tree only nudges. It's the SUM that does the work.

### `learning_rate=0.05` — the "go slow" knob

In practice we scale each tree's output by 0.05 before adding it. So a tree wanting to predict 0.4 only contributes 0.02 to the final answer. This is a "go slow, be careful" knob — overshooting destabilizes things.

### Why "Light" GBM specifically

There were earlier libraries (XGBoost, scikit-learn). LightGBM (Microsoft, 2017) added two tricks that make it 5–20× faster without hurting accuracy. For us, that means training takes ~1 minute on CPU instead of 5.

---

## 3. NDCG — grading the order of a list

Up to now we've talked about predicting one number for one row. But what we actually care about is: **does the LIST of docs come out in the right order for each query?**

NDCG (Normalized Discounted Cumulative Gain) is the score for "how good is this ordered list?"

It's built from two ideas. Let me explain them one at a time.

### 3.1 Idea A — different docs are worth different amounts

Our labels go 0, 1, 2, 3 (3 is highly relevant, 0 is irrelevant). NDCG uses an **exponential gain function**:

```
gain(label) = 2^label − 1
```

| Label | Gain |
|---|---|
| 3 (highly relevant) | 2³ − 1 = **7** |
| 2 (related) | 2² − 1 = **3** |
| 1 (weakly related) | 2¹ − 1 = **1** |
| 0 (irrelevant) | 2⁰ − 1 = **0** |

**One relevance-3 doc is worth 7 relevance-1 docs.** Not 3 of them — 7. That exponential weighting reflects what users actually want: **precision at the very top matters disproportionately**.

### 3.2 Idea B — earlier positions matter more

Position 1 of a search result page is way more valuable than position 10. NDCG reflects this with a **logarithmic discount**:

```
discount(position) = 1 / log₂(position + 1)
```

| Position | log₂(pos+1) | Discount |
|---|---|---|
| 1 | log₂(2) = 1.000 | 1.000 |
| 2 | log₂(3) = 1.585 | 0.631 |
| 3 | log₂(4) = 2.000 | 0.500 |
| 5 | log₂(6) = 2.585 | 0.387 |
| 10 | log₂(11) = 3.459 | 0.289 |
| 50 | log₂(51) = 5.672 | 0.176 |

A label-3 doc at position 1 contributes `7 × 1.000 = 7.000`. The same doc at position 10 only contributes `7 × 0.289 = 2.023`. **Same doc, different value depending on where it landed.**

### 3.3 Putting them together — DCG

**Discounted Cumulative Gain** for a list of K docs:

```
DCG@K = sum over positions 1 to K of:    gain(label_at_position) × discount(position)
```

### 3.4 Worked example

You ask "how to reduce LLM inference cost". The ranker returns these top 5:

```
position 1:   doc_A   true label = 3      ← perfect! highly relevant doc at top
position 2:   doc_B   true label = 0      ← oops, irrelevant
position 3:   doc_C   true label = 3      ← another good one (but late)
position 4:   doc_D   true label = 1      ← weakly relevant
position 5:   doc_E   true label = 0      ← irrelevant
```

DCG calculation:

```
position 1:  gain(3) × disc(1) = 7 × 1.000 = 7.000
position 2:  gain(0) × disc(2) = 0 × 0.631 = 0.000
position 3:  gain(3) × disc(3) = 7 × 0.500 = 3.500
position 4:  gain(1) × disc(4) = 1 × 0.431 = 0.431
position 5:  gain(0) × disc(5) = 0 × 0.387 = 0.000

DCG@5 = 10.931
```

What's the **best possible** DCG for these 5 docs? Sort them by true label: `[3, 3, 1, 0, 0]`.

```
position 1:  gain(3) × disc(1) = 7 × 1.000 = 7.000
position 2:  gain(3) × disc(2) = 7 × 0.631 = 4.416
position 3:  gain(1) × disc(3) = 1 × 0.500 = 0.500
position 4:  gain(0) × disc(4) = 0 × 0.431 = 0.000
position 5:  gain(0) × disc(5) = 0 × 0.387 = 0.000

IDCG@5 = 11.916
```

```
NDCG@5 = DCG@5 / IDCG@5 = 10.931 / 11.916 = 0.917
```

**0.917 out of 1.0** — pretty good. We got the relevance-3 doc at position 1 (huge), but the second relevance-3 landed at position 3 instead of 2 (mild penalty).

### 3.5 Why "Normalized"?

The "N" in NDCG. Different queries have different best-possible DCGs. Dividing by IDCG normalizes everything to [0, 1] so easy and hard queries get **equal weight** when we average across queries.

```
overall NDCG@10 = average of per-query NDCG@10 across all queries
```

---

## 4. LambdaRank — teaching the model to optimize NDCG

Now the cleverest part. Here's the problem:

> We want the model to predict scores so that sorting docs by score within each query maximizes NDCG.

**The catch:** NDCG depends on **positions** (integers like 1, 2, 3), not on **scores** directly. Tweaking a score by a tiny amount might not change any positions (NDCG stays the same), or it might flip two docs (NDCG jumps suddenly). You can't take a derivative of a stair-step function. Standard gradient boosting needs a smooth gradient to work.

So we need a clever workaround. Let's first see why the obvious fallback doesn't work.

### 4.1 The naive try — pairwise loss

Simplest idea: **for every pair of docs in the wrong order, push them apart.** This is "naive pairwise loss." Sounds reasonable.

But there's a fatal flaw. Let's see it.

### 4.2 Why naive pairwise is broken — the mistake-location problem

Consider a query with 4 docs whose true labels (sorted ideal) are:

```
ideal ranking:  [3, 2, 1, 0]    ← this is what NDCG considers perfect
```

Now compare two different "bad" rankings the model might produce:

```
Bad ranker X:  [2, 3, 1, 0]    ← swapped positions 1 ↔ 2 (top mistake)
Bad ranker Y:  [3, 2, 0, 1]    ← swapped positions 3 ↔ 4 (bottom mistake)
```

**Both rankings have exactly one pair in the wrong order.**

From naive pairwise loss's perspective, they're equally bad. Same penalty, same gradient, model treats them identically.

But let's compute their actual NDCG:

```
gains:      label 3 → 7,    label 2 → 3,    label 1 → 1,    label 0 → 0
discounts:  pos 1 → 1.000,  pos 2 → 0.631,  pos 3 → 0.500,  pos 4 → 0.431

IDEAL: IDCG = 7×1.000 + 3×0.631 + 1×0.500 + 0×0.431 = 9.393

Bad X (top mistake):
  DCG = 3×1.000 + 7×0.631 + 1×0.500 + 0×0.431 = 7.916
  NDCG = 7.916 / 9.393 = 0.843

Bad Y (bottom mistake):
  DCG = 7×1.000 + 3×0.631 + 0×0.500 + 1×0.431 = 9.324
  NDCG = 9.324 / 9.393 = 0.993
```

### 4.3 The punchline

| | Wrong pairs | NDCG | Penalty (1−NDCG) |
|---|---|---|---|
| Bad X (top mistake) | 1 | **0.843** | 0.157 |
| Bad Y (bottom mistake) | 1 | **0.993** | 0.007 |

Same number of wrong pairs — but **the top mistake is ~22× worse than the bottom mistake** as far as users (and NDCG) care.

### 4.4 Why this matters during training

Imagine the model is mid-training. At any moment it has many wrong pairs and has to decide which to fix first.

**With naive pairwise:** all wrong pairs are weighted equally. If the model has 10 deep bottom-mistakes and 1 top-mistake, it spends ~90% of its energy on the bottom — moves NDCG by maybe 0.07 total, while ignoring the top mistake that alone could move NDCG by 0.15.

**With LambdaRank:** each wrong pair's gradient is multiplied by `|ΔNDCG when swapped|`. The top mistake gets a gradient 22× bigger than each bottom mistake. The model **automatically** fixes the top first.

### 4.5 Visual

```
Two wrong rankings, both with "1 wrong pair":

Bad X:    [2, 3, 1, 0]   ← top mistake
            ⏶⏶                                    NDCG drops to 0.843  💥
            big problem

Bad Y:    [3, 2, 0, 1]   ← bottom mistake
                  ⏶⏶                              NDCG drops to 0.993  ~⌐
                  tiny problem


How the two losses see them:

NAIVE PAIRWISE:           "1 wrong pair = 1 wrong pair. Both equally bad."
                          Gives both the same gradient. ❌ misallocates effort.

LAMBDARANK:               "Top mistake matters 22× more."
                          Gives X's pair a gradient 22× bigger. ✅ correct.
```

### 4.6 Why this is the only thing that matters

In real datasets, **most wrong pairs live in the depths of the list** — the long tail where everything has label 0. By combinatorics, there are simply more of them. Naive pairwise gets dragged toward fixing those because there are more pairs. LambdaRank ignores the deep noise and forces the model to focus on the few top pairs that actually move NDCG.

That's the entire reason LambdaRank dominates production search ranking. It tells the model where to look — at the top — by making mistakes there feel proportionally bigger.

### 4.7 The training loop with LambdaRank gradients

Round by round:

1. Current ensemble produces scores for all (query, doc) rows.
2. **For each query group**, look at every pair where label_A > label_B:
   - Compute "how much would NDCG change if we swapped A and B right now?" — a real number, the **lambda**.
   - Multiply by the standard pairwise direction (push A up, B down).
   - Add to A's accumulated lambda. Subtract from B's.
3. Each row now has a single accumulated lambda — its weighted "you should change this much" signal.
4. Fit a new tree to those lambdas (treating them as the "target" the tree should predict).
5. Add the tree to the ensemble (scaled by `learning_rate`).
6. Loop.

After ~200 rounds, the ensemble has learned to push high-gain docs to the top of each query.

You never wrote down a closed-form NDCG loss. You don't need to. By making each pair's gradient proportional to its NDCG impact, you're effectively optimizing NDCG **without ever differentiating it**. This is the core insight of LambdaRank.

---

## 5. The "group" concept in one sentence

> The model is told "rows 1–64 belong to query 1, rows 65–151 belong to query 2, ..." so that pairs are only formed **within** a query. Doc A from query 1 is never compared against Doc B from query 2 — they're not on the same list, so comparing scores across them is meaningless.

That's it. The `group` array is just `[size_of_q1, size_of_q2, ...]`, and LightGBM uses it to know where one query's rows end and the next begins.

For us:

```
features matrix (94,822 rows × 12 cols)        groups array (length 1215)
┌──────────────────────────────────────┐      ┌─────┐
│ row 0   (q_00001, doc_00008, ...)    │      │ 64  │ ← query 1 has 64 rows
│ row 1   (q_00001, doc_00007, ...)    │      │ 87  │ ← query 2 has 87 rows
│   ...                                 │      │ 62  │
│ row 63  (q_00001, doc_00321, ...)    │      │ 64  │
│ row 64  (q_00002, doc_00100, ...)    │      │ 78  │
│   ...                                 │      │ ... │
└──────────────────────────────────────┘      └─────┘
                                              sum = 94,822
```

**Cardinal sin: never shuffle rows between Phase 3 and training.** If you do, the group array goes out of sync with the data and training silently produces garbage.

---

## 6. Cheat sheet

| Concept | One-line gist |
|---|---|
| **Decision tree** | A flowchart of yes/no questions over features; leaves output numbers |
| **Gradient boosting** | A team of weak trees, each fixing the previous ones' mistakes; final answer = sum |
| **LightGBM** | A fast, modern gradient boosting library (5–20× faster than older alternatives) |
| **`learning_rate`** | How much each new tree contributes — small (e.g. 0.05) is "go slow, be careful" |
| **Gain (in NDCG)** | A relevance-3 doc is worth 7. Relevance-2 is worth 3. **Exponential** weighting. |
| **Discount (in NDCG)** | Position 1 worth 1.000, position 10 worth ~0.29, position 50 worth ~0.18. Logarithmic decay. |
| **DCG** | Sum of `gain × discount` over a list |
| **NDCG** | DCG / IDCG. "How close is your list to the perfect possible list?" Always in [0, 1]. |
| **Pairwise loss** | "For every wrong-order pair, push them apart" — but treats all pairs equally (broken) |
| **LambdaRank** | Pairwise gradient × `|ΔNDCG when swapped|` — model spends effort where positions matter |
| **Group** | Tells LightGBM "rows 1–64 are query 1, rows 65–151 are query 2..." — pairs stay within a query |

---

## 7. Pointers for going deeper later

When you're ready to dig past beginner-level:

- **The LambdaRank paper** — Burges et al., *"From RankNet to LambdaRank to LambdaMART: An Overview"* (2010). Short, accessible read.
- **The LightGBM paper** — Ke et al., NeurIPS 2017. Explains the histogram trick + leaf-wise growth.
- **`docs/lightgbm-ranker.md`** in this repo — same material as this doc but slightly more technical, covers hyperparameters and training-loop pseudocode in more detail.
