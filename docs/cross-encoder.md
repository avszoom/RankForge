# Phase 7 — Cross-Encoder Reranker (design)

> Forward-looking design doc for the **cross-encoder reranking** stage. Not implemented yet — this is the plan we'll execute next. Code paths referenced (`src/ranking/cross_encoder.py`) don't exist yet.

---

## 0. The 30-second version

We have a working pipeline:

```
query → BM25 + FAISS → HybridRetriever → LightGBM → top-20
```

**Phase 7 adds one more stage at the end:**

```
top-20 from LightGBM → CrossEncoderReranker → top-10 (final result)
```

The cross-encoder is a small neural network that takes `(query, doc)` *together* and outputs a relevance score. It's slower than everything before it but much more accurate at fine-grained ranking. We only run it on the top-20 from LightGBM, so the cost stays bounded.

The rest of this doc explains what a cross-encoder actually is, why it helps, and how it slots into our pipeline.

---

## 1. Bi-encoder vs cross-encoder — a key distinction

We already use a neural model in this system: **`sentence-transformers/all-MiniLM-L6-v2`** for FAISS vector search. That's a **bi-encoder**. The cross-encoder is a different *architecture* of the same family.

### Bi-encoder (what we use for FAISS)

```
query    ──→ encode ──→ vector_q (384-dim)
doc      ──→ encode ──→ vector_d (384-dim)
                              │
                              ▼
                      cosine_sim(vector_q, vector_d)  →  0.82
```

The query and doc are encoded **independently**. Each one becomes a single vector. Similarity is computed between the two vectors.

**Critical property:** doc vectors can be **precomputed once** (we did this in Phase 1 — that's what `models/faiss.index` is). At query time we only encode the query (one model call) and FAISS does fast vector lookups.

### Cross-encoder (what Phase 7 will add)

```
[CLS] query tokens [SEP] doc tokens [SEP] ──→ BERT ──→ pooler ──→ score  →  0.94
```

The query and doc are **concatenated** and fed through the model **together**. The model can attend across query and doc tokens at every layer — it sees them in context.

**No precomputation possible.** The model has to run for **every (query, doc) pair** at query time.

### Why this matters

| | Bi-encoder | Cross-encoder |
|---|---|---|
| Where it works | Indexing 1000s–millions of docs | Reranking a small handful |
| Encodes query + doc | Independently | Together |
| Doc vectors precomputed | ✅ Yes | ❌ No |
| Cost per query | 1 model call (encode the query) | N model calls (one per candidate) |
| Accuracy | Good | Better — captures fine-grained interactions |
| Why? | Limited: can't model query-doc word alignment | Can directly see "does the query word 'cost' appear near the doc word 'inference' in a meaningful context?" |

### Concrete example of the difference

Query: *"how to reduce LLM inference cost"*
Doc title: *"Cutting LLM Inference Costs: A Before-and-After Look"*

- **Bi-encoder** sees these as two ~250-character strings. It pools each into a 384-dim vector independently. The vectors are *similar* (cosine 0.65) — good. But the model never directly aligns "reduce" ↔ "cutting" or "cost" ↔ "costs".
- **Cross-encoder** runs `[CLS] how to reduce LLM inference cost [SEP] Cutting LLM Inference Costs: A Before-and-After Look [SEP]` through BERT. Every layer's attention can directly compute "the query phrase 'reduce cost' aligns with the doc's 'cutting costs'" and the model outputs a score reflecting that alignment.

The cross-encoder almost always scores better because it sees the alignment. The bi-encoder is faster because it doesn't have to.

---

## 2. Why two-stage ranking works

You might think: "if cross-encoder is more accurate, why don't we just use it everywhere?"

**Cost.** A cross-encoder forward pass is ~50–100ms on CPU per (query, doc) pair. Multiply by 2520 corpus docs:

```
2520 docs × 100ms = 252 seconds per query.   ❌ unusable
```

So we use a **cascade** (also called "telescoping" or "two-stage retrieval"):

```
                                 cost per query
─────────────────────────────────────────────────────────────────
Stage 1: BM25 + FAISS retrieval         ~10 ms     (2520 docs)
                ↓
Stage 2: HybridRetriever (RRF merge)    ~negligible (100 docs)
                ↓
Stage 3: LightGBM ranker                ~5  ms     (100 → 20 docs)
                ↓
Stage 4: Cross-encoder rerank           ~1.5 sec   (20 → 10 docs)   ← PHASE 7
                ↓
              final top-10
```

Each stage gets more accurate and more expensive — but each one only sees the output of the previous stage, so the slow stages process tiny inputs.

**The cascade idea:**
- Cheap models eliminate obviously irrelevant docs first
- Expensive models only see the survivors
- Total cost: ~1.5 sec/query (dominated by cross-encoder)
- Quality: close to "running cross-encoder on everything" (which would take 4 minutes)

This pattern is everywhere in industrial search systems — Google, LinkedIn, Bing, Amazon all use cascades like this.

### Why 20 → 10 specifically?

- LightGBM is already strong (NDCG@10 = 0.883 from Phase 5). Top-20 from LightGBM almost certainly contains the right answer somewhere.
- Cross-encoder's job: take that top-20 and **decide which 10 should be at the top**. Small reordering, big perceived quality bump.
- 20 cross-encoder calls is fast enough (~1.5 sec) to use in real time.

If we sent more docs to the cross-encoder (say 100), latency would grow proportionally with diminishing returns. If we sent fewer (say 10), there'd be nothing for the cross-encoder to *promote* — it can only reorder what it sees.

---

## 3. The model we'll use: `cross-encoder/ms-marco-MiniLM-L-6-v2`

A small, off-the-shelf cross-encoder from the `sentence-transformers` library. Fits our toolchain perfectly.

| Property | Value |
|---|---|
| Architecture | 6-layer MiniLM (~22M parameters) |
| Size on disk | ~80 MB |
| Training data | MS MARCO passage ranking dataset (1M+ Bing search queries with click data) |
| Output | A single relevance score per (query, passage) pair |
| Latency on CPU | ~50–100 ms per pair |
| Latency on Mac MPS | ~20–40 ms per pair |
| Library | `sentence_transformers.CrossEncoder` (already a transitive dep) |

Why this specific model:
- **MS MARCO trained.** MS MARCO is *the* learning-to-rank benchmark for short queries against passages — exactly our use case.
- **Same family as our bi-encoder** (MiniLM). They share tokenization conventions, making integration painless.
- **Small.** No GPU needed. Cold-start under a second.
- **Battle-tested.** Used as a baseline in dozens of LtR papers.

### What the model takes and returns

```python
from sentence_transformers import CrossEncoder
model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

pairs = [
    ("how to reduce LLM inference cost", "Cutting LLM Inference Costs: A Before-and-After Look. We deployed..."),
    ("how to reduce LLM inference cost", "Travel itineraries for solo backpackers..."),
]
scores = model.predict(pairs)
# → array([7.42, -8.13])     real numbers; higher = more relevant
```

Scores are **logits**, not probabilities. They have no fixed scale — only relative ordering matters within a query.

---

## 4. How it'll integrate with our system

### New module: `src/ranking/cross_encoder.py`

```python
class CrossEncoderReranker:
    """Wraps cross-encoder/ms-marco-MiniLM-L-6-v2.

    Given (query, doc_ids) pairs from upstream ranking, builds the
    actual (query, title+body) text pairs, scores them, and returns
    a reordered list.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        corpus_path: Path = ...,
    ): ...

    def rerank(
        self,
        query: str,
        doc_ids: list[str],
        top_k: int = 10,
        max_chars: int = 1024,    # truncate doc body to fit in the model's context
    ) -> list[tuple[str, float]]:
        # Build (query, title + " " + body[:max_chars]) pairs
        # Run model.predict() in a single batch
        # Sort by score descending
        # Return top_k as [(doc_id, ce_score), ...]
```

### New wrapper: `TwoStageRanker`

Combines the existing `LightGBMRanker` (stage 1) with `CrossEncoderReranker` (stage 2):

```python
class TwoStageRanker:
    def __init__(
        self,
        lgbm: LightGBMRanker | None = None,
        ce: CrossEncoderReranker | None = None,
        stage1_top_k: int = 20,
    ): ...

    def rank(self, query: str, top_k: int = 10) -> list[tuple[Candidate, float, float]]:
        # 1. LightGBM → top-20 candidates with their lgbm_scores
        # 2. Cross-encoder reranks those 20 → top-K
        # 3. Returns [(Candidate, lgbm_score, ce_score), ...]
        # The Candidate carries the per-stage info (bm25/vec scores+ranks)
        # so the CLI can show what changed at every stage.
```

### Updated `scripts/query.py`

A new mode:

```bash
python scripts/query.py "how to reduce LLM inference cost" \
    --retriever ranker_ce \
    --top-k 10
```

Output shows the cross-encoder score alongside the LightGBM score, so you can see how the reranker moved things.

### Updated `scripts/compare.py`

Extend to a 3-way comparison:

```
Hybrid (RRF) → LightGBM → +Cross-Encoder
     H              R              CE
```

For each doc in the final top-10, show its position in each prior stage and how it moved:

```
CE   R    H    Δ_R   Δ_H   doc_id      title
 1   3   16    +2    +15   doc_00003   Cutting Costs Without Cutting...   ← CE promoted from R-rank 3 to position 1
 2   1    2    -1    +0    doc_00909   Navigating the API Gateway...
 ...
```

### Updated `src/evaluation/evaluate.py`

Add a 5th setup `ranker_ce` to the comparison. Phase 5's harness already runs 4 setups; we just plug in one more and rerun.

---

## 5. Files Phase 7 will add

```
src/ranking/
  cross_encoder.py        # CrossEncoderReranker class
  two_stage.py            # TwoStageRanker (lgbm + ce)

scripts/
  query.py                # +1 mode: --retriever ranker_ce
  compare.py              # 3-way comparison (or new compare_ce.py)
  evaluate.py             # picks up the new setup automatically

src/evaluation/
  evaluate.py             # adds ranker_ce setup
```

No new top-level deps — `sentence-transformers` is already pinned, and the `CrossEncoder` class lives in the same package.

---

## 6. What we expect to gain

Honest expectations based on published cross-encoder literature on similar small datasets:

| | NDCG@10 | Notes |
|---|---|---|
| Current (LightGBM ranker) | **0.883** | Phase 5 measured |
| LightGBM + CrossEncoder | **0.90–0.93** (estimated) | typical lift on top of strong tabular ranker |

Lift varies a lot by query type:

| query_type | Current ranker | Estimated +CE |
|---|---|---|
| `exact` | 0.954 | ~0.96 (already ceiling-bound) |
| `paraphrase` | 0.825 | **~0.88–0.90** ← biggest gain expected |
| `noisy` | 0.865 | ~0.88 |

The cross-encoder's value is in **fine-grained semantic discrimination** — judging "does this body actually answer the query?" — which is hardest for paraphrased queries where keyword overlap is low and tabular features don't capture the alignment.

A realistic ceiling for our synthetic data is ~0.93 NDCG@10. Beyond that, the limiting factor is the labels themselves (the relevance.jsonl sampling) rather than the model.

### What about cost?

| Metric | Stage 1 (LightGBM) | Stage 2 (Cross-Encoder) |
|---|---|---|
| Latency | ~5 ms | ~1.5 sec on CPU (~600 ms on Mac MPS) |
| Memory | tiny | ~80 MB model + ~500 MB peak during inference |

The total query latency goes from "instant" to "noticeable but acceptable." For a Streamlit demo (Phase 6), this is fine. For real production, you'd batch reranking requests or use a GPU.

---

## 7. Caveats & things to watch

### Truncation

Our docs are 150–300 words but the model's max input is 512 tokens (~700 chars including the query). We'll truncate the body to ~1024 chars before feeding in. Most relevance signal lives in the first paragraph anyway.

### Score scale isn't comparable across queries

Cross-encoder scores can range roughly `[-12, +12]` on the MS MARCO logit scale. They're not probabilities. Within a query they sort docs correctly; across queries they're not directly comparable. We just use them as a sort key per query.

### MS MARCO bias

The model was trained on web-style queries against web passages. Our synthetic blog-style docs are structurally similar, so this should generalize cleanly. If the corpus drifted toward, say, code or medical text, we'd want a domain-specific cross-encoder.

### Doesn't fix retrieval gaps

The cross-encoder can only reorder what LightGBM gave it. If the truly relevant doc is at LightGBM rank 25 (outside the top-20), the cross-encoder never sees it and can't promote it. Increasing `stage1_top_k` from 20 to 30 or 50 trades latency for recall.

---

## 8. After Phase 7 lands

Pipeline becomes:

```
query
  ↓
BM25 + FAISS retrieval     (Phase 1)
  ↓
HybridRetriever / RRF      (Phase 2)
  ↓
FeatureBuilder             (Phase 3)
  ↓
LightGBM Ranker            (Phase 4)        ← top-20
  ↓
CrossEncoderReranker       (Phase 7)        ← top-10
  ↓
final results to user
```

The full search pipeline is industrial-grade: it has retrieval, fusion, learned tabular ranking, and neural reranking.

After Phase 7 closes:
- **Phase 6 (Streamlit UI)** — wraps everything in a browser demo: search box, ranked results, the side-by-side compare view, an evaluation page showing the per-stage NDCG metrics.
- **Deployment** — package as a Docker image, deploy somewhere (Render / Fly.io / locally), share a URL.

That's the project shipped end-to-end.
