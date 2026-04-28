# Indexing — BM25 and FAISS

> Forward-looking design doc. This describes how **Phase 1** will build the indexes so we have a shared mental model before writing code. Code paths referenced (`src/indexing/...`) don't exist yet.

## Why two indexes?

A single retriever can't do everything well:

| Retriever | Wins on | Loses on |
|---|---|---|
| **BM25** (keyword) | "LLM inference cost optimization" → finds docs with those exact words | "ways to make AI API cheaper" → no shared keywords with the relevant doc |
| **Vector / FAISS** (semantic) | "ways to make AI API cheaper" → matches a doc about *prompt caching* via meaning | Acronyms, exact product names, rare terms — embeddings smear them |

So we build both, retrieve from each independently, and merge the candidate sets in Phase 2.

---

## 1. BM25 — Keyword Index

### What it is

**BM25** ("Best Matching 25") scores a document against a query by combining three ideas:

1. **Term frequency (TF)** — does the document contain the query terms? More occurrences → higher score, but with diminishing returns (saturating curve, not linear).
2. **Inverse document frequency (IDF)** — common words like "the" and "system" are downweighted; rare words like "FAISS" or "quantization" carry most of the signal.
3. **Document length normalization** — short docs aren't penalized for not repeating terms; long docs don't get unfair credit for having more words.

The scoring formula (simplified):

```
BM25(q, d) = Σ over terms t in q:
                IDF(t) · (TF(t, d) · (k₁ + 1))
                          ─────────────────────────────────────
                          TF(t, d) + k₁ · (1 − b + b · |d| / avg_doc_len)
```

`k₁` (≈1.5) controls TF saturation. `b` (≈0.75) controls length normalization. We use the library defaults.

### Why we use it

- 30+ years old, battle-tested, free, no GPU.
- Strong baseline — even modern systems combine BM25 with neural rerankers because BM25 is *complementary* (catches different signals).
- Sparse: scores depend only on overlapping vocabulary, so it's cheap and interpretable. You can debug "why was this doc ranked #1?" by reading the matched terms.

### How we'll build it

```
data/corpus.jsonl                          (2520 documents)
        │
        ▼
   for each doc:
     tokens = tokenize(title + " " + body)
        │
        ▼
   list[list[str]]   ← all docs, all tokens
        │
        ▼
   BM25Okapi(corpus_tokens)   ← from the rank-bm25 library
        │
        ▼
   pickle to models/bm25.pkl with parallel doc_ids list
```

The shared `tokenize()` function (in `src/text_utils.py`):
- Lowercases everything
- Strips punctuation via `re.findall(r"[a-z0-9]+", text)`
- Drops a small stopword list (~40 words: "the", "a", "of", …)
- Drops tokens shorter than 2 characters

We tokenize **title + body together** (titles are signal-rich; concatenating gives them implicit weight since their terms appear in both contexts).

### Index artifacts

| Path | Content | Size |
|---|---|---|
| `models/bm25.pkl` | `{"bm25": BM25Okapi, "doc_ids": [...]}` | ~3–5 MB for 2520 docs |

The pickled `BM25Okapi` object stores per-document term frequencies and the corpus statistics needed for scoring. The parallel `doc_ids` list maps row index → `doc_id` so a query result like `[2, 17, 421]` can be resolved back to actual document IDs.

### Query-time

```python
retriever = BM25Retriever()
results = retriever.search("how to reduce LLM inference cost", top_k=10)
# → [("doc_00007", 12.4), ("doc_00012", 11.8), ...]
```

Internally:
1. Tokenize the query with the **same** `tokenize()` used at index time. (Mismatched tokenization is the #1 source of silent BM25 bugs.)
2. `bm25.get_scores(query_tokens)` → numpy array of length 2520 (one score per doc).
3. `argsort` descending, take top-K, map indices back to `doc_id`s.

---

## 2. FAISS — Vector / Semantic Index

### What it is

A **vector index** stores documents as fixed-length numeric arrays ("embeddings") and answers nearest-neighbor queries:

> Given a query embedding, return the K corpus embeddings closest to it.

"Close" is measured by cosine similarity (or, equivalently after L2 normalization, inner product). Documents with similar *meaning* land close in this 384-dimensional space, even if they share no literal words.

**FAISS** (Facebook AI Similarity Search) is the standard library for this — it provides multiple index structures (flat, HNSW, IVF, PQ) optimized for different scale/speed tradeoffs.

### Embeddings: where the meaning comes from

We use [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) — a small (~80 MB), fast, CPU-friendly model that produces a 384-dim vector per text. It was trained on >1 billion sentence pairs to make semantically similar texts produce similar vectors.

```python
emb = model.encode("how to reduce LLM inference cost", normalize_embeddings=True)
# emb.shape == (384,), unit length
```

### Why we use FAISS

- Sub-millisecond k-NN search even with exact (brute-force) similarity at our scale (2520 docs).
- Production-grade: scales to billions of vectors with approximate indexes when you need to.
- Dead simple API: `index.add(embeddings)`, `index.search(query, k)`.

### How we'll build it

```
data/corpus.jsonl                          (2520 documents)
        │
        ▼
   for each doc:
     text = clean_text(title + ". " + body)
        │
        ▼
   model.encode(texts, normalize_embeddings=True, batch_size=64)
        │
        ▼
   numpy array of shape (2520, 384)        ← all embeddings
        │
        ▼
   index = faiss.IndexFlatIP(384)
   index.add(embeddings)
        │
        ▼
   write to models/faiss.index
   + parallel models/faiss_doc_ids.json    ← row → doc_id map
```

Notes:
- We feed `clean_text(...)` (lowercase + whitespace normalize) but **not** our BM25 `tokenize()` — the model has its own subword tokenizer (WordPiece) and benefits from natural-language input, not bag-of-tokens.
- `normalize_embeddings=True` returns unit vectors. With unit vectors, **inner product equals cosine similarity** — so we use `IndexFlatIP` (inner product) and get cosine for free.
- `IndexFlatIP` is brute-force: it computes similarity to every doc. At 2520 docs that's instant. We'd switch to `IndexHNSW` only above ~100K docs.

### Index artifacts

| Path | Content | Size |
|---|---|---|
| `models/faiss.index` | Binary FAISS index file (vectors + metadata) | ~4 MB (2520 × 384 × 4 bytes) |
| `models/faiss_doc_ids.json` | `["doc_00001", "doc_00002", ...]` parallel list | ~50 KB |

### Query-time

```python
retriever = VectorRetriever()
results = retriever.search("ways to make AI API cheaper", top_k=10)
# → [("doc_00007", 0.87), ("doc_00321", 0.83), ...]   scores ∈ [-1, 1] (cosine)
```

Internally:
1. Encode the query with the **same** model used at index time (cached in process after first call).
2. `index.search(query_vec, top_k)` → returns parallel arrays `(distances, indices)`.
3. Map each row index to `doc_id` via the sidecar JSON.

---

## 3. Building the indexes — flow

The pipeline runs once, after the corpus exists:

```
┌─────────────────────────┐
│   data/corpus.jsonl     │   2520 documents (Phase 0 output)
└──────────┬──────────────┘
           │
   ┌───────┴────────┐
   │                │
   ▼                ▼
build_bm25.py    build_faiss.py
   │                │
   ▼                ▼
models/bm25.pkl  models/faiss.index + models/faiss_doc_ids.json
```

CLI (planned):

```bash
python -m src.indexing.build_bm25     # ~1 sec
python -m src.indexing.build_faiss    # ~30 sec on CPU, first run downloads MiniLM ~80 MB
```

Both scripts are idempotent and re-run only when the corpus changes (we'll add a Makefile target later to wire this up).

---

## 4. The contrast — what each index will retrieve

For the query **"how to reduce LLM inference cost"** (one of our `query_type: exact` queries):

- **BM25 top-5** → docs whose title/body literally contain `llm`, `inference`, `cost`. Almost entirely from the `LLM Systems & Inference` category. Strong signal because the keywords are rare and topical.
- **Vector top-5** → mostly the same `LLM inference cost optimization` topic, plus possibly some semantic neighbors like `prompt caching` or `dynamic batching` even when the literal keywords differ.

For **"ways to make AI API cheaper"** (one of our `query_type: paraphrase` queries):

- **BM25 top-5** → weak. "ways", "make", "cheaper" don't match much; many docs score near zero. Results may include irrelevant docs that happen to contain "make" or "ways".
- **Vector top-5** → strong. The model knows "AI API cheaper" ≈ "LLM inference cost optimization". Returns the right docs despite zero shared keywords.

This contrast is the entire point of having both — and Phase 2's hybrid merger will combine their candidate sets.

---

## 5. Tradeoffs we're locking in

| Decision | Alternative | Why we chose ours |
|---|---|---|
| Title + body concatenated into one embedding | Separate title + body embeddings, dual indexes | Simpler. Title prepended once gets implicit upweight via repetition. Dual-vector belongs in a later phase if needed. |
| `IndexFlatIP` (exact, brute-force) | `IndexHNSW` (approximate, faster) | At 2520 docs, flat is sub-millisecond. HNSW only matters above ~100 K docs. Less tuning surface area. |
| Pickle BM25 to disk | Rebuild on startup | ~1 sec rebuild is fine, but pickling matches the README's `models/` artifact convention and keeps retrievers stateless on import. |
| Same `tokenize()` for indexing and querying | Different (e.g. lemmatize at index time) | Tokenization mismatches between index- and query-time are the most common BM25 bug. One function, one source of truth. |
| `all-MiniLM-L6-v2` (384-dim) | Larger models like `mpnet-base-v2` (768-dim) | 5x faster, half the memory, ~2 percentage points lower on benchmarks but plenty good for a 2520-doc POC. We can swap models later behind the same retriever interface. |

---

## 6. What this enables

Once both indexes exist, every later phase plugs into the same two retrievers:

- **Phase 2** — hybrid retrieval merges BM25 top-50 + vector top-50, dedupes, keeps both scores per doc.
- **Phase 3** — feature builder uses `bm25_score`, `bm25_rank`, `vector_score`, `vector_rank` as four of the ranker's input features.
- **Phase 4** — LightGBM ranker learns to combine the retrievers' scores with surface features (title overlap, length, freshness).
- **Phase 5** — evaluation compares "BM25 only" vs "vector only" vs "hybrid + ranker" using NDCG@10, MRR, P@K.

The two indexes are foundational. Everything else is built on top.
