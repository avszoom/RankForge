# Cross-Encoder Architecture — A Beginner Walkthrough

> A concrete walkthrough of how a cross-encoder (BERT-family model) actually computes a relevance score, traced through a tiny toy example layer by layer.
>
> Pairs with [`docs/cross-encoder.md`](./cross-encoder.md) (high-level integration design) and [`docs/lightgbm-beginner.md`](./lightgbm-beginner.md) (LightGBM beginner walkthrough).

---

## 1. Setup — a really small example

Real query and doc are 50–250 tokens. For this walkthrough, let's use **6 tokens total** so we can fit them on the page:

```
Input:  [CLS]  cheap   LLM   [SEP]  reduce  cost  [SEP]
        token1 token2 token3 token4  token5 token6 token7
```

`[CLS]` is a "summary slot" (more on it later). `[SEP]` is a "separator" between query and doc. Both are special tokens the model was trained to recognize.

Pretend the model uses **4-dim vectors** instead of the real 384 (just for readability).

After the **embedding lookup** (step 0 — before any layer runs), each token has its own 4-dim vector. This is just looking up the word in a giant dictionary:

```
[CLS]   = [0.5,  0.0,  0.1,  0.2]    ← generic "summary slot" vector
cheap   = [0.8,  0.6, -0.1,  0.0]    ← "cheap" embedding
LLM     = [0.2,  0.7,  0.5, -0.3]
[SEP]   = [0.0,  0.1,  0.0,  0.4]
reduce  = [0.7,  0.5, -0.2,  0.1]
cost    = [0.6,  0.4, -0.0,  0.0]
[SEP]   = [0.0,  0.1,  0.0,  0.4]
```

These embeddings are just lookups — same word always gets the same starting vector. They don't yet know about context.

> Numbers are illustrative. Real embeddings are 384-dim with values roughly in `[-1, 1]`.

---

## 2. All tokens go into Layer 1 in parallel

You hand Layer 1 the **whole stack of 7 vectors at once**. Layer 1 doesn't process them one-at-a-time. It processes them all simultaneously and outputs 7 new 4-dim vectors (same shape, refined values).

```
       all 7 input vectors
              │  ↓  ↓  ↓
              ↓
       ┌──────────────────┐
       │     Layer 1      │   ← processes all 7 in parallel
       └──────────────────┘
              ↓  ↓  ↓  │
              ↓
       7 refined output vectors
```

This parallelism is *the* defining property of transformers. (It's why they train fast on GPUs — there's no sequential dependency between tokens within a layer.)

---

## 3. What does Layer 1 actually do?

Layer 1 has two sub-steps, applied in order. Walking through what happens for the **`cheap`** token:

### Sub-step A: Attention — "look around and update yourself"

Each token gets to look at all the other tokens and decide who to "mix in."

```
The 'cheap' token thinks:
  "I'm 'cheap'. Let me see how much each other token is relevant to me."

  attention to [CLS]    = small  (CLS is generic, low signal)
  attention to 'cheap'  = medium (myself, decent signal)
  attention to 'LLM'    = small  (different topic but in my query)
  attention to [SEP]    = tiny   (boundary marker, low signal)
  attention to 'reduce' = MEDIUM (semantically aligned with 'cheap'!)
  attention to 'cost'   = LARGE  (synonymous with cheap!)
  attention to [SEP]    = tiny

(These weights add up to 1 across the row.)
```

**The model has learned during training that "cheap" and "cost" / "reduce" are semantically related, so it pays more attention to them.** That learned knowledge is encoded in the attention weights.

Now the `cheap` token blends in those other tokens' values, weighted by attention:

```
new_cheap_vector
  = 0.05 × [CLS_vector]
  + 0.20 × [cheap_vector]
  + 0.10 × [LLM_vector]
  + 0.02 × [SEP_vector]
  + 0.25 × [reduce_vector]      ← biggest contributors
  + 0.35 × [cost_vector]        ← from across the SEP boundary!
  + 0.03 × [SEP_vector]
─────────────────────────────────
  = [0.65, 0.45, -0.05, 0.05]
```

The `cheap` token's vector has been updated. It still mostly represents "cheap," but now it has *picked up information about cost and reduce*. It is no longer just the word "cheap" — it's "cheap, in the context where there's a doc talking about reducing cost."

**This same operation runs in parallel for all 7 tokens.** The `cost` token gets to update itself by attending to others (especially `cheap`). The `[CLS]` token gets to soak up information from everywhere.

### Sub-step B: Feed-forward — "think harder per token"

After each token has gathered information from the others, each one independently runs through a tiny 2-layer neural network. This is where each token does some extra "processing" of its mixed-in context.

```
each token's vector
  → pass through a small linear layer → expand to bigger size
  → apply GELU (a smooth wiggly function, similar to ReLU)
  → pass through another linear layer → compress back
  → got a slightly different vector
```

The input shape is the same (4-dim) and the output shape is the same (4-dim) — so the layer can be stacked on top of more layers.

After Layer 1 finishes (both sub-steps), the output is **7 new vectors of the same shape**. They now encode "each word AND a bit of who it relates to."

---

## 4. What does Layer 2 do?

The exact same two sub-steps — attention, then FFN — but now operating on the **already-refined Layer-1 output**.

So in Layer 2:
- Each token attends to the layer-1 outputs of all other tokens.
- But those layer-1 outputs already had context baked in.
- So Layer 2 is essentially attending to "tokens that already know who they're connected to."

Effect: **deeper, more abstract relationships emerge.**

| | After this layer |
|---|---|
| **Layer 1** | "the word 'cheap' is related to the word 'cost'" (lexical/synonym level) |
| **Layer 2** | "this query is asking about cost reduction in general" (phrase level) |
| **Layer 3** | "this is a cost-optimization context, not a price-comparison context" (intent level) |
| **Layer 4** | "the query and doc are aligned on the same theme" (cross-section signal) |
| **Layer 5** | "the doc actually answers the query (not just topically related)" (relevance) |
| **Layer 6** | final refinements — last chance to adjust |

(These role descriptions are loose intuition. The real model doesn't have neat per-layer roles, but research has shown this rough progression: lower layers handle syntax and surface, higher layers handle semantics and task-specific signals.)

---

## 5. After 6 Layers — pulling out the score

After Layer 6 finishes, we have 7 refined 4-dim vectors. We don't care about most of them. We only need **the `[CLS]` token's final vector**.

The `[CLS]` started as a generic "summary slot" with no real meaning. But across 6 layers of attention, every other token's information has been mixed into it (just like `cheap` mixed in info about `cost`). Now `[CLS]` is a holistic summary of "how well does this query and doc fit together."

```
final [CLS] vector after Layer 6:
  [3.1, -0.4, 1.7, 0.5]      ← made up; real one is 384-dim
```

Now we hit the **classification head**, which is the simplest possible neural network — one linear layer:

```
score = w₁ × 3.1  +  w₂ × (-0.4)  +  w₃ × 1.7  +  w₄ × 0.5  +  bias
      = 0.5 × 3.1 + (-2.0) × (-0.4) + 1.5 × 1.7 + 0.2 × 0.5 + (-1.0)
      = 1.55 + 0.80 + 2.55 + 0.10 - 1.00
      = 4.0
```

That's it. **One number out: `4.0`**. Higher means more relevant.

`w₁, w₂, w₃, w₄` and `bias` are 5 trained parameters (real model: 384 + 1 = 385 trained parameters in the head). They learned during MS MARCO training to convert "rich [CLS] summary" into "relevance score."

No activation function on top. No softmax. Just `dot product + bias`.

---

## 6. The whole forward pass, condensed

```
Step 0 — Embedding lookup
  7 tokens → 7 fresh 4-dim vectors. Just dictionary lookups.

Step 1 — Layer 1
  Each token attends to all 7 tokens (in parallel).
  Each token's vector is updated: weighted blend + small FFN.
  → 7 refined vectors

Step 2 — Layer 2
  Same operation, but now on Layer 1's outputs.
  → 7 more-refined vectors

Step 3 — Layer 3
  Same operation, on Layer 2's outputs.
  → 7 even-more-refined vectors

Steps 4, 5, 6 — Layers 4, 5, 6
  Same. Each layer further refines.
  → 7 fully-baked vectors

Step 7 — Pull [CLS] vector
  Take only the 4-dim (real: 384-dim) final vector for [CLS].

Step 8 — Score head
  Linear projection: 4 dims (real: 384) → 1 number.

Output: a single relevance score.
```

---

## 7. Visualization — how `cheap`'s vector evolves through the layers

Pretend we can read each token's "vibe" at each layer:

| Layer | `cheap` token represents... |
|---|---|
| Step 0 (embedding) | just the dictionary meaning of "cheap" |
| After Layer 1 | "cheap" + a little bit of "cost" + a little bit of "reduce" |
| After Layer 2 | "cheap, in the context of saving money on something" |
| After Layer 3 | "cheap, where the doc talks about cost-cutting strategies" |
| After Layer 4 | "cheap, signaling this query and doc are aligned" |
| After Layer 5 | "cheap → relevance signal: high" |
| After Layer 6 | "cheap → finalized contribution to the [CLS] summary" |

Meanwhile the `[CLS]` token has been vacuuming up information from all of these refined tokens at every layer — so by the end, it's a rich, single-vector summary of "this whole query+doc pair, from a relevance perspective."

---

## 8. The real-model numbers

Our actual model is `cross-encoder/ms-marco-MiniLM-L-6-v2`:

| Real value | Toy value (this doc) |
|---|---|
| 6 transformer blocks | 6 layers ✓ |
| Hidden size: 384 | 4 |
| Attention heads: 12 (each 32-dim) | 1 (we ignored multi-head) |
| FFN inner dim: 1536 | "expanded size" (we waved at it) |
| Vocab size: 30,522 | n/a |
| Max sequence length: 512 tokens | 7 in our example |
| Total parameters: ~22.5M | ~50 |

The math operations are identical at every scale — just bigger matrices, more parallel heads, longer sequences.

---

## 9. The two questions that confuse most beginners

### "Are tokens fed sequentially or in parallel?"

**In parallel.** All N tokens enter Layer 1 simultaneously. The matrix operations inside the layer (attention + FFN) compute on the entire batch at once. There is no left-to-right scanning at the token level. (Old recurrent networks like LSTMs *did* scan sequentially. Transformers ditched that — and got way faster training as a result.)

### "How does the `[CLS]` token end up containing 'the meaning' of the whole input?"

Through attention, layer by layer. At every layer, `[CLS]` attends to every other token and mixes their values into itself. After 6 layers, it has accumulated a 384-dim summary that the final score head can interpret as a relevance score.

There's no special wiring that says "summarize here." It's a **convention** — during training, the model was rewarded for putting the relevance signal into `[CLS]`'s final vector, so it learned to. If we trained another head off a different token's final vector, we'd get a different model. But by convention, `[CLS]` is the score-bearing slot for classification/scoring tasks.

---

## 10. How this connects to RankForge

This same forward pass runs **once per (query, doc) pair** at inference. In Phase 7:

1. LightGBM ranker returns top-20 candidates.
2. For each of those 20 docs, we build a `(query, doc_text)` pair.
3. Each pair runs through the cross-encoder forward pass described above (~6 layers, ~22.5M parameters, ~50–100 ms on CPU).
4. We get 20 scores.
5. Sort by score → final top-10.

Total cost: ~1.5 seconds per query for the cross-encoder stage. Slow but acceptable for the demo.

That's the entire model. Everything else — Phase 1's BM25 index, Phase 4's LightGBM ranker — feeds candidates into this final scoring stage.
