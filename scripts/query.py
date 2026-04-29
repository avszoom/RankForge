"""Run a query against BM25, FAISS, the merged hybrid set, or the trained ranker.

Examples:
    python scripts/query.py "how to reduce LLM inference cost"
    python scripts/query.py "ways to make AI API cheaper" --top-k 5
    python scripts/query.py "consensus algorithms" --retriever bm25
    python scripts/query.py "mobile crash dashboards" --retriever vector
    python scripts/query.py "real-time chat scaling" --retriever hybrid --top-k 15

    # Trained LightGBM ranker — defaults to top-20, no extra args needed
    python scripts/query.py "reduce LLM inference cost" --retriever ranker
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _load_corpus_map(corpus_path: Path) -> dict[str, dict]:
    return {
        json.loads(line)["doc_id"]: json.loads(line)
        for line in corpus_path.read_text().splitlines() if line.strip()
    }


def _print_simple(label: str, results: list[tuple[str, float]], corpus: dict[str, dict]) -> None:
    print(f"\n=== {label}  (top {len(results)}) ===")
    if not results:
        print("  (no results)")
        return
    print(f"  {'rank':>4}  {'score':>8}  {'doc_id':<10}  {'category':<32}  title")
    print(f"  {'-'*4}  {'-'*8}  {'-'*10}  {'-'*32}  {'-'*40}")
    for rank, (doc_id, score) in enumerate(results, 1):
        d = corpus.get(doc_id, {})
        hn = " [HN]" if d.get("is_hard_negative") else ""
        title_short = textwrap.shorten(d.get("title", "") + hn, width=70, placeholder="...")
        print(f"  {rank:>4}  {score:>8.3f}  {doc_id:<10}  {d.get('category','')[:32]:<32}  {title_short}")


def _print_hybrid(candidates, corpus: dict[str, dict]) -> None:
    print(f"\n=== HYBRID (BM25 + FAISS, RRF-fused)  (top {len(candidates)}) ===")
    if not candidates:
        print("  (no results)")
        return
    n_both = sum(1 for c in candidates if c.retrieved_by_bm25 and c.retrieved_by_vector)
    n_bm25_only = sum(1 for c in candidates if c.retrieved_by_bm25 and not c.retrieved_by_vector)
    n_vec_only = sum(1 for c in candidates if c.retrieved_by_vector and not c.retrieved_by_bm25)
    print(f"  source split: both={n_both}, bm25-only={n_bm25_only}, vector-only={n_vec_only}")
    print(f"  {'rank':>4}  {'src':<6}  {'bm25':>7}/{'rk':<4}  {'vec':>5}/{'rk':<4}  {'doc_id':<10}  {'category':<28}  title")
    print(f"  {'-'*4}  {'-'*6}  {'-'*7}/{'-'*4}  {'-'*5}/{'-'*4}  {'-'*10}  {'-'*28}  {'-'*40}")
    for rank, c in enumerate(candidates, 1):
        d = corpus.get(c.doc_id, {})
        hn = " [HN]" if d.get("is_hard_negative") else ""
        title_short = textwrap.shorten(d.get("title", "") + hn, width=60, placeholder="...")
        if c.retrieved_by_bm25 and c.retrieved_by_vector:
            src = "BOTH"
        elif c.retrieved_by_bm25:
            src = "BM25"
        else:
            src = "VEC"
        bm_s = f"{c.bm25_score:>7.2f}" if c.retrieved_by_bm25 else f"{'-':>7}"
        bm_r = f"{c.bm25_rank:<4}" if c.bm25_rank is not None else f"{'-':<4}"
        v_s  = f"{c.vector_score:>5.2f}" if c.retrieved_by_vector else f"{'-':>5}"
        v_r  = f"{c.vector_rank:<4}" if c.vector_rank is not None else f"{'-':<4}"
        print(f"  {rank:>4}  {src:<6}  {bm_s}/{bm_r}  {v_s}/{v_r}  {c.doc_id:<10}  {d.get('category','')[:28]:<28}  {title_short}")


def _print_ranker(ranked, corpus: dict[str, dict]) -> None:
    print(f"\n=== LightGBM RANKER  (top {len(ranked)}) ===")
    if not ranked:
        print("  (no results)")
        return
    print(f"  {'rank':>4}  {'score':>8}  {'src':<6}  {'bm25':>5}/{'rk':<4}  {'vec':>5}/{'rk':<4}  {'doc_id':<10}  {'category':<28}  title")
    print(f"  {'-'*4}  {'-'*8}  {'-'*6}  {'-'*5}/{'-'*4}  {'-'*5}/{'-'*4}  {'-'*10}  {'-'*28}  {'-'*40}")
    for rank, (c, score) in enumerate(ranked, 1):
        d = corpus.get(c.doc_id, {})
        hn = " [HN]" if d.get("is_hard_negative") else ""
        title_short = textwrap.shorten(d.get("title", "") + hn, width=55, placeholder="...")
        if c.retrieved_by_bm25 and c.retrieved_by_vector:
            src = "BOTH"
        elif c.retrieved_by_bm25:
            src = "BM25"
        else:
            src = "VEC"
        bm_s = f"{c.bm25_score:>5.2f}" if c.retrieved_by_bm25 else f"{'-':>5}"
        bm_r = f"{c.bm25_rank:<4}" if c.bm25_rank is not None else f"{'-':<4}"
        v_s  = f"{c.vector_score:>5.2f}" if c.retrieved_by_vector else f"{'-':>5}"
        v_r  = f"{c.vector_rank:<4}" if c.vector_rank is not None else f"{'-':<4}"
        print(f"  {rank:>4}  {score:>8.3f}  {src:<6}  {bm_s}/{bm_r}  {v_s}/{v_r}  {c.doc_id:<10}  {d.get('category','')[:28]:<28}  {title_short}")


def main() -> int:
    p = argparse.ArgumentParser(description="Query the BM25/FAISS indexes (and optionally the trained LightGBM ranker).")
    p.add_argument("query", help="The search query.")
    p.add_argument("--top-k", type=int, default=None,
                   help="Number of results to display. Default: 20 for ranker, else 10.")
    p.add_argument(
        "--retriever",
        choices=["bm25", "vector", "both", "hybrid", "ranker"],
        default="both",
    )
    p.add_argument("--per-retriever", type=int, default=50,
                   help="Per-retriever fetch size for hybrid/ranker (default 50)")
    p.add_argument("--retriever-top-k", type=int, default=100,
                   help="Hybrid candidate-pool size before ranker scores (default 100)")
    p.add_argument("--corpus", default=str(REPO / "data" / "corpus.jsonl"))
    args = p.parse_args()

    if args.top_k is None:
        args.top_k = 20 if args.retriever == "ranker" else 10

    corpus = _load_corpus_map(Path(args.corpus))
    print(f"\nQuery: {args.query!r}")

    if args.retriever == "bm25":
        from src.retrieval.bm25_retriever import BM25Retriever
        _print_simple("BM25 (keyword)", BM25Retriever().search(args.query, top_k=args.top_k), corpus)
    elif args.retriever == "vector":
        from src.retrieval.vector_retriever import VectorRetriever
        _print_simple("FAISS (semantic)", VectorRetriever().search(args.query, top_k=args.top_k), corpus)
    elif args.retriever == "both":
        from src.retrieval.bm25_retriever import BM25Retriever
        from src.retrieval.vector_retriever import VectorRetriever
        _print_simple("BM25 (keyword)", BM25Retriever().search(args.query, top_k=args.top_k), corpus)
        _print_simple("FAISS (semantic)", VectorRetriever().search(args.query, top_k=args.top_k), corpus)
    elif args.retriever == "hybrid":
        from src.retrieval.hybrid_retriever import HybridRetriever
        cands = HybridRetriever().search(
            args.query, top_k=args.top_k, per_retriever=args.per_retriever,
        )
        _print_hybrid(cands, corpus)
    else:  # ranker
        from src.ranking.ranker import LightGBMRanker
        ranked = LightGBMRanker().rank(
            query=args.query,
            top_k=args.top_k,
            per_retriever=args.per_retriever,
            retriever_top_k=args.retriever_top_k,
        )
        _print_ranker(ranked, corpus)

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
