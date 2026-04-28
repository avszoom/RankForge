"""Run a query against BM25, FAISS, or both, printing top-K results side-by-side.

Examples:
    python scripts/query.py "how to reduce LLM inference cost"
    python scripts/query.py "ways to make AI API cheaper" --top-k 5
    python scripts/query.py "consensus algorithms" --retriever bm25
    python scripts/query.py "mobile crash dashboards" --retriever vector
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


def _print_results(label: str, results: list[tuple[str, float]], corpus: dict[str, dict]) -> None:
    print(f"\n=== {label}  (top {len(results)}) ===")
    if not results:
        print("  (no results)")
        return
    print(f"  {'rank':>4}  {'score':>8}  {'doc_id':<10}  {'category':<32}  title")
    print(f"  {'-'*4}  {'-'*8}  {'-'*10}  {'-'*32}  {'-'*40}")
    for rank, (doc_id, score) in enumerate(results, 1):
        d = corpus.get(doc_id, {})
        title = d.get("title", "")
        category = d.get("category", "")
        hn = " [HN]" if d.get("is_hard_negative") else ""
        title_short = textwrap.shorten(title + hn, width=70, placeholder="...")
        print(f"  {rank:>4}  {score:>8.3f}  {doc_id:<10}  {category[:32]:<32}  {title_short}")


def main() -> int:
    p = argparse.ArgumentParser(description="Query the BM25/FAISS indexes.")
    p.add_argument("query", help="The search query.")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--retriever", choices=["bm25", "vector", "both"], default="both")
    p.add_argument("--corpus", default=str(REPO / "data" / "corpus.jsonl"))
    args = p.parse_args()

    corpus = _load_corpus_map(Path(args.corpus))
    print(f"\nQuery: {args.query!r}")

    if args.retriever in ("bm25", "both"):
        from src.retrieval.bm25_retriever import BM25Retriever
        bm25 = BM25Retriever().search(args.query, top_k=args.top_k)
        _print_results("BM25 (keyword)", bm25, corpus)

    if args.retriever in ("vector", "both"):
        from src.retrieval.vector_retriever import VectorRetriever
        vec = VectorRetriever().search(args.query, top_k=args.top_k)
        _print_results("FAISS (semantic)", vec, corpus)

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
