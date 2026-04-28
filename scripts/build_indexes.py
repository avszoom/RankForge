"""Build both BM25 and FAISS indexes from data/corpus.jsonl.

Usage:
    python scripts/build_indexes.py             # build both
    python scripts/build_indexes.py --only bm25
    python scripts/build_indexes.py --only faiss
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def main() -> int:
    p = argparse.ArgumentParser(description="Build BM25 + FAISS indexes.")
    p.add_argument("--only", choices=["bm25", "faiss"], help="Build only one of the two")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.only != "faiss":
        from src.indexing.build_bm25 import build_bm25
        build_bm25()

    if args.only != "bm25":
        from src.indexing.build_faiss import build_faiss
        build_faiss()

    return 0


if __name__ == "__main__":
    sys.exit(main())
