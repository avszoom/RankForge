"""Build a BM25 index from data/corpus.jsonl and pickle it to models/bm25.pkl.

Output pickle structure:
    {
        "bm25":     BM25Okapi instance (corpus stats + per-doc term frequencies),
        "doc_ids":  list[str], parallel to bm25's internal row order,
        "k1":       1.5,            # BM25 TF-saturation parameter (library default)
        "b":        0.75,           # BM25 length-normalization parameter (library default)
        "avgdl":    float,          # average document length in tokens
        "n_docs":   int,
    }
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path

from rank_bm25 import BM25Okapi

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.text_utils import tokenize  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_CORPUS = REPO / "data" / "corpus.jsonl"
DEFAULT_OUT = REPO / "models" / "bm25.pkl"


def build_bm25(corpus_path: Path = DEFAULT_CORPUS, out_path: Path = DEFAULT_OUT) -> int:
    t0 = time.time()
    if not corpus_path.is_file():
        raise FileNotFoundError(f"corpus not found at {corpus_path}")

    doc_ids: list[str] = []
    corpus_tokens: list[list[str]] = []
    for line in corpus_path.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        doc_ids.append(d["doc_id"])
        corpus_tokens.append(tokenize(d["title"] + " " + d["body"]))

    bm25 = BM25Okapi(corpus_tokens)  # k1=1.5, b=0.75 by default

    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "bm25": bm25,
        "doc_ids": doc_ids,
        "k1": bm25.k1,
        "b": bm25.b,
        "avgdl": bm25.avgdl,
        "n_docs": len(doc_ids),
    }
    with out_path.open("wb") as f:
        pickle.dump(payload, f)

    elapsed = time.time() - t0
    logger.info(
        "built BM25 index: %d docs, avgdl=%.1f tokens, k1=%.2f, b=%.2f -> %s (%.2fs)",
        len(doc_ids), bm25.avgdl, bm25.k1, bm25.b, out_path, elapsed,
    )
    return len(doc_ids)


def main() -> int:
    p = argparse.ArgumentParser(description="Build the BM25 index.")
    p.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    build_bm25(Path(args.corpus), Path(args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
