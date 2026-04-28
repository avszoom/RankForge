"""Build a FAISS vector index from data/corpus.jsonl.

Outputs:
    models/faiss.index           — binary FAISS IndexFlatIP file (cosine via normalized inner product)
    models/faiss_doc_ids.json    — {"model": "...", "dim": 384, "doc_ids": [...]}
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.text_utils import clean_text  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_CORPUS = REPO / "data" / "corpus.jsonl"
DEFAULT_OUT_INDEX = REPO / "models" / "faiss.index"
DEFAULT_OUT_IDS = REPO / "models" / "faiss_doc_ids.json"
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 64


def build_faiss(
    corpus_path: Path = DEFAULT_CORPUS,
    out_index: Path = DEFAULT_OUT_INDEX,
    out_ids: Path = DEFAULT_OUT_IDS,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = BATCH_SIZE,
) -> int:
    t0 = time.time()
    if not corpus_path.is_file():
        raise FileNotFoundError(f"corpus not found at {corpus_path}")

    doc_ids: list[str] = []
    texts: list[str] = []
    for line in corpus_path.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        doc_ids.append(d["doc_id"])
        # Title + body together; clean_text only (the model has its own subword tokenizer).
        texts.append(clean_text(d["title"] + ". " + d["body"]))

    logger.info("loading embedding model %s ...", model_name)
    model = SentenceTransformer(model_name)
    dim = model.get_sentence_embedding_dimension()

    logger.info("encoding %d docs (batch_size=%d, dim=%d)...", len(texts), batch_size, dim)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,        # unit vectors -> inner product == cosine
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    index = faiss.IndexFlatIP(dim)        # exact (brute-force) inner-product search
    index.add(embeddings)

    out_index.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_index))
    out_ids.write_text(json.dumps({
        "model": model_name,
        "dim": dim,
        "metric": "cosine (inner-product on normalized vectors)",
        "n_docs": len(doc_ids),
        "doc_ids": doc_ids,
    }, indent=2))

    elapsed = time.time() - t0
    logger.info(
        "built FAISS index: %d docs, dim=%d, metric=IP-on-normalized -> %s (%.2fs)",
        len(doc_ids), dim, out_index, elapsed,
    )
    return len(doc_ids)


def main() -> int:
    p = argparse.ArgumentParser(description="Build the FAISS vector index.")
    p.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    p.add_argument("--out-index", default=str(DEFAULT_OUT_INDEX))
    p.add_argument("--out-ids", default=str(DEFAULT_OUT_IDS))
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    build_faiss(
        Path(args.corpus), Path(args.out_index), Path(args.out_ids),
        model_name=args.model, batch_size=args.batch_size,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
