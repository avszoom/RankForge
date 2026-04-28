"""BM25Retriever — keyword-based retrieval over the pickled BM25 index."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from src.text_utils import tokenize

DEFAULT_INDEX = Path(__file__).resolve().parents[2] / "models" / "bm25.pkl"


class BM25Retriever:
    def __init__(self, index_path: Path | str = DEFAULT_INDEX) -> None:
        self.index_path = Path(index_path)
        self._loaded = False
        self._bm25 = None
        self._doc_ids: list[str] = []

    def _load(self) -> None:
        if self._loaded:
            return
        with self.index_path.open("rb") as f:
            payload = pickle.load(f)
        self._bm25 = payload["bm25"]
        self._doc_ids = payload["doc_ids"]
        self._loaded = True

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        self._load()
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        if top_k >= len(scores):
            order = np.argsort(scores)[::-1]
        else:
            # argpartition is O(n); sort only the top_k slice
            top_idx = np.argpartition(scores, -top_k)[-top_k:]
            order = top_idx[np.argsort(scores[top_idx])[::-1]]
        return [(self._doc_ids[int(i)], float(scores[int(i)])) for i in order]
