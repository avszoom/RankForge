"""VectorRetriever — semantic retrieval over the FAISS vector index."""
from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.text_utils import clean_text

REPO = Path(__file__).resolve().parents[2]
DEFAULT_INDEX = REPO / "models" / "faiss.index"
DEFAULT_IDS = REPO / "models" / "faiss_doc_ids.json"


class VectorRetriever:
    def __init__(
        self,
        index_path: Path | str = DEFAULT_INDEX,
        ids_path: Path | str = DEFAULT_IDS,
    ) -> None:
        self.index_path = Path(index_path)
        self.ids_path = Path(ids_path)
        self._loaded = False
        self._index = None
        self._model = None
        self._model_name: str = ""
        self._doc_ids: list[str] = []

    def _load(self) -> None:
        if self._loaded:
            return
        meta = json.loads(self.ids_path.read_text())
        self._doc_ids = meta["doc_ids"]
        self._model_name = meta["model"]
        self._index = faiss.read_index(str(self.index_path))
        self._model = SentenceTransformer(self._model_name)
        self._loaded = True

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        self._load()
        q = clean_text(query)
        if not q:
            return []
        emb = self._model.encode(
            [q], normalize_embeddings=True, convert_to_numpy=True
        ).astype(np.float32)
        scores, idx = self._index.search(emb, top_k)
        return [
            (self._doc_ids[int(i)], float(s))
            for i, s in zip(idx[0], scores[0])
            if i != -1
        ]
