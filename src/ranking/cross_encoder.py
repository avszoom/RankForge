"""CrossEncoderReranker — neural reranker that scores (query, doc) text pairs jointly.

Wraps `cross-encoder/ms-marco-MiniLM-L-6-v2` from sentence-transformers. The model
takes `[CLS] query [SEP] doc [SEP]` as input and outputs a single relevance score.

Used as the second stage in TwoStageRanker — LightGBM produces top-20 candidates
on cheap features, then this reranker rescores those 20 with the actual text.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS = REPO / "data" / "corpus.jsonl"
DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_MAX_CHARS = 1024  # truncate body so combined input fits in the 512-token window


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        corpus_path: Path | str = DEFAULT_CORPUS,
        max_chars: int = DEFAULT_MAX_CHARS,
        batch_size: int = 16,
    ) -> None:
        self.model_name = model_name
        self.corpus_path = Path(corpus_path)
        self.max_chars = max_chars
        self.batch_size = batch_size
        self._model = None
        self._corpus: dict[str, dict] = {}
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        from sentence_transformers import CrossEncoder
        logger.info("loading cross-encoder %s", self.model_name)
        self._model = CrossEncoder(self.model_name)
        for line in self.corpus_path.read_text().splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            self._corpus[d["doc_id"]] = d
        self._loaded = True

    def _build_text(self, doc: dict) -> str:
        title = doc.get("title", "").strip()
        body = doc.get("body", "").strip()
        text = f"{title}. {body}"
        if len(text) > self.max_chars:
            text = text[: self.max_chars]
        return text

    def rerank(
        self,
        query: str,
        doc_ids: Iterable[str],
        top_k: int | None = None,
    ) -> list[tuple[str, float]]:
        """Score a list of doc_ids for the query and return them sorted by score (desc).

        If top_k is None, return all input doc_ids reordered. Otherwise return top_k.
        """
        self._load()
        ids = [d for d in doc_ids if d in self._corpus]
        if not ids:
            return []
        pairs = [(query, self._build_text(self._corpus[d])) for d in ids]
        scores = self._model.predict(
            pairs, batch_size=self.batch_size, show_progress_bar=False,
        )
        ranked = sorted(zip(ids, [float(s) for s in scores]), key=lambda x: x[1], reverse=True)
        if top_k is not None:
            ranked = ranked[:top_k]
        return ranked
