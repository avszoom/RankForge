"""HybridRetriever — merges BM25 + FAISS candidate sets per query.

Each call to `search(query, top_k=100, per_retriever=50)`:
  1. fetches BM25 top-`per_retriever` and FAISS top-`per_retriever` independently,
  2. dedupes by doc_id,
  3. attaches both scores + 1-based ranks per source (None when not retrieved),
  4. attaches retrieved_by_bm25 / retrieved_by_vector flags,
  5. returns the merged list (capped at top_k by combined-rank reciprocal sum).

Output is the input shape Phase 3 (feature builder) and Phase 4 (LightGBM ranker)
will consume — no surface features are computed here, only retrieval signals.
"""
from __future__ import annotations

from dataclasses import dataclass

from .bm25_retriever import BM25Retriever
from .vector_retriever import VectorRetriever


@dataclass
class Candidate:
    doc_id: str
    bm25_score: float = 0.0
    bm25_rank: int | None = None
    vector_score: float = 0.0
    vector_rank: int | None = None
    retrieved_by_bm25: bool = False
    retrieved_by_vector: bool = False

    @property
    def rrf_score(self) -> float:
        """Reciprocal Rank Fusion: a robust score-free fusion baseline.

        Used here only to order the merged set when the caller wants a single
        ranked list. The Phase 4 ranker will replace this with a learned model.
        """
        k = 60.0  # Cormack et al. 2009 default; insensitive in practice
        s = 0.0
        if self.bm25_rank is not None:
            s += 1.0 / (k + self.bm25_rank)
        if self.vector_rank is not None:
            s += 1.0 / (k + self.vector_rank)
        return s

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "bm25_score": self.bm25_score,
            "bm25_rank": self.bm25_rank,
            "vector_score": self.vector_score,
            "vector_rank": self.vector_rank,
            "retrieved_by_bm25": self.retrieved_by_bm25,
            "retrieved_by_vector": self.retrieved_by_vector,
        }


class HybridRetriever:
    def __init__(
        self,
        bm25: BM25Retriever | None = None,
        vector: VectorRetriever | None = None,
    ) -> None:
        self.bm25 = bm25 or BM25Retriever()
        self.vector = vector or VectorRetriever()

    def search(
        self,
        query: str,
        top_k: int = 100,
        per_retriever: int = 50,
    ) -> list[Candidate]:
        bm25_hits = self.bm25.search(query, top_k=per_retriever)
        vec_hits = self.vector.search(query, top_k=per_retriever)

        merged: dict[str, Candidate] = {}

        for rank, (doc_id, score) in enumerate(bm25_hits, 1):
            c = merged.setdefault(doc_id, Candidate(doc_id=doc_id))
            c.bm25_score = score
            c.bm25_rank = rank
            c.retrieved_by_bm25 = True

        for rank, (doc_id, score) in enumerate(vec_hits, 1):
            c = merged.setdefault(doc_id, Candidate(doc_id=doc_id))
            c.vector_score = score
            c.vector_rank = rank
            c.retrieved_by_vector = True

        # Order by RRF — gives a stable, score-free ranking until Phase 4 lands.
        ordered = sorted(merged.values(), key=lambda c: c.rrf_score, reverse=True)
        return ordered[:top_k]
