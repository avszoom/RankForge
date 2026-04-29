"""TwoStageRanker — LightGBM (fast tabular) + CrossEncoder (slow neural).

Stage 1 (LightGBMRanker) takes ~100 hybrid candidates and reduces to top-20.
Stage 2 (CrossEncoderReranker) reorders those 20 → final top-10.

Returns rich tuples so the CLI/eval can show per-stage scores side by side.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from src.ranking.cross_encoder import CrossEncoderReranker
from src.ranking.ranker import LightGBMRanker
from src.retrieval.hybrid_retriever import Candidate

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RankedItem:
    candidate: Candidate
    lgbm_score: float
    ce_score: float
    lgbm_rank: int    # 1-based position in stage-1 output


class TwoStageRanker:
    def __init__(
        self,
        lgbm: LightGBMRanker | None = None,
        ce: CrossEncoderReranker | None = None,
        stage1_top_k: int = 20,
    ) -> None:
        self.lgbm = lgbm or LightGBMRanker()
        self.ce = ce or CrossEncoderReranker()
        self.stage1_top_k = stage1_top_k

    def rank(
        self,
        query: str,
        top_k: int = 10,
        per_retriever: int = 50,
        retriever_top_k: int = 100,
    ) -> list[RankedItem]:
        """Run the full 2-stage pipeline. Returns top-K items, sorted by ce_score."""
        # Stage 1: LightGBM produces top-stage1_top_k candidates ordered by lgbm_score.
        stage1 = self.lgbm.rank(
            query,
            top_k=self.stage1_top_k,
            per_retriever=per_retriever,
            retriever_top_k=retriever_top_k,
        )
        if not stage1:
            return []

        cand_by_id: dict[str, Candidate] = {c.doc_id: c for c, _ in stage1}
        lgbm_score_by_id: dict[str, float] = {c.doc_id: float(s) for c, s in stage1}
        lgbm_rank_by_id: dict[str, int] = {c.doc_id: i for i, (c, _) in enumerate(stage1, 1)}

        # Stage 2: CE reranks the same set of doc_ids.
        ce_ranked = self.ce.rerank(query, [c.doc_id for c, _ in stage1])

        out: list[RankedItem] = []
        for doc_id, ce_score in ce_ranked[:top_k]:
            out.append(RankedItem(
                candidate=cand_by_id[doc_id],
                lgbm_score=lgbm_score_by_id[doc_id],
                ce_score=ce_score,
                lgbm_rank=lgbm_rank_by_id[doc_id],
            ))
        return out

    def rank_eval_list(
        self,
        query: str,
        top_k: int = 50,
        per_retriever: int = 50,
        retriever_top_k: int = 100,
    ) -> list[str]:
        """Produce a flat list of doc_ids of length up to `top_k` for evaluation.

        - Positions 1..stage1_top_k: CE-reordered docs.
        - Positions stage1_top_k+1..top_k: original LightGBM ordering (untouched).
        Below the stage1_top_k line, the CE never sees those docs.
        """
        # Get LightGBM top up to top_k (so we have positions stage1_top_k+1..top_k for padding).
        lgbm_full = self.lgbm.rank(
            query,
            top_k=top_k,
            per_retriever=per_retriever,
            retriever_top_k=retriever_top_k,
        )
        if not lgbm_full:
            return []

        # Stage 2: CE on the first stage1_top_k docs.
        head_ids = [c.doc_id for c, _ in lgbm_full[: self.stage1_top_k]]
        ce_ranked = self.ce.rerank(query, head_ids)
        ce_ordered_ids = [d for d, _ in ce_ranked]

        # Tail: LightGBM positions stage1_top_k+1..top_k unchanged.
        tail_ids = [c.doc_id for c, _ in lgbm_full[self.stage1_top_k :]]
        return ce_ordered_ids + tail_ids
