"""Search ranking metrics — pure functions, stdlib + numpy.

Each function takes:
    pred_ids:  ranked list of doc_ids the model returned (length >= K).
    labels:    dict[doc_id -> int relevance], with default 0 for unlabeled docs.
    k:         cutoff position.

Conventions:
    - relevance ∈ {0, 1, 2, 3} as in relevance.jsonl.
    - "relevant" means relevance >= relevant_threshold (default 2) for binary metrics.
    - NDCG uses graded relevance; the binary threshold doesn't apply there.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def ndcg_at_k(pred_ids: Sequence[str], labels: Mapping[str, int], k: int) -> float:
    """Normalized Discounted Cumulative Gain at K.

    Returns 0.0 when there are no positively-labeled docs for this query
    (would give 0/0 — by convention treat as 0).
    """
    dcg = _dcg_at_k([labels.get(d, 0) for d in pred_ids[:k]])
    ideal = sorted(labels.values(), reverse=True)[:k]
    idcg = _dcg_at_k(ideal)
    if idcg == 0:
        return 0.0
    return dcg / idcg


def _dcg_at_k(rels: Sequence[int]) -> float:
    return sum(
        (2 ** rel - 1) / math.log2(i + 2)  # i is 0-based; position is i+1; log₂(i+1+1) = log₂(i+2)
        for i, rel in enumerate(rels)
    )


def mrr(pred_ids: Sequence[str], labels: Mapping[str, int], relevant_threshold: int = 2) -> float:
    """Reciprocal rank of the first relevant doc (relevance >= threshold). 0 if none in list."""
    for i, doc_id in enumerate(pred_ids, 1):
        if labels.get(doc_id, 0) >= relevant_threshold:
            return 1.0 / i
    return 0.0


def precision_at_k(
    pred_ids: Sequence[str],
    labels: Mapping[str, int],
    k: int,
    relevant_threshold: int = 2,
) -> float:
    """Fraction of the top-K that are relevant. Returns 0 if pred_ids empty."""
    if not pred_ids:
        return 0.0
    top = pred_ids[:k]
    n_relevant = sum(1 for d in top if labels.get(d, 0) >= relevant_threshold)
    return n_relevant / k


def recall_at_k(
    pred_ids: Sequence[str],
    labels: Mapping[str, int],
    k: int,
    relevant_threshold: int = 2,
) -> float:
    """Of all relevant docs (in the labels), what fraction landed in top-K.

    The 'all relevant' set is taken from the labels dict, so make sure labels
    contains the whole labeled set for the query (not just predicted docs).
    """
    total_relevant = sum(1 for r in labels.values() if r >= relevant_threshold)
    if total_relevant == 0:
        return 0.0
    top = set(pred_ids[:k])
    n_in_top = sum(
        1 for d, r in labels.items()
        if d in top and r >= relevant_threshold
    )
    return n_in_top / total_relevant
