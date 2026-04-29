"""Run all 4 setups (BM25, FAISS, Hybrid+RRF, LightGBM) on the test queries
and produce per-query and per-setup metric DataFrames.

Test queries are the ones with `split == "test"` in queries.jsonl
(129 of them, pre-split at corpus generation time).
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.evaluation.metrics import mrr, ndcg_at_k, precision_at_k, recall_at_k
from src.ranking.ranker import LightGBMRanker
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.vector_retriever import VectorRetriever

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[2]
QUERIES_PATH = REPO / "data" / "queries.jsonl"
RELEVANCE_PATH = REPO / "data" / "relevance.jsonl"

SETUPS = ("bm25", "faiss", "hybrid_rrf", "ranker")
EVAL_TOP_K = 50  # deepest top-K we need (for recall@50); all other metrics slice from this


def _load_queries(path: Path = QUERIES_PATH) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _load_labels_by_query(path: Path = RELEVANCE_PATH) -> dict[str, dict[str, int]]:
    """Group relevance.jsonl into {query_id: {doc_id: relevance}}."""
    labels: dict[str, dict[str, int]] = defaultdict(dict)
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        labels[r["query_id"]][r["doc_id"]] = int(r["relevance"])
    return dict(labels)


def _row_metrics(pred_ids: list[str], labels: dict[str, int]) -> dict[str, float]:
    return {
        "ndcg@5":  ndcg_at_k(pred_ids, labels, 5),
        "ndcg@10": ndcg_at_k(pred_ids, labels, 10),
        "ndcg@20": ndcg_at_k(pred_ids, labels, 20),
        "mrr":     mrr(pred_ids, labels),
        "p@5":     precision_at_k(pred_ids, labels, 5),
        "p@10":    precision_at_k(pred_ids, labels, 10),
        "r@10":    recall_at_k(pred_ids, labels, 10),
        "r@50":    recall_at_k(pred_ids, labels, 50),
    }


def evaluate(
    queries: list[dict] | None = None,
    labels_by_query: dict[str, dict[str, int]] | None = None,
    *,
    only_split: str | None = "test",
    top_k: int = EVAL_TOP_K,
    progress: bool = True,
) -> pd.DataFrame:
    """Return a long-form DataFrame with one row per (query_id, setup, metric set)."""
    queries = queries or _load_queries()
    labels_by_query = labels_by_query or _load_labels_by_query()
    if only_split:
        queries = [q for q in queries if q.get("split") == only_split]

    bm25 = BM25Retriever()
    vec = VectorRetriever()
    hybrid = HybridRetriever()
    ranker = LightGBMRanker()

    rows: list[dict] = []
    iterator = tqdm(queries, desc=f"eval ({len(queries)} queries × 4 setups)") if progress else queries
    for q in iterator:
        qid = q["query_id"]
        labels = labels_by_query.get(qid, {})

        # Each setup yields a ranked list of doc_ids of length up to top_k.
        bm25_preds = [d for d, _ in bm25.search(q["query"], top_k=top_k)]
        faiss_preds = [d for d, _ in vec.search(q["query"], top_k=top_k)]
        hybrid_cands = hybrid.search(q["query"], top_k=top_k, per_retriever=50)
        hybrid_preds = [c.doc_id for c in hybrid_cands]
        ranker_preds = [c.doc_id for c, _ in ranker.rank(
            q["query"], top_k=top_k, per_retriever=50, retriever_top_k=100,
        )]

        for setup, preds in (
            ("bm25", bm25_preds),
            ("faiss", faiss_preds),
            ("hybrid_rrf", hybrid_preds),
            ("ranker", ranker_preds),
        ):
            row = {
                "query_id": qid,
                "query_type": q["query_type"],
                "split": q["split"],
                "setup": setup,
                "n_predicted": len(preds),
                **_row_metrics(preds, labels),
            }
            rows.append(row)

    return pd.DataFrame(rows)


def aggregate_overall(df: pd.DataFrame) -> pd.DataFrame:
    """Mean metrics per setup across all queries."""
    metric_cols = ["ndcg@5", "ndcg@10", "ndcg@20", "mrr", "p@5", "p@10", "r@10", "r@50"]
    return (
        df.groupby("setup")[metric_cols]
          .mean()
          .reindex(SETUPS)  # consistent setup ordering: bm25, faiss, hybrid_rrf, ranker
    )


def aggregate_by_query_type(df: pd.DataFrame) -> pd.DataFrame:
    """Mean metrics per (query_type, setup)."""
    metric_cols = ["ndcg@5", "ndcg@10", "ndcg@20", "mrr", "p@5", "p@10", "r@10", "r@50"]
    return (
        df.groupby(["query_type", "setup"])[metric_cols]
          .mean()
          .reindex(
              pd.MultiIndex.from_product(
                  [sorted(df["query_type"].unique()), SETUPS],
                  names=["query_type", "setup"],
              )
          )
    )
