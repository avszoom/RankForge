"""Feature builder — turns (query, candidate) pairs into a feature matrix.

For each query in queries.jsonl:
  1. Run HybridRetriever to get ~50–100 candidate docs.
  2. Compute 12 features per (query, doc) pair.
  3. Look up the graded relevance label from relevance.jsonl (default 0 if absent).
  4. Emit one row per candidate, ordered by query_id (so LightGBM groups land contiguous).

This same builder is reused at inference time — the only difference is whether
we look up `relevance` or not. Keeping one code path avoids train/serve skew.
"""
from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.retrieval.hybrid_retriever import HybridRetriever
from src.text_utils import tokenize

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[2]
CORPUS_PATH = REPO / "data" / "corpus.jsonl"
QUERIES_PATH = REPO / "data" / "queries.jsonl"
RELEVANCE_PATH = REPO / "data" / "relevance.jsonl"

ID_COLUMNS = ["query_id", "doc_id"]
FEATURE_COLUMNS = [
    "bm25_score",
    "bm25_rank",
    "vector_score",
    "vector_rank",
    "retrieved_by_bm25",
    "retrieved_by_vector",
    "title_overlap",
    "body_overlap",
    "doc_length",
    "query_length",
    "freshness_days",
    "category_match",
]
LABEL_COLUMN = "relevance"
ALL_COLUMNS = ID_COLUMNS + FEATURE_COLUMNS + [LABEL_COLUMN, "split"]

# Sentinel for ranks when a retriever didn't return the doc.
# Matches the per_retriever ceiling so the model sees a "worse-than-anything-retrieved" signal.
MISSING_RANK = 999


@dataclass(slots=True)
class _DocCache:
    title_tokens: set[str]
    body_tokens: set[str]
    doc_length: int
    category: str
    created_at: date


class FeatureBuilder:
    """Computes feature rows. Loads corpus / labels / retriever lazily."""

    def __init__(
        self,
        corpus_path: Path = CORPUS_PATH,
        relevance_path: Path = RELEVANCE_PATH,
        retriever: HybridRetriever | None = None,
        per_retriever: int = 50,
        top_k: int = 100,
    ) -> None:
        self.corpus_path = corpus_path
        self.relevance_path = relevance_path
        self.retriever = retriever or HybridRetriever()
        self.per_retriever = per_retriever
        self.top_k = top_k
        self._docs: dict[str, _DocCache] = {}
        self._reference_date: date | None = None
        self._labels: dict[tuple[str, str], int] = {}
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return

        max_date = date.min
        for line in self.corpus_path.read_text().splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            title_tokens = set(tokenize(d["title"]))
            body_tokens = set(tokenize(d["body"]))
            doc_len = len(tokenize(d["title"] + " " + d["body"]))
            created = date.fromisoformat(d["created_at"])
            if created > max_date:
                max_date = created
            self._docs[d["doc_id"]] = _DocCache(
                title_tokens=title_tokens,
                body_tokens=body_tokens,
                doc_length=doc_len,
                category=d["category"],
                created_at=created,
            )
        # Anchor freshness to the corpus's most recent doc so features are
        # corpus-deterministic (re-running the builder gives identical numbers).
        self._reference_date = max_date

        for line in self.relevance_path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            self._labels[(r["query_id"], r["doc_id"])] = int(r["relevance"])

        logger.info(
            "FeatureBuilder loaded: %d docs, %d explicit labels, reference_date=%s",
            len(self._docs), len(self._labels), self._reference_date,
        )
        self._loaded = True

    def _row(
        self,
        query_id: str,
        query_text: str,
        target_category: str,
        doc_id: str,
        candidate,
    ) -> dict:
        """Build one feature row. Returns dict with all ID + feature columns + label."""
        doc = self._docs[doc_id]
        q_tokens = set(tokenize(query_text))
        q_len = len(q_tokens)

        title_overlap = (
            len(q_tokens & doc.title_tokens) / q_len if q_len else 0.0
        )
        body_overlap = (
            len(q_tokens & doc.body_tokens) / q_len if q_len else 0.0
        )
        freshness_days = (self._reference_date - doc.created_at).days

        return {
            "query_id": query_id,
            "doc_id": doc_id,
            "bm25_score": candidate.bm25_score,
            "bm25_rank": candidate.bm25_rank if candidate.bm25_rank is not None else MISSING_RANK,
            "vector_score": candidate.vector_score,
            "vector_rank": candidate.vector_rank if candidate.vector_rank is not None else MISSING_RANK,
            "retrieved_by_bm25": int(candidate.retrieved_by_bm25),
            "retrieved_by_vector": int(candidate.retrieved_by_vector),
            "title_overlap": title_overlap,
            "body_overlap": body_overlap,
            "doc_length": doc.doc_length,
            "query_length": q_len,
            "freshness_days": freshness_days,
            "category_match": int(doc.category == target_category),
            LABEL_COLUMN: self._labels.get((query_id, doc_id), 0),
        }

    def build_for_queries(self, queries: Iterable[dict]) -> pd.DataFrame:
        """Run hybrid retrieval on each query and emit one DataFrame of rows."""
        self._load()
        rows: list[dict] = []
        n = 0
        for q in queries:
            query_id = q["query_id"]
            query_text = q["query"]
            target_category = q["target_category"]
            split = q.get("split", "train")

            cands = self.retriever.search(
                query_text, top_k=self.top_k, per_retriever=self.per_retriever,
            )
            for c in cands:
                row = self._row(query_id, query_text, target_category, c.doc_id, c)
                row["split"] = split
                rows.append(row)
            n += 1
            if n % 100 == 0:
                logger.info("  processed %d queries (%d rows so far)", n, len(rows))

        df = pd.DataFrame(rows, columns=ALL_COLUMNS)
        # Stable order: by query_id first (for LightGBM groups), then by RRF-ish
        # signal so within a query the strongest candidate is row 0.
        df = df.sort_values(
            ["query_id", "bm25_score", "vector_score"],
            ascending=[True, False, False],
            kind="stable",
        ).reset_index(drop=True)
        return df


def load_queries(path: Path = QUERIES_PATH) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def split_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into train/test by the `split` column already present per row."""
    train = df[df["split"] == "train"].reset_index(drop=True)
    test = df[df["split"] == "test"].reset_index(drop=True)
    return train, test


def label_distribution(df: pd.DataFrame) -> dict[int, int]:
    return Counter(df[LABEL_COLUMN].tolist())


def coverage_stats(df: pd.DataFrame, fb: FeatureBuilder) -> tuple[int, int]:
    """Return (rows_with_explicit_label, total_rows) for diagnostics."""
    fb._load()
    explicit = sum(
        1 for q, d in zip(df["query_id"], df["doc_id"]) if (q, d) in fb._labels
    )
    return explicit, len(df)
