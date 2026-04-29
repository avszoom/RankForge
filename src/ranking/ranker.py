"""LightGBMRanker — inference-time wrapper around the trained ranker.

Loads models/ranker.pkl + models/ranker_meta.json once, then exposes:

    score(features_df) -> np.ndarray            # one score per row, sort key
    rank(query, target_category, top_k=20)      # full pipeline: hybrid → features → score → sort

The default top_k is 20 (per-spec). The ranker itself doesn't truncate at training time —
it produces a continuous score for every candidate and we slice to top_k after sorting.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.ranking.feature_builder import (
    ALL_COLUMNS, FEATURE_COLUMNS, FeatureBuilder, MISSING_RANK,
)
from src.retrieval.hybrid_retriever import Candidate, HybridRetriever

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = REPO / "models" / "ranker.pkl"
DEFAULT_META = REPO / "models" / "ranker_meta.json"


class LightGBMRanker:
    def __init__(
        self,
        model_path: Path | str = DEFAULT_MODEL,
        meta_path: Path | str = DEFAULT_META,
        feature_builder: FeatureBuilder | None = None,
        retriever: HybridRetriever | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.meta_path = Path(meta_path)
        self.feature_builder = feature_builder
        self.retriever = retriever
        self._model = None
        self._feature_columns: list[str] = []
        self._best_iteration: int = 0
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Ranker model not found at {self.model_path}. "
                f"Train one first: python scripts/train_ranker.py"
            )
        meta = json.loads(self.meta_path.read_text())
        self._feature_columns = meta["feature_columns"]
        self._best_iteration = int(meta.get("best_iteration", 0))
        self._model = joblib.load(self.model_path)
        if self.feature_builder is None:
            self.feature_builder = FeatureBuilder(retriever=self.retriever)
        self._loaded = True

    def score(self, features: pd.DataFrame) -> np.ndarray:
        """Score a feature DataFrame. Returns one float per row.

        Scores are real numbers with no fixed scale — only relative ordering matters.
        """
        self._load()
        X = features[self._feature_columns]
        return self._model.predict(X, num_iteration=self._best_iteration or None)

    def rank(
        self,
        query: str,
        top_k: int = 20,
        per_retriever: int = 50,
        retriever_top_k: int = 100,
    ) -> list[tuple[Candidate, float]]:
        """End-to-end: hybrid retrieve → build features → score → return top_k pairs.

        Returns: list of (Candidate, ranker_score), sorted by ranker_score descending.

        No `target_category` arg — the feature builder derives the proxy
        (top_retrieved_category) from the candidate set itself, so no ground
        truth is needed at inference.
        """
        self._load()
        retriever = self.retriever or HybridRetriever()
        candidates = retriever.search(
            query, top_k=retriever_top_k, per_retriever=per_retriever,
        )
        if not candidates:
            return []

        self.feature_builder._load()
        rows = []
        for c in candidates:
            row = self.feature_builder._row(
                query_id="<inference>",
                query_text=query,
                doc_id=c.doc_id,
                candidate=c,
            )
            rows.append(row)

        df = pd.DataFrame(rows, columns=ALL_COLUMNS)
        scores = self.score(df)

        ranked = sorted(zip(candidates, scores), key=lambda x: float(x[1]), reverse=True)
        return ranked[:top_k]
