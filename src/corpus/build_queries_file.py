"""Assemble cached query outputs into data/queries.jsonl."""
from __future__ import annotations

import json
import logging
import random
import re
from pathlib import Path

from .ontology import iter_topics, slugify
from ._cache import cache_path
from ._text import topic_tokens

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
QUERIES_PATH = DATA_DIR / "queries.jsonl"
SEED = 42
TEST_SPLIT = 0.10  # 10% test, 90% train


def classify_query_type(query: str, topic: str) -> str:
    """Heuristic: exact (>=2 topic tokens), noisy (very short, no articles, abbreviations), else paraphrase."""
    q = query.strip().lower()
    q_tokens = set(re.findall(r"[a-zA-Z]+", q))
    overlap = len(q_tokens & topic_tokens(topic))
    if overlap >= 2:
        return "exact"
    word_count = len(q.split())
    has_article = any(a in q.split() for a in ("the", "a", "an", "is", "are"))
    if word_count <= 4 and not has_article:
        return "noisy"
    return "paraphrase"


def build_queries_file() -> int:
    """Read raw queries from cache and write data/queries.jsonl. Returns row count."""
    rng = random.Random(SEED)
    rows: list[dict] = []
    next_id = 1

    for category, topic in iter_topics():
        key = f"{slugify(category)}__{slugify(topic)}"
        p = cache_path("queries", key)
        if not p.is_file():
            raise FileNotFoundError(f"missing cached queries: {p}")
        data = json.loads(p.read_text())
        queries = data["queries"]
        for q in queries:
            q_str = str(q).strip()
            if not q_str:
                continue
            split = "test" if rng.random() < TEST_SPLIT else "train"
            rows.append({
                "query_id": f"q_{next_id:05d}",
                "query": q_str,
                "target_category": category,
                "target_topic": topic,
                "query_type": classify_query_type(q_str, topic),
                "split": split,
            })
            next_id += 1

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with QUERIES_PATH.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    logger.info("build_queries_file: wrote %d queries to %s", len(rows), QUERIES_PATH)
    return len(rows)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    build_queries_file()
