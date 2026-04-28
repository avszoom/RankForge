"""Assemble cached doc + hard-neg outputs into data/corpus.jsonl."""
from __future__ import annotations

import json
import logging
import random
from datetime import date, timedelta
from pathlib import Path

from .ontology import CATEGORIES, iter_topics, slugify
from ._cache import cache_path
from ._text import extract_keywords

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
CORPUS_PATH = DATA_DIR / "corpus.jsonl"
SEED = 42
DOCS_PER_TOPIC = 10
NEGS_PER_TOPIC = 5

# Date window for created_at: last 18 months from a fixed reference so reruns are stable.
REFERENCE_DATE = date(2026, 4, 28)
WINDOW_DAYS = 18 * 30


def _random_date(rng: random.Random) -> str:
    delta = rng.randint(0, WINDOW_DAYS)
    return (REFERENCE_DATE - timedelta(days=delta)).isoformat()


def build_corpus() -> int:
    """Read raw docs + hard_negs from cache and write data/corpus.jsonl. Returns row count."""
    rng = random.Random(SEED)
    rows: list[dict] = []
    next_id = 1

    for category, topic in iter_topics():
        cat_slug = slugify(category)
        topic_slug = slugify(topic)

        for i in range(DOCS_PER_TOPIC):
            key = f"{cat_slug}__{topic_slug}__{i:02d}"
            p = cache_path("docs", key)
            if not p.is_file():
                raise FileNotFoundError(f"missing cached doc: {p}")
            data = json.loads(p.read_text())
            title = str(data["title"]).strip()
            body = str(data["body"]).strip()
            rows.append({
                "doc_id": f"doc_{next_id:05d}",
                "category": category,
                "topic": topic,
                "title": title,
                "body": body,
                "keywords": extract_keywords(title),
                "created_at": _random_date(rng),
                "is_hard_negative": False,
            })
            next_id += 1

        neg_key = f"{cat_slug}__{topic_slug}"
        neg_path = cache_path("hard_negs", neg_key)
        if not neg_path.is_file():
            raise FileNotFoundError(f"missing cached hard_negs: {neg_path}")
        neg_data = json.loads(neg_path.read_text())
        negs = neg_data["docs"][:NEGS_PER_TOPIC]
        for d in negs:
            title = str(d["title"]).strip()
            body = str(d["body"]).strip()
            rows.append({
                "doc_id": f"doc_{next_id:05d}",
                "category": category,
                "topic": topic,
                "title": title,
                "body": body,
                "keywords": extract_keywords(title),
                "created_at": _random_date(rng),
                "is_hard_negative": True,
            })
            next_id += 1

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with CORPUS_PATH.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    logger.info("build_corpus: wrote %d docs to %s", len(rows), CORPUS_PATH)
    return len(rows)


def write_ontology_json() -> None:
    """Render the canonical ontology dict to data/ontology.json."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = DATA_DIR / "ontology.json"
    out.write_text(json.dumps(CATEGORIES, ensure_ascii=False, indent=2))
    logger.info("wrote %s", out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    write_ontology_json()
    build_corpus()
