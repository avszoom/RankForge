"""Compute graded relevance labels from corpus + queries.

Rules (per the approved plan):
- Same topic, NOT hard-neg               -> 3
- Same topic, IS hard-neg                -> 1   (semantically related but not the topic)
- Same category, different topic         -> 2   (sample 15 per query)
- Same domain group, different category  -> 1   (sample 15 per query)
- Different domain group                 -> 0   (sample 30 per query)
"""
from __future__ import annotations

import json
import logging
import random
from collections import defaultdict
from pathlib import Path

from .ontology import CATEGORY_TO_DOMAIN, DOMAIN_GROUPS

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
CORPUS_PATH = DATA_DIR / "corpus.jsonl"
QUERIES_PATH = DATA_DIR / "queries.jsonl"
RELEVANCE_PATH = DATA_DIR / "relevance.jsonl"
SEED = 42

SAMPLE_SAME_CAT_OTHER_TOPIC = 15
SAMPLE_SAME_DOMAIN_OTHER_CAT = 15
SAMPLE_OTHER_DOMAIN = 30


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def build_relevance() -> int:
    rng = random.Random(SEED)
    corpus = _read_jsonl(CORPUS_PATH)
    queries = _read_jsonl(QUERIES_PATH)

    # Pre-bucket docs for fast sampling.
    by_topic: dict[tuple[str, str], list[dict]] = defaultdict(list)
    by_category: dict[str, list[dict]] = defaultdict(list)
    by_domain: dict[str, list[dict]] = defaultdict(list)
    for d in corpus:
        by_topic[(d["category"], d["topic"])].append(d)
        by_category[d["category"]].append(d)
        domain = CATEGORY_TO_DOMAIN[d["category"]]
        by_domain[domain].append(d)

    rows: list[dict] = []
    for q in queries:
        cat = q["target_category"]
        topic = q["target_topic"]
        domain = CATEGORY_TO_DOMAIN[cat]
        seen: set[str] = set()

        # Tier 3 + Tier 1 (hard-negs) for the query's own topic
        for d in by_topic[(cat, topic)]:
            label = 1 if d["is_hard_negative"] else 3
            rows.append({"query_id": q["query_id"], "doc_id": d["doc_id"], "relevance": label})
            seen.add(d["doc_id"])

        # Tier 2 - other topics in same category
        same_cat_other_topic = [
            d for d in by_category[cat] if d["topic"] != topic and d["doc_id"] not in seen
        ]
        for d in rng.sample(same_cat_other_topic, min(SAMPLE_SAME_CAT_OTHER_TOPIC, len(same_cat_other_topic))):
            rows.append({"query_id": q["query_id"], "doc_id": d["doc_id"], "relevance": 2})
            seen.add(d["doc_id"])

        # Tier 1 - same domain, different category
        same_domain_other_cat = [
            d for d in by_domain[domain]
            if d["category"] != cat and d["doc_id"] not in seen
        ]
        for d in rng.sample(same_domain_other_cat, min(SAMPLE_SAME_DOMAIN_OTHER_CAT, len(same_domain_other_cat))):
            rows.append({"query_id": q["query_id"], "doc_id": d["doc_id"], "relevance": 1})
            seen.add(d["doc_id"])

        # Tier 0 - other domain entirely
        other_domain = [d for d in corpus if CATEGORY_TO_DOMAIN[d["category"]] != domain]
        for d in rng.sample(other_domain, min(SAMPLE_OTHER_DOMAIN, len(other_domain))):
            if d["doc_id"] in seen:
                continue
            rows.append({"query_id": q["query_id"], "doc_id": d["doc_id"], "relevance": 0})
            seen.add(d["doc_id"])

    with RELEVANCE_PATH.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    logger.info("build_relevance: wrote %d labels to %s", len(rows), RELEVANCE_PATH)
    return len(rows)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    build_relevance()
