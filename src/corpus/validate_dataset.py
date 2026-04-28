"""Validate the generated dataset (corpus / queries / relevance / ontology).

Schema, referential integrity, and tier-rule sanity checks. Stdlib only.
Exits 0 on success, 1 on any hard error.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

from .ontology import CATEGORIES, CATEGORY_TO_DOMAIN

DOC_ID_RE = re.compile(r"^doc_\d{5}$")
QUERY_ID_RE = re.compile(r"^q_\d{5}$")
ALLOWED_QUERY_TYPES = {"exact", "paraphrase", "noisy"}
ALLOWED_SPLITS = {"train", "test"}
ALLOWED_RELEVANCE = {0, 1, 2, 3}


class Validator:
    def __init__(self, data_dir: Path) -> None:
        self.dir = data_dir
        self.errors: list[str] = []

    def err(self, msg: str) -> None:
        self.errors.append(msg)

    def check_ontology(self) -> dict[str, list[str]]:
        path = self.dir / "ontology.json"
        if not path.is_file():
            self.err(f"missing {path}")
            return {}
        ontology = json.loads(path.read_text())
        # Compare to canonical Python ontology
        if set(ontology) != set(CATEGORIES):
            self.err(f"ontology.json categories differ from canonical: extra={set(ontology) - set(CATEGORIES)}, missing={set(CATEGORIES) - set(ontology)}")
        for cat, topics in CATEGORIES.items():
            if cat not in ontology:
                continue
            if list(ontology[cat]) != list(topics):
                self.err(f"ontology.json topics for {cat!r} differ from canonical")
        return ontology

    def check_corpus(self) -> list[dict]:
        path = self.dir / "corpus.jsonl"
        if not path.is_file():
            self.err(f"missing {path}")
            return []
        corpus: list[dict] = []
        seen_ids: set[str] = set()
        for ln, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError as e:
                self.err(f"corpus.jsonl L{ln}: invalid JSON: {e}")
                continue
            for f in ("doc_id", "category", "topic", "title", "body", "keywords", "created_at", "is_hard_negative"):
                if f not in d:
                    self.err(f"corpus.jsonl L{ln}: missing field {f!r}")
                    break
            else:
                if not DOC_ID_RE.match(d["doc_id"]):
                    self.err(f"corpus.jsonl L{ln}: bad doc_id {d['doc_id']!r}")
                if d["doc_id"] in seen_ids:
                    self.err(f"corpus.jsonl L{ln}: duplicate doc_id {d['doc_id']!r}")
                seen_ids.add(d["doc_id"])
                if d["category"] not in CATEGORIES:
                    self.err(f"corpus.jsonl L{ln}: unknown category {d['category']!r}")
                elif d["topic"] not in CATEGORIES[d["category"]]:
                    self.err(f"corpus.jsonl L{ln}: topic {d['topic']!r} not in category {d['category']!r}")
                if not isinstance(d["keywords"], list):
                    self.err(f"corpus.jsonl L{ln}: keywords must be list")
                if not isinstance(d["is_hard_negative"], bool):
                    self.err(f"corpus.jsonl L{ln}: is_hard_negative must be bool")
                try:
                    date.fromisoformat(d["created_at"])
                except (ValueError, TypeError):
                    self.err(f"corpus.jsonl L{ln}: bad created_at {d['created_at']!r}")
                if not d["title"].strip() or not d["body"].strip():
                    self.err(f"corpus.jsonl L{ln}: empty title or body")
                corpus.append(d)
        return corpus

    def check_queries(self) -> list[dict]:
        path = self.dir / "queries.jsonl"
        if not path.is_file():
            self.err(f"missing {path}")
            return []
        queries: list[dict] = []
        seen_ids: set[str] = set()
        for ln, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError as e:
                self.err(f"queries.jsonl L{ln}: invalid JSON: {e}")
                continue
            for f in ("query_id", "query", "target_category", "target_topic", "query_type", "split"):
                if f not in d:
                    self.err(f"queries.jsonl L{ln}: missing field {f!r}")
                    break
            else:
                if not QUERY_ID_RE.match(d["query_id"]):
                    self.err(f"queries.jsonl L{ln}: bad query_id {d['query_id']!r}")
                if d["query_id"] in seen_ids:
                    self.err(f"queries.jsonl L{ln}: duplicate query_id {d['query_id']!r}")
                seen_ids.add(d["query_id"])
                if d["target_category"] not in CATEGORIES:
                    self.err(f"queries.jsonl L{ln}: unknown target_category {d['target_category']!r}")
                elif d["target_topic"] not in CATEGORIES[d["target_category"]]:
                    self.err(f"queries.jsonl L{ln}: target_topic {d['target_topic']!r} not in {d['target_category']!r}")
                if d["query_type"] not in ALLOWED_QUERY_TYPES:
                    self.err(f"queries.jsonl L{ln}: bad query_type {d['query_type']!r}")
                if d["split"] not in ALLOWED_SPLITS:
                    self.err(f"queries.jsonl L{ln}: bad split {d['split']!r}")
                if not d["query"].strip():
                    self.err(f"queries.jsonl L{ln}: empty query")
                queries.append(d)
        return queries

    def check_relevance(self, corpus: list[dict], queries: list[dict]) -> int:
        path = self.dir / "relevance.jsonl"
        if not path.is_file():
            self.err(f"missing {path}")
            return 0
        doc_ids = {d["doc_id"] for d in corpus}
        query_ids = {q["query_id"] for q in queries}
        threes_per_query: Counter = Counter()
        total = 0
        for ln, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError as e:
                self.err(f"relevance.jsonl L{ln}: invalid JSON: {e}")
                continue
            for f in ("query_id", "doc_id", "relevance"):
                if f not in d:
                    self.err(f"relevance.jsonl L{ln}: missing field {f!r}")
                    break
            else:
                if d["query_id"] not in query_ids:
                    self.err(f"relevance.jsonl L{ln}: unknown query_id {d['query_id']}")
                if d["doc_id"] not in doc_ids:
                    self.err(f"relevance.jsonl L{ln}: unknown doc_id {d['doc_id']}")
                if d["relevance"] not in ALLOWED_RELEVANCE:
                    self.err(f"relevance.jsonl L{ln}: bad relevance {d['relevance']!r}")
                if d["relevance"] == 3:
                    threes_per_query[d["query_id"]] += 1
                total += 1

        for q in queries:
            if threes_per_query[q["query_id"]] < 1:
                self.err(f"query {q['query_id']!r} has no label-3 (highly relevant) doc")
        return total

    def summary(self, corpus: list[dict], queries: list[dict], n_labels: int) -> str:
        cat_counts = Counter(d["category"] for d in corpus)
        topic_counts = Counter((d["category"], d["topic"]) for d in corpus)
        type_counts = Counter(q["query_type"] for q in queries)
        split_counts = Counter(q["split"] for q in queries)
        hardneg_count = sum(1 for d in corpus if d["is_hard_negative"])

        lines = [
            "",
            f"Ontology: {len(CATEGORIES)} categories, {sum(len(v) for v in CATEGORIES.values())} topics",
            "",
            f"Corpus: {len(corpus)} docs ({hardneg_count} hard-negs)",
            f"  per category: min={min(cat_counts.values())} max={max(cat_counts.values())}",
            f"  per topic:    min={min(topic_counts.values())} max={max(topic_counts.values())}",
            "",
            f"Queries: {len(queries)} (train={split_counts['train']}, test={split_counts['test']})",
            f"  by query_type: " + ", ".join(f"{k}={v}" for k, v in sorted(type_counts.items())),
            "",
            f"Relevance labels: {n_labels}",
        ]
        return "\n".join(lines)

    def run(self) -> int:
        ontology = self.check_ontology()
        corpus = self.check_corpus()
        queries = self.check_queries()
        n_labels = self.check_relevance(corpus, queries) if (corpus and queries) else 0

        if self.errors:
            print(f"FAILED with {len(self.errors)} error(s):", file=sys.stderr)
            for e in self.errors[:20]:
                print(f"  - {e}", file=sys.stderr)
            if len(self.errors) > 20:
                print(f"  ... and {len(self.errors) - 20} more", file=sys.stderr)
            return 1

        print(self.summary(corpus, queries, n_labels))
        print("\nAll checks passed.")
        return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Validate the RankForge dataset.")
    p.add_argument("--dir", default="data", help="Dataset directory (default: data)")
    args = p.parse_args(argv)
    return Validator(Path(args.dir)).run()


if __name__ == "__main__":
    sys.exit(main())
