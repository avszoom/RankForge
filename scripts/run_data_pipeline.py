"""Orchestrate the RankForge data generation pipeline.

Phases (default --phase=all runs them in order):
  1. ontology      → write data/ontology.json from canonical Python source
  2. docs          → 1680 OpenAI calls (cached in data/_raw/docs)
  3. queries       → 168 OpenAI calls
  4. hard_negs     → 168 OpenAI calls
  5. assemble      → build corpus.jsonl, queries.jsonl, relevance.jsonl from cache
  6. validate      → run validate_dataset

Use --dry-run to print what would be called without spending tokens.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Make `src.corpus` importable when running this script from the repo root.
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.corpus.ontology import CATEGORIES, iter_topics, slugify  # noqa: E402
from src.corpus._cache import load_cached  # noqa: E402

PHASES = ("ontology", "docs", "queries", "hard_negs", "assemble", "validate", "all")


def _count_pending() -> dict[str, int]:
    pending = {"docs": 0, "queries": 0, "hard_negs": 0}
    for cat, topic in iter_topics():
        cs, ts = slugify(cat), slugify(topic)
        # docs (10 per topic)
        for i in range(10):
            if load_cached("docs", f"{cs}__{ts}__{i:02d}") is None:
                pending["docs"] += 1
        if load_cached("queries", f"{cs}__{ts}") is None:
            pending["queries"] += 1
        if load_cached("hard_negs", f"{cs}__{ts}") is None:
            pending["hard_negs"] += 1
    return pending


def _print_dry_run() -> None:
    p = _count_pending()
    total_topics = sum(len(v) for v in CATEGORIES.values())
    print("=== DRY RUN ===")
    print(f"Categories: {len(CATEGORIES)}, Topics: {total_topics}")
    print(f"Pending API calls (after cache):")
    print(f"  docs:       {p['docs']:>5} / {total_topics * 10}")
    print(f"  queries:    {p['queries']:>5} / {total_topics}")
    print(f"  hard_negs:  {p['hard_negs']:>5} / {total_topics}")
    total = sum(p.values())
    print(f"  TOTAL:      {total:>5}")
    # rough cost estimate at gpt-4o-mini pricing
    est = (
        p['docs'] * 0.0003
        + p['queries'] * 0.0001
        + p['hard_negs'] * 0.0008
    )
    print(f"Est. cost on gpt-4o-mini: ~${est:.2f}")


async def _run_generation(args: argparse.Namespace) -> None:
    from src.corpus.generate_docs import generate_all_docs
    from src.corpus.generate_queries import generate_all_queries
    from src.corpus.generate_hard_negatives import generate_all_hard_negs

    coros = []
    if args.phase in ("docs", "all"):
        coros.append(generate_all_docs(
            concurrency=args.concurrency, force=args.force, model_override=args.model,
        ))
    if args.phase in ("queries", "all"):
        coros.append(generate_all_queries(
            concurrency=args.concurrency, force=args.force, model_override=args.model,
        ))
    if args.phase in ("hard_negs", "all"):
        coros.append(generate_all_hard_negs(
            concurrency=args.concurrency, force=args.force, model_override=args.model,
        ))
    if coros:
        await asyncio.gather(*coros)


def _run_assemble() -> None:
    from src.corpus.build_corpus import build_corpus, write_ontology_json
    from src.corpus.build_queries_file import build_queries_file
    from src.corpus.build_relevance import build_relevance

    write_ontology_json()
    build_corpus()
    build_queries_file()
    build_relevance()


def _run_validate() -> int:
    from src.corpus.validate_dataset import main as validate_main
    return validate_main(["--dir", "data"])


def main() -> int:
    p = argparse.ArgumentParser(description="RankForge data pipeline orchestrator.")
    p.add_argument("--phase", default="all", choices=PHASES,
                   help="Run a single phase or 'all' (default: all)")
    p.add_argument("--concurrency", type=int, default=10, help="Async concurrency cap (default: 10)")
    p.add_argument("--force", action="store_true", help="Ignore cached responses, regenerate everything")
    p.add_argument("--model", default=None, help="OpenAI model override (default: env OPENAI_MODEL or gpt-4o-mini)")
    p.add_argument("--dry-run", action="store_true", help="Print pending call counts and exit")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.dry_run:
        _print_dry_run()
        return 0

    # ontology phase
    if args.phase in ("ontology", "all"):
        from src.corpus.build_corpus import write_ontology_json
        write_ontology_json()

    # generation phases
    if args.phase in ("docs", "queries", "hard_negs", "all"):
        asyncio.run(_run_generation(args))

    if args.phase in ("assemble", "all"):
        _run_assemble()

    if args.phase in ("validate", "all"):
        rc = _run_validate()
        if rc != 0:
            return rc

    return 0


if __name__ == "__main__":
    sys.exit(main())
