"""Generate 5 hard-negative documents per topic. 168 calls total.

Hard negatives are docs that look related to the topic at first glance but
aren't actually about it - the kind of thing a ranker has to learn to
demote. Same JSON-mode wrapping as the other generators; the prompt asks
for an array, so we wrap inside `{"docs": [...]}` to satisfy JSON-object mode.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from tqdm.asyncio import tqdm_asyncio

from .llm_client import chat_json
from .ontology import iter_topics, slugify
from ._cache import load_cached, save_cache

logger = logging.getLogger(__name__)

PHASE = "hard_negs"
NEGS_PER_TOPIC = 5

SYSTEM = (
    "You craft documents that look topically adjacent but are NOT about a "
    "given target topic - useful as ranking-system distractors. Output strict JSON only."
)

USER_TEMPLATE = """Given topic: {TOPIC}

Generate {N} documents that are:
semantically related but NOT directly about the topic
could confuse a search ranking system

Each document should have a realistic title and a 100-200 word body. Make them
distinct from each other.

Output JSON: {{"docs": [{{"title": "...", "body": "..."}}, ...]}}"""


async def _generate_one(
    sem: asyncio.Semaphore,
    category: str,
    topic: str,
    *,
    force: bool,
    model_override: str | None,
) -> dict[str, Any]:
    key = f"{slugify(category)}__{slugify(topic)}"
    if not force:
        cached = load_cached(PHASE, key)
        if cached is not None:
            return cached

    user = USER_TEMPLATE.format(TOPIC=topic, N=NEGS_PER_TOPIC)
    async with sem:
        data = await chat_json(
            system=SYSTEM,
            user=user,
            required_keys=["docs"],
            model_override=model_override,
            temperature=0.85,
            max_tokens=1500,
        )

    docs = data.get("docs")
    if not isinstance(docs, list) or len(docs) < NEGS_PER_TOPIC:
        raise ValueError(
            f"hard_negs returned {len(docs) if isinstance(docs, list) else 'non-list'} "
            f"for {topic}, expected {NEGS_PER_TOPIC}"
        )
    for i, d in enumerate(docs):
        if not isinstance(d, dict) or "title" not in d or "body" not in d:
            raise ValueError(f"hard_negs[{i}] malformed for {topic}: {d}")

    save_cache(PHASE, key, data)
    return data


async def generate_all_hard_negs(
    *,
    concurrency: int = 10,
    force: bool = False,
    model_override: str | None = None,
) -> int:
    sem = asyncio.Semaphore(concurrency)
    tasks = [
        _generate_one(sem, c, t, force=force, model_override=model_override)
        for c, t in iter_topics()
    ]

    cache_hits = sum(
        1 for c, t in iter_topics()
        if load_cached(PHASE, f"{slugify(c)}__{slugify(t)}") is not None
    ) if not force else 0

    logger.info(
        "generate_hard_negs: %d cached, %d to fetch", cache_hits, len(tasks) - cache_hits
    )
    await tqdm_asyncio.gather(*tasks, desc="hard_negs")
    return len(tasks) - cache_hits
