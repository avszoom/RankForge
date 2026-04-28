"""Generate 8 queries per topic. 168 calls total."""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from tqdm.asyncio import tqdm_asyncio

from .llm_client import chat_json
from .ontology import iter_topics, slugify
from ._cache import load_cached, save_cache

logger = logging.getLogger(__name__)

PHASE = "queries"

SYSTEM = (
    "You produce realistic search-engine queries that real users would type. "
    "Output strict JSON only."
)

USER_TEMPLATE = """Generate 8 realistic search queries for the topic: {TOPIC}.

Requirements:
Mix of:
beginner queries (simple)
advanced queries (technical)
vague queries (broad intent)
messy/noisy queries (like real users type)
Keep queries short (3-10 words)
Make them diverse (avoid repeating structure)

Return JSON: {{"queries": ["...", "..."]}}"""


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

    user = USER_TEMPLATE.format(TOPIC=topic)
    async with sem:
        data = await chat_json(
            system=SYSTEM,
            user=user,
            required_keys=["queries"],
            model_override=model_override,
            temperature=0.8,
            max_tokens=400,
        )

    if not isinstance(data.get("queries"), list) or not data["queries"]:
        raise ValueError(f"queries field empty/non-list for {topic}: {data}")

    save_cache(PHASE, key, data)
    return data


async def generate_all_queries(
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
        "generate_queries: %d cached, %d to fetch", cache_hits, len(tasks) - cache_hits
    )
    await tqdm_asyncio.gather(*tasks, desc="queries")
    return len(tasks) - cache_hits
