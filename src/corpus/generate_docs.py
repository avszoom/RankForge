"""Generate 10 documents per (category, topic) via OpenAI. 1680 calls total.

Prompt is the user-supplied "Document Generation Prompt" verbatim, with a small
per-call style hint appended so the 10 docs in the same topic don't collapse to
near-duplicates. Each call is cached to data/_raw/docs/{cat_slug}__{topic_slug}__{i}.json
so reruns are free and the pipeline is resumable.
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

DOCS_PER_TOPIC = 10
PHASE = "docs"

# Style hints cycled to nudge diversity across the 10 docs of a single topic.
# Each hint is a single short noun phrase; appended as "Angle: {hint}" so it
# steers the model without overriding the user's verbatim prompt.
STYLE_HINTS = [
    "implementation details",
    "common pitfalls",
    "case study perspective",
    "metrics and evaluation",
    "team practices",
    "quickstart guide",
    "trade-off analysis",
    "before/after example",
    "tooling landscape",
    "real-world incident lessons",
]

SYSTEM = (
    "You are a senior practitioner who writes clear, candid technical and "
    "professional articles for working engineers, founders, investors, or "
    "enthusiasts (depending on the domain). Output strict JSON only."
)

USER_TEMPLATE = """Write a realistic, high-quality article about the topic: {TOPIC} in the domain of {CATEGORY}.

Requirements:
Length: 150-300 words
Tone: natural, like a blog post or engineering article (not textbook, not bullet points)
Include:
what the concept is
why it matters
practical examples or real-world usage
tradeoffs or limitations

Avoid:
repetition
generic filler phrases
obvious templating

Make it feel like it was written by a practitioner or engineer.

Angle for this variant: {HINT}

Output JSON: {{"title": "...", "body": "..."}}"""


async def _generate_one(
    sem: asyncio.Semaphore,
    category: str,
    topic: str,
    idx: int,
    *,
    force: bool,
    model_override: str | None,
) -> tuple[str, str, int, dict[str, Any]]:
    key = f"{slugify(category)}__{slugify(topic)}__{idx:02d}"
    if not force:
        cached = load_cached(PHASE, key)
        if cached is not None:
            return category, topic, idx, cached

    hint = STYLE_HINTS[idx % len(STYLE_HINTS)]
    user = USER_TEMPLATE.format(TOPIC=topic, CATEGORY=category, HINT=hint)

    async with sem:
        data = await chat_json(
            system=SYSTEM,
            user=user,
            required_keys=["title", "body"],
            model_override=model_override,
            temperature=0.9,
            max_tokens=900,
        )

    save_cache(PHASE, key, data)
    return category, topic, idx, data


async def generate_all_docs(
    *,
    concurrency: int = 10,
    force: bool = False,
    model_override: str | None = None,
) -> int:
    """Generate (or load from cache) all 1680 docs. Returns the call count made (excludes cache hits)."""
    sem = asyncio.Semaphore(concurrency)
    tasks = []
    for category, topic in iter_topics():
        for i in range(DOCS_PER_TOPIC):
            tasks.append(_generate_one(sem, category, topic, i, force=force, model_override=model_override))

    cache_hits = 0
    if not force:
        # Pre-count cache hits without actually skipping the gather (the function checks anyway)
        for category, topic in iter_topics():
            for i in range(DOCS_PER_TOPIC):
                key = f"{slugify(category)}__{slugify(topic)}__{i:02d}"
                if load_cached(PHASE, key) is not None:
                    cache_hits += 1

    logger.info(
        "generate_docs: %d cached, %d to fetch (concurrency=%d)",
        cache_hits, len(tasks) - cache_hits, concurrency,
    )

    await tqdm_asyncio.gather(*tasks, desc="docs")
    return len(tasks) - cache_hits
