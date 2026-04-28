"""Tiny text helpers shared by builders. No external deps."""
from __future__ import annotations

import re
from collections import Counter

_STOPWORDS = frozenset("""
a an the of to in for on with by is are was were be been being at as it its from this that these those
if while during about against between into through above below up down out off over under again further
then once here there when where why how all any both each few more most other some such no nor not only
own same so than too very can will just don dont don't should now you your yours i my mine we our ours
they them their what who which whom but and or also one two three new use used using how-to vs via per
""".split())


def extract_keywords(title: str, max_n: int = 5) -> list[str]:
    """Extract up-to-N lowercase keywords from a title (cheap, deterministic)."""
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9\-]*", title.lower())
    filtered = [t for t in tokens if len(t) >= 3 and t not in _STOPWORDS]
    if not filtered:
        return []
    counts = Counter(filtered)
    return [w for w, _ in counts.most_common(max_n)]


def topic_tokens(topic: str) -> set[str]:
    """Significant lowercase tokens from a topic name (used for query_type heuristic)."""
    return {
        t for t in re.findall(r"[a-zA-Z]+", topic.lower())
        if len(t) >= 3 and t not in _STOPWORDS
    }
