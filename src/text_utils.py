"""Shared text helpers used by both indexing and query-time.

Critical invariant: the SAME tokenize() function must run at index time and
query time. Tokenization mismatches are the #1 source of silent BM25 bugs.
"""
from __future__ import annotations

import re
import unicodedata

STOPWORDS: frozenset[str] = frozenset("""
a an the of to in for on with by is are was were be been being at as it its from this that these those
if while during about against between into through above below up down out off over under again further
then once here there when where why how all any both each few more most other some such no nor not only
own same so than too very can will just don dont should now you your yours i my mine we our ours
they them their what who which whom but and or also one two three new use used using how-to vs via per
""".split())

_WORD_RE = re.compile(r"[a-z0-9]+")


def clean_text(s: str) -> str:
    """Lowercase, normalize unicode, collapse whitespace."""
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", "ignore").decode("ascii")
    return " ".join(s.lower().split())


def tokenize(s: str) -> list[str]:
    """Lowercase regex tokenization with stopword filtering. Used by BM25 only."""
    cleaned = clean_text(s)
    return [t for t in _WORD_RE.findall(cleaned) if len(t) >= 2 and t not in STOPWORDS]
