"""Compare retrieval / ranking pipelines for a single query.

Default mode: 2-way (Hybrid RRF vs LightGBM Ranker).
With --with-ce flag: 3-way (Hybrid → LightGBM → CrossEncoder).

Examples:
    python scripts/compare.py "how to reduce LLM inference cost"
    python scripts/compare.py "ways to make AI API cheaper" --top-k 10
    python scripts/compare.py "consensus algorithms" --body-chars 250

    # 3-way: see how CE re-orders LightGBM's top-20
    python scripts/compare.py "ways to make AI API cheaper" --with-ce --top-k 10
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _load_corpus_map(corpus_path: Path) -> dict[str, dict]:
    return {
        json.loads(line)["doc_id"]: json.loads(line)
        for line in corpus_path.read_text().splitlines() if line.strip()
    }


def _excerpt(body: str, n_chars: int) -> str:
    """First N chars of the body, ending on a word boundary, with leading whitespace stripped."""
    body = body.strip().replace("\n", " ")
    if len(body) <= n_chars:
        return body
    cut = body[:n_chars].rsplit(" ", 1)[0]
    return cut + "…"


def _movement(r_rank: int, h_rank: int | None) -> str:
    """Describe how a doc moved from hybrid rank to ranker rank.

    Returns a short tag like '↑3', '↓2', '=' or '✱' (new — wasn't in hybrid top-K).
    """
    if h_rank is None:
        return "✱"  # new — was outside the hybrid top-K we displayed
    delta = h_rank - r_rank
    if delta == 0:
        return "="
    if delta > 0:
        return f"↑{delta}"
    return f"↓{-delta}"


def _print_row(r_rank, h_rank_str, mv, doc_id, category, title, body_excerpt, ranker_score=None):
    score_str = f"  {ranker_score:>6.3f}" if ranker_score is not None else ""
    title_short = textwrap.shorten(title, width=60, placeholder="…")
    cat_short = (category or "")[:26]
    print(f"  {r_rank:>3}  {h_rank_str:>3}  {mv:<5}{score_str}  {doc_id:<10}  {cat_short:<26}  {title_short}")
    if body_excerpt:
        print(f"        │ {body_excerpt}")
        print()


def _run_three_way(args, corpus: dict[str, dict]) -> int:
    """3-way: Hybrid (RRF) → LightGBM Ranker → CrossEncoder, top-K."""
    from src.retrieval.hybrid_retriever import HybridRetriever
    from src.ranking.two_stage import TwoStageRanker

    hybrid_pool = max(args.top_k * 5, args.retriever_top_k)
    hybrid = HybridRetriever().search(
        args.query, top_k=hybrid_pool, per_retriever=args.per_retriever,
    )
    h_rank_by_id: dict[str, int] = {c.doc_id: i + 1 for i, c in enumerate(hybrid)}

    two_stage = TwoStageRanker(stage1_top_k=20)
    items = two_stage.rank(
        args.query,
        top_k=args.top_k,
        per_retriever=args.per_retriever,
        retriever_top_k=args.retriever_top_k,
    )

    print(f"=== Hybrid (RRF) → LightGBM Ranker → CrossEncoder, top {args.top_k} ===\n")
    print(f"  {'CE':>3}  {'L':>3}  {'H':>3}  {'Δ_L':<5}  {'Δ_H':<5}  "
          f"{'ce_score':>9}  {'lgbm':>6}  {'doc_id':<10}  {'category':<24}  title")
    print("  " + "  ".join([
        "-" * 3, "-" * 3, "-" * 3, "-" * 5, "-" * 5,
        "-" * 9, "-" * 6, "-" * 10, "-" * 24, "-" * 40,
    ]))

    for ce_rank, item in enumerate(items, 1):
        doc_id = item.candidate.doc_id
        d = corpus.get(doc_id, {})
        h = h_rank_by_id.get(doc_id)
        title = d.get("title", "") + (" [HN]" if d.get("is_hard_negative") else "")
        title_short = textwrap.shorten(title, width=55, placeholder="…")
        cat_short = (d.get("category", "") or "")[:24]

        delta_l = item.lgbm_rank - ce_rank   # positive = CE promoted vs LightGBM
        mv_l = "=" if delta_l == 0 else (f"↑{delta_l}" if delta_l > 0 else f"↓{-delta_l}")
        if h is None:
            mv_h = "✱"
        else:
            delta_h = h - ce_rank
            mv_h = "=" if delta_h == 0 else (f"↑{delta_h}" if delta_h > 0 else f"↓{-delta_h}")

        h_str = str(h) if h is not None else "—"
        print(f"  {ce_rank:>3}  {item.lgbm_rank:>3}  {h_str:>3}  {mv_l:<5}  {mv_h:<5}  "
              f"{item.ce_score:>9.3f}  {item.lgbm_score:>6.3f}  "
              f"{doc_id:<10}  {cat_short:<24}  {title_short}")
        if args.body_chars:
            body = _excerpt(d.get("body", ""), args.body_chars)
            print(f"        │ {body}\n")

    print()
    print("  legend: CE=cross-encoder rank, L=LightGBM rank, H=hybrid rank")
    print("  Δ_L = how CE moved this doc from LightGBM's order  (↑n promoted, ↓n demoted)")
    print("  Δ_H = total movement from hybrid order  (✱ = was outside hybrid pool)")
    print()
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Compare hybrid retrieval vs trained ranker on one query.")
    p.add_argument("query", help="The search query.")
    p.add_argument("--top-k", type=int, default=20,
                   help="How many docs to display from each side (default 20)")
    p.add_argument("--per-retriever", type=int, default=50,
                   help="Per-retriever fetch size for hybrid/ranker (default 50)")
    p.add_argument("--retriever-top-k", type=int, default=100,
                   help="Hybrid candidate-pool size before ranker scores (default 100)")
    p.add_argument("--body-chars", type=int, default=200,
                   help="Number of body chars to include per doc (default 200; 0 to disable)")
    p.add_argument("--with-ce", action="store_true",
                   help="3-way compare: Hybrid → LightGBM → CrossEncoder rerank")
    p.add_argument("--corpus", default=str(REPO / "data" / "corpus.jsonl"))
    args = p.parse_args()

    corpus = _load_corpus_map(Path(args.corpus))
    print(f"\nQuery: {args.query!r}\n")

    if args.with_ce:
        return _run_three_way(args, corpus)

    # Run both pipelines on the same retrieved candidate set.
    from src.retrieval.hybrid_retriever import HybridRetriever
    from src.ranking.ranker import LightGBMRanker

    # Use a deeper hybrid pool so we can compute meaningful movement arrows
    # for ranker docs that were ranked below `top_k` by hybrid alone.
    hybrid_pool = max(args.top_k * 3, args.retriever_top_k)
    hybrid = HybridRetriever().search(
        args.query, top_k=hybrid_pool, per_retriever=args.per_retriever,
    )
    ranker = LightGBMRanker()
    ranked = ranker.rank(
        args.query,
        top_k=args.top_k,
        per_retriever=args.per_retriever,
        retriever_top_k=args.retriever_top_k,
    )

    # Build hybrid rank lookup. Capped at `hybrid_pool` for the "✱ new" semantics
    # (a ranker doc beyond hybrid_pool is genuinely "new").
    h_rank_by_id: dict[str, int] = {c.doc_id: i + 1 for i, c in enumerate(hybrid)}
    hybrid_top_k_ids: set[str] = {c.doc_id for c in hybrid[: args.top_k]}

    # ── 1. ranker top-K with hybrid annotations ────────────────────────────
    print(f"=== Hybrid (RRF) → LightGBM Ranker, top {args.top_k} ===\n")
    print(f"  {'R':>3}  {'H':>3}  {'Δ':<5}  {'score':>6}  {'doc_id':<10}  {'category':<26}  title")
    print(f"  {'-'*3}  {'-'*3}  {'-'*5}  {'-'*6}  {'-'*10}  {'-'*26}  {'-'*40}")
    for i, (cand, score) in enumerate(ranked, 1):
        d = corpus.get(cand.doc_id, {})
        h = h_rank_by_id.get(cand.doc_id)
        title = d.get("title", "") + (" [HN]" if d.get("is_hard_negative") else "")
        body = _excerpt(d.get("body", ""), args.body_chars) if args.body_chars else ""
        _print_row(
            i,
            str(h) if h is not None else "—",
            _movement(i, h),
            cand.doc_id,
            d.get("category", ""),
            title,
            body,
            ranker_score=float(score),
        )

    # ── 2. docs the ranker pushed OUT of the top-K ─────────────────────────
    ranker_top_k_ids = {c.doc_id for c, _ in ranked}
    pushed_out = [
        (i + 1, c) for i, c in enumerate(hybrid[: args.top_k])
        if c.doc_id not in ranker_top_k_ids
    ]
    if pushed_out:
        print(f"\n=== Pushed out by ranker  (was in hybrid top-{args.top_k}, not in ranker top-{args.top_k}) ===\n")
        print(f"  {'H':>3}  {'doc_id':<10}  {'category':<26}  title")
        print(f"  {'-'*3}  {'-'*10}  {'-'*26}  {'-'*40}")
        for h_rank, c in pushed_out:
            d = corpus.get(c.doc_id, {})
            title = d.get("title", "") + (" [HN]" if d.get("is_hard_negative") else "")
            title_short = textwrap.shorten(title, width=60, placeholder="…")
            cat_short = d.get("category", "")[:26]
            print(f"  {h_rank:>3}  {c.doc_id:<10}  {cat_short:<26}  {title_short}")
            if args.body_chars:
                body = _excerpt(d.get("body", ""), args.body_chars)
                print(f"        │ {body}\n")

    # ── 3. summary ────────────────────────────────────────────────────────
    n_promoted = sum(1 for i, (c, _) in enumerate(ranked, 1)
                     if (h := h_rank_by_id.get(c.doc_id)) is not None and h > i)
    n_demoted = sum(1 for i, (c, _) in enumerate(ranked, 1)
                    if (h := h_rank_by_id.get(c.doc_id)) is not None and h < i)
    n_unchanged = sum(1 for i, (c, _) in enumerate(ranked, 1)
                      if h_rank_by_id.get(c.doc_id) == i)
    n_new = sum(1 for c, _ in ranked if c.doc_id not in hybrid_top_k_ids)

    print()
    print(f"  promoted: {n_promoted:>2}   demoted: {n_demoted:>2}   "
          f"unchanged: {n_unchanged:>2}   new (was outside hybrid top-{args.top_k}): {n_new:>2}")
    print()
    print("  legend: R=ranker rank, H=hybrid rank, Δ=position change "
          "(↑n promoted, ↓n demoted, = same, ✱ new)")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
