"""Evaluate all 4 setups (BM25, FAISS, Hybrid+RRF, LightGBM) on the held-out test queries.

Outputs:
  - prints overall metric table (rows = setups)
  - prints per-query-type breakdown (rows = (query_type, setup))
  - writes data/eval_results.parquet  (one row per query × setup; 129 × 4 = 516 rows)
  - writes data/eval_summary.json     (aggregate tables, easy to grep)

Examples:
    python scripts/evaluate.py
    python scripts/evaluate.py --top-k 100
    python scripts/evaluate.py --split train     # debug on training queries
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _print_table(df, title: str, fmt: str = "{:.4f}") -> None:
    print(f"\n{title}\n" + "=" * len(title))
    pd_style = df.copy()
    for c in pd_style.select_dtypes("float").columns:
        pd_style[c] = pd_style[c].map(lambda v: fmt.format(v) if v == v else "—")
    # Render with no truncation
    import pandas as pd
    with pd.option_context("display.max_rows", None, "display.max_columns", None,
                            "display.width", 200, "display.colheader_justify", "right"):
        print(pd_style.to_string())


def main() -> int:
    p = argparse.ArgumentParser(description="Run the Phase 5 evaluation.")
    p.add_argument("--top-k", type=int, default=50,
                   help="Deepest top-K to fetch per setup (default 50; needed for recall@50)")
    p.add_argument("--split", default="test", choices=["test", "train", "all"],
                   help="Which queries to evaluate (default: test)")
    p.add_argument("--out-rows", default=str(REPO / "data" / "eval_results.parquet"))
    p.add_argument("--out-summary", default=str(REPO / "data" / "eval_summary.json"))
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from src.evaluation.evaluate import (
        aggregate_by_query_type, aggregate_overall, evaluate,
    )

    only_split = None if args.split == "all" else args.split
    print(f"Running evaluation: split={only_split or 'all'}, top_k={args.top_k}")
    df = evaluate(only_split=only_split, top_k=args.top_k)

    overall = aggregate_overall(df)
    by_type = aggregate_by_query_type(df)

    _print_table(overall, "Overall metrics (mean across queries)")
    _print_table(by_type, "Per query_type breakdown")

    # Persist
    Path(args.out_rows).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out_rows, compression="snappy", index=False)

    summary = {
        "split": args.split,
        "top_k": args.top_k,
        "n_queries": int(df["query_id"].nunique()),
        "overall": {setup: row.to_dict() for setup, row in overall.iterrows()},
        "by_query_type": {
            f"{qt}/{setup}": row.to_dict()
            for (qt, setup), row in by_type.iterrows()
        },
    }
    Path(args.out_summary).write_text(json.dumps(summary, indent=2))

    print(f"\nWrote {args.out_rows}")
    print(f"Wrote {args.out_summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
