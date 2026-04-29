"""Build the (query, candidate) feature matrix for LightGBM ranker training.

Outputs:
    data/features_train.parquet
    data/features_test.parquet
    data/features_meta.json

Usage:
    python scripts/build_features.py
    python scripts/build_features.py --per-retriever 50 --top-k 100
    python scripts/build_features.py --limit 50    # debug: only first 50 queries
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def main() -> int:
    p = argparse.ArgumentParser(description="Build feature matrix for the ranker.")
    p.add_argument("--per-retriever", type=int, default=50,
                   help="Per-retriever top-k fed into the hybrid merger (default 50)")
    p.add_argument("--top-k", type=int, default=100,
                   help="Max merged candidates per query (default 100)")
    p.add_argument("--limit", type=int, default=None,
                   help="Debug: limit to the first N queries")
    p.add_argument("--out-train", default=str(REPO / "data" / "features_train.parquet"))
    p.add_argument("--out-test", default=str(REPO / "data" / "features_test.parquet"))
    p.add_argument("--out-meta", default=str(REPO / "data" / "features_meta.json"))
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from src.ranking.feature_builder import (
        ALL_COLUMNS, FEATURE_COLUMNS, ID_COLUMNS, LABEL_COLUMN,
        FeatureBuilder, coverage_stats, label_distribution,
        load_queries, split_dataframe,
    )

    queries = load_queries()
    if args.limit:
        queries = queries[: args.limit]
    n_train = sum(1 for q in queries if q.get("split") == "train")
    n_test = sum(1 for q in queries if q.get("split") == "test")
    print(f"Loaded {len(queries)} queries (train={n_train}, test={n_test})")

    fb = FeatureBuilder(per_retriever=args.per_retriever, top_k=args.top_k)
    t0 = time.time()
    df = fb.build_for_queries(queries)
    elapsed = time.time() - t0
    print(f"\nBuilt {len(df)} feature rows in {elapsed:.1f}s "
          f"({len(df) / max(len(queries), 1):.1f} rows/query)")

    train, test = split_dataframe(df)

    # ── Train summary ─────────────────────────────────────────────────────
    print(f"\nTrain: {len(train):>7,} rows from {train['query_id'].nunique()} queries")
    if len(train):
        ld = label_distribution(train)
        for k in sorted(ld):
            print(f"  label {k}: {ld[k]:>6,} ({ld[k]/len(train)*100:5.1f}%)")
        explicit, total = coverage_stats(train, fb)
        print(f"  explicit labels matched: {explicit:,} / {total:,} ({explicit/total*100:.1f}%)")

    # ── Test summary ──────────────────────────────────────────────────────
    print(f"\nTest:  {len(test):>7,} rows from {test['query_id'].nunique()} queries")
    if len(test):
        ld = label_distribution(test)
        for k in sorted(ld):
            print(f"  label {k}: {ld[k]:>6,} ({ld[k]/len(test)*100:5.1f}%)")
        explicit, total = coverage_stats(test, fb)
        print(f"  explicit labels matched: {explicit:,} / {total:,} ({explicit/total*100:.1f}%)")

    # ── Write outputs ─────────────────────────────────────────────────────
    Path(args.out_train).parent.mkdir(parents=True, exist_ok=True)

    train_out = train.drop(columns=["split"])
    test_out = test.drop(columns=["split"])
    train_out.to_parquet(args.out_train, compression="snappy", index=False)
    test_out.to_parquet(args.out_test, compression="snappy", index=False)

    meta = {
        "feature_columns": FEATURE_COLUMNS,
        "label_column": LABEL_COLUMN,
        "id_columns": ID_COLUMNS,
        "row_order": "query_id ASC, then bm25_score DESC, vector_score DESC",
        "n_train_rows": int(len(train)),
        "n_test_rows": int(len(test)),
        "n_train_queries": int(train["query_id"].nunique()) if len(train) else 0,
        "n_test_queries": int(test["query_id"].nunique()) if len(test) else 0,
        "retriever_config": {
            "per_retriever": args.per_retriever,
            "top_k": args.top_k,
        },
        "feature_builder_version": "0.1",
    }
    Path(args.out_meta).write_text(json.dumps(meta, indent=2))

    print(f"\nWrote {args.out_train}")
    print(f"Wrote {args.out_test}")
    print(f"Wrote {args.out_meta}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
