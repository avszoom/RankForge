"""Train the LightGBM LambdaRank ranker on Phase 3's feature parquet.

Examples:
    python scripts/train_ranker.py
    python scripts/train_ranker.py --num-rounds 2000 --learning-rate 0.03
    python scripts/train_ranker.py --num-leaves 15 --min-data-in-leaf 50    # smaller, more reg
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def main() -> int:
    p = argparse.ArgumentParser(description="Train the LightGBM ranker.")
    p.add_argument("--num-rounds", type=int, default=1000)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--num-leaves", type=int, default=31)
    p.add_argument("--min-data-in-leaf", type=int, default=20)
    p.add_argument("--feature-fraction", type=float, default=0.9)
    p.add_argument("--bagging-fraction", type=float, default=0.8)
    p.add_argument("--bagging-freq", type=int, default=5)
    p.add_argument("--lambdarank-truncation-level", type=int, default=20)
    p.add_argument("--early-stopping-rounds", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from src.ranking.train import TrainConfig, train

    cfg = TrainConfig(
        num_rounds=args.num_rounds,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        min_data_in_leaf=args.min_data_in_leaf,
        feature_fraction=args.feature_fraction,
        bagging_fraction=args.bagging_fraction,
        bagging_freq=args.bagging_freq,
        lambdarank_truncation_level=args.lambdarank_truncation_level,
        early_stopping_rounds=args.early_stopping_rounds,
        seed=args.seed,
    )
    train(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
