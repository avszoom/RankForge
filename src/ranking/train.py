"""Train the LightGBM LambdaRank model on the Phase 3 feature parquet.

Reads:
    data/features_train.parquet
    data/features_test.parquet
    data/features_meta.json

Writes:
    models/ranker.txt          — LightGBM text format (portable, human-readable trees)
    models/ranker.pkl          — joblib pickle of the lgb.Booster (faster to load)
    models/ranker_meta.json    — best_iteration, best NDCG@K, feature importances, params

Same FeatureBuilder used at training time will be used at inference (Phase 4 ranker).
The objective is lambdarank — pairwise gradients weighted by ΔNDCG, the standard
learning-to-rank objective in industry search.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import pandas as pd

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[2]
TRAIN_PATH = REPO / "data" / "features_train.parquet"
TEST_PATH = REPO / "data" / "features_test.parquet"
META_PATH = REPO / "data" / "features_meta.json"
MODELS_DIR = REPO / "models"
RANKER_PATH = MODELS_DIR / "ranker.pkl"
RANKER_TXT = MODELS_DIR / "ranker.txt"
RANKER_META = MODELS_DIR / "ranker_meta.json"

# Top-k positions to evaluate. Per user spec, we extend to 20 (LightGBM default flow surfaces top-20).
NDCG_AT = [5, 10, 20]


@dataclass(slots=True)
class TrainConfig:
    num_rounds: int = 1000
    learning_rate: float = 0.05
    num_leaves: int = 31
    min_data_in_leaf: int = 20
    feature_fraction: float = 0.9
    bagging_fraction: float = 0.8
    bagging_freq: int = 5
    lambdarank_truncation_level: int = 20    # only top-20 pairs contribute to gradients
    early_stopping_rounds: int = 50
    seed: int = 42


def _build_groups(df: pd.DataFrame) -> list[int]:
    """Sizes of consecutive query_id runs. Phase 3 already sorted by query_id, so this
    is just `groupby(sort=False).size()` and length matches the number of unique queries.
    """
    sizes = df.groupby("query_id", sort=False).size().tolist()
    assert sum(sizes) == len(df), "groups don't sum to row count — was the parquet shuffled?"
    return sizes


def _params(cfg: TrainConfig) -> dict[str, Any]:
    return {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": NDCG_AT,
        "lambdarank_truncation_level": cfg.lambdarank_truncation_level,
        "learning_rate": cfg.learning_rate,
        "num_leaves": cfg.num_leaves,
        "min_data_in_leaf": cfg.min_data_in_leaf,
        "feature_fraction": cfg.feature_fraction,
        "bagging_fraction": cfg.bagging_fraction,
        "bagging_freq": cfg.bagging_freq,
        "verbose": -1,
        "seed": cfg.seed,
        "deterministic": True,
    }


def train(
    cfg: TrainConfig | None = None,
    *,
    train_path: Path = TRAIN_PATH,
    test_path: Path = TEST_PATH,
    meta_path: Path = META_PATH,
) -> dict[str, Any]:
    cfg = cfg or TrainConfig()
    meta = json.loads(meta_path.read_text())
    feature_cols: list[str] = meta["feature_columns"]
    label_col: str = meta["label_column"]

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    X_train, y_train = train_df[feature_cols], train_df[label_col]
    X_test, y_test = test_df[feature_cols], test_df[label_col]
    groups_train = _build_groups(train_df)
    groups_test = _build_groups(test_df)

    print(
        f"Loaded {len(train_df):,} train rows from {len(groups_train)} queries; "
        f"{len(test_df):,} test rows from {len(groups_test)} queries.",
    )

    train_set = lgb.Dataset(X_train, label=y_train, group=groups_train, feature_name=feature_cols)
    valid_set = lgb.Dataset(X_test, label=y_test, group=groups_test,
                            reference=train_set, feature_name=feature_cols)

    params = _params(cfg)

    print(f"\nTraining LightGBM (objective=lambdarank, ndcg_eval_at={NDCG_AT})\n")

    t0 = time.time()
    booster = lgb.train(
        params,
        train_set,
        num_boost_round=cfg.num_rounds,
        valid_sets=[train_set, valid_set],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(cfg.early_stopping_rounds, verbose=True),
            lgb.log_evaluation(50),
        ],
    )
    elapsed = time.time() - t0

    best_iter = int(booster.best_iteration or booster.current_iteration())
    # `best_score` is keyed by valid-set name then by metric name like "ndcg@5"
    best_valid = {k: float(v) for k, v in booster.best_score.get("valid", {}).items()}
    best_train = {k: float(v) for k, v in booster.best_score.get("train", {}).items()}

    importances_gain = booster.feature_importance(importance_type="gain")
    importances_split = booster.feature_importance(importance_type="split")
    fi = sorted(
        [
            {
                "feature": f,
                "gain": float(g),
                "splits": int(s),
            }
            for f, g, s in zip(feature_cols, importances_gain, importances_split)
        ],
        key=lambda d: d["gain"],
        reverse=True,
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(RANKER_TXT), num_iteration=best_iter)
    joblib.dump(booster, RANKER_PATH)

    out = {
        "feature_columns": feature_cols,
        "label_column": label_col,
        "best_iteration": best_iter,
        "train_seconds": round(elapsed, 1),
        "ndcg_eval_at": NDCG_AT,
        "best_train_metric": best_train,
        "best_valid_metric": best_valid,
        "n_train_rows": int(len(train_df)),
        "n_test_rows": int(len(test_df)),
        "n_train_queries": len(groups_train),
        "n_test_queries": len(groups_test),
        "params": params,
        "feature_importance": fi,
    }
    RANKER_META.write_text(json.dumps(out, indent=2))

    print()
    print(f"Best iteration:     {best_iter}")
    for k, v in best_valid.items():
        print(f"  valid {k:<10}  {v:.4f}")
    for k, v in best_train.items():
        print(f"  train {k:<10}  {v:.4f}")
    print(f"\nTrained in {elapsed:.1f}s")
    print(f"Wrote {RANKER_PATH}")
    print(f"Wrote {RANKER_TXT}")
    print(f"Wrote {RANKER_META}")
    return out
