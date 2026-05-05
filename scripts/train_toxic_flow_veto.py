#!/usr/bin/env python3
"""Train and evaluate a time-aware toxic-flow veto model."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score


LABEL_DEFAULT = "label_bad_markout_30s"
NEVER_FEATURE_PREFIXES = (
    "same_bid_markout_",
    "microprice_markout_",
    "label_",
)
NEVER_FEATURE_SUBSTRINGS = (
    "target_hit_",
    "target_seconds_",
)
NEVER_FEATURE_COLUMNS = {
    "ticker",
    "wall_ns",
    "result_yes",
    "settlement_pnl_mD",
    "max_adverse_bid_change_30s_mD",
    "max_favorable_bid_change_30s_mD",
    "actual_filled",
    "fill_lag_exchange_s",
    "fill_fee_mD",
    "fill_is_taker",
    "reason",
}
CATEGORICAL_COLUMNS = ("asset", "side")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--label", default=LABEL_DEFAULT)
    parser.add_argument("--model", choices=("lightgbm", "xgboost"), default="lightgbm")
    parser.add_argument("--train-frac", type=float, default=0.60)
    parser.add_argument("--tune-frac", type=float, default=0.20)
    parser.add_argument("--random-state", type=int, default=7)
    parser.add_argument("--num-leaves", type=int, default=7)
    parser.add_argument("--learning-rate", type=float, default=0.035)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--min-child-samples", type=int, default=25)
    parser.add_argument("--subsample", type=float, default=0.90)
    parser.add_argument("--colsample-bytree", type=float, default=0.80)
    parser.add_argument(
        "--veto-quantiles",
        default="0.50,0.60,0.70,0.75,0.80,0.85,0.90",
        help="Quantiles of tune-set toxic probability to test as veto cutoffs.",
    )
    parser.add_argument(
        "--include-feature-regex",
        help="Optional regex; when set, only matching features are used, plus asset/side.",
    )
    return parser.parse_args()


def is_leakage_column(col: str) -> bool:
    if col in NEVER_FEATURE_COLUMNS:
        return True
    if any(col.startswith(prefix) for prefix in NEVER_FEATURE_PREFIXES):
        return True
    if any(part in col for part in NEVER_FEATURE_SUBSTRINGS):
        return True
    return False


def make_features(
    df: pd.DataFrame,
    label: str,
    model_name: str,
    include_feature_regex: str | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    feature_cols = [
        col
        for col in df.columns
        if col != label and not is_leakage_column(col)
    ]
    if include_feature_regex:
        pattern = re.compile(include_feature_regex)
        keep_always = {"asset", "side", "level", "tte_s", "tte_bucket", "entry_price_mD"}
        feature_cols = [col for col in feature_cols if col in keep_always or pattern.search(col)]
    x = df[feature_cols].copy()
    for col in CATEGORICAL_COLUMNS:
        if col in x.columns:
            x[col] = x[col].astype("category")
    for col in x.columns:
        if col in CATEGORICAL_COLUMNS:
            continue
        x[col] = pd.to_numeric(x[col], errors="coerce")
    if model_name == "xgboost":
        present_categoricals = [col for col in CATEGORICAL_COLUMNS if col in x.columns]
        x = pd.get_dummies(x, columns=present_categoricals, dummy_na=True)
        x = x.astype(float)
        return x, list(x.columns)
    return x, feature_cols


def build_model(args: argparse.Namespace):
    if args.model == "lightgbm":
        return lgb.LGBMClassifier(
            objective="binary",
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            num_leaves=args.num_leaves,
            min_child_samples=args.min_child_samples,
            subsample=args.subsample,
            colsample_bytree=args.colsample_bytree,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=args.random_state,
            n_jobs=4,
            verbosity=-1,
        )
    return xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=max(2, int(np.ceil(np.log2(args.num_leaves)))),
        min_child_weight=max(1, args.min_child_samples // 10),
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=args.random_state,
        n_jobs=4,
        tree_method="hist",
        verbosity=0,
    )


def chronological_split(
    df: pd.DataFrame, train_frac: float, tune_frac: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.sort_values(["wall_ns", "ticker", "side", "entry_price_mD"]).reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_frac)
    tune_end = int(n * (train_frac + tune_frac))
    return df.iloc[:train_end].copy(), df.iloc[train_end:tune_end].copy(), df.iloc[tune_end:].copy()


def metric_or_nan(fn, y: pd.Series, p: np.ndarray) -> float:
    try:
        if y.nunique(dropna=False) < 2:
            return float("nan")
        return float(fn(y, p))
    except ValueError:
        return float("nan")


def score_frame(name: str, df: pd.DataFrame, y: pd.Series, p: np.ndarray, label: str) -> dict:
    out = {
        "split": name,
        "rows": int(len(df)),
        "positive_rate": float(y.mean()) if len(y) else float("nan"),
        "auc": metric_or_nan(roc_auc_score, y, p),
        "average_precision": metric_or_nan(average_precision_score, y, p),
        "mean_pred": float(np.mean(p)) if len(p) else float("nan"),
    }
    if "settlement_pnl_mD" in df.columns:
        out["raw_settlement_pnl_mD"] = float(pd.to_numeric(df["settlement_pnl_mD"], errors="coerce").sum())
    if "max_adverse_bid_change_30s_mD" in df.columns:
        out["mean_max_adverse_30s_mD"] = float(
            pd.to_numeric(df["max_adverse_bid_change_30s_mD"], errors="coerce").mean()
        )
    return out


def evaluate_vetoes(
    split_name: str,
    df: pd.DataFrame,
    y: pd.Series,
    pred: np.ndarray,
    thresholds: list[float],
) -> list[dict]:
    rows: list[dict] = []
    pnl = pd.to_numeric(df.get("settlement_pnl_mD"), errors="coerce")
    adverse = pd.to_numeric(df.get("max_adverse_bid_change_30s_mD"), errors="coerce")
    for threshold in thresholds:
        keep = pred < threshold
        veto = ~keep
        kept_n = int(keep.sum())
        veto_n = int(veto.sum())
        row = {
            "split": split_name,
            "threshold": float(threshold),
            "rows": int(len(df)),
            "kept": kept_n,
            "vetoed": veto_n,
            "kept_rate": float(keep.mean()) if len(df) else float("nan"),
            "raw_bad_rate": float(y.mean()) if len(y) else float("nan"),
            "kept_bad_rate": float(y[keep].mean()) if kept_n else float("nan"),
            "veto_bad_rate": float(y[veto].mean()) if veto_n else float("nan"),
            "bad_avoided": int(y[veto].sum()) if veto_n else 0,
            "total_bad": int(y.sum()),
        }
        if pnl is not None:
            row["raw_settlement_pnl_mD"] = float(pnl.sum())
            row["kept_settlement_pnl_mD"] = float(pnl[keep].sum())
            row["vetoed_settlement_pnl_mD"] = float(pnl[veto].sum())
        if adverse is not None:
            row["raw_mean_adverse_30s_mD"] = float(adverse.mean())
            row["kept_mean_adverse_30s_mD"] = float(adverse[keep].mean()) if kept_n else float("nan")
            row["vetoed_mean_adverse_30s_mD"] = float(adverse[veto].mean()) if veto_n else float("nan")
        rows.append(row)
    return rows


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input)
    if args.label not in df.columns:
        raise SystemExit(f"missing label column: {args.label}")
    df = df[df[args.label].notna()].copy()
    df[args.label] = pd.to_numeric(df[args.label], errors="coerce").fillna(0).astype(int)
    train_df, tune_df, test_df = chronological_split(df, args.train_frac, args.tune_frac)

    full_x, feature_cols = make_features(df, args.label, args.model, args.include_feature_regex)
    x_train = full_x.loc[train_df.index]
    x_tune = full_x.loc[tune_df.index]
    x_test = full_x.loc[test_df.index]
    y_train = train_df[args.label]
    y_tune = tune_df[args.label]
    y_test = test_df[args.label]

    model = build_model(args)
    eval_set = [(x_tune, y_tune)] if len(tune_df) and y_tune.nunique() > 1 else None
    if eval_set and args.model == "lightgbm":
        model.fit(
            x_train,
            y_train,
            eval_set=eval_set,
            eval_metric="auc",
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )
    else:
        model.fit(x_train, y_train)

    pred_train = model.predict_proba(x_train)[:, 1]
    pred_tune = model.predict_proba(x_tune)[:, 1]
    pred_test = model.predict_proba(x_test)[:, 1]

    summary = {
        "input": str(args.input),
        "label": args.label,
        "model": args.model,
        "feature_count": len(feature_cols),
        "rows": int(len(df)),
        "splits": {
            "train": {
                "rows": int(len(train_df)),
                "start_wall_ns": int(train_df["wall_ns"].min()) if len(train_df) else None,
                "end_wall_ns": int(train_df["wall_ns"].max()) if len(train_df) else None,
            },
            "tune": {
                "rows": int(len(tune_df)),
                "start_wall_ns": int(tune_df["wall_ns"].min()) if len(tune_df) else None,
                "end_wall_ns": int(tune_df["wall_ns"].max()) if len(tune_df) else None,
            },
            "test": {
                "rows": int(len(test_df)),
                "start_wall_ns": int(test_df["wall_ns"].min()) if len(test_df) else None,
                "end_wall_ns": int(test_df["wall_ns"].max()) if len(test_df) else None,
            },
        },
        "params": model.get_params(),
        "include_feature_regex": args.include_feature_regex,
        "best_iteration": getattr(model, "best_iteration_", None),
        "metrics": [
            score_frame("train", train_df, y_train, pred_train, args.label),
            score_frame("tune", tune_df, y_tune, pred_tune, args.label),
            score_frame("test", test_df, y_test, pred_test, args.label),
        ],
    }

    quantiles = [float(x) for x in args.veto_quantiles.split(",") if x.strip()]
    thresholds = [float(np.quantile(pred_tune, q)) for q in quantiles] if len(pred_tune) else []
    summary["veto_thresholds_from_tune_quantiles"] = dict(zip(map(str, quantiles), thresholds))

    veto_rows = []
    veto_rows.extend(evaluate_vetoes("tune", tune_df, y_tune, pred_tune, thresholds))
    veto_rows.extend(evaluate_vetoes("test", test_df, y_test, pred_test, thresholds))
    pd.DataFrame(veto_rows).to_csv(args.out_dir / "veto_threshold_report.csv", index=False)

    if args.model == "lightgbm":
        importances = pd.DataFrame(
            {
                "feature": feature_cols,
                "gain": model.booster_.feature_importance(importance_type="gain"),
                "split": model.booster_.feature_importance(importance_type="split"),
            }
        ).sort_values(["gain", "split"], ascending=False)
    else:
        booster = model.get_booster()
        gain_scores = booster.get_score(importance_type="gain")
        weight_scores = booster.get_score(importance_type="weight")
        importances = pd.DataFrame(
            {
                "feature": feature_cols,
                "gain": [gain_scores.get(f, 0.0) for f in feature_cols],
                "split": [weight_scores.get(f, 0.0) for f in feature_cols],
            }
        ).sort_values(["gain", "split"], ascending=False)
    importances.to_csv(args.out_dir / "feature_importance.csv", index=False)

    pred_rows = pd.concat(
        [
            train_df[["ticker", "asset", "wall_ns", "side", args.label, "settlement_pnl_mD"]].assign(
                split="train", toxic_pred=pred_train
            ),
            tune_df[["ticker", "asset", "wall_ns", "side", args.label, "settlement_pnl_mD"]].assign(
                split="tune", toxic_pred=pred_tune
            ),
            test_df[["ticker", "asset", "wall_ns", "side", args.label, "settlement_pnl_mD"]].assign(
                split="test", toxic_pred=pred_test
            ),
        ],
        ignore_index=True,
    )
    pred_rows.to_csv(args.out_dir / "predictions.csv", index=False)

    if args.model == "lightgbm":
        model.booster_.save_model(args.out_dir / "toxic_flow_lgbm.txt")
    else:
        model.save_model(args.out_dir / "toxic_flow_xgboost.json")
    with (args.out_dir / "feature_columns.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "features": feature_cols,
                "categorical": [] if args.model == "xgboost" else list(CATEGORICAL_COLUMNS),
                "model": args.model,
            },
            f,
            indent=2,
        )
    with (args.out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary["metrics"], indent=2))
    print(f"feature_count={len(feature_cols)}")
    print(f"out_dir={args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
