#!/usr/bin/env python3
"""Train a side-win model over every logged price level and evaluate EV."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score


LABEL = "side_won"
NEVER_FEATURE_COLUMNS = {
    "ticker",
    "wall_ns",
    "wall_time_utc",
    "result_yes",
    "won",
    "side_won",
    "settlement_pnl_mD",
    "pnl_cents",
    "fee_cents",
    "logged_p_yes",
    "logged_ml_edge",
    "logged_meta_p_good",
    "logged_bucket_threshold",
    "logged_trade",
    "reason",
    "_is_clean",
    "_reject_reasons",
}
DEFAULT_CATEGORICAL_COLUMNS = ("asset", "side", "source", "quality_tier")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--train-frac", type=float, default=0.60)
    parser.add_argument("--tune-frac", type=float, default=0.20)
    parser.add_argument("--buy-fee-mD", type=float, default=10.0)
    parser.add_argument("--random-state", type=int, default=11)
    parser.add_argument("--n-estimators", type=int, default=700)
    parser.add_argument("--learning-rate", type=float, default=0.025)
    parser.add_argument("--num-leaves", type=int, default=15)
    parser.add_argument("--min-child-samples", type=int, default=200)
    parser.add_argument("--subsample", type=float, default=0.85)
    parser.add_argument("--colsample-bytree", type=float, default=0.85)
    parser.add_argument("--min-edge-mD", default="-20,0,10,15,20,30,40,50,75,100")
    parser.add_argument(
        "--include-feature-regex",
        help="Optional regex; when set, keep matching features plus core asset/side/level/tte/entry fields.",
    )
    return parser.parse_args()


def chronological_ticker_split(
    df: pd.DataFrame, train_frac: float, tune_frac: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    starts = (
        df.groupby("ticker", as_index=False)["wall_ns"]
        .min()
        .sort_values(["wall_ns", "ticker"])
        .reset_index(drop=True)
    )
    tickers = starts["ticker"].tolist()
    n = len(tickers)
    train_end = int(n * train_frac)
    tune_end = int(n * (train_frac + tune_frac))
    train_tickers = set(tickers[:train_end])
    tune_tickers = set(tickers[train_end:tune_end])
    test_tickers = set(tickers[tune_end:])
    return (
        df[df["ticker"].isin(train_tickers)].copy(),
        df[df["ticker"].isin(tune_tickers)].copy(),
        df[df["ticker"].isin(test_tickers)].copy(),
    )


def make_feature_frame(df: pd.DataFrame, include_feature_regex: str | None = None) -> tuple[pd.DataFrame, list[str]]:
    feature_cols = [col for col in df.columns if col not in NEVER_FEATURE_COLUMNS]
    if include_feature_regex:
        pattern = re.compile(include_feature_regex)
        keep_always = {"asset", "side", "level", "tte_s", "tte_bucket", "entry_price_mD"}
        feature_cols = [col for col in feature_cols if col in keep_always or pattern.search(col)]
    x = df[feature_cols].copy()
    categorical_cols = [col for col in DEFAULT_CATEGORICAL_COLUMNS if col in x.columns]
    numeric_cols = [col for col in x.columns if col not in categorical_cols]
    for col in numeric_cols:
        x[col] = pd.to_numeric(x[col], errors="coerce")
    x = pd.get_dummies(x, columns=categorical_cols, dummy_na=True)
    keep = [col for col in x.columns if not x[col].isna().all()]
    x = x[keep]
    return x, keep


def normalize_input(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if LABEL not in df.columns:
        if "won" not in df.columns:
            raise SystemExit(f"missing required label column: {LABEL} or won")
        df[LABEL] = pd.to_numeric(df["won"], errors="coerce")
    if "entry_price_mD" not in df.columns:
        if "entry_cost" not in df.columns:
            raise SystemExit("missing required entry price column: entry_price_mD or entry_cost")
        df["entry_price_mD"] = pd.to_numeric(df["entry_cost"], errors="coerce") * 10.0
    if "settlement_pnl_mD" not in df.columns:
        if "won" not in df.columns:
            raise SystemExit("missing settlement_pnl_mD and won; cannot compute gross settlement PnL")
        won = pd.to_numeric(df["won"], errors="coerce").fillna(0).astype(int)
        entry = pd.to_numeric(df["entry_price_mD"], errors="coerce")
        df["settlement_pnl_mD"] = np.where(won == 1, 1000.0 - entry, -entry)
    return df


def metric_or_nan(fn, y: pd.Series, p: np.ndarray) -> float:
    try:
        if y.nunique(dropna=False) < 2:
            return float("nan")
        return float(fn(y, p))
    except ValueError:
        return float("nan")


def split_summary(name: str, df: pd.DataFrame, pred: np.ndarray | None = None) -> dict:
    out = {
        "split": name,
        "rows": int(len(df)),
        "tickers": int(df["ticker"].nunique()) if len(df) else 0,
        "start_wall_ns": int(df["wall_ns"].min()) if len(df) else None,
        "end_wall_ns": int(df["wall_ns"].max()) if len(df) else None,
        "win_rate": float(df[LABEL].mean()) if len(df) else float("nan"),
        "raw_settlement_pnl_mD": float(df["settlement_pnl_mD"].sum()) if len(df) else 0.0,
    }
    if pred is not None:
        y = df[LABEL].astype(int)
        out["auc"] = metric_or_nan(roc_auc_score, y, pred)
        out["logloss"] = metric_or_nan(log_loss, y, pred)
        out["mean_pred"] = float(np.mean(pred)) if len(pred) else float("nan")
    return out


def max_drawdown(values: pd.Series) -> float:
    if values.empty:
        return 0.0
    curve = values.cumsum()
    return float((curve - curve.cummax()).min())


def evaluate_thresholds(
    split: str,
    df: pd.DataFrame,
    pred: np.ndarray,
    thresholds: list[float],
    buy_fee_mD: float,
) -> list[dict]:
    work = df.copy()
    work["pred_side_win"] = pred
    work["pred_edge_mD"] = 1000.0 * work["pred_side_win"] - work["entry_price_mD"] - buy_fee_mD
    work["net_pnl_mD"] = work["settlement_pnl_mD"] - buy_fee_mD
    rows: list[dict] = []
    for threshold in thresholds:
        chosen = work[work["pred_edge_mD"] >= threshold].sort_values(["wall_ns", "ticker", "side", "level"])
        rows.append(
            {
                "split": split,
                "min_edge_mD": threshold,
                "trades": int(len(chosen)),
                "tickers": int(chosen["ticker"].nunique()) if len(chosen) else 0,
                "win_rate": float(chosen[LABEL].mean()) if len(chosen) else float("nan"),
                "net_pnl_mD": float(chosen["net_pnl_mD"].sum()),
                "net_pnl_per_trade_mD": float(chosen["net_pnl_mD"].mean()) if len(chosen) else float("nan"),
                "max_drawdown_mD": max_drawdown(chosen["net_pnl_mD"]),
                "avg_entry_price_mD": float(chosen["entry_price_mD"].mean()) if len(chosen) else float("nan"),
                "avg_pred_edge_mD": float(chosen["pred_edge_mD"].mean()) if len(chosen) else float("nan"),
            }
        )
    return rows


def group_report(df: pd.DataFrame, pred: np.ndarray, buy_fee_mD: float) -> pd.DataFrame:
    work = df.copy()
    work["pred_side_win"] = pred
    work["pred_edge_mD"] = 1000.0 * work["pred_side_win"] - work["entry_price_mD"] - buy_fee_mD
    work["net_pnl_mD"] = work["settlement_pnl_mD"] - buy_fee_mD
    work["entry_cents"] = (work["entry_price_mD"] / 10.0).round().astype(int)
    rows = []
    group_cols = ["asset", "side", "entry_cents"]
    for keys, group in work.groupby(group_cols, dropna=False):
        rows.append(
            {
                "asset": keys[0],
                "side": keys[1],
                "entry_cents": int(keys[2]),
                "rows": int(len(group)),
                "win_rate": float(group[LABEL].mean()),
                "net_pnl_mD": float(group["net_pnl_mD"].sum()),
                "net_pnl_per_trade_mD": float(group["net_pnl_mD"].mean()),
                "avg_pred_side_win": float(group["pred_side_win"].mean()),
                "avg_pred_edge_mD": float(group["pred_edge_mD"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["asset", "side", "entry_cents"])


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = normalize_input(pd.read_csv(args.input, low_memory=False))
    required = {"ticker", "wall_ns", "entry_price_mD", LABEL, "settlement_pnl_mD"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"missing required columns: {missing}")
    df = df[df[LABEL].notna()].copy()
    df[LABEL] = pd.to_numeric(df[LABEL], errors="coerce").fillna(0).astype(int)
    df["wall_ns"] = pd.to_numeric(df["wall_ns"], errors="coerce")
    df["entry_price_mD"] = pd.to_numeric(df["entry_price_mD"], errors="coerce")
    df["settlement_pnl_mD"] = pd.to_numeric(df["settlement_pnl_mD"], errors="coerce")
    df = df[df["wall_ns"].notna() & df["entry_price_mD"].notna() & df["settlement_pnl_mD"].notna()].copy()

    train_df, tune_df, test_df = chronological_ticker_split(df, args.train_frac, args.tune_frac)
    full_x, feature_cols = make_feature_frame(df, args.include_feature_regex)
    x_train = full_x.loc[train_df.index]
    x_tune = full_x.loc[tune_df.index]
    x_test = full_x.loc[test_df.index]
    y_train = train_df[LABEL]
    y_tune = tune_df[LABEL]
    y_test = test_df[LABEL]

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        min_child_samples=args.min_child_samples,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_alpha=0.2,
        reg_lambda=2.0,
        random_state=args.random_state,
        n_jobs=4,
        verbosity=-1,
    )
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_tune, y_tune)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    pred_train = model.predict_proba(x_train)[:, 1]
    pred_tune = model.predict_proba(x_tune)[:, 1]
    pred_test = model.predict_proba(x_test)[:, 1]
    thresholds = [float(x) for x in args.min_edge_mD.split(",") if x.strip()]
    threshold_rows = []
    threshold_rows.extend(evaluate_thresholds("tune", tune_df, pred_tune, thresholds, args.buy_fee_mD))
    threshold_rows.extend(evaluate_thresholds("test", test_df, pred_test, thresholds, args.buy_fee_mD))
    pd.DataFrame(threshold_rows).to_csv(args.out_dir / "edge_threshold_report.csv", index=False)
    group_report(test_df, pred_test, args.buy_fee_mD).to_csv(args.out_dir / "test_price_level_report.csv", index=False)

    importances = pd.DataFrame(
        {
            "feature": feature_cols,
            "gain": model.booster_.feature_importance(importance_type="gain"),
            "split": model.booster_.feature_importance(importance_type="split"),
        }
    ).sort_values(["gain", "split"], ascending=False)
    importances.to_csv(args.out_dir / "feature_importance.csv", index=False)

    predictions = pd.concat(
        [
            train_df[["ticker", "asset", "wall_ns", "side", "level", "entry_price_mD", LABEL, "settlement_pnl_mD"]]
            .assign(split="train", pred_side_win=pred_train),
            tune_df[["ticker", "asset", "wall_ns", "side", "level", "entry_price_mD", LABEL, "settlement_pnl_mD"]]
            .assign(split="tune", pred_side_win=pred_tune),
            test_df[["ticker", "asset", "wall_ns", "side", "level", "entry_price_mD", LABEL, "settlement_pnl_mD"]]
            .assign(split="test", pred_side_win=pred_test),
        ],
        ignore_index=True,
    )
    predictions["pred_edge_mD"] = 1000.0 * predictions["pred_side_win"] - predictions["entry_price_mD"] - args.buy_fee_mD
    predictions.to_csv(args.out_dir / "predictions.csv", index=False)

    model.booster_.save_model(args.out_dir / "all_price_level_lgbm.txt")
    with (args.out_dir / "feature_columns.json").open("w", encoding="utf-8") as f:
        json.dump({"features": feature_cols, "label": LABEL, "buy_fee_mD": args.buy_fee_mD}, f, indent=2)

    summary = {
        "input": str(args.input),
        "rows": int(len(df)),
        "tickers": int(df["ticker"].nunique()),
        "feature_count": len(feature_cols),
        "label": LABEL,
        "buy_fee_mD": args.buy_fee_mD,
        "best_iteration": int(model.best_iteration_ or args.n_estimators),
        "splits": [
            split_summary("train", train_df, pred_train),
            split_summary("tune", tune_df, pred_tune),
            split_summary("test", test_df, pred_test),
        ],
        "params": model.get_params(),
        "include_feature_regex": args.include_feature_regex,
    }
    with (args.out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"out_dir={args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
