#!/usr/bin/env python3
"""Train base-only and base-plus challenger LightGBM models on candidate rows."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


LABEL = "result_yes"
LABEL_AND_AUDIT = {
    "ticker",
    "wall_ns",
    "wall_time_utc",
    "result_yes",
    "won",
    "pnl_cents",
    "fee_cents",
    "_is_clean",
    "_reject_reasons",
}
CATEGORICAL_COLUMNS = ["asset", "source", "quality_tier", "side"]
NEW_PREFIXES = ("plv_", "depth_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--tune-frac", type=float, default=0.15)
    parser.add_argument("--edge-thresholds", default="-0.02,0,0.01,0.02,0.03,0.04,0.05,0.075,0.10")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--per-asset",
        action="store_true",
        help="Train separate model variants for each asset instead of one global model.",
    )
    parser.add_argument(
        "--assets",
        default="",
        help="Optional comma-separated asset allowlist, e.g. BTC,ETH. Defaults to all assets in input.",
    )
    parser.add_argument(
        "--read-chunksize",
        type=int,
        default=50000,
        help="Rows per pandas chunk when filtering large per-asset inputs.",
    )
    parser.add_argument(
        "--skip-predictions",
        action="store_true",
        help="Do not write full predictions.csv files. Useful on low-disk machines.",
    )
    return parser.parse_args()


def split_by_ticker_time(df: pd.DataFrame, train_frac: float, tune_frac: float):
    ticker_times = df.groupby("ticker", as_index=False)["wall_ns"].min().sort_values(["wall_ns", "ticker"])
    n = len(ticker_times)
    train_n = max(1, int(n * train_frac))
    tune_n = max(1, int(n * tune_frac))
    train_tickers = set(ticker_times.iloc[:train_n]["ticker"])
    tune_tickers = set(ticker_times.iloc[train_n : train_n + tune_n]["ticker"])
    test_tickers = set(ticker_times.iloc[train_n + tune_n :]["ticker"])
    return (
        df[df["ticker"].isin(train_tickers)].copy(),
        df[df["ticker"].isin(tune_tickers)].copy(),
        df[df["ticker"].isin(test_tickers)].copy(),
    )


def feature_columns(df: pd.DataFrame, include_new: bool) -> list[str]:
    cols: list[str] = []
    for col in df.columns:
        if col in LABEL_AND_AUDIT:
            continue
        if not include_new and col.startswith(NEW_PREFIXES):
            continue
        if col in CATEGORICAL_COLUMNS:
            cols.append(col)
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def prepare_xy(parts, features: list[str]):
    train_df, tune_df, test_df = parts
    combined = pd.concat([train_df[features], tune_df[features], test_df[features]], axis=0)
    cats = [col for col in CATEGORICAL_COLUMNS if col in combined.columns]
    combined = pd.get_dummies(combined, columns=cats, dummy_na=True)
    combined = combined.replace([np.inf, -np.inf], np.nan)
    train_x = combined.iloc[: len(train_df)].copy()
    tune_x = combined.iloc[len(train_df) : len(train_df) + len(tune_df)].copy()
    test_x = combined.iloc[len(train_df) + len(tune_df) :].copy()
    return train_x, train_df[LABEL].astype(int), tune_x, tune_df[LABEL].astype(int), test_x, test_df[LABEL].astype(int)


def fit_lgbm(train_x, train_y, tune_x, tune_y, random_state: int):
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=600,
        learning_rate=0.025,
        num_leaves=31,
        max_depth=6,
        min_child_samples=100,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=5.0,
        random_state=random_state,
        n_jobs=4,
        verbosity=-1,
    )
    model.fit(
        train_x,
        train_y,
        eval_set=[(tune_x, tune_y)],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(40, verbose=False)],
    )
    return model


def metrics(y: pd.Series, p: np.ndarray) -> dict[str, float]:
    clipped = np.clip(p, 1e-6, 1 - 1e-6)
    out = {
        "rows": float(len(y)),
        "base_rate": float(np.mean(y)),
        "brier": float(brier_score_loss(y, clipped)),
        "logloss": float(log_loss(y, clipped, labels=[0, 1])),
    }
    try:
        out["auc"] = float(roc_auc_score(y, clipped))
    except ValueError:
        out["auc"] = math.nan
    return out


def fee_cents(price_cents: float) -> int:
    if not math.isfinite(price_cents) or price_cents <= 0 or price_cents >= 100:
        return 0
    p = price_cents / 100.0
    return int(math.ceil(0.07 * p * (1.0 - p) * 100.0 - 1e-9))


def simulate(df: pd.DataFrame, p_yes: np.ndarray, threshold: float) -> dict[str, float]:
    pnls: list[float] = []
    for row, p in zip(df.itertuples(index=False), p_yes):
        yes_ask = float(getattr(row, "yes_ask"))
        no_ask = float(getattr(row, "no_ask"))
        result_yes = int(getattr(row, "result_yes"))
        yes_edge = -math.inf
        no_edge = -math.inf
        if 0 < yes_ask < 100:
            yes_edge = p - yes_ask / 100.0 - fee_cents(yes_ask) / 100.0
        if 0 < no_ask < 100:
            no_edge = (1.0 - p) - no_ask / 100.0 - fee_cents(no_ask) / 100.0
        if max(yes_edge, no_edge) <= threshold:
            continue
        if yes_edge >= no_edge:
            cost = yes_ask
            pnl = (100 - cost if result_yes == 1 else -cost) - fee_cents(cost)
        else:
            cost = no_ask
            pnl = (100 - cost if result_yes == 0 else -cost) - fee_cents(cost)
        pnls.append(float(pnl))
    if not pnls:
        return {"trades": 0.0, "pnl_cents": 0.0, "pnl_per_trade": math.nan, "win_rate": math.nan, "max_drawdown": 0.0}
    arr = np.array(pnls)
    curve = np.cumsum(arr)
    dd = curve - np.maximum.accumulate(np.insert(curve, 0, 0.0))[1:]
    return {
        "trades": float(len(arr)),
        "pnl_cents": float(arr.sum()),
        "pnl_per_trade": float(arr.mean()),
        "win_rate": float((arr > 0).mean()),
        "max_drawdown": float(dd.min()),
    }


def train_variant(name: str, df: pd.DataFrame, parts, args: argparse.Namespace, include_new: bool) -> dict:
    out_dir = args.out_dir / name
    out_dir.mkdir(parents=True, exist_ok=True)
    features = feature_columns(df, include_new)
    train_x, train_y, tune_x, tune_y, test_x, test_y = prepare_xy(parts, features)
    model = fit_lgbm(train_x, train_y, tune_x, tune_y, args.random_state)
    preds = {
        "train": model.predict_proba(train_x)[:, 1],
        "tune": model.predict_proba(tune_x)[:, 1],
        "test": model.predict_proba(test_x)[:, 1],
    }
    train_df, tune_df, test_df = parts
    split_map = {"train": (train_df, train_y), "tune": (tune_df, tune_y), "test": (test_df, test_y)}
    summary = {
        "variant": name,
        "feature_count": len(features),
        "best_iteration": int(model.best_iteration_ or 0),
        "metrics": {split: metrics(y, preds[split]) for split, (_, y) in split_map.items()},
    }
    threshold_rows = []
    thresholds = [float(x) for x in args.edge_thresholds.split(",") if x.strip()]
    for split in ("tune", "test"):
        frame, _ = split_map[split]
        for threshold in thresholds:
            row = {"variant": name, "split": split, "edge_threshold": threshold}
            row.update(simulate(frame, preds[split], threshold))
            threshold_rows.append(row)
    pd.DataFrame(threshold_rows).to_csv(out_dir / "threshold_report.csv", index=False)
    if not args.skip_predictions:
        pred_rows = pd.concat(
            [
                train_df[["ticker", "asset", "wall_ns", "side", "level", "yes_ask", "no_ask", LABEL]].assign(
                    split="train", p_yes=preds["train"]
                ),
                tune_df[["ticker", "asset", "wall_ns", "side", "level", "yes_ask", "no_ask", LABEL]].assign(
                    split="tune", p_yes=preds["tune"]
                ),
                test_df[["ticker", "asset", "wall_ns", "side", "level", "yes_ask", "no_ask", LABEL]].assign(
                    split="test", p_yes=preds["test"]
                ),
            ]
        )
        pred_rows.to_csv(out_dir / "predictions.csv", index=False)
    importances = pd.DataFrame(
        {
            "feature": list(train_x.columns),
            "gain": model.booster_.feature_importance(importance_type="gain"),
            "split": model.booster_.feature_importance(importance_type="split"),
        }
    ).sort_values(["gain", "split"], ascending=False)
    importances.to_csv(out_dir / "feature_importance.csv", index=False)
    model.booster_.save_model(out_dir / "model.txt")
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def train_scope(scope_name: str, df: pd.DataFrame, args: argparse.Namespace) -> dict:
    scope_out = args.out_dir / scope_name if scope_name else args.out_dir
    scope_out.mkdir(parents=True, exist_ok=True)
    parts = split_by_ticker_time(df, args.train_frac, args.tune_frac)
    split_counts = {
        name: {"rows": int(len(part)), "tickers": int(part["ticker"].nunique())}
        for name, part in zip(("train", "tune", "test"), parts)
    }
    scoped_args = argparse.Namespace(**vars(args))
    scoped_args.out_dir = scope_out
    summaries = [
        train_variant("base_only", df, parts, scoped_args, include_new=False),
        train_variant("base_plus_level_flow", df, parts, scoped_args, include_new=True),
    ]
    result = {
        "scope": scope_name or "global",
        "rows": int(len(df)),
        "tickers": int(df["ticker"].nunique()),
        "split_counts": split_counts,
        "summaries": summaries,
    }
    with (scope_out / "summary.json").open("w", encoding="utf-8") as f:
        json.dump({"input": str(args.input), **result}, f, indent=2)
    return result


def requested_assets(args: argparse.Namespace) -> set[str] | None:
    if not args.assets:
        return None
    return {asset.strip() for asset in args.assets.split(",") if asset.strip()}


def discover_assets(path: Path, chunksize: int) -> list[str]:
    if path.suffix == ".parquet":
        return sorted(str(v) for v in pd.read_parquet(path, columns=["asset"])["asset"].dropna().unique())
    assets: set[str] = set()
    for chunk in pd.read_csv(path, usecols=["asset"], chunksize=chunksize):
        assets.update(str(v) for v in chunk["asset"].dropna().unique())
    return sorted(assets)


def load_input(path: Path, asset_filter: set[str] | None, chunksize: int) -> pd.DataFrame:
    if path.suffix == ".parquet":
        if asset_filter:
            return pd.read_parquet(path, filters=[("asset", "in", sorted(asset_filter))])
        return pd.read_parquet(path)
    if not asset_filter:
        return pd.read_csv(path, low_memory=False)
    parts = []
    for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
        part = chunk[chunk["asset"].isin(asset_filter)]
        if not part.empty:
            parts.append(part.copy())
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def downcast_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if pd.api.types.is_float_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast="float")
        elif pd.api.types.is_integer_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast="integer")
    return df


def clean_input(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df[LABEL].isin([0, 1])].copy()
    df["wall_ns"] = pd.to_numeric(df["wall_ns"], errors="coerce")
    df = df[df["wall_ns"].notna()].sort_values(["wall_ns", "ticker"]).reset_index(drop=True)
    return downcast_numeric(df)


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.per_asset:
        wanted = requested_assets(args)
        asset_names = sorted(wanted) if wanted else discover_assets(args.input, args.read_chunksize)
        scopes = []
        for asset in asset_names:
            asset_df = clean_input(load_input(args.input, {asset}, args.read_chunksize))
            if asset_df.empty:
                print(f"skipping asset={asset}: no rows")
                continue
            tickers = asset_df["ticker"].nunique()
            if tickers < 5:
                print(f"skipping asset={asset}: only {tickers} tickers")
                continue
            scopes.append(train_scope(f"asset_{asset}", asset_df, args))
        with (args.out_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump({"input": str(args.input), "per_asset": True, "scopes": scopes}, f, indent=2)
        print(json.dumps(scopes, indent=2))
        return 0

    df = clean_input(load_input(args.input, requested_assets(args), args.read_chunksize))
    if df.empty:
        raise SystemExit("no rows to train")
    scope = train_scope("", df, args)
    summaries = scope["summaries"]
    with (args.out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump({"input": str(args.input), **scope}, f, indent=2)
    print(json.dumps(summaries, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
