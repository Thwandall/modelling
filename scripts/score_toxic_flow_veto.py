#!/usr/bin/env python3
"""Score a frozen toxic-flow veto report on a toxic-flow dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--trades-csv", type=Path)
    parser.add_argument("--policy")
    parser.add_argument("--exclude-asset", action="append", default=[])
    return parser.parse_args()


def metric_or_nan(fn, y: pd.Series, p: np.ndarray) -> float:
    try:
        if y.nunique(dropna=False) < 2:
            return float("nan")
        return float(fn(y, p))
    except ValueError:
        return float("nan")


def load_scoring_rows(args: argparse.Namespace) -> pd.DataFrame:
    dtype = {"ticker": "string", "wall_ns": "string", "side": "string", "entry_price_mD": "string"}
    df = pd.read_csv(args.input, dtype=dtype)
    if args.trades_csv:
        trades = pd.read_csv(args.trades_csv, dtype=dtype)
        if args.policy:
            trades = trades[trades["policy"] == args.policy].copy()
        key_cols = ["ticker", "wall_ns", "side", "entry_price_mD"]
        for col in ("ticker", "side"):
            df[col] = df[col].astype(str)
            trades[col] = trades[col].astype(str)
        for col in ("wall_ns", "entry_price_mD"):
            df[col] = df[col].astype(str)
            trades[col] = trades[col].astype(str)
        keys = trades[key_cols].drop_duplicates()
        df = df.merge(keys.assign(_keep=1), on=key_cols, how="inner").drop(columns=["_keep"])
    for asset in args.exclude_asset:
        df = df[df["asset"] != asset].copy()
    return df


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = load_scoring_rows(args)
    summary = json.load(open(args.report_dir / "summary.json"))
    feature_doc = json.load(open(args.report_dir / "feature_columns.json"))
    features = feature_doc["features"]
    label = summary["label"]
    model_name = summary.get("model", "lightgbm")
    if model_name != "lightgbm":
        raise SystemExit("this scorer currently supports LightGBM toxic reports only")
    for col in features:
        if col not in df.columns:
            df[col] = np.nan
    x = df[features].copy()
    for col in feature_doc.get("categorical", []):
        if col in x.columns:
            x[col] = x[col].astype("category")
    for col in x.columns:
        if str(x[col].dtype) == "category":
            continue
        x[col] = pd.to_numeric(x[col], errors="coerce")
    model = lgb.Booster(model_file=str(args.report_dir / "toxic_flow_lgbm.txt"))
    pred = model.predict(x)
    y = pd.to_numeric(df[label], errors="coerce").fillna(0).astype(int)

    thresholds = list((summary.get("veto_thresholds_from_tune_quantiles") or {}).values())
    rows = []
    pnl = pd.to_numeric(df.get("settlement_pnl_mD"), errors="coerce")
    adverse = pd.to_numeric(df.get("max_adverse_bid_change_30s_mD"), errors="coerce")
    for threshold in thresholds:
        keep = pred < threshold
        veto = ~keep
        rows.append(
            {
                "threshold": threshold,
                "rows": len(df),
                "kept": int(keep.sum()),
                "vetoed": int(veto.sum()),
                "kept_rate": float(keep.mean()) if len(df) else np.nan,
                "raw_bad_rate": float(y.mean()) if len(df) else np.nan,
                "kept_bad_rate": float(y[keep].mean()) if keep.sum() else np.nan,
                "veto_bad_rate": float(y[veto].mean()) if veto.sum() else np.nan,
                "raw_settlement_pnl_mD": float(pnl.sum()),
                "kept_settlement_pnl_mD": float(pnl[keep].sum()),
                "vetoed_settlement_pnl_mD": float(pnl[veto].sum()),
                "raw_mean_adverse_30s_mD": float(adverse.mean()),
                "kept_mean_adverse_30s_mD": float(adverse[keep].mean()) if keep.sum() else np.nan,
                "vetoed_mean_adverse_30s_mD": float(adverse[veto].mean()) if veto.sum() else np.nan,
            }
        )
    out = {
        "input": str(args.input),
        "report_dir": str(args.report_dir),
        "trades_csv": str(args.trades_csv) if args.trades_csv else None,
        "policy": args.policy,
        "excluded_assets": args.exclude_asset,
        "label": label,
        "rows": int(len(df)),
        "positive_rate": float(y.mean()) if len(df) else np.nan,
        "auc": metric_or_nan(roc_auc_score, y, pred),
        "average_precision": metric_or_nan(average_precision_score, y, pred),
    }
    pd.DataFrame(rows).to_csv(args.out_dir / "toxic_veto_threshold_report.csv", index=False)
    pd.DataFrame({"toxic_pred": pred, label: y}).to_csv(args.out_dir / "toxic_predictions.csv", index=False)
    with (args.out_dir / "toxic_score_summary.json").open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print(f"out_dir={args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
