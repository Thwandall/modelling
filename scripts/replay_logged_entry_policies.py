#!/usr/bin/env python3
"""Replay model-bundle risk policies against logged live ML feature vectors."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/root/crypto_data"))
    parser.add_argument("--market-glob", default="KX*15M-*")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model-dir", action="append", type=Path, required=True)
    parser.add_argument(
        "--require-live-safety-approved",
        action="store_true",
        help="Only evaluate rows whose logged live decision was approved.",
    )
    return parser.parse_args()


def load_json_lines(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def load_json_doc(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            row = json.load(f)
    except json.JSONDecodeError:
        return None
    return row if isinstance(row, dict) else None


def load_outcome(market_dir: Path) -> int | None:
    for row in reversed(load_json_lines(market_dir / "outcome.ndjson")):
        if "result_yes" in row:
            return int(row["result_yes"])
    row = load_json_doc(market_dir / "outcome.json")
    if row and "result_yes" in row:
        return int(row["result_yes"])
    return None


def asset_from_ticker(ticker: str) -> str:
    m = re.match(r"KX([A-Z]+)15M-", ticker)
    return m.group(1) if m else ""


def to_float(value: Any, default: float = math.nan) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def to_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default


def load_policy(model_dir: Path) -> dict[str, Any]:
    risk = load_json_doc(model_dir / "risk_policy.json") or {}
    thresholds_doc = load_json_doc(model_dir / "thresholds.json") or {}
    thresholds: dict[tuple[str, int], float] = {}
    for row in thresholds_doc.get("thresholds", []):
        asset = str(row.get("asset", ""))
        bucket = to_int(row.get("tte_bucket"), -1)
        if asset and bucket >= 0:
            thresholds[(asset, bucket)] = to_float(row.get("min_edge"), 0.0)
    base_model = lgb.Booster(model_file=str(model_dir / "base_model.txt"))
    meta_model = lgb.Booster(model_file=str(model_dir / "meta_model.txt"))
    calibration = load_json_doc(model_dir / "base_calibration.json") or {}
    feature_schema = load_json_doc(model_dir / "feature_schema.json") or {}
    base_names = list(feature_schema.get("encoded_feature_order") or base_model.feature_name())
    return {
        "name": model_dir.name,
        "path": str(model_dir),
        "min_ml_edge": to_float(risk.get("min_ml_edge"), 0.0),
        "min_meta_p": to_float(risk.get("min_meta_p"), 0.0),
        "max_asset_side": to_int(risk.get("max_asset_side"), 0),
        "thresholds": thresholds,
        "base_model": base_model,
        "meta_model": meta_model,
        "base_names": base_names,
        "base_pos": {name: i for i, name in enumerate(base_names)},
        "meta_names": list(meta_model.feature_name()),
        "calibration": calibration,
    }


def calibrate_probability(raw_p: float, calibration: dict[str, Any]) -> float:
    if calibration.get("type") != "isotonic":
        return raw_p
    xs = calibration.get("x_thresholds") or []
    ys = calibration.get("y_thresholds") or []
    if not xs or not ys or len(xs) != len(ys):
        return raw_p
    return float(np.interp(raw_p, xs, ys))


def score_policy_row(policy: dict[str, Any], row: dict[str, Any], asset: str) -> dict[str, float]:
    base_values = row.get("base_values") or []
    base_x = np.array([[to_float(v) for v in base_values]], dtype=float)
    raw_p = float(policy["base_model"].predict(base_x)[0])
    p_yes = calibrate_probability(raw_p, policy["calibration"])
    side = str(row.get("side") or "")
    entry = to_int(row.get("entry_price_mD"))
    if side == "YES":
        ml_edge = p_yes - entry / 1000.0
    elif side == "NO":
        ml_edge = (1.0 - p_yes) - entry / 1000.0
    else:
        ml_edge = math.nan
    bucket = to_int(row.get("tte_bucket"), -1)
    bucket_threshold = policy["thresholds"].get((asset, bucket), 0.0)

    base_pos = policy["base_pos"]
    meta_values: list[float] = []
    for name in policy["meta_names"]:
        if name == "p_yes":
            meta_values.append(p_yes)
        elif name == "ml_edge":
            meta_values.append(ml_edge)
        elif name == "bucket_threshold":
            meta_values.append(bucket_threshold)
        elif name == "wall_ns":
            meta_values.append(to_float(row.get("wall_ns")))
        elif name == "ml_side_YES":
            meta_values.append(1.0 if side == "YES" else 0.0)
        elif name == "ml_side_NO":
            meta_values.append(1.0 if side == "NO" else 0.0)
        elif name == "ml_side_nan":
            meta_values.append(0.0)
        elif name in base_pos:
            idx = base_pos[name]
            meta_values.append(to_float(base_values[idx]) if idx < len(base_values) else math.nan)
        else:
            meta_values.append(math.nan)
    meta_p_good = float(policy["meta_model"].predict(np.array([meta_values], dtype=float))[0])
    return {
        "p_yes": p_yes,
        "ml_edge": ml_edge,
        "meta_p_good": meta_p_good,
        "bucket_threshold": bucket_threshold,
    }


def policy_decision(policy: dict[str, Any], row: dict[str, Any], asset: str) -> tuple[bool, str, float, dict[str, float]]:
    scores = score_policy_row(policy, row, asset)
    bucket = to_int(row.get("tte_bucket"), -1)
    key = (asset, bucket)
    if key not in policy["thresholds"]:
        return False, "bucket_disabled", 0.0, scores
    bucket_threshold = scores["bucket_threshold"]
    ml_edge = scores["ml_edge"]
    meta_p_good = scores["meta_p_good"]
    if math.isnan(ml_edge):
        return False, "missing_ml_edge", bucket_threshold, scores
    if ml_edge <= bucket_threshold:
        return False, "below_bucket_threshold", bucket_threshold, scores
    if math.isnan(meta_p_good):
        return False, "missing_meta_p", bucket_threshold, scores
    if meta_p_good < policy["min_meta_p"]:
        return False, "below_meta_threshold", bucket_threshold, scores
    if ml_edge < policy["min_ml_edge"]:
        return False, "below_candidate_v2_edge", bucket_threshold, scores
    return True, "approved", bucket_threshold, scores


def settlement_pnl_mD(side: str, entry_mD: int, result_yes: int) -> int:
    wins = (side == "YES" and result_yes == 1) or (side == "NO" and result_yes == 0)
    return 1000 - entry_mD if wins else -entry_mD


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    policies = [load_policy(p) for p in args.model_dir]
    market_dirs = sorted(Path(p) for p in glob.glob(str(args.data_root / args.market_glob)) if Path(p).is_dir())

    detail_rows: list[dict[str, Any]] = []
    summary: dict[str, Counter] = {p["name"]: Counter() for p in policies}
    pnl_by_asset: dict[str, Counter] = {p["name"]: Counter() for p in policies}
    reasons: dict[str, Counter] = {p["name"]: Counter() for p in policies}

    markets_with_rows = 0
    markets_with_outcome = 0
    source_rows = 0
    for market_dir in market_dirs:
        result_yes = load_outcome(market_dir)
        if result_yes is None:
            continue
        markets_with_outcome += 1
        rows = load_json_lines(market_dir / "ml_feature_vectors.ndjson")
        if not rows:
            continue
        markets_with_rows += 1
        ticker = market_dir.name
        asset = asset_from_ticker(ticker)
        for row in rows:
            if args.require_live_safety_approved and not to_int(row.get("trade")):
                continue
            source_rows += 1
            side = str(row.get("side") or "")
            entry = to_int(row.get("entry_price_mD"))
            if side not in ("YES", "NO") or entry <= 0:
                continue
            pnl = settlement_pnl_mD(side, entry, result_yes)
            for policy in policies:
                ok, reason, bucket_threshold, scores = policy_decision(policy, row, asset)
                reasons[policy["name"]][reason] += 1
                summary[policy["name"]]["candidates"] += 1
                if not ok:
                    continue
                summary[policy["name"]]["trades"] += 1
                summary[policy["name"]]["pnl_mD"] += pnl
                summary[policy["name"]]["wins"] += int(pnl > 0)
                summary[policy["name"]]["losses"] += int(pnl < 0)
                pnl_by_asset[policy["name"]][asset + "_trades"] += 1
                pnl_by_asset[policy["name"]][asset + "_pnl_mD"] += pnl
                detail_rows.append(
                    {
                        "policy": policy["name"],
                        "ticker": ticker,
                        "asset": asset,
                        "wall_ns": row.get("wall_ns"),
                        "side": side,
                        "level": row.get("level"),
                        "tte_s": row.get("tte_s"),
                        "tte_bucket": row.get("tte_bucket"),
                        "entry_price_mD": entry,
                        "result_yes": result_yes,
                        "pnl_mD": pnl,
                        "p_yes": scores["p_yes"],
                        "ml_edge": scores["ml_edge"],
                        "meta_p_good": scores["meta_p_good"],
                        "bucket_threshold": bucket_threshold,
                        "logged_reason": row.get("reason"),
                        "logged_trade": row.get("trade"),
                    }
                )

    with (args.out_dir / "policy_replay_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "market_glob": args.market_glob,
                "markets_scanned": len(market_dirs),
                "markets_with_outcome": markets_with_outcome,
                "markets_with_feature_rows": markets_with_rows,
                "source_rows": source_rows,
                "policies": [
                    {
                        **{
                            k: v
                            for k, v in p.items()
                            if k
                            in (
                                "name",
                                "path",
                                "min_ml_edge",
                                "min_meta_p",
                                "max_asset_side",
                            )
                        },
                        "summary": dict(summary[p["name"]]),
                        "reasons": dict(reasons[p["name"]]),
                        "by_asset": dict(pnl_by_asset[p["name"]]),
                    }
                    for p in policies
                ],
            },
            f,
            indent=2,
        )
    if detail_rows:
        with (args.out_dir / "policy_replay_trades.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(detail_rows[0]))
            writer.writeheader()
            writer.writerows(detail_rows)

    print(f"markets_scanned={len(market_dirs)}")
    print(f"markets_with_outcome={markets_with_outcome}")
    print(f"markets_with_feature_rows={markets_with_rows}")
    print(f"source_rows={source_rows}")
    for policy in policies:
        s = summary[policy["name"]]
        trades = s["trades"]
        pnl = s["pnl_mD"]
        win_rate = s["wins"] / trades if trades else float("nan")
        print(
            f"policy={policy['name']} candidates={s['candidates']} trades={trades} "
            f"pnl_mD={pnl} pnl_cents={pnl / 10:.1f} win_rate={win_rate:.4f}"
        )
        print("  reasons=" + json.dumps(dict(reasons[policy["name"]]), sort_keys=True))
        print("  by_asset=" + json.dumps(dict(pnl_by_asset[policy["name"]]), sort_keys=True))
    print(f"out_dir={args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
