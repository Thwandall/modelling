#!/usr/bin/env python3
"""Build a settlement-labelled dataset from every logged ML price level.

This intentionally uses ``ml_feature_vectors.ndjson`` rather than approved fills
so lower entry levels such as 60c/70c can be studied even when the deployed
policy never traded them.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/root/crypto_data"))
    parser.add_argument("--market-glob", default="KX*15M-*")
    parser.add_argument(
        "--feature-schema",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "models"
        / "candidate_trade_features_v1"
        / "feature_schema.json",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--exclude-asset",
        action="append",
        default=[],
        help="Asset symbol to exclude. Can be passed multiple times.",
    )
    parser.add_argument(
        "--exclude-reason",
        action="append",
        default=[],
        help="Logged decision reason to exclude. Can be passed multiple times.",
    )
    parser.add_argument("--max-markets", type=int, default=0)
    return parser.parse_args()


def asset_from_ticker(ticker: str) -> str:
    match = re.match(r"KX([A-Z]+)15M-", ticker)
    return match.group(1) if match else ""


def to_int(value: Any, default: int = 0) -> int:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default


def to_float(value: Any, default: float = math.nan) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_json_doc(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def load_outcome(market_dir: Path) -> int | None:
    for name in ("outcome.json", "outcome.ndjson"):
        path = market_dir / name
        if not path.exists():
            continue
        if path.suffix == ".json":
            obj = load_json_doc(path)
            if obj and obj.get("result_yes") is not None:
                return to_int(obj.get("result_yes"))
        else:
            last: dict[str, Any] | None = None
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(obj, dict):
                        last = obj
            if last and last.get("result_yes") is not None:
                return to_int(last.get("result_yes"))
    return None


def load_feature_names(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        schema = json.load(f)
    names = schema.get("encoded_feature_order")
    if not isinstance(names, list) or not names:
        raise ValueError(f"{path} does not contain encoded_feature_order")
    return [str(name) for name in names]


def iter_feature_rows(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if isinstance(row, dict):
                yield row


def main() -> int:
    args = parse_args()
    raw_feature_names = load_feature_names(args.feature_schema)
    feature_names = [f"base_{name}" for name in raw_feature_names]
    exclude_assets = {asset.upper() for asset in args.exclude_asset}
    exclude_reasons = set(args.exclude_reason)

    market_dirs = [Path(p) for p in glob.glob(str(args.data_root / args.market_glob))]
    market_dirs = sorted([p for p in market_dirs if p.is_dir()])
    if args.max_markets:
        market_dirs = market_dirs[: args.max_markets]

    fieldnames = [
        "ticker",
        "asset",
        "wall_ns",
        "side",
        "level",
        "tte_s",
        "tte_bucket",
        "entry_price_mD",
        "logged_p_yes",
        "logged_ml_edge",
        "logged_meta_p_good",
        "logged_bucket_threshold",
        "logged_trade",
        "reason",
        *feature_names,
        "result_yes",
        "side_won",
        "settlement_pnl_mD",
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    with args.out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for market_dir in market_dirs:
            ticker = market_dir.name
            asset = asset_from_ticker(ticker)
            if not asset:
                counts["skipped_bad_ticker"] += 1
                continue
            if asset in exclude_assets:
                counts[f"skipped_asset_{asset}"] += 1
                continue
            outcome = load_outcome(market_dir)
            if outcome not in (0, 1):
                counts["skipped_missing_outcome"] += 1
                continue
            feature_path = market_dir / "ml_feature_vectors.ndjson"
            if not feature_path.exists():
                counts["skipped_missing_features"] += 1
                continue
            market_rows = 0
            for row in iter_feature_rows(feature_path):
                reason = str(row.get("reason") or "")
                if reason in exclude_reasons:
                    counts[f"skipped_reason_{reason}"] += 1
                    continue
                side = str(row.get("side") or "").upper()
                if side not in {"YES", "NO"}:
                    counts["skipped_bad_side"] += 1
                    continue
                entry_price_mD = to_float(row.get("entry_price_mD"))
                if not math.isfinite(entry_price_mD):
                    counts["skipped_bad_entry"] += 1
                    continue
                side_won = outcome if side == "YES" else 1 - outcome
                settlement_pnl_mD = (1000.0 - entry_price_mD) if side_won else -entry_price_mD
                out = {
                    "ticker": ticker,
                    "asset": asset,
                    "wall_ns": to_int(row.get("wall_ns")),
                    "side": side,
                    "level": to_int(row.get("level")),
                    "tte_s": to_int(row.get("tte_s")),
                    "tte_bucket": to_int(row.get("tte_bucket")),
                    "entry_price_mD": entry_price_mD,
                    "logged_p_yes": to_float(row.get("p_yes")),
                    "logged_ml_edge": to_float(row.get("ml_edge")),
                    "logged_meta_p_good": to_float(row.get("meta_p_good")),
                    "logged_bucket_threshold": to_float(row.get("bucket_threshold")),
                    "logged_trade": to_int(row.get("trade")),
                    "reason": reason,
                    "result_yes": outcome,
                    "side_won": side_won,
                    "settlement_pnl_mD": settlement_pnl_mD,
                }
                values = row.get("base_values") or []
                for idx, name in enumerate(feature_names):
                    out[name] = values[idx] if idx < len(values) else math.nan
                writer.writerow(out)
                market_rows += 1
            counts["markets_written"] += int(market_rows > 0)
            counts["rows_written"] += market_rows

    print(json.dumps({"out": str(args.out), "counts": counts}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
