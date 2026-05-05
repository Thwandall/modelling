#!/usr/bin/env python3
"""Build base-model-style feature rows from raw tick price-level crossings.

This is the "base model without old strategy prefilter" dataset: ticks create
the candidate events, then the same point-in-time base feature construction used
by ``build_ml_feature_table.py`` enriches and labels each event.
"""

from __future__ import annotations

import argparse
import csv
import glob
import importlib.util
import json
import math
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


NY = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")
MONTHS = {
    "JAN": 1,
    "FEB": 2,
    "MAR": 3,
    "APR": 4,
    "MAY": 5,
    "JUN": 6,
    "JUL": 7,
    "AUG": 8,
    "SEP": 9,
    "OCT": 10,
    "NOV": 11,
    "DEC": 12,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/root/crypto_data"))
    parser.add_argument("--market-glob", default="KX*15M-*")
    parser.add_argument("--base-builder", type=Path, default=Path("/root/crypto_data/build_ml_feature_table.py"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--levels", default="60-99")
    parser.add_argument("--exclude-asset", action="append", default=[])
    parser.add_argument("--max-markets", type=int, default=0)
    parser.add_argument("--min-tick-span-s", type=float, default=60.0)
    parser.add_argument("--include-initial-crossed", action="store_true")
    return parser.parse_args()


def load_base_builder(path: Path):
    spec = importlib.util.spec_from_file_location("crypto_base_feature_builder", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def asset_from_ticker(ticker: str) -> str:
    match = re.match(r"KX([A-Z]+)15M-", ticker)
    return match.group(1) if match else ""


def parse_market_end_ns(ticker: str) -> int:
    match = re.match(r"KX[A-Z]+15M-(\d{2})([A-Z]{3})(\d{2})(\d{2})(\d{2})-", ticker)
    if not match:
        return 0
    year_s, mon_s, day_s, hour_s, minute_s = match.groups()
    month = MONTHS.get(mon_s)
    if month is None:
        return 0
    local_end = datetime(
        2000 + int(year_s),
        month,
        int(day_s),
        int(hour_s),
        int(minute_s),
        tzinfo=NY,
    )
    return int(local_end.astimezone(UTC).timestamp() * 1_000_000_000)


def parse_levels(spec: str) -> list[int]:
    levels: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            lo = int(lo_s)
            hi = int(hi_s)
            levels.update(range(min(lo, hi), max(lo, hi) + 1))
        else:
            levels.add(int(part))
    return sorted(level for level in levels if 1 <= level <= 99)


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


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def load_ticks(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            wall_ns = to_int(row.get("wall_ns"))
            yes_bid_mD = to_float(row.get("yes_bid_mD"))
            no_bid_mD = to_float(row.get("no_bid_mD"))
            if wall_ns <= 0 or not math.isfinite(yes_bid_mD) or not math.isfinite(no_bid_mD):
                continue
            rows.append(
                {
                    **row,
                    "wall_ns": wall_ns,
                    "yes_bid_mD_num": yes_bid_mD,
                    "yes_ask_mD_num": to_float(row.get("yes_ask_mD")),
                    "no_bid_mD_num": no_bid_mD,
                    "no_ask_mD_num": to_float(row.get("no_ask_mD")),
                    "book_seq_num": to_int(row.get("book_seq")),
                }
            )
    rows.sort(key=lambda r: (r["wall_ns"], r["book_seq_num"]))
    return rows


def normalize_tick_rows_for_base(ticks: list[dict[str, Any]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for row in ticks:
        converted = dict(row)
        converted["wall_ns"] = str(row["wall_ns"])
        converted["yes_bid"] = str(int(round(row["yes_bid_mD_num"] / 10.0)))
        converted["yes_ask"] = str(int(round(row["yes_ask_mD_num"] / 10.0)))
        converted["no_bid"] = str(int(round(row["no_bid_mD_num"] / 10.0)))
        converted["no_ask"] = str(int(round(row["no_ask_mD_num"] / 10.0)))
        out.append(converted)
    return out


def make_crossing_bases(
    ticks: list[dict[str, Any]],
    levels: list[int],
    end_ns: int,
    include_initial_crossed: bool,
) -> list[dict[str, Any]]:
    bases: list[dict[str, Any]] = []
    prev_bid = {"YES": math.nan, "NO": math.nan}
    for row in ticks:
        for side, bid_key, ask_key in (
            ("YES", "yes_bid_mD_num", "yes_ask_mD_num"),
            ("NO", "no_bid_mD_num", "no_ask_mD_num"),
        ):
            side_bid_mD = row[bid_key]
            prior = prev_bid[side]
            prev_bid[side] = side_bid_mD
            if not math.isfinite(side_bid_mD):
                continue
            yes_bid = int(round(row["yes_bid_mD_num"] / 10.0))
            yes_ask = int(round(row["yes_ask_mD_num"] / 10.0))
            no_bid = int(round(row["no_bid_mD_num"] / 10.0))
            no_ask = int(round(row["no_ask_mD_num"] / 10.0))
            for level in levels:
                level_mD = level * 10.0
                crossed = side_bid_mD >= level_mD and (
                    (math.isfinite(prior) and prior < level_mD)
                    or (include_initial_crossed and not math.isfinite(prior))
                )
                if not crossed:
                    continue
                bases.append(
                    {
                        "source": "tick_crossing",
                        "wall_ns": row["wall_ns"],
                        "side": side,
                        "side_id": 0 if side == "YES" else 1,
                        "level": level,
                        "tte_s": int((end_ns - row["wall_ns"]) / 1_000_000_000),
                        "yes_bid": yes_bid,
                        "yes_ask": yes_ask,
                        "no_bid": no_bid,
                        "no_ask": no_ask,
                        "cb_mid_cents": math.nan,
                        "rvol_1m_microbps": math.nan,
                        "rvol_5m_microbps": math.nan,
                        "strike_dist_bps": math.nan,
                        "volume": to_float(row.get("volume_fp100")) / 100.0,
                    }
                )
    return bases


def main() -> int:
    args = parse_args()
    base = load_base_builder(args.base_builder)
    levels = parse_levels(args.levels)
    exclude_assets = {asset.upper() for asset in args.exclude_asset}
    market_dirs = [Path(p) for p in glob.glob(str(args.data_root / args.market_glob))]
    market_dirs = sorted([p for p in market_dirs if p.is_dir()])
    if args.max_markets:
        market_dirs = market_dirs[: args.max_markets]

    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for market_dir in market_dirs:
        ticker = market_dir.name
        asset = asset_from_ticker(ticker)
        if not asset:
            counts["skipped_bad_ticker"] += 1
            continue
        if asset in exclude_assets:
            counts[f"skipped_asset_{asset}"] += 1
            continue
        outcome = load_json(market_dir / "outcome.json")
        if not outcome or outcome.get("result_yes") not in (0, 1):
            counts["skipped_missing_outcome"] += 1
            continue
        end_ns = parse_market_end_ns(ticker)
        if end_ns <= 0:
            counts["skipped_bad_end_time"] += 1
            continue
        ticks = load_ticks(market_dir / "ticks.csv")
        if len(ticks) < 10:
            counts["skipped_few_ticks"] += 1
            continue
        span_s = (ticks[-1]["wall_ns"] - ticks[0]["wall_ns"]) / 1_000_000_000
        if span_s < args.min_tick_span_s:
            counts["skipped_short_span"] += 1
            continue
        bases = make_crossing_bases(ticks, levels, end_ns, args.include_initial_crossed)
        if not bases:
            counts["markets_no_crossings"] += 1
            continue
        rti_rows = base.read_csv_dicts(market_dir / "rti_snapshots.csv")
        tick_rows = normalize_tick_rows_for_base(ticks)
        rti_walls = base.csv_walls(rti_rows)
        tick_walls = base.csv_walls(tick_rows)
        market_rows = [
            base.build_row(market_dir, row, outcome, rti_rows, rti_walls, tick_rows, tick_walls)
            for row in bases
            if int(row["wall_ns"]) > 0
        ]
        rows.extend(market_rows)
        counts["markets_written"] += 1
        counts["rows_written"] += len(market_rows)

    rows.sort(key=lambda r: (r["wall_ns"], r["ticker"], r["side_id"], r["level"]))
    if not rows:
        raise SystemExit("no rows built")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"out": str(args.out), "levels": levels, "counts": counts}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
