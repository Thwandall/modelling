#!/usr/bin/env python3
"""Build lower-price-level candidates directly from Kalshi tick history.

Rows are emitted when a side bid crosses upward through a configured level.
This produces synthetic decision points for levels that the deployed ML policy
never logged, without assuming impossible fills after a market is already above
the target level.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import glob
import json
import math
import re
from collections import Counter
from datetime import datetime, timedelta
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
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--levels", default="60-99", help="Comma list and/or ranges, e.g. 60-99,85,90")
    parser.add_argument("--exclude-asset", action="append", default=[])
    parser.add_argument("--max-markets", type=int, default=0)
    parser.add_argument(
        "--include-initial-crossed",
        action="store_true",
        help="Emit rows for levels already crossed at the first tick. Usually optimistic.",
    )
    parser.add_argument(
        "--min-tick-span-s",
        type=float,
        default=60.0,
        help="Require at least this much tick history in a market.",
    )
    return parser.parse_args()


def asset_from_ticker(ticker: str) -> str:
    match = re.match(r"KX([A-Z]+)15M-", ticker)
    return match.group(1) if match else ""


def parse_market_end_ns(ticker: str) -> int:
    match = re.match(r"KX[A-Z]+15M-(\d{2})([A-Z]{3})(\d{2})(\d{2})(\d{2})-", ticker)
    if not match:
        return 0
    year_s, mon_s, day_s, hour_s, minute_s = match.groups()
    year = 2000 + int(year_s)
    month = MONTHS.get(mon_s)
    if not month:
        return 0
    local_end = datetime(
        year,
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


def load_outcome(market_dir: Path) -> int | None:
    path = market_dir / "outcome.json"
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
    except json.JSONDecodeError:
        return None
    result = obj.get("result_yes") if isinstance(obj, dict) else None
    return to_int(result) if result in (0, 1) else None


def load_ticks(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            wall_ns = to_int(row.get("wall_ns"))
            if wall_ns <= 0:
                continue
            parsed = {
                "wall_ns": wall_ns,
                "yes_bid_mD": to_float(row.get("yes_bid_mD")),
                "yes_ask_mD": to_float(row.get("yes_ask_mD")),
                "no_bid_mD": to_float(row.get("no_bid_mD")),
                "no_ask_mD": to_float(row.get("no_ask_mD")),
                "spread_mD": to_float(row.get("spread")),
                "microprice_x1000": to_float(row.get("microprice_x1000")),
                "imbalance_x10000": to_float(row.get("imbalance_x10000")),
                "yes_bid_qty": to_float(row.get("yes_bid_qty")),
                "no_bid_qty": to_float(row.get("no_bid_qty")),
                "book_seq": to_int(row.get("book_seq")),
                "volume_fp100": to_float(row.get("volume_fp100")),
                "open_interest_fp100": to_float(row.get("open_interest_fp100")),
            }
            if not math.isfinite(parsed["yes_bid_mD"]) or not math.isfinite(parsed["no_bid_mD"]):
                continue
            rows.append(parsed)
    rows.sort(key=lambda r: (r["wall_ns"], r["book_seq"]))
    return rows


def prior_value(rows: list[dict[str, Any]], walls: list[int], idx: int, lookback_s: float, field: str) -> float:
    target = rows[idx]["wall_ns"] - int(lookback_s * 1_000_000_000)
    prior_idx = bisect.bisect_right(walls, target) - 1
    if prior_idx < 0:
        return math.nan
    return to_float(rows[prior_idx].get(field))


def side_values(row: dict[str, Any], side: str) -> tuple[float, float, float, float, float, float]:
    if side == "YES":
        return (
            row["yes_bid_mD"],
            row["yes_ask_mD"],
            row["no_bid_mD"],
            row["no_ask_mD"],
            row["yes_bid_qty"],
            row["no_bid_qty"],
        )
    return (
        row["no_bid_mD"],
        row["no_ask_mD"],
        row["yes_bid_mD"],
        row["yes_ask_mD"],
        row["no_bid_qty"],
        row["yes_bid_qty"],
    )


def tte_bucket(tte_s: int) -> int:
    if tte_s < 0:
        return -1
    return min(8, tte_s // 120)


def main() -> int:
    args = parse_args()
    levels = parse_levels(args.levels)
    exclude_assets = {asset.upper() for asset in args.exclude_asset}
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
        "entry_price_mD",
        "tte_s",
        "tte_bucket",
        "hour_utc",
        "day_of_week_utc",
        "yes_bid_mD",
        "yes_ask_mD",
        "no_bid_mD",
        "no_ask_mD",
        "side_bid_mD",
        "side_ask_mD",
        "opp_bid_mD",
        "opp_ask_mD",
        "side_bid_qty",
        "opp_bid_qty",
        "spread_mD",
        "microprice_x1000",
        "imbalance_x10000",
        "book_seq",
        "volume_contracts",
        "open_interest_contracts",
        "side_bid_minus_level_mD",
        "side_ask_minus_level_mD",
        "side_bid_change_1s_mD",
        "side_bid_change_5s_mD",
        "side_bid_change_15s_mD",
        "side_bid_change_30s_mD",
        "microprice_change_5s",
        "microprice_change_15s",
        "result_yes",
        "side_won",
        "settlement_pnl_mD",
    ]

    counts: Counter[str] = Counter()
    args.out.parent.mkdir(parents=True, exist_ok=True)
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
            walls = [row["wall_ns"] for row in ticks]
            prev_bid = {"YES": math.nan, "NO": math.nan}
            market_rows = 0
            for idx, row in enumerate(ticks):
                dt = datetime.fromtimestamp(row["wall_ns"] / 1_000_000_000, tz=UTC)
                for side in ("YES", "NO"):
                    side_bid, side_ask, opp_bid, opp_ask, side_qty, opp_qty = side_values(row, side)
                    prior = prev_bid[side]
                    prev_bid[side] = side_bid
                    if not math.isfinite(side_bid):
                        continue
                    for level in levels:
                        entry_mD = level * 10.0
                        crossed = side_bid >= entry_mD and (
                            (math.isfinite(prior) and prior < entry_mD)
                            or (args.include_initial_crossed and not math.isfinite(prior))
                        )
                        if not crossed:
                            continue
                        side_won = outcome if side == "YES" else 1 - outcome
                        tte_s = int((end_ns - row["wall_ns"]) / 1_000_000_000)
                        bid_field = "yes_bid_mD" if side == "YES" else "no_bid_mD"
                        out = {
                            "ticker": ticker,
                            "asset": asset,
                            "wall_ns": row["wall_ns"],
                            "side": side,
                            "level": level,
                            "entry_price_mD": entry_mD,
                            "tte_s": tte_s,
                            "tte_bucket": tte_bucket(tte_s),
                            "hour_utc": dt.hour,
                            "day_of_week_utc": dt.weekday(),
                            "yes_bid_mD": row["yes_bid_mD"],
                            "yes_ask_mD": row["yes_ask_mD"],
                            "no_bid_mD": row["no_bid_mD"],
                            "no_ask_mD": row["no_ask_mD"],
                            "side_bid_mD": side_bid,
                            "side_ask_mD": side_ask,
                            "opp_bid_mD": opp_bid,
                            "opp_ask_mD": opp_ask,
                            "side_bid_qty": side_qty,
                            "opp_bid_qty": opp_qty,
                            "spread_mD": row["spread_mD"],
                            "microprice_x1000": row["microprice_x1000"],
                            "imbalance_x10000": row["imbalance_x10000"],
                            "book_seq": row["book_seq"],
                            "volume_contracts": row["volume_fp100"] / 100.0,
                            "open_interest_contracts": row["open_interest_fp100"] / 100.0,
                            "side_bid_minus_level_mD": side_bid - entry_mD,
                            "side_ask_minus_level_mD": side_ask - entry_mD,
                            "side_bid_change_1s_mD": side_bid - prior_value(ticks, walls, idx, 1, bid_field),
                            "side_bid_change_5s_mD": side_bid - prior_value(ticks, walls, idx, 5, bid_field),
                            "side_bid_change_15s_mD": side_bid - prior_value(ticks, walls, idx, 15, bid_field),
                            "side_bid_change_30s_mD": side_bid - prior_value(ticks, walls, idx, 30, bid_field),
                            "microprice_change_5s": row["microprice_x1000"]
                            - prior_value(ticks, walls, idx, 5, "microprice_x1000"),
                            "microprice_change_15s": row["microprice_x1000"]
                            - prior_value(ticks, walls, idx, 15, "microprice_x1000"),
                            "result_yes": outcome,
                            "side_won": side_won,
                            "settlement_pnl_mD": (1000.0 - entry_mD) if side_won else -entry_mD,
                        }
                        writer.writerow(out)
                        market_rows += 1
            counts["markets_written"] += int(market_rows > 0)
            counts["rows_written"] += market_rows

    print(json.dumps({"out": str(args.out), "levels": levels, "counts": counts}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
