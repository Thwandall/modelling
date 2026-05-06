#!/usr/bin/env python3
"""Add price-level trade-flow and sparse entry-depth features to candidate rows."""

from __future__ import annotations

import argparse
import bisect
import csv
import glob
import gzip
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

NS_PER_S = 1_000_000_000


@dataclass(frozen=True, slots=True)
class PublicTrade:
    wall_ns: int
    yes_price_c: int
    no_price_c: int
    qty: float
    yes_taker: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("/root/crypto_data"))
    parser.add_argument("--market-glob", default="KX*15M-*")
    parser.add_argument("--backfill-trades-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--windows", default="30,60,300,900")
    parser.add_argument("--near-cents", type=int, default=2)
    parser.add_argument("--depth-max-lag-s", type=float, default=2.0)
    parser.add_argument("--max-rows", type=int, default=0)
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


BAD_JSON_LINES: Counter[str] = Counter()


def iter_json_lines(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                BAD_JSON_LINES[str(path)] += 1
                continue
            if isinstance(row, dict):
                yield row


def load_json_lines(path: Path) -> list[dict[str, Any]]:
    return list(iter_json_lines(path) or [])


def load_json_doc(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def price_mD_to_cents(value: Any) -> int:
    price = to_float(value)
    if not math.isfinite(price):
        return 0
    return int(round(price / 10.0))


def price_dollars_to_cents(value: Any) -> int:
    price = to_float(value)
    if not math.isfinite(price):
        return 0
    return int(round(price * 100.0))


def iso_time_to_ns(value: Any) -> int:
    if value is None or value == "":
        return 0
    text = str(value)
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return 0
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * NS_PER_S)


def scan_input_requirements(input_path: Path, max_rows: int, lookback_ns: int) -> tuple[set[str], int, int]:
    assets: set[str] = set()
    min_wall_ns = 0
    max_wall_ns = 0
    rows = 0
    with input_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            wall_ns = to_int(row.get("wall_ns"))
            if wall_ns <= 0:
                continue
            asset = str(row.get("asset") or asset_from_ticker(str(row.get("ticker", ""))))
            if asset:
                assets.add(asset)
            min_wall_ns = wall_ns if min_wall_ns == 0 else min(min_wall_ns, wall_ns)
            max_wall_ns = max(max_wall_ns, wall_ns)
            rows += 1
            if max_rows and rows >= max_rows:
                break
    if min_wall_ns > 0:
        min_wall_ns = max(0, min_wall_ns - lookback_ns)
    return assets, min_wall_ns, max_wall_ns


def load_live_public_trades(path: Path, min_wall_ns: int, max_wall_ns: int) -> list[PublicTrade]:
    trades: list[PublicTrade] = []
    for row in iter_json_lines(path) or []:
        wall_ns = to_int(row.get("recv_wall_ns") or row.get("wall_ns"))
        if wall_ns <= 0:
            continue
        if min_wall_ns and wall_ns < min_wall_ns:
            continue
        if max_wall_ns and wall_ns > max_wall_ns:
            continue
        qty = to_float(row.get("count_fp"), 0.0) / 100.0
        if qty <= 0:
            continue
        trades.append(
            PublicTrade(
                wall_ns=wall_ns,
                yes_price_c=price_mD_to_cents(row.get("yes_price_mD")),
                no_price_c=price_mD_to_cents(row.get("no_price_mD")),
                qty=qty,
                yes_taker=to_int(row.get("yes_taker")),
            )
        )
    return trades


def load_backfilled_public_trades(path: Path, min_wall_ns: int, max_wall_ns: int) -> list[PublicTrade]:
    trades: list[PublicTrade] = []
    for row in iter_json_lines(path) or []:
        wall_ns = iso_time_to_ns(row.get("created_time"))
        if wall_ns <= 0:
            ts_ms = to_int(row.get("ts_ms") or row.get("created_time_ms") or row.get("created_ts_ms"))
            wall_ns = ts_ms * 1_000_000 if ts_ms > 0 else 0
        if wall_ns <= 0:
            continue
        if min_wall_ns and wall_ns < min_wall_ns:
            continue
        if max_wall_ns and wall_ns > max_wall_ns:
            continue
        qty = to_float(row.get("count_fp"), 0.0)
        if qty <= 0:
            continue
        taker_side = str(row.get("taker_side", "")).lower()
        trades.append(
            PublicTrade(
                wall_ns=wall_ns,
                yes_price_c=price_dollars_to_cents(row.get("yes_price_dollars")),
                no_price_c=price_dollars_to_cents(row.get("no_price_dollars")),
                qty=qty,
                yes_taker=1 if taker_side == "yes" else 0,
            )
        )
    return trades


def build_trade_index(
    market_dirs: list[Path],
    backfill_trades_dir: Path | None,
    assets_needed: set[str],
    min_wall_ns: int,
    max_wall_ns: int,
) -> tuple[dict[str, list[PublicTrade]], dict[str, list[int]], Counter[str]]:
    by_asset: dict[str, list[PublicTrade]] = defaultdict(list)
    counts: Counter[str] = Counter()
    for market_dir in market_dirs:
        asset = asset_from_ticker(market_dir.name)
        if not asset:
            continue
        if assets_needed and asset not in assets_needed:
            continue
        live_path = market_dir / "public_trades.ndjson"
        for trade in load_live_public_trades(live_path, min_wall_ns, max_wall_ns):
            by_asset[asset].append(trade)
            counts["live_trade_rows"] += 1
        if live_path.exists():
            counts["live_trade_files"] += 1
    if backfill_trades_dir and backfill_trades_dir.exists():
        for path in sorted(backfill_trades_dir.glob("*.trades.ndjson")):
            ticker = path.name.removesuffix(".trades.ndjson")
            asset = asset_from_ticker(ticker)
            if not asset:
                continue
            if assets_needed and asset not in assets_needed:
                continue
            file_rows = 0
            for trade in load_backfilled_public_trades(path, min_wall_ns, max_wall_ns):
                by_asset[asset].append(trade)
                counts["backfill_trade_rows"] += 1
                file_rows += 1
            counts["backfill_trade_files"] += 1
            counts["nonempty_backfill_trade_files"] += int(file_rows > 0)
    walls: dict[str, list[int]] = {}
    for asset, trades in by_asset.items():
        trades.sort(key=lambda t: t.wall_ns)
        walls[asset] = [t.wall_ns for t in trades]
    for asset, trades in by_asset.items():
        counts[f"trade_rows_asset_{asset}"] = len(trades)
    return by_asset, walls, counts


def load_entry_snapshots(market_dir: Path) -> tuple[list[dict[str, Any]], list[int]]:
    rows: list[dict[str, Any]] = []
    ndjson = market_dir / "entry_snapshot.ndjson"
    if ndjson.exists():
        rows.extend(load_json_lines(ndjson))
    doc = load_json_doc(market_dir / "entry_snapshot.json")
    if doc:
        rows.append(doc)
    parsed = [row for row in rows if to_int(row.get("wall_ns")) > 0]
    parsed.sort(key=lambda r: to_int(r.get("wall_ns")))
    return parsed, [to_int(r.get("wall_ns")) for r in parsed]


def build_snapshot_index(market_dirs: list[Path]) -> dict[str, tuple[list[dict[str, Any]], list[int]]]:
    out: dict[str, tuple[list[dict[str, Any]], list[int]]] = {}
    for market_dir in market_dirs:
        rows, walls = load_entry_snapshots(market_dir)
        if rows:
            out[market_dir.name] = (rows, walls)
    return out


def side_price(trade: PublicTrade, side: str) -> int:
    return trade.yes_price_c if side == "YES" else trade.no_price_c


def is_signal_taker(trade: PublicTrade, side: str) -> bool:
    return bool(trade.yes_taker) if side == "YES" else not bool(trade.yes_taker)


def trade_features_for_row(
    trades: list[PublicTrade],
    walls: list[int],
    wall_ns: int,
    side: str,
    level: int,
    entry_cost: float,
    windows: list[int],
    near_cents: int,
) -> dict[str, float]:
    out: dict[str, float] = {}
    end = bisect.bisect_left(walls, wall_ns)
    for window in windows:
        start = bisect.bisect_left(walls, wall_ns - window * NS_PER_S)
        total_vol = total_count = 0.0
        at_vol = at_count = near_vol = near_count = 0.0
        above_vol = below_vol = 0.0
        sig_at = opp_at = sig_near = opp_near = 0.0
        near_notional = 0.0
        for trade in trades[start:end]:
            price = side_price(trade, side)
            if price <= 0:
                continue
            qty = trade.qty
            total_vol += qty
            total_count += 1.0
            if price > level:
                above_vol += qty
            elif price < level:
                below_vol += qty
            if price == level:
                at_vol += qty
                at_count += 1.0
                if is_signal_taker(trade, side):
                    sig_at += qty
                else:
                    opp_at += qty
            if abs(price - level) <= near_cents:
                near_vol += qty
                near_count += 1.0
                near_notional += qty * price
                if is_signal_taker(trade, side):
                    sig_near += qty
                else:
                    opp_near += qty
        prefix = f"plv_{window}s"
        out[f"{prefix}_total_volume"] = total_vol
        out[f"{prefix}_total_count"] = total_count
        out[f"{prefix}_at_volume"] = at_vol
        out[f"{prefix}_at_count"] = at_count
        out[f"{prefix}_near_volume"] = near_vol
        out[f"{prefix}_near_count"] = near_count
        out[f"{prefix}_above_volume"] = above_vol
        out[f"{prefix}_below_volume"] = below_vol
        out[f"{prefix}_at_volume_share"] = at_vol / total_vol if total_vol > 0 else 0.0
        out[f"{prefix}_near_volume_share"] = near_vol / total_vol if total_vol > 0 else 0.0
        out[f"{prefix}_above_volume_share"] = above_vol / total_vol if total_vol > 0 else 0.0
        out[f"{prefix}_below_volume_share"] = below_vol / total_vol if total_vol > 0 else 0.0
        out[f"{prefix}_signal_taker_at_volume"] = sig_at
        out[f"{prefix}_opp_taker_at_volume"] = opp_at
        out[f"{prefix}_signal_taker_near_volume"] = sig_near
        out[f"{prefix}_opp_taker_near_volume"] = opp_near
        out[f"{prefix}_net_signal_pressure_near"] = sig_near - opp_near
        out[f"{prefix}_signal_taker_share_near"] = sig_near / near_vol if near_vol > 0 else 0.0
        near_vwap = near_notional / near_vol if near_vol > 0 else 0.0
        out[f"{prefix}_near_vwap"] = near_vwap
        out[f"{prefix}_entry_vs_near_vwap"] = entry_cost - near_vwap if near_vwap > 0 else 0.0
    return out


def normalize_snapshot_price(price: Any) -> int:
    value = to_int(price)
    if value > 100:
        return int(round(value / 10.0))
    return value


def level_qty(levels: Any, price_c: int) -> float:
    total = 0.0
    if not isinstance(levels, list):
        return total
    for pair in levels:
        if not isinstance(pair, list | tuple) or len(pair) < 2:
            continue
        if normalize_snapshot_price(pair[0]) == price_c:
            total += to_float(pair[1], 0.0) / 100.0
    return total


def range_qty(levels: Any, lo: int, hi: int) -> float:
    total = 0.0
    if not isinstance(levels, list):
        return total
    for pair in levels:
        if not isinstance(pair, list | tuple) or len(pair) < 2:
            continue
        price = normalize_snapshot_price(pair[0])
        if lo <= price <= hi:
            total += to_float(pair[1], 0.0) / 100.0
    return total


def depth_features_for_row(
    snapshot_index: dict[str, tuple[list[dict[str, Any]], list[int]]],
    ticker: str,
    wall_ns: int,
    side: str,
    level: int,
    max_lag_s: float,
) -> dict[str, float]:
    out = {
        "depth_has_snapshot": 0.0,
        "depth_snapshot_lag_s": math.nan,
        "depth_signal_level_visible": 0.0,
    }
    found = snapshot_index.get(ticker)
    if not found:
        return out
    rows, walls = found
    idx = bisect.bisect_right(walls, wall_ns) - 1
    if idx < 0:
        return out
    snap = rows[idx]
    lag_s = (wall_ns - walls[idx]) / 1e9
    if lag_s < 0 or lag_s > max_lag_s:
        return out
    signal_bids = snap.get("yes_bids") if side == "YES" else snap.get("no_bids")
    signal_asks = snap.get("yes_asks") if side == "YES" else snap.get("no_asks")
    opp_bids = snap.get("no_bids") if side == "YES" else snap.get("yes_bids")
    opp_asks = snap.get("no_asks") if side == "YES" else snap.get("yes_asks")
    top_signal_bid = range_qty(signal_bids, 0, 100)
    top_signal_ask = range_qty(signal_asks, 0, 100)
    top_opp_bid = range_qty(opp_bids, 0, 100)
    top_opp_ask = range_qty(opp_asks, 0, 100)
    signal_at_bid = level_qty(signal_bids, level)
    signal_at_ask = level_qty(signal_asks, level)
    opp_at_bid = level_qty(opp_bids, 100 - level)
    opp_at_ask = level_qty(opp_asks, 100 - level)
    out.update(
        {
            "depth_has_snapshot": 1.0,
            "depth_snapshot_lag_s": lag_s,
            "depth_signal_level_visible": 1.0 if signal_at_bid > 0 or signal_at_ask > 0 else 0.0,
            "depth_signal_bid_at_level": signal_at_bid,
            "depth_signal_ask_at_level": signal_at_ask,
            "depth_opp_bid_at_complement_level": opp_at_bid,
            "depth_opp_ask_at_complement_level": opp_at_ask,
            "depth_signal_bid_within_1c": range_qty(signal_bids, level - 1, level + 1),
            "depth_signal_ask_within_1c": range_qty(signal_asks, level - 1, level + 1),
            "depth_signal_bid_within_3c": range_qty(signal_bids, level - 3, level + 3),
            "depth_signal_ask_within_3c": range_qty(signal_asks, level - 3, level + 3),
            "depth_signal_bid_within_5c": range_qty(signal_bids, level - 5, level + 5),
            "depth_signal_ask_within_5c": range_qty(signal_asks, level - 5, level + 5),
            "depth_top10_signal_bid_depth": top_signal_bid,
            "depth_top10_signal_ask_depth": top_signal_ask,
            "depth_top10_opp_bid_depth": top_opp_bid,
            "depth_top10_opp_ask_depth": top_opp_ask,
            "depth_top10_signal_bid_share": top_signal_bid / (top_signal_bid + top_opp_bid)
            if top_signal_bid + top_opp_bid > 0
            else 0.0,
            "depth_top10_signal_ask_share": top_signal_ask / (top_signal_ask + top_opp_ask)
            if top_signal_ask + top_opp_ask > 0
            else 0.0,
            "depth_top10_book_imbalance": (top_signal_bid - top_signal_ask) / (top_signal_bid + top_signal_ask)
            if top_signal_bid + top_signal_ask > 0
            else 0.0,
        }
    )
    return out


def new_feature_columns(windows: list[int]) -> list[str]:
    cols: list[str] = []
    for window in windows:
        prefix = f"plv_{window}s"
        cols.extend(
            [
                f"{prefix}_total_volume",
                f"{prefix}_total_count",
                f"{prefix}_at_volume",
                f"{prefix}_at_count",
                f"{prefix}_near_volume",
                f"{prefix}_near_count",
                f"{prefix}_above_volume",
                f"{prefix}_below_volume",
                f"{prefix}_at_volume_share",
                f"{prefix}_near_volume_share",
                f"{prefix}_above_volume_share",
                f"{prefix}_below_volume_share",
                f"{prefix}_signal_taker_at_volume",
                f"{prefix}_opp_taker_at_volume",
                f"{prefix}_signal_taker_near_volume",
                f"{prefix}_opp_taker_near_volume",
                f"{prefix}_net_signal_pressure_near",
                f"{prefix}_signal_taker_share_near",
                f"{prefix}_near_vwap",
                f"{prefix}_entry_vs_near_vwap",
            ]
        )
    cols.extend(
        [
            "depth_has_snapshot",
            "depth_snapshot_lag_s",
            "depth_signal_level_visible",
            "depth_signal_bid_at_level",
            "depth_signal_ask_at_level",
            "depth_opp_bid_at_complement_level",
            "depth_opp_ask_at_complement_level",
            "depth_signal_bid_within_1c",
            "depth_signal_ask_within_1c",
            "depth_signal_bid_within_3c",
            "depth_signal_ask_within_3c",
            "depth_signal_bid_within_5c",
            "depth_signal_ask_within_5c",
            "depth_top10_signal_bid_depth",
            "depth_top10_signal_ask_depth",
            "depth_top10_opp_bid_depth",
            "depth_top10_opp_ask_depth",
            "depth_top10_signal_bid_share",
            "depth_top10_signal_ask_share",
            "depth_top10_book_imbalance",
        ]
    )
    return cols


def main() -> int:
    args = parse_args()
    windows = [int(x) for x in args.windows.split(",") if x.strip()]
    max_lookback_ns = max(windows, default=0) * NS_PER_S
    assets_needed, min_trade_wall_ns, max_trade_wall_ns = scan_input_requirements(
        args.input, args.max_rows, max_lookback_ns
    )
    market_dirs = [Path(p) for p in glob.glob(str(args.data_root / args.market_glob))]
    market_dirs = sorted([p for p in market_dirs if p.is_dir()])
    print(f"loading trades from {len(market_dirs)} market dirs")
    trades_by_asset, walls_by_asset, trade_counts = build_trade_index(
        market_dirs,
        args.backfill_trades_dir,
        assets_needed,
        min_trade_wall_ns,
        max_trade_wall_ns,
    )
    print(
        json.dumps(
            {
                "input_scan": {
                    "assets": sorted(assets_needed),
                    "min_trade_wall_ns": min_trade_wall_ns,
                    "max_trade_wall_ns": max_trade_wall_ns,
                    "lookback_s": max(windows, default=0),
                    "max_rows": args.max_rows,
                },
                "trade_index": trade_counts,
            },
            indent=2,
        ),
        flush=True,
    )
    snapshot_index = build_snapshot_index(market_dirs)
    counts: Counter[str] = Counter()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    extra_cols = new_feature_columns(windows)
    with args.input.open("r", encoding="utf-8", newline="") as in_f:
        reader = csv.DictReader(in_f)
        if not reader.fieldnames:
            raise SystemExit(f"missing CSV header: {args.input}")
        fieldnames = list(reader.fieldnames)
        fieldnames.extend([col for col in extra_cols if col not in fieldnames])
        opener = gzip.open if args.out.suffix == ".gz" else open
        with opener(args.out, "wt", encoding="utf-8", newline="") as out_f:
            writer = csv.DictWriter(out_f, fieldnames=fieldnames)
            writer.writeheader()
            for row in reader:
                ticker = str(row.get("ticker", ""))
                asset = str(row.get("asset", ""))
                side = str(row.get("side", "")).upper()
                wall_ns = to_int(row.get("wall_ns"))
                level = to_int(row.get("level"))
                entry_cost = to_float(row.get("entry_cost"))
                if not math.isfinite(entry_cost):
                    entry_cost = to_float(row.get("entry_price_mD")) / 10.0
                trades = trades_by_asset.get(asset, [])
                walls = walls_by_asset.get(asset, [])
                features = trade_features_for_row(
                    trades, walls, wall_ns, side, level, entry_cost, windows, args.near_cents
                )
                features.update(
                    depth_features_for_row(snapshot_index, ticker, wall_ns, side, level, args.depth_max_lag_s)
                )
                for col in extra_cols:
                    row[col] = features.get(col, 0.0)
                writer.writerow(row)
                counts["rows"] += 1
                counts["rows_with_trade_history"] += int(bool(trades))
                counts["rows_with_depth_snapshot"] += int(features.get("depth_has_snapshot", 0.0) > 0)
                if args.max_rows and counts["rows"] >= args.max_rows:
                    break
                if counts["rows"] % 50_000 == 0:
                    print(f"enriched_rows={counts['rows']}", flush=True)
    print(
        json.dumps(
            {
                "out": str(args.out),
                "rows": counts["rows"],
                "new_columns": extra_cols,
                "counts": counts,
                "bad_json_lines": BAD_JSON_LINES,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
