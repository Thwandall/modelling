#!/usr/bin/env python3
"""Build an adverse-selection / toxic-flow dataset from live crypto market logs.

The output is intentionally a flat CSV so it is easy to audit before training.
Generated artifacts should live outside this repository; keep this script here so
the modelling process is versioned with the model bundles.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import glob
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


NS_PER_S = 1_000_000_000
MD_PER_CENT = 10


@dataclass(frozen=True)
class Tick:
    wall_ns: int
    yes_bid_mD: float
    yes_ask_mD: float
    no_bid_mD: float
    no_ask_mD: float
    spread_mD: float
    microprice_x1000: float
    imbalance_x10000: float
    yes_bid_qty: float
    no_bid_qty: float
    book_seq: int


@dataclass(frozen=True)
class RtiPoint:
    wall_ns: int
    rti_cents: float
    rti_vs_strike_bps: float
    consolidated_bid_cents: float
    consolidated_ask_cents: float
    utilized_depth_lots: float
    dynamic_cap_lots: float
    eligible_venues: float


@dataclass(frozen=True)
class PublicTrade:
    wall_ns: int
    yes_price_mD: float
    no_price_mD: float
    qty: float
    yes_taker: int


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
    parser.add_argument("--approved-only", action="store_true", default=False)
    parser.add_argument(
        "--all-candidates",
        action="store_true",
        help="Include rejected rows too. Overrides --approved-only.",
    )
    parser.add_argument("--max-markets", type=int, default=0)
    parser.add_argument("--profit-target-cents", type=float, default=2.0)
    parser.add_argument("--target-window-s", type=int, default=120)
    parser.add_argument("--markout-seconds", default="1,3,5,15,30,60")
    parser.add_argument(
        "--fill-match-window-s",
        type=float,
        default=2.0,
        help="Max distance between candidate wall_ns and fill order_ts_ns.",
    )
    parser.add_argument(
        "--bad-markout-mD",
        type=float,
        default=-30.0,
        help="Label bad if same-side bid drops at least this much within 30s.",
    )
    return parser.parse_args()


def asset_from_ticker(ticker: str) -> str:
    m = re.match(r"KX([A-Z]+)15M-", ticker)
    return m.group(1) if m else ""


def load_json_lines(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
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
                out.append(row)
    return out


def load_json_doc(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            row = json.load(f)
    except json.JSONDecodeError:
        return None
    return row if isinstance(row, dict) else None


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


def load_ticks(path: Path) -> tuple[list[Tick], list[int]]:
    if not path.exists():
        return [], []
    ticks: list[Tick] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            wall_ns = to_int(row.get("wall_ns"))
            if wall_ns <= 0:
                continue
            ticks.append(
                Tick(
                    wall_ns=wall_ns,
                    yes_bid_mD=to_float(row.get("yes_bid_mD")),
                    yes_ask_mD=to_float(row.get("yes_ask_mD")),
                    no_bid_mD=to_float(row.get("no_bid_mD")),
                    no_ask_mD=to_float(row.get("no_ask_mD")),
                    spread_mD=to_float(row.get("spread")),
                    microprice_x1000=to_float(row.get("microprice_x1000")),
                    imbalance_x10000=to_float(row.get("imbalance_x10000")),
                    yes_bid_qty=to_float(row.get("yes_bid_qty")),
                    no_bid_qty=to_float(row.get("no_bid_qty")),
                    book_seq=to_int(row.get("book_seq")),
                )
            )
    ticks.sort(key=lambda t: t.wall_ns)
    return ticks, [t.wall_ns for t in ticks]


def load_rti_points(path: Path) -> tuple[list[RtiPoint], list[int]]:
    if not path.exists():
        return [], []
    points: list[RtiPoint] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            wall_ns = to_int(row.get("wall_ns"))
            if wall_ns <= 0:
                continue
            points.append(
                RtiPoint(
                    wall_ns=wall_ns,
                    rti_cents=to_float(row.get("rti_cents")),
                    rti_vs_strike_bps=to_float(row.get("rti_vs_strike_bps")),
                    consolidated_bid_cents=to_float(row.get("consolidated_bid_cents")),
                    consolidated_ask_cents=to_float(row.get("consolidated_ask_cents")),
                    utilized_depth_lots=to_float(row.get("utilized_depth_lots")),
                    dynamic_cap_lots=to_float(row.get("dynamic_cap_lots")),
                    eligible_venues=to_float(row.get("eligible_venues")),
                )
            )
    points.sort(key=lambda p: p.wall_ns)
    return points, [p.wall_ns for p in points]


def load_public_trades(path: Path) -> list[PublicTrade]:
    trades: list[PublicTrade] = []
    for row in load_json_lines(path):
        wall_ns = to_int(row.get("recv_wall_ns") or row.get("wall_ns"))
        if wall_ns <= 0:
            continue
        trades.append(
            PublicTrade(
                wall_ns=wall_ns,
                yes_price_mD=to_float(row.get("yes_price_mD")),
                no_price_mD=to_float(row.get("no_price_mD")),
                qty=to_float(row.get("count_fp"), 0.0) / 100.0,
                yes_taker=to_int(row.get("yes_taker")),
            )
        )
    return trades


def build_continuous_trade_index(
    market_dirs: list[Path],
) -> tuple[dict[str, list[PublicTrade]], dict[str, list[int]]]:
    by_asset: dict[str, list[PublicTrade]] = defaultdict(list)
    seen: set[tuple[str, int, float, float, float, int]] = set()
    for market_dir in market_dirs:
        asset = asset_from_ticker(market_dir.name)
        if not asset:
            continue
        for trade in load_public_trades(market_dir / "public_trades.ndjson"):
            key = (
                asset,
                trade.wall_ns,
                trade.yes_price_mD,
                trade.no_price_mD,
                trade.qty,
                trade.yes_taker,
            )
            if key in seen:
                continue
            seen.add(key)
            by_asset[asset].append(trade)
    walls: dict[str, list[int]] = {}
    for asset, trades in by_asset.items():
        trades.sort(key=lambda t: t.wall_ns)
        walls[asset] = [t.wall_ns for t in trades]
    return by_asset, walls


def tick_at_or_before(ticks: list[Tick], walls: list[int], wall_ns: int) -> Tick | None:
    idx = bisect.bisect_right(walls, wall_ns) - 1
    if idx < 0:
        return None
    return ticks[idx]


def tick_at_or_after(ticks: list[Tick], walls: list[int], wall_ns: int) -> Tick | None:
    idx = bisect.bisect_left(walls, wall_ns)
    if idx >= len(ticks):
        return None
    return ticks[idx]


def ticks_between(ticks: list[Tick], walls: list[int], start_ns: int, end_ns: int) -> list[Tick]:
    start_idx = bisect.bisect_left(walls, start_ns)
    end_idx = bisect.bisect_right(walls, end_ns)
    return ticks[start_idx:end_idx]


def rti_at_or_before(points: list[RtiPoint], walls: list[int], wall_ns: int) -> RtiPoint | None:
    idx = bisect.bisect_right(walls, wall_ns) - 1
    if idx < 0:
        return None
    return points[idx]


def rti_between(points: list[RtiPoint], walls: list[int], start_ns: int, end_ns: int) -> list[RtiPoint]:
    start_idx = bisect.bisect_left(walls, start_ns)
    end_idx = bisect.bisect_right(walls, end_ns)
    return points[start_idx:end_idx]


def same_bid(tick: Tick | None, side: str) -> float:
    if tick is None:
        return math.nan
    return tick.yes_bid_mD if side == "YES" else tick.no_bid_mD


def same_ask(tick: Tick | None, side: str) -> float:
    if tick is None:
        return math.nan
    return tick.yes_ask_mD if side == "YES" else tick.no_ask_mD


def opp_bid(tick: Tick | None, side: str) -> float:
    if tick is None:
        return math.nan
    return tick.no_bid_mD if side == "YES" else tick.yes_bid_mD


def opp_ask(tick: Tick | None, side: str) -> float:
    if tick is None:
        return math.nan
    return tick.no_ask_mD if side == "YES" else tick.yes_ask_mD


def load_feature_names(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        schema = json.load(f)
    names = schema.get("encoded_feature_order")
    if not isinstance(names, list) or not names:
        raise ValueError(f"{path} does not contain encoded_feature_order")
    return [str(x) for x in names]


def load_outcome(market_dir: Path) -> tuple[int | None, bool]:
    path = market_dir / "outcome.ndjson"
    if path.exists():
        rows = load_json_lines(path)
        for row in reversed(rows):
            if "result_yes" in row:
                return to_int(row.get("result_yes")), True
    path = market_dir / "outcome.json"
    row = load_json_doc(path)
    if row and "result_yes" in row:
        return to_int(row.get("result_yes")), True
    if path.exists():
        return None, True
    return None, False


def index_entry_fills(market_dir: Path) -> dict[tuple[str, int], list[dict[str, Any]]]:
    fills = load_json_lines(market_dir / "fills.ndjson")
    by_side_price: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for fill in fills:
        if to_int(fill.get("is_exit")):
            continue
        side = "YES" if to_int(fill.get("buy_yes")) else "NO"
        price = to_int(fill.get("fill_price_mD"))
        by_side_price[(side, price)].append(fill)
    for rows in by_side_price.values():
        rows.sort(key=lambda r: to_int(r.get("order_ts_ns")))
    return by_side_price


def match_entry_fill(
    fill_index: dict[tuple[str, int], list[dict[str, Any]]],
    side: str,
    entry_mD: int,
    wall_ns: int,
    max_distance_ns: int,
) -> dict[str, Any] | None:
    rows = fill_index.get((side, entry_mD), [])
    best: dict[str, Any] | None = None
    best_dist = max_distance_ns + 1
    for row in rows:
        dist = abs(to_int(row.get("order_ts_ns")) - wall_ns)
        if dist < best_dist:
            best = row
            best_dist = dist
    if best is None or best_dist > max_distance_ns:
        return None
    return best


def first_target_hit(
    ticks: list[Tick],
    walls: list[int],
    side: str,
    start_ns: int,
    target_mD: float,
    window_s: int,
) -> tuple[int, float]:
    for tick in ticks_between(ticks, walls, start_ns, start_ns + window_s * NS_PER_S):
        bid = same_bid(tick, side)
        if not math.isnan(bid) and bid >= target_mD:
            return 1, (tick.wall_ns - start_ns) / NS_PER_S
    return 0, math.nan


def markout_stats(
    ticks: list[Tick],
    walls: list[int],
    side: str,
    start_ns: int,
    start_bid: float,
    horizon_s: int,
) -> tuple[float, float]:
    if math.isnan(start_bid):
        return math.nan, math.nan
    worst = math.nan
    best = math.nan
    for tick in ticks_between(ticks, walls, start_ns, start_ns + horizon_s * NS_PER_S):
        bid = same_bid(tick, side)
        if math.isnan(bid):
            continue
        change = bid - start_bid
        worst = change if math.isnan(worst) else min(worst, change)
        best = change if math.isnan(best) else max(best, change)
    return worst, best


def add_book_velocity_features(row: dict[str, Any], ticks: list[Tick], walls: list[int],
                               wall_ns: int, side: str, signal_tick: Tick | None) -> None:
    signal_bid = same_bid(signal_tick, side)
    signal_ask = same_ask(signal_tick, side)
    signal_opp_bid = opp_bid(signal_tick, side)
    for sec in (1, 3, 5, 15):
        prior = tick_at_or_before(ticks, walls, wall_ns - sec * NS_PER_S)
        prior_bid = same_bid(prior, side)
        prior_ask = same_ask(prior, side)
        prior_opp = opp_bid(prior, side)
        row[f"book_same_bid_change_{sec}s_mD"] = (
            signal_bid - prior_bid
            if not math.isnan(signal_bid) and not math.isnan(prior_bid)
            else math.nan
        )
        row[f"book_same_ask_change_{sec}s_mD"] = (
            signal_ask - prior_ask
            if not math.isnan(signal_ask) and not math.isnan(prior_ask)
            else math.nan
        )
        row[f"book_opp_bid_change_{sec}s_mD"] = (
            signal_opp_bid - prior_opp
            if not math.isnan(signal_opp_bid) and not math.isnan(prior_opp)
            else math.nan
        )
        row[f"book_microprice_change_{sec}s"] = (
            signal_tick.microprice_x1000 - prior.microprice_x1000
            if signal_tick and prior
            else math.nan
        )
        row[f"book_imbalance_change_{sec}s"] = (
            signal_tick.imbalance_x10000 - prior.imbalance_x10000
            if signal_tick and prior
            else math.nan
        )
        row[f"book_spread_change_{sec}s_mD"] = (
            signal_tick.spread_mD - prior.spread_mD
            if signal_tick and prior
            else math.nan
        )
        row[f"book_yes_bid_qty_change_{sec}s"] = (
            signal_tick.yes_bid_qty - prior.yes_bid_qty
            if signal_tick and prior
            else math.nan
        )
        row[f"book_no_bid_qty_change_{sec}s"] = (
            signal_tick.no_bid_qty - prior.no_bid_qty
            if signal_tick and prior
            else math.nan
        )


def add_rti_path_features(row: dict[str, Any], points: list[RtiPoint], walls: list[int],
                          wall_ns: int) -> None:
    current = rti_at_or_before(points, walls, wall_ns)
    current_rti = current.rti_cents if current else math.nan
    current_dist = current.rti_vs_strike_bps if current else math.nan
    for sec in (1, 3, 5, 15, 30, 60):
        prior = rti_at_or_before(points, walls, wall_ns - sec * NS_PER_S)
        row[f"rti_path_change_{sec}s_cents"] = (
            current_rti - prior.rti_cents
            if prior and not math.isnan(current_rti) and not math.isnan(prior.rti_cents)
            else math.nan
        )
        row[f"rti_dist_change_{sec}s_bps"] = (
            current_dist - prior.rti_vs_strike_bps
            if prior and not math.isnan(current_dist) and not math.isnan(prior.rti_vs_strike_bps)
            else math.nan
        )
    pts60 = rti_between(points, walls, wall_ns - 60 * NS_PER_S, wall_ns)
    rti_vals = [p.rti_cents for p in pts60 if not math.isnan(p.rti_cents)]
    dist_vals = [p.rti_vs_strike_bps for p in pts60 if not math.isnan(p.rti_vs_strike_bps)]
    if rti_vals and not math.isnan(current_rti):
        row["rti_drawdown_from_60s_high_cents"] = current_rti - max(rti_vals)
        row["rti_bounce_from_60s_low_cents"] = current_rti - min(rti_vals)
        row["rti_range_60s_cents"] = max(rti_vals) - min(rti_vals)
    else:
        row["rti_drawdown_from_60s_high_cents"] = math.nan
        row["rti_bounce_from_60s_low_cents"] = math.nan
        row["rti_range_60s_cents"] = math.nan
    if dist_vals and not math.isnan(current_dist):
        row["rti_dist_drawdown_from_60s_high_bps"] = current_dist - max(dist_vals)
        row["rti_dist_bounce_from_60s_low_bps"] = current_dist - min(dist_vals)
    else:
        row["rti_dist_drawdown_from_60s_high_bps"] = math.nan
        row["rti_dist_bounce_from_60s_low_bps"] = math.nan


def add_continuous_trade_features(
    row: dict[str, Any],
    trades: list[PublicTrade],
    walls: list[int],
    wall_ns: int,
    side: str,
    entry_mD: int,
) -> None:
    idx_now = bisect.bisect_left(walls, wall_ns)
    prior = trades[idx_now - 1] if idx_now > 0 else None
    if prior:
        row["asset_trade_last_age_s"] = (wall_ns - prior.wall_ns) / NS_PER_S
        row["asset_trade_last_yes_price"] = prior.yes_price_mD / 10.0
        row["asset_trade_last_no_price"] = prior.no_price_mD / 10.0
    else:
        row["asset_trade_last_age_s"] = 1e9
        row["asset_trade_last_yes_price"] = 0.0
        row["asset_trade_last_no_price"] = 0.0
    for sec in (30, 60, 300, 900):
        start_idx = bisect.bisect_left(walls, wall_ns - sec * NS_PER_S)
        window = trades[start_idx:idx_now]
        count = len(window)
        volume = sum(t.qty for t in window)
        yes_taker_vol = sum(t.qty for t in window if t.yes_taker)
        no_taker_vol = volume - yes_taker_vol
        yes_notional = sum(t.qty * (t.yes_price_mD / 10.0) for t in window)
        no_notional = sum(t.qty * (t.no_price_mD / 10.0) for t in window)
        vwap_yes = yes_notional / volume if volume > 0 else 0.0
        vwap_no = no_notional / volume if volume > 0 else 0.0
        net_yes_pressure = yes_taker_vol - no_taker_vol
        side_pressure = net_yes_pressure if side == "YES" else -net_yes_pressure
        side_vwap = vwap_yes if side == "YES" else vwap_no
        row[f"asset_trade_{sec}s_count"] = count
        row[f"asset_trade_{sec}s_volume"] = volume
        row[f"asset_trade_{sec}s_vwap_yes"] = vwap_yes
        row[f"asset_trade_{sec}s_vwap_no"] = vwap_no
        row[f"asset_trade_{sec}s_yes_taker_vol"] = yes_taker_vol
        row[f"asset_trade_{sec}s_no_taker_vol"] = no_taker_vol
        row[f"asset_trade_{sec}s_net_yes_pressure"] = net_yes_pressure
        row[f"asset_trade_{sec}s_yes_taker_share"] = yes_taker_vol / volume if volume > 0 else 0.0
        row[f"asset_trade_{sec}s_avg_size"] = volume / count if count else 0.0
        row[f"asset_trade_{sec}s_signal_side_pressure"] = side_pressure
        row[f"asset_trade_{sec}s_entry_vs_vwap"] = entry_mD / 10.0 - side_vwap if side_vwap > 0 else 0.0
        row[f"asset_trade_{sec}s_signal_side_vwap_edge"] = side_vwap - entry_mD / 10.0 if side_vwap > 0 else 0.0
        row[f"asset_trade_{sec}s_price_momentum"] = (
            (window[-1].yes_price_mD - window[0].yes_price_mD) / 10.0
            if count >= 2
            else 0.0
        )
        row[f"asset_trade_{sec}s_large_count"] = sum(1 for t in window if t.qty >= 10.0)
        row[f"asset_trade_{sec}s_large_volume"] = sum(t.qty for t in window if t.qty >= 10.0)
    row["asset_trade_volume_burst_30s_vs_5m"] = (
        row["asset_trade_30s_volume"] / row["asset_trade_300s_volume"]
        if row["asset_trade_300s_volume"] > 0
        else 0.0
    )
    row["asset_trade_count_burst_30s_vs_5m"] = (
        row["asset_trade_30s_count"] / row["asset_trade_300s_count"]
        if row["asset_trade_300s_count"] > 0
        else 0.0
    )
    row["asset_trade_pressure_accel_30s_vs_5m"] = (
        row["asset_trade_30s_signal_side_pressure"] -
        row["asset_trade_300s_signal_side_pressure"] / 10.0
    )


def settlement_pnl_mD(side: str, entry_mD: int, result_yes: int | None) -> float:
    if result_yes is None:
        return math.nan
    side_wins = (side == "YES" and result_yes == 1) or (side == "NO" and result_yes == 0)
    return 1000 - entry_mD if side_wins else -entry_mD


def build_rows_for_market(
    market_dir: Path,
    feature_names: list[str],
    args: argparse.Namespace,
    markout_seconds: list[int],
    asset_trades: dict[str, list[PublicTrade]],
    asset_trade_walls: dict[str, list[int]],
) -> tuple[list[dict[str, Any]], Counter]:
    stats: Counter = Counter()
    ticker = market_dir.name
    asset = asset_from_ticker(ticker)
    candidates = load_json_lines(market_dir / "ml_feature_vectors.ndjson")
    if not candidates:
        stats["missing_feature_vectors"] += 1
        return [], stats
    ticks, tick_walls = load_ticks(market_dir / "ticks.csv")
    if not ticks:
        stats["missing_ticks"] += 1
    rti_points, rti_walls = load_rti_points(market_dir / "rti_snapshots.csv")
    if not rti_points:
        stats["missing_rti_points"] += 1
    result_yes, has_outcome_file = load_outcome(market_dir)
    if result_yes is None:
        stats["missing_outcome"] += 1
        if has_outcome_file:
            stats["invalid_outcome"] += 1
    fill_index = index_entry_fills(market_dir)

    sorted_candidates = sorted(candidates, key=lambda r: to_int(r.get("wall_ns")))
    candidate_walls = [to_int(r.get("wall_ns")) for r in sorted_candidates]
    rows: list[dict[str, Any]] = []
    max_fill_distance_ns = int(args.fill_match_window_s * NS_PER_S)
    target_delta_mD = args.profit_target_cents * MD_PER_CENT

    for candidate in sorted_candidates:
        if args.approved_only and not args.all_candidates and not to_int(candidate.get("trade")):
            continue
        stats["candidate_rows_seen"] += 1
        wall_ns = to_int(candidate.get("wall_ns"))
        side = str(candidate.get("side") or "")
        entry_mD = to_int(candidate.get("entry_price_mD"))
        if wall_ns <= 0 or side not in ("YES", "NO") or entry_mD <= 0:
            stats["invalid_candidate_rows"] += 1
            continue

        signal_tick = tick_at_or_before(ticks, tick_walls, wall_ns)
        signal_bid = same_bid(signal_tick, side)
        signal_ask = same_ask(signal_tick, side)
        target_mD = entry_mD + target_delta_mD
        target_hit, target_seconds = first_target_hit(
            ticks, tick_walls, side, wall_ns, target_mD, args.target_window_s
        )
        adverse_30, favorable_30 = markout_stats(ticks, tick_walls, side, wall_ns, signal_bid, 30)
        pnl_to_settle = settlement_pnl_mD(side, entry_mD, result_yes)
        fill = match_entry_fill(fill_index, side, entry_mD, wall_ns, max_fill_distance_ns)

        row: dict[str, Any] = {
            "ticker": ticker,
            "asset": asset,
            "wall_ns": wall_ns,
            "side": side,
            "level": candidate.get("level"),
            "tte_s": candidate.get("tte_s"),
            "tte_bucket": candidate.get("tte_bucket"),
            "entry_price_mD": entry_mD,
            "p_yes": candidate.get("p_yes"),
            "ml_edge": candidate.get("ml_edge"),
            "meta_p_good": candidate.get("meta_p_good"),
            "bucket_threshold": candidate.get("bucket_threshold"),
            "trade": to_int(candidate.get("trade")),
            "reason": candidate.get("reason"),
            "result_yes": result_yes if result_yes is not None else "",
            "settlement_pnl_mD": pnl_to_settle,
            "signal_same_bid_mD": signal_bid,
            "signal_same_ask_mD": signal_ask,
            "signal_opp_bid_mD": opp_bid(signal_tick, side),
            "signal_opp_ask_mD": opp_ask(signal_tick, side),
            "signal_spread_mD": signal_tick.spread_mD if signal_tick else math.nan,
            "signal_microprice_x1000": signal_tick.microprice_x1000 if signal_tick else math.nan,
            "signal_imbalance_x10000": signal_tick.imbalance_x10000 if signal_tick else math.nan,
            "entry_minus_same_bid_mD": entry_mD - signal_bid if not math.isnan(signal_bid) else math.nan,
            "same_ask_minus_entry_mD": signal_ask - entry_mD if not math.isnan(signal_ask) else math.nan,
            "would_cross_now": int(not math.isnan(signal_ask) and signal_ask <= entry_mD),
            "target_profit_cents": args.profit_target_cents,
            "target_price_mD": target_mD,
            f"target_hit_{args.target_window_s}s": target_hit,
            f"target_seconds_{args.target_window_s}s": target_seconds,
            "max_adverse_bid_change_30s_mD": adverse_30,
            "max_favorable_bid_change_30s_mD": favorable_30,
            "actual_filled": int(fill is not None),
            "fill_lag_exchange_s": math.nan,
            "fill_fee_mD": math.nan,
            "fill_is_taker": "",
            "label_bad_markout_30s": int(not math.isnan(adverse_30) and adverse_30 <= args.bad_markout_mD),
            f"label_failed_target_{args.target_window_s}s": int(target_hit == 0),
            "label_loses_at_settlement": int(not math.isnan(pnl_to_settle) and pnl_to_settle < 0),
            "label_toxic": "",
        }
        if fill is not None:
            exchange_ts_ms = to_int(fill.get("exchange_ts_ms"))
            if exchange_ts_ms > 0:
                row["fill_lag_exchange_s"] = exchange_ts_ms / 1000.0 - wall_ns / NS_PER_S
            row["fill_fee_mD"] = to_float(fill.get("fee_mD"))
            row["fill_is_taker"] = to_int(fill.get("is_taker"))
            stats["matched_entry_fills"] += 1

        add_book_velocity_features(row, ticks, tick_walls, wall_ns, side, signal_tick)
        add_rti_path_features(row, rti_points, rti_walls, wall_ns)
        add_continuous_trade_features(
            row,
            asset_trades.get(asset, []),
            asset_trade_walls.get(asset, []),
            wall_ns,
            side,
            entry_mD,
        )

        for sec in markout_seconds:
            future_tick = tick_at_or_after(ticks, tick_walls, wall_ns + sec * NS_PER_S)
            future_bid = same_bid(future_tick, side)
            future_micro = future_tick.microprice_x1000 if future_tick else math.nan
            row[f"same_bid_markout_{sec}s_mD"] = (
                future_bid - signal_bid
                if not math.isnan(future_bid) and not math.isnan(signal_bid)
                else math.nan
            )
            row[f"microprice_markout_{sec}s"] = (
                future_micro - signal_tick.microprice_x1000
                if signal_tick and future_tick and not math.isnan(future_micro)
                else math.nan
            )

        for window_s in (1, 5):
            start = wall_ns - window_s * NS_PER_S
            start_idx = bisect.bisect_left(candidate_walls, start)
            end_idx = bisect.bisect_right(candidate_walls, wall_ns)
            recent = sorted_candidates[start_idx:end_idx]
            row[f"candidate_count_{window_s}s"] = len(recent)
            row[f"same_side_candidate_count_{window_s}s"] = sum(
                1 for r in recent if r.get("side") == side
            )
            row[f"approved_count_{window_s}s"] = sum(1 for r in recent if to_int(r.get("trade")))

        base_values = candidate.get("base_values") or []
        for idx, name in enumerate(feature_names):
            row[f"base_{name}"] = base_values[idx] if idx < len(base_values) else math.nan

        toxic_parts = (
            row["label_bad_markout_30s"],
            row[f"label_failed_target_{args.target_window_s}s"],
            row["label_loses_at_settlement"],
        )
        row["label_toxic"] = int(any(toxic_parts))
        rows.append(row)

    stats["rows_written"] += len(rows)
    return rows, stats


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    if args.all_candidates:
        args.approved_only = False
    markout_seconds = [int(x) for x in args.markout_seconds.split(",") if x.strip()]
    feature_names = load_feature_names(args.feature_schema)
    market_dirs = [Path(p) for p in glob.glob(str(args.data_root / args.market_glob))]
    market_dirs = sorted(p for p in market_dirs if p.is_dir())
    if args.max_markets:
        market_dirs = market_dirs[: args.max_markets]
    asset_trades, asset_trade_walls = build_continuous_trade_index(market_dirs)

    all_rows: list[dict[str, Any]] = []
    stats: Counter = Counter()
    by_asset: Counter = Counter()
    for idx, market_dir in enumerate(market_dirs, 1):
        rows, market_stats = build_rows_for_market(
            market_dir,
            feature_names,
            args,
            markout_seconds,
            asset_trades,
            asset_trade_walls,
        )
        all_rows.extend(rows)
        stats.update(market_stats)
        if rows:
            by_asset[rows[0]["asset"]] += len(rows)
        if idx % 500 == 0:
            print(f"processed_markets={idx} rows={len(all_rows)}")

    write_csv(args.out, all_rows)
    print(f"markets_scanned={len(market_dirs)}")
    print(f"rows_written={len(all_rows)}")
    for key in sorted(stats):
        print(f"{key}={stats[key]}")
    for asset, count in sorted(by_asset.items()):
        print(f"asset_rows.{asset}={count}")
    if all_rows:
        toxic = sum(to_int(r.get("label_toxic")) for r in all_rows)
        filled = sum(to_int(r.get("actual_filled")) for r in all_rows)
        target_col = f"target_hit_{args.target_window_s}s"
        target_hits = sum(to_int(r.get(target_col)) for r in all_rows)
        print(f"label_toxic_rate={toxic / len(all_rows):.4f}")
        print(f"actual_fill_rate={filled / len(all_rows):.4f}")
        print(f"target_hit_rate={target_hits / len(all_rows):.4f}")
    print(f"out={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
