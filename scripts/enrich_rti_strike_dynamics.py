#!/usr/bin/env python3
"""Add RTI distance/momentum features relative to the Kalshi strike."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


WINDOWS = (5, 15, 30, 60)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--compression", choices=["infer", "gzip", "none"], default="infer")
    return parser.parse_args()


def bps(price: pd.Series, strike: pd.Series) -> pd.Series:
    return np.where((price > 0) & (strike > 0), (price - strike) * 10000.0 / strike, np.nan)


def main() -> int:
    args = parse_args()
    df = pd.read_csv(args.input, low_memory=False)

    required = {"rti_cents", "floor_strike_cents", "rti_vs_strike_bps", "side"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"missing required columns: {missing}")

    rti = pd.to_numeric(df["rti_cents"], errors="coerce")
    strike = pd.to_numeric(df["floor_strike_cents"], errors="coerce")
    current_dist = pd.to_numeric(df["rti_vs_strike_bps"], errors="coerce")
    side = df["side"].astype(str).str.upper()
    side_sign = np.where(side == "YES", 1.0, np.where(side == "NO", -1.0, np.nan))

    new_cols: dict[str, pd.Series | np.ndarray] = {
        "rti_side_strike_bps": current_dist * side_sign,
        "rti_abs_strike_bps": current_dist.abs(),
        "rti_near_strike_10bps": (current_dist.abs() <= 10).astype(int),
        "rti_near_strike_25bps": (current_dist.abs() <= 25).astype(int),
        "rti_near_strike_50bps": (current_dist.abs() <= 50).astype(int),
    }
    created: list[str] = list(new_cols)

    dist_changes: dict[int, pd.Series] = {}
    toward_changes: dict[int, pd.Series] = {}
    side_mom: dict[int, pd.Series] = {}

    for window in WINDOWS:
        change_col = f"rti_change_{window}s"
        if change_col not in df.columns:
            continue
        price_change = pd.to_numeric(df[change_col], errors="coerce")
        prior_rti = rti - price_change
        prior_dist = pd.Series(bps(prior_rti, strike), index=df.index)
        dist_change = current_dist - prior_dist
        toward = prior_dist.abs() - current_dist.abs()
        side_momentum = dist_change * side_sign

        dist_changes[window] = dist_change
        toward_changes[window] = toward
        side_mom[window] = side_momentum

        names = {
            f"rti_dist_change_{window}s_bps": dist_change,
            f"rti_abs_dist_change_{window}s_bps": current_dist.abs() - prior_dist.abs(),
            f"rti_toward_strike_{window}s_bps": toward,
            f"rti_away_from_strike_{window}s_bps": -toward,
            f"rti_side_momentum_{window}s_bps": side_momentum,
            f"rti_dist_velocity_{window}s_bps_per_s": dist_change / float(window),
            f"rti_toward_velocity_{window}s_bps_per_s": toward / float(window),
            f"rti_side_velocity_{window}s_bps_per_s": side_momentum / float(window),
            f"rti_crossed_strike_{window}s": (
                (prior_dist.notna())
                & (current_dist.notna())
                & (np.sign(prior_dist) != 0)
                & (np.sign(current_dist) != 0)
                & (np.sign(prior_dist) != np.sign(current_dist))
            ).astype(int),
        }
        new_cols.update(names)
        created.extend(names)

    for short, long in ((5, 30), (15, 60), (5, 60)):
        if short not in dist_changes or long not in dist_changes:
            continue
        pairs = {
            f"rti_dist_accel_{short}s_vs_{long}s_bps_per_s": (
                dist_changes[short] / float(short) - dist_changes[long] / float(long)
            ),
            f"rti_toward_accel_{short}s_vs_{long}s_bps_per_s": (
                toward_changes[short] / float(short) - toward_changes[long] / float(long)
            ),
            f"rti_side_accel_{short}s_vs_{long}s_bps_per_s": (
                side_mom[short] / float(short) - side_mom[long] / float(long)
            ),
        }
        new_cols.update(pairs)
        created.extend(pairs)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    derived = pd.DataFrame(new_cols, index=df.index)
    df = pd.concat([df, derived], axis=1)
    compression = "infer" if args.compression == "infer" else (None if args.compression == "none" else args.compression)
    df.to_csv(args.out, index=False, compression=compression)
    print(
        json.dumps(
            {
                "input": str(args.input),
                "out": str(args.out),
                "rows": int(len(df)),
                "new_columns": created,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
