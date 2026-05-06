#!/usr/bin/env python3
"""Join frozen policy candidates to backfilled PLV/depth features for veto training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


KEYS = ["ticker", "wall_ns", "side", "level"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-predictions", type=Path, required=True)
    parser.add_argument("--plv-features", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--approved-column", default="knob_trade")
    parser.add_argument("--pnl-column", default="knob_pnl_cents")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    policy = pd.read_csv(args.policy_predictions, low_memory=False)
    if args.approved_column not in policy.columns:
        raise SystemExit(f"missing approved column: {args.approved_column}")
    if args.pnl_column not in policy.columns:
        raise SystemExit(f"missing pnl column: {args.pnl_column}")
    for key in KEYS:
        if key not in policy.columns:
            raise SystemExit(f"policy file missing key column: {key}")

    approved = policy[pd.to_numeric(policy[args.approved_column], errors="coerce").fillna(0).astype(int) == 1].copy()
    if "entry_price_mD" not in approved.columns and "entry_cost" in approved.columns:
        approved["entry_price_mD"] = pd.to_numeric(approved["entry_cost"], errors="coerce") * 10.0
    approved["settlement_pnl_mD"] = pd.to_numeric(approved[args.pnl_column], errors="coerce").fillna(0.0) * 10.0
    approved["label_loses_at_settlement"] = (approved["settlement_pnl_mD"] < 0).astype(int)
    approved["label_loss_50c"] = (pd.to_numeric(approved[args.pnl_column], errors="coerce").fillna(0.0) <= -50).astype(int)
    approved["label_loss_80c"] = (pd.to_numeric(approved[args.pnl_column], errors="coerce").fillna(0.0) <= -80).astype(int)
    approved["label_loss_90c"] = (pd.to_numeric(approved[args.pnl_column], errors="coerce").fillna(0.0) <= -90).astype(int)
    approved["label_toxic"] = approved["label_loses_at_settlement"]

    header = pd.read_csv(args.plv_features, nrows=0)
    plv_cols = [col for col in header.columns if col in KEYS or col.startswith("plv_") or col.startswith("depth_")]
    plv = pd.read_csv(args.plv_features, usecols=plv_cols, low_memory=False)
    before_dedupe = len(plv)
    plv = plv.drop_duplicates(KEYS, keep="first")

    out = approved.merge(plv, on=KEYS, how="left", indicator=True)
    missing = int((out["_merge"] != "both").sum())
    out = out.drop(columns=["_merge"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    summary = {
        "policy_predictions": str(args.policy_predictions),
        "plv_features": str(args.plv_features),
        "out": str(args.out),
        "approved_rows": int(len(approved)),
        "output_rows": int(len(out)),
        "policy_pnl_cents": float(pd.to_numeric(approved[args.pnl_column], errors="coerce").sum()),
        "loss_rows": int(out["label_loses_at_settlement"].sum()),
        "plv_rows_before_dedupe": int(before_dedupe),
        "plv_rows_after_dedupe": int(len(plv)),
        "missing_plv_rows": missing,
        "plv_900s_total_volume_nonzero": int((out.get("plv_900s_total_volume", 0).fillna(0) != 0).sum())
        if "plv_900s_total_volume" in out
        else 0,
        "depth_has_snapshot_nonzero": int((out.get("depth_has_snapshot", 0).fillna(0) != 0).sum())
        if "depth_has_snapshot" in out
        else 0,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
