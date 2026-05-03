#!/usr/bin/env python3
import csv
import json
import sys
from pathlib import Path


REQUIRED = [
    "base_model.txt",
    "meta_model.txt",
    "feature_schema.json",
    "thresholds.json",
    "risk_policy.json",
    "model_manifest.json",
]

RECOMMENDED = [
    "categorical_schema.json",
    "base_calibration.json",
    "golden_rows.csv",
    "golden_predictions.csv",
]


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def schema_feature_order(schema):
    for key in ["encoded_feature_order", "feature_names", "base_features"]:
        value = schema.get(key)
        if isinstance(value, list):
            return value
        if isinstance(value, dict) and isinstance(value.get("feature_names"), list):
            return value["feature_names"]
    return []


def lightgbm_feature_names(path: Path):
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("feature_names="):
                return line[len("feature_names="):].strip().split()
    return []


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: validate_bundle.py /path/to/model_bundle", file=sys.stderr)
        return 2

    bundle = Path(sys.argv[1]).resolve()
    if not bundle.is_dir():
        print(f"ERROR: not a directory: {bundle}", file=sys.stderr)
        return 1

    errors = []
    warnings = []

    for name in REQUIRED:
        if not (bundle / name).is_file():
            errors.append(f"missing required file: {name}")

    for name in RECOMMENDED:
        if not (bundle / name).is_file():
            warnings.append(f"missing recommended file: {name}")

    for name in ["feature_schema.json", "thresholds.json", "risk_policy.json", "model_manifest.json"]:
        path = bundle / name
        if path.is_file():
            try:
                load_json(path)
            except Exception as exc:
                errors.append(f"{name} is not valid JSON: {exc}")

    schema_path = bundle / "feature_schema.json"
    if schema_path.is_file():
        schema = load_json(schema_path)
        schema_order = schema_feature_order(schema)
        base = lightgbm_feature_names(bundle / "base_model.txt") if (bundle / "base_model.txt").is_file() else []
        meta = lightgbm_feature_names(bundle / "meta_model.txt") if (bundle / "meta_model.txt").is_file() else []
        if not base:
            errors.append("base_model.txt has no LightGBM feature_names line")
        if not meta:
            errors.append("meta_model.txt has no LightGBM feature_names line")
        if not schema_order:
            errors.append("feature_schema.json has no encoded_feature_order")
        if base and schema_order and base != schema_order:
            warnings.append("base_model.txt feature_names differ from feature_schema encoded_feature_order")
        print(
            f"schema_features={len(schema_order)} "
            f"base_features={len(base)} meta_features={len(meta)}"
        )

    thresholds_path = bundle / "thresholds.json"
    if thresholds_path.is_file():
        th = load_json(thresholds_path)
        rows = th.get("thresholds", [])
        if not isinstance(rows, list) or not rows:
            errors.append("thresholds.json must contain non-empty thresholds list")
        else:
            assets = sorted({str(r.get("asset", "")) for r in rows if isinstance(r, dict)})
            buckets = sorted({int(r.get("tte_bucket", -1)) for r in rows if isinstance(r, dict) and "tte_bucket" in r})
            print(f"threshold_rows={len(rows)} assets={','.join(assets)} buckets={buckets}")

    risk_path = bundle / "risk_policy.json"
    if risk_path.is_file():
        risk = load_json(risk_path)
        for key in ["min_meta_p", "min_ml_edge", "max_asset_side"]:
            if key not in risk:
                warnings.append(f"risk_policy.json missing knob: {key}")
        print("risk_policy=" + json.dumps(risk, sort_keys=True))

    for name in ["golden_rows.csv", "golden_predictions.csv"]:
        path = bundle / name
        if path.is_file():
            with path.open("r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader, [])
                count = sum(1 for _ in reader)
            print(f"{name}: rows={count} columns={len(header)}")

    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print(f"OK: {bundle}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
