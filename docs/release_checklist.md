# Release Checklist

Use this checklist before pointing `CRYPTO_ML_MODEL_DIR` at a new bundle.

## Research Export

- Train the base model on the intended training window.
- Train or refresh the meta-filter on frozen base predictions.
- Freeze the exact feature list and order.
- Export LightGBM text models as `base_model.txt` and `meta_model.txt`.
- Export `feature_schema.json`.
- Export `thresholds.json` and `risk_policy.json`.
- Export `golden_rows.csv` and `golden_predictions.csv`.
- Write `model_manifest.json` with training data path, report paths, creation
  time, and hashes.

## Bundle Validation

Run:

```bash
python3 scripts/validate_bundle.py models/<model_id>
```

Confirm:

- all required files exist
- JSON files parse
- base/meta feature lists are non-empty
- thresholds include expected assets and TTE buckets
- risk policy knobs are present and intentional

For `candidate_trade_features_v1_52k`, the intentional risk knobs are:

```json
{
  "min_meta_p": 0.8,
  "min_ml_edge": 0.0,
  "max_asset_side": 180,
  "one_contract": true,
  "disable_recovery": true,
  "disable_aggression": true
}
```

These knobs approximate the `+54,232c` rolling trade-feature meta/toxic policy
with a static live policy that scored `+52,911c` historically.

## C++ Parity Validation

Before live deployment:

- Build the local C++ bot.
- Load the new bundle through `CRYPTO_ML_MODEL_DIR`.
- Score `golden_rows.csv` in C++.
- Compare with `golden_predictions.csv`.
- Verify feature units for Kalshi trade sizes, RTI fields, prices, and quality
  tiers.

## Live Deployment

Set the model directory explicitly:

```bash
export CRYPTO_ML_MODEL_DIR=/opt/kalshi/crypto-models/models/<model_id>
systemctl restart crypto-bot
```

After restart, verify logs show:

- ML enabled
- expected `model_dir`
- expected base/meta feature counts
- no schema load errors
- per-asset public trade counters are nonzero
- RTI freshness is healthy

## Rollback

Keep the previous model bundle installed. Rollback should be only an env var
change plus restart:

```bash
export CRYPTO_ML_MODEL_DIR=/opt/kalshi/crypto-models/models/<previous_model_id>
systemctl restart crypto-bot
```
