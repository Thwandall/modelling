# Kalshi Crypto Models

This repository is the model registry for the Kalshi crypto C++ bot.

The bot should continue to select its live model bundle through:

```bash
CRYPTO_ML_MODEL_DIR=/path/to/kalshi-crypto-models/models/<model_id>
```

Each model directory is an immutable release artifact. Do not edit an existing
released model in place. If the model, feature schema, thresholds, calibration,
or risk knobs change, create a new directory such as
`candidate_trade_features_v2`.

## Current Bundles

- `models/candidate_trade_features_v1`: currently deployed bundle copied from
  `/opt/kalshi/crypto/models/candidate_trade_features_v1`.

## Required Bundle Files

Every live C++ bundle must include:

- `base_model.txt`
- `meta_model.txt`
- `feature_schema.json`
- `thresholds.json`
- `risk_policy.json`
- `model_manifest.json`

Recommended support files:

- `categorical_schema.json`
- `base_calibration.json`
- `golden_rows.csv`
- `golden_predictions.csv`
- `threshold_health.csv`
- `threshold_tuning_grid.csv`

## Validation

Before deploying or copying a bundle locally:

```bash
python3 scripts/validate_bundle.py models/candidate_trade_features_v1
```

This checks file presence, JSON parseability, and the main schema/threshold
contracts expected by the C++ loader.

## Deployment Pattern

On a machine running the C++ bot:

```bash
export CRYPTO_ML_MODEL_DIR=/opt/kalshi/crypto-models/models/candidate_trade_features_v1
systemctl restart crypto-bot
```

The bot also has a compiled default path, but production deploys should use the
environment variable so model switches are explicit and auditable.

