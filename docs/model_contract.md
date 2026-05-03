# Model Contract

This document defines the contract between research and the C++ live bot.

## Directory Contract

A deployable model bundle is a directory containing the model files, feature
schema, thresholds, risk knobs, validation rows, and manifest for one immutable
release.

The C++ loader expects `CRYPTO_ML_MODEL_DIR` to point directly at that bundle
directory.

Example:

```bash
CRYPTO_ML_MODEL_DIR=/opt/kalshi/crypto-models/models/candidate_trade_features_v1
```

## Required Files

- `base_model.txt`: LightGBM text model for base probability.
- `meta_model.txt`: LightGBM text model for the meta-filter.
- `feature_schema.json`: exact feature order consumed by C++.
- `thresholds.json`: per-asset/TTE minimum edge thresholds.
- `risk_policy.json`: post-model knobs such as `min_meta_p`,
  `min_ml_edge`, and `max_asset_side`.
- `model_manifest.json`: provenance and hashes for the release.

## Feature Contract

Feature names in `feature_schema.json` are the source of truth. The C++ bot
looks features up by name and must generate the same units used in training.

Fields that need extra care:

- Kalshi prices: live C++ uses milli-dollars (`mD`) internally and often exposes
  cents-like features by dividing by 10.
- Kalshi trade size: research and live must agree on contracts versus fixed
  point hundredths of contracts.
- RTI fields: live must distinguish real multi-venue RTI from Coinbase fallback
  if the model was trained on real RTI.
- Quality tiers: live must compute the same `quality_tier_*` values used during
  training.
- `bucket_threshold`: meta model inputs must receive the same threshold value
  used in research.
- Freshness fields: `rti_join_lag_ms`, `tick_join_lag_ms`, and
  `trade_last_age_s` must be real values, not placeholders, if the model uses
  them.

## Golden Rows

Every released bundle should include:

- `golden_rows.csv`: representative feature rows.
- `golden_predictions.csv`: expected base/meta/decision output from Python.

The local C++ checkout should have a test or diagnostic binary that loads the
bundle, scores the golden rows, and compares predictions within a small
tolerance before deployment.

## Immutability

Once a model is deployed, never modify its bundle in place. Create a new model
directory for any change:

- new training data
- changed feature list
- changed feature units
- changed calibration
- changed thresholds
- changed risk knobs
- changed toxic gates

