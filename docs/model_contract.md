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

## Research Policy Versus Live Policy

Do not confuse trained model files with the full research policy.

The historical `+54,232c` trade-feature result came from a rolling research
procedure:

1. Train a base LightGBM on history before a trailing tune window.
2. Calibrate base probabilities on the tune window.
3. Select per-asset/TTE edge thresholds on the tune window.
4. Generate candidate trades on tune and test windows.
5. Train a meta LightGBM on tune-window candidate trades.
6. Select a meta approval threshold on the tune window.
7. Apply that fold-specific threshold to the next 6h test window.
8. Apply hard toxic gates after meta approval.

The live C++ bot does not currently retrain the base model, retrain the meta
model, or tune a fresh fold-specific meta threshold every six hours. It loads
static LightGBM text files and static risk knobs from `risk_policy.json`.

The deployable `candidate_trade_features_v1_52k` bundle is therefore a static
approximation of the rolling research policy:

```json
{
  "min_meta_p": 0.8,
  "min_ml_edge": 0.0,
  "max_asset_side": 180
}
```

This configuration corresponded to `+52,911c` in historical rolling validation.
The raw rolling meta/toxic policy without the static cap/knob approximation was
`+54,232c`.

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

## Toxic Gates

The `toxic_gates` field in `risk_policy.json` is descriptive. In the current
C++ implementation, toxic behavior is hardcoded in the decision path. It blocks
selected RTI/momentum patterns, mostly NO-side candidates where RTI is already
above strike and 30s/60s RTI momentum is positive.

These gates are not a trained adverse-selection model. They do not directly
model fast YES-side nose dives, threshold cascades, spread blowouts, book
depletion, or toxic taker-flow bursts. A future nose-dive model should be
exported as its own artifact and documented separately.

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
