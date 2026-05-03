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
- `models/candidate_trade_features_v1_52k`: same trade-feature base/meta model
  artifacts as v1, with the live-static policy knobs selected from the
  trade-feature knob study:
  `min_meta_p=0.8`, `min_ml_edge=0.0`, `max_asset_side=180`.

## Policy Lineage

The model bundle contains two trained LightGBM models:

- **Base model**: predicts the probability that the Kalshi contract resolves
  YES from point-in-time market, RTI, book, and trade-flow features.
- **Meta model**: scores candidate trades emitted by the base model and tries to
  filter out candidates that are historically likely to lose.

The headline research policies are wrappers around those models:

- **Trade-feature meta/toxic research policy**: rolling 24h tune window,
  rolling 6h test window, base model retrained by fold, meta model trained on
  trailing candidate trades, fold-specific meta threshold selected on the tune
  window, then hard toxic gates applied. This produced `+54,232c` over `12,708`
  one-contract trades in historical rolling validation.
- **52k live-static approximation**: uses the exported trade-feature base/meta
  models with static live knobs:
  `min_meta_p=0.8`, `min_ml_edge=0.0`, `max_asset_side=180`. This produced
  `+52,911c` over `11,787` one-contract trades in the historical knob study.

The `+54,232c` policy is not exactly reproduced by setting `min_meta_p=0.0` in
live C++. In research, the meta threshold was chosen independently for each
rolling fold, often around `0.45` to `0.85`. In live C++, `min_meta_p=0.0`
means the meta model is scored but almost never blocks a candidate. The 52k
bundle intentionally uses `min_meta_p=0.8` as a deployable static approximation.

## Live Risk Knobs

The knobs in `risk_policy.json` are not model weights. They are post-model
decision controls:

- `min_meta_p`: minimum meta-model probability required after base approval.
  Higher values trade less and should reduce lower-confidence candidates. The
  52k policy uses `0.8`.
- `min_ml_edge`: minimum base-model expected edge required after per-bucket
  thresholds. The 52k policy uses `0.0` because the rolling bucket thresholds
  and meta filter carried the main filtering load in research.
- `max_asset_side`: rolling per-asset/per-side acceptance cap used by live C++.
  The 52k policy uses `180` to reduce unbounded exposure without cutting as
  aggressively as the previous `120` cap.
- `one_contract`: live size override for paper/live risk containment. Historical
  PnL numbers above assume one contract per accepted trade.
- `disable_recovery` and `disable_aggression`: execution controls. They keep the
  live bot closer to single-entry research semantics and prevent recovery or
  extra aggressive sizing from changing the tested policy.
- `toxic_gates`: documentation flag for the hardcoded toxic gates. These are not
  a trained toxic model; they are narrow RTI/momentum rules, mostly affecting
  NO-side candidates. They do not catch all adverse-selection/nose-dive cases.

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
