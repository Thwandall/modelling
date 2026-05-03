# Live Environment Workflow

The C++ bot should keep using an environment variable for model selection.

## Recommended Paths

Research machine:

```bash
/root/crypto_data/kalshi-crypto-models/models/<model_id>
```

Local/dev machine:

```bash
~/kalshi-crypto-models/models/<model_id>
```

Production machine:

```bash
/opt/kalshi/crypto-models/models/<model_id>
```

## Environment Variable

```bash
CRYPTO_ML_MODEL_DIR=/opt/kalshi/crypto-models/models/<model_id>
```

The path should point directly at the directory containing `base_model.txt`,
`meta_model.txt`, `feature_schema.json`, `thresholds.json`, and
`risk_policy.json`.

## Systemd Pattern

Use an environment file instead of editing code:

```text
CRYPTO_ML_MODEL_DIR=/opt/kalshi/crypto-models/models/candidate_trade_features_v1
```

Then the service can load it with:

```ini
EnvironmentFile=/etc/crypto-bot.env
```

This makes model switches explicit and reversible without changing the C++ bot.

## Local Development Pattern

When testing locally:

```bash
export CRYPTO_ML_MODEL_DIR=$HOME/kalshi-crypto-models/models/candidate_trade_features_v1
./crypto_bot
```

Do not copy research report directories directly into production. Only deploy
validated model bundles.

