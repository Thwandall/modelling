# Toxic Flow Dataset

This dataset is the first modelling artifact for an adverse-selection veto. It is
separate from the entry model: the entry model answers "is this candidate good?";
this dataset is for a second model that answers "is this otherwise-good candidate
about to be toxic?"

## Source Data

Run the builder against the live market directories in `/root/crypto_data`.

Required per-market files:

- `ml_feature_vectors.ndjson`: one row per ML candidate, including model scores and
  the frozen `base_values` feature vector used by the live policy.
- `ticks.csv`: point-in-time BBO/book snapshots for markout and profit-take labels.

Optional but useful per-market files:

- `fills.ndjson`: used to identify which approved candidates were actually filled.
- `outcome.ndjson` or `outcome.json`: used for settlement labels.

The script does not write generated artifacts into this repo by default. Keep
large CSVs in `/root/crypto_data/...` and commit only code, docs, and frozen model
bundles.

## Build Command

Recent smoke run:

```bash
python3 /root/crypto_data/kalshi-crypto-models/scripts/build_toxic_flow_dataset.py \
  --data-root /root/crypto_data \
  --market-glob 'KX*15M-26MAY04*' \
  --out /root/crypto_data/toxic_flow_dataset_may04_approved.csv \
  --approved-only \
  --max-markets 200
```

Full approved-candidate build:

```bash
python3 /root/crypto_data/kalshi-crypto-models/scripts/build_toxic_flow_dataset.py \
  --data-root /root/crypto_data \
  --market-glob 'KX*15M-*' \
  --out /root/crypto_data/toxic_flow_dataset_all_approved.csv \
  --approved-only
```

To include rejected candidates too:

```bash
python3 /root/crypto_data/kalshi-crypto-models/scripts/build_toxic_flow_dataset.py \
  --data-root /root/crypto_data \
  --market-glob 'KX*15M-*' \
  --out /root/crypto_data/toxic_flow_dataset_all_candidates.csv \
  --all-candidates
```

## Row Definition

The default row is one approved ML candidate:

```text
market signal -> candidate approved by live policy -> hypothetical entry at entry_price_mD
```

This is intentionally broader than only actual fills. If we train only on filled
orders, the model learns the execution policy's selection bias. The filled columns
are still included so we can separately analyze what happened live.

Key identity columns:

- `ticker`
- `asset`
- `wall_ns`
- `side`
- `level`
- `tte_s`
- `tte_bucket`
- `entry_price_mD`
- `ml_edge`
- `meta_p_good`
- `bucket_threshold`
- `trade`
- `reason`

## Preserved Entry Features

The builder expands the original live model vector using the frozen
`feature_schema.json`. These columns are prefixed with `base_`.

Examples:

- `base_rti_vs_strike_bps`
- `base_rti_change_5s`
- `base_trade_30s_signal_side_pressure`
- `base_trade_30s_price_momentum`
- `base_trade_volume_burst_30s_vs_5m`
- `base_imbalance_x10000`
- `base_microprice_x1000`

These columns let the toxic model train on the same information the entry model
had, without requiring the research code to reconstruct every feature.

## Added Toxicity Features

The builder adds point-in-time features from the local book at candidate time:

- `signal_same_bid_mD`
- `signal_same_ask_mD`
- `signal_opp_bid_mD`
- `signal_opp_ask_mD`
- `signal_spread_mD`
- `signal_microprice_x1000`
- `signal_imbalance_x10000`
- `entry_minus_same_bid_mD`
- `same_ask_minus_entry_mD`
- `would_cross_now`

It also adds cascade features:

- `candidate_count_1s`
- `same_side_candidate_count_1s`
- `approved_count_1s`
- `candidate_count_5s`
- `same_side_candidate_count_5s`
- `approved_count_5s`

These are useful for detecting clustered threshold cascades where the signal is
firing into a fast move.

## Labels

The current labels are deliberately simple and auditable:

- `same_bid_markout_{1,3,5,15,30,60}s_mD`: same-side bid change after the signal.
- `max_adverse_bid_change_30s_mD`: worst same-side bid move in the first 30s.
- `max_favorable_bid_change_30s_mD`: best same-side bid move in the first 30s.
- `target_hit_120s`: whether the same-side bid reached `entry_price_mD + 2c`
  within 120s.
- `target_seconds_120s`: seconds until that target was reached.
- `settlement_pnl_mD`: hypothetical hold-to-settlement PnL for one contract.
- `actual_filled`: whether live execution filled a matching entry order.

Binary labels:

- `label_bad_markout_30s`: `max_adverse_bid_change_30s_mD <= -30mD`.
- `label_failed_target_120s`: 2c target was not hit within 120s.
- `label_loses_at_settlement`: hold-to-settlement PnL is negative.
- `label_toxic`: any of the three labels above.

These labels are starting points, not permanent truth. For model training, prefer
testing separate targets first:

- train one model for adverse 30s markout;
- train one model for failed 2c target;
- compare both against settlement loss.

Do not hide these differences inside one blended label until we know which label
best explains live losses.

## Current Build Results

Full approved-candidate command:

```bash
python3 /root/crypto_data/kalshi-crypto-models/scripts/build_toxic_flow_dataset.py \
  --data-root /root/crypto_data \
  --market-glob 'KX*15M-*' \
  --out /root/crypto_data/toxic_flow_dataset_all_approved.csv \
  --approved-only
```

Output:

```text
markets_scanned=12132
rows_written=888
matched_entry_fills=449
missing_feature_vectors=11372
missing_outcome=1
asset_rows.BTC=332
asset_rows.ETH=310
asset_rows.SOL=246
label_toxic_rate=0.3694
actual_fill_rate=0.5056
target_hit_rate=0.9628
```

Label counts:

```text
label_toxic = 328 / 888
label_bad_markout_30s = 303 / 888
label_failed_target_120s = 33 / 888
label_loses_at_settlement = 56 / 888
```

Asset breakdown:

```text
BTC: n=332 toxic=29.2% bad_markout=27.1% failed_target=4.5% settle_loss=6.0%
ETH: n=310 toxic=43.9% bad_markout=40.0% failed_target=2.9% settle_loss=6.8%
SOL: n=246 toxic=38.6% bad_markout=36.2% failed_target=3.7% settle_loss=6.1%
```

XRP has no approved rows in this build. That is expected from the current live
data issue: XRP candidates are still blocked by synthetic/missing RTI.

The main pattern is that most toxic labels come from short-horizon adverse
markout, not failure to hit the 2c target. That supports training the first veto
model on nose-dive/adverse-selection detection.

## First Smoke Result

Command:

```bash
python3 /root/crypto_data/kalshi-crypto-models/scripts/build_toxic_flow_dataset.py \
  --data-root /root/crypto_data \
  --market-glob 'KX*15M-26MAY04*' \
  --out /root/crypto_data/toxic_flow_dataset_may04_approved.csv \
  --approved-only \
  --max-markets 200
```

Output:

```text
markets_scanned=200
rows_written=361
matched_entry_fills=193
asset_rows.BTC=146
asset_rows.ETH=178
asset_rows.SOL=37
label_toxic_rate=0.4211
actual_fill_rate=0.5346
target_hit_rate=0.9529
```

In that slice, the dominant problem was adverse 30s markout, not failure to hit
the 2c target:

```text
label_bad_markout_30s = 140 / 361
label_failed_target_120s = 17 / 361
label_loses_at_settlement = 28 / 361
```

That supports training the first veto model on short-horizon adverse selection.

## Next Modelling Step

Train a time-aware LightGBM/XGBoost classifier using:

- features: `base_*`, current model scores, book snapshot features, cascade counts;
- label: start with `label_bad_markout_30s`;
- grouping: split by chronological market windows, never random rows;
- evaluation: compare entry model alone vs entry model plus toxic veto;
- objective: reduce large losers while preserving enough approved volume.

The live decision rule should eventually be:

```text
trade only if base policy approves AND toxic_probability < threshold
```

The toxic threshold must be selected on tuning folds and frozen before forward
testing.
