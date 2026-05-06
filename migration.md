# RTI/PLV Challenger Migration Plan

Date: 2026-05-06

This document is the handoff plan for migrating from the current static trade-feature Kalshi crypto bot to a deployable RTI/PLV challenger. It is intentionally split into:

- Work done on this VM/research machine.
- Work done in the local C++ execution codebase.

The target is not to blindly ship a new model. The target is to make research, model export, live feature generation, and execution behavior match closely enough that live results can be interpreted.

## Current State

### Live C++ Bot

The live bot code is in:

```text
/opt/kalshi/crypto
```

The currently configured live bundle is:

```text
/opt/kalshi/crypto/models/candidate_trade_features_v1
```

The current model bundle has:

- `112` raw features.
- `121` encoded base features.
- No `plv_*` features.
- No new RTI dynamics features such as `rti_toward_strike_*`, `rti_side_momentum_*`, or `rti_dist_accel_*`.

Current live risk policy:

```json
{
  "policy": "candidate_v2",
  "min_meta_p": 0.0,
  "min_ml_edge": 0.04,
  "max_asset_side": 120,
  "max_asset_side_tte": null,
  "one_contract": true,
  "disable_recovery": true,
  "disable_aggression": true,
  "toxic_gates": "validate_meta_filter.apply_toxic_gates",
  "rolling_6h_asset_stop_cents": -1500,
  "rolling_6h_total_stop_cents": -4000,
  "daily_total_stop_cents": -7500
}
```

Important live execution facts:

- `maybe_enter()` evaluates threshold crossings and calls `try_enter()` if ML/risk approves.
- Signals are tracked by independent slots: YES/NO by signal level.
- A slot blocks only itself while active; another level or opposite side can still trade.
- After a buy fill, the bot currently sends a maker sell at `fill + 2c`, capped at `98c`.
- Profit-taking is hardcoded in C++ today.
- Remaining unexited position settles at market outcome.
- `trade_flow_.clear()` is called on market roll, so live trade-flow memory is currently window-local, not cross-window.
- Current C++ risk policy only supports static `min_meta_p`, `min_ml_edge`, and global rolling `max_asset_side`.

### Research/Modelling Repo

The modelling repo is:

```text
/root/crypto_data/kalshi-crypto-models
```

Important current files:

```text
README.md
docs/model_contract.md
docs/research_ledger.md
scripts/enrich_candidate_level_flow_features.py
scripts/enrich_rti_strike_dynamics.py
scripts/train_challenger_base_model.py
scripts/validate_bundle.py
models/candidate_trade_features_v1/
models/candidate_trade_features_v1_52k/
```

Current useful research files outside the repo:

```text
/root/crypto_data/ml_features_quality_ab_plus_level_flow_backfilled.csv.gz
/root/crypto_data/ml_features_quality_ab_plus_level_flow_rti_dynamics.csv.gz
/root/crypto_data/ml_features_quality_ab_plus_level_flow_rti_dynamics.parquet
/root/crypto_data/meta_filter_rti_dynamics_reports/
/root/crypto_data/knob_tuning_rti_dynamics/
```

Latest relevant RTI/PLV enriched table:

```text
/root/crypto_data/ml_features_quality_ab_plus_level_flow_rti_dynamics.parquet
```

Known row count:

```text
200,351 rows
```

## Migration Target

Primary target:

```text
candidate_rti_plv_v1
```

Default feature policy:

- Include the current trade-feature base feature set.
- Include RTI dynamics.
- Include price-level public trade-flow features, abbreviated as PLV.
- Exclude sparse `depth_*` features for v1.

Default execution policy:

- Make execution mode configurable.
- Support both `hold_to_settlement` and `profit_take`.
- Use `hold_to_settlement` for the first clean validation run because it better matches the main research PnL assumption.
- Keep `profit_take` available for separate live A/B testing.

Default risk policy:

```json
{
  "policy": "candidate_rti_plv_v1",
  "min_meta_p": 0.8,
  "min_ml_edge": 0.0,
  "max_asset_side": 90,
  "one_contract": true,
  "disable_recovery": true,
  "disable_aggression": true,
  "execution_mode": "hold_to_settlement",
  "profit_take_mD": 20,
  "profit_take_cap_mD": 980
}
```

The cap `90` is intentionally more conservative than the no-cap knob result. It can be relaxed later only after forward validation.

## Dev-Time Feature Switches

The migration should be chunked so local development can choose which feature families to support.

### Required For RTI/PLV Challenger

These must be implemented for the full target:

- Existing 112 trade-feature inputs.
- RTI dynamics.
- PLV features.
- Configurable execution mode.
- Bundle/schema guardrails.
- Golden prediction parity.

### Optional Switches

The modelling/export path should support these switches:

```text
--include-rti-dynamics / --no-rti-dynamics
--include-plv / --no-plv
--include-depth / --no-depth
--execution-mode hold_to_settlement|profit_take
```

Default:

```text
--include-rti-dynamics
--include-plv
--no-depth
--execution-mode hold_to_settlement
```

Do not export a model bundle with features that local C++ cannot compute.

## VM-Side Work

This is work to do on this research VM and in the modelling repo.

## Chunk 1: Freeze Candidate Feature Families

Goal: define exactly which features are in `candidate_rti_plv_v1`.

### Include

Current live/research trade-feature family:

- Kalshi top-of-book features.
- Time-to-expiry features.
- Side/level/categorical features.
- RTI base fields.
- Existing `rti_change_5s`, `rti_change_15s`, `rti_change_30s`, `rti_change_60s`.
- Existing broad `trade_*` aggregate features.

RTI dynamics from:

```text
scripts/enrich_rti_strike_dynamics.py
```

Feature groups:

- `rti_side_strike_bps`
- `rti_abs_strike_bps`
- `rti_near_strike_10bps`
- `rti_near_strike_25bps`
- `rti_near_strike_50bps`
- `rti_dist_change_*_bps`
- `rti_abs_dist_change_*_bps`
- `rti_toward_strike_*_bps`
- `rti_away_from_strike_*_bps`
- `rti_side_momentum_*_bps`
- `rti_dist_velocity_*_bps_per_s`
- `rti_toward_velocity_*_bps_per_s`
- `rti_side_velocity_*_bps_per_s`
- `rti_crossed_strike_*`
- `rti_dist_accel_*`
- `rti_toward_accel_*`
- `rti_side_accel_*`

PLV features from:

```text
scripts/enrich_candidate_level_flow_features.py
```

Windows:

```text
30s, 60s, 300s, 900s
```

Near level definition:

```text
abs(trade_side_price - candidate_level) <= 2c
```

Per-window PLV fields:

- `plv_{window}s_total_volume`
- `plv_{window}s_total_count`
- `plv_{window}s_at_volume`
- `plv_{window}s_at_count`
- `plv_{window}s_near_volume`
- `plv_{window}s_near_count`
- `plv_{window}s_above_volume`
- `plv_{window}s_below_volume`
- `plv_{window}s_at_volume_share`
- `plv_{window}s_near_volume_share`
- `plv_{window}s_above_volume_share`
- `plv_{window}s_below_volume_share`
- `plv_{window}s_signal_taker_at_volume`
- `plv_{window}s_opp_taker_at_volume`
- `plv_{window}s_signal_taker_near_volume`
- `plv_{window}s_opp_taker_near_volume`
- `plv_{window}s_net_signal_pressure_near`
- `plv_{window}s_signal_taker_share_near`
- `plv_{window}s_near_vwap`
- `plv_{window}s_entry_vs_near_vwap`

### Exclude By Default

Depth features:

- `depth_*`

Reason:

- Historical depth snapshot coverage was only `449 / 200,351` rows.
- Live C++ can compute current book depth, but the historical training distribution would be extremely sparse unless continuous depth logging/backfill is built.

### DoD

- A feature allowlist exists in the modelling repo.
- The allowlist names every included feature family.
- Export scripts use the allowlist instead of implicitly training on every numeric column.
- `feature_schema.json` states which optional families are included.
- `model_manifest.json` records feature switches.

## Chunk 2: Build Canonical Training Dataset

Goal: produce a single canonical point-in-time input table for training/export.

Preferred existing input:

```text
/root/crypto_data/ml_features_quality_ab_plus_level_flow_rti_dynamics.parquet
```

If rebuilding from the current trade-feature table:

```bash
/root/crypto_data/.venv/bin/python \
  /root/crypto_data/kalshi-crypto-models/scripts/enrich_candidate_level_flow_features.py \
  --input /root/crypto_data/ml_features_quality_ab_plus_trades.csv \
  --data-root /root/crypto_data \
  --market-glob 'KX*15M-*' \
  --backfill-trades-dir /root/crypto_data/kalshi_public_trades \
  --windows 30,60,300,900 \
  --near-cents 2 \
  --out /root/crypto_data/ml_features_quality_ab_plus_level_flow_backfilled.csv.gz

/root/crypto_data/.venv/bin/python \
  /root/crypto_data/kalshi-crypto-models/scripts/enrich_rti_strike_dynamics.py \
  --input /root/crypto_data/ml_features_quality_ab_plus_level_flow_backfilled.csv.gz \
  --out /root/crypto_data/ml_features_quality_ab_plus_level_flow_rti_dynamics.csv.gz

/root/crypto_data/.venv/bin/python \
  /root/crypto_data/kalshi-crypto-models/scripts/convert_csv_to_parquet.py \
  --input /root/crypto_data/ml_features_quality_ab_plus_level_flow_rti_dynamics.csv.gz \
  --out /root/crypto_data/ml_features_quality_ab_plus_level_flow_rti_dynamics.parquet
```

### Quality Checks

Run lightweight checks before training:

```bash
/root/crypto_data/.venv/bin/python -c "
import pandas as pd
p='/root/crypto_data/ml_features_quality_ab_plus_level_flow_rti_dynamics.parquet'
df=pd.read_parquet(p)
print('rows', len(df))
print('tickers', df['ticker'].nunique())
print('assets', sorted(df['asset'].dropna().unique()))
print('plv cols', sum(c.startswith('plv_') for c in df.columns))
print('rti dyn cols', sum(c.startswith('rti_') and ('toward' in c or 'side_' in c or 'accel' in c or 'crossed' in c) for c in df.columns))
print('depth cols', sum(c.startswith('depth_') for c in df.columns))
"
```

Expected:

- Around `200,351` rows.
- Assets: `BTC`, `ETH`, `SOL`, `XRP`.
- PLV columns present.
- RTI dynamics columns present.

### DoD

- Dataset exists in Parquet.
- Dataset hash recorded.
- Row count and ticker count recorded.
- Missingness report generated for RTI dynamics and PLV columns.
- Any use of depth columns is explicitly disabled unless `--include-depth` is chosen.

## Chunk 3: Train And Validate Candidate

Goal: train the RTI/PLV challenger in a way that is comparable to previous research.

Use the existing rolling meta validation style:

```bash
/root/crypto_data/.venv/bin/python /root/crypto_data/validate_meta_filter.py \
  --input /root/crypto_data/ml_features_quality_ab_plus_level_flow_rti_dynamics.csv.gz \
  --out-dir /root/crypto_data/meta_filter_rti_plv_v1_reports \
  --tune-hours 24 \
  --test-hours 6 \
  --step-hours 6 \
  --toxic-gates
```

Then tune static deployable knobs:

```bash
/root/crypto_data/.venv/bin/python /root/crypto_data/tune_candidate_knobs.py \
  --predictions /root/crypto_data/meta_filter_rti_plv_v1_reports/meta_filter_predictions.csv \
  --out-dir /root/crypto_data/knob_tuning_rti_plv_v1
```

Important:

- Do not choose knobs purely by max total PnL.
- Penalize worst fold and drawdown.
- Prefer flat robust regions over knife-edge optima.
- Avoid no-cap configs for live.

### Metrics To Record

For base, meta/toxic, and selected static policy:

- Number of folds.
- Total trades.
- Total PnL.
- PnL per trade.
- Win rate.
- Positive fold count.
- Negative fold count.
- Worst fold.
- Max drawdown.
- Per-asset PnL.
- Per-asset trade count.
- May 4 and May 5 behavior, if present in the dataset.

### DoD

- Rolling validation report exists.
- Knob tuning report exists.
- Selected policy is justified by robustness, not only PnL.
- The report explicitly compares against `candidate_trade_features_v1_52k`.

## Chunk 4: Export Deployable Bundle

Goal: create an immutable bundle that local C++ can load.

Target:

```text
/root/crypto_data/kalshi-crypto-models/models/candidate_rti_plv_v1/
```

Required files:

```text
base_model.txt
meta_model.txt
feature_schema.json
categorical_schema.json
thresholds.json
risk_policy.json
model_manifest.json
golden_rows.csv
golden_predictions.csv
threshold_health.csv
threshold_tuning_grid.csv
```

Exporter requirements:

- Accept Parquet input.
- Accept feature switches.
- Use feature allowlist.
- Record feature switches in manifest.
- Record train/tune/test window settings.
- Record source dataset hash.
- Record selected risk policy.
- Record validation PnL and drawdown.
- Emit golden rows that include all raw columns needed to recompute features.

Example intended command:

```bash
/root/crypto_data/.venv/bin/python \
  /root/crypto_data/kalshi-crypto-models/scripts/export_candidate_rti_plv.py \
  --input /root/crypto_data/ml_features_quality_ab_plus_level_flow_rti_dynamics.parquet \
  --out-dir /root/crypto_data/kalshi-crypto-models/models/candidate_rti_plv_v1 \
  --include-rti-dynamics \
  --include-plv \
  --no-depth \
  --min-meta-p 0.8 \
  --min-ml-edge 0.0 \
  --max-asset-side 90 \
  --execution-mode hold_to_settlement \
  --profit-take-mD 20 \
  --profit-take-cap-mD 980 \
  --golden-rows 1024
```

### DoD

- Bundle directory is complete.
- `scripts/validate_bundle.py` passes.
- `model_manifest.json` has hashes for all files.
- Bundle is committed to modelling repo.
- Existing released bundles are not modified in place.

## Chunk 5: Generate Local C++ Parity Cases

Goal: give the local C++ implementation exact expected values.

Create:

```text
models/candidate_rti_plv_v1/parity/rti_dynamics_cases.jsonl
models/candidate_rti_plv_v1/parity/plv_cases.jsonl
models/candidate_rti_plv_v1/parity/feature_vector_cases.jsonl
```

Each JSONL row should contain:

- Candidate identity: ticker, asset, side, level, wall time.
- Minimal raw state needed to compute the feature.
- Expected output feature values.

RTI parity cases must cover:

- YES and NO side.
- RTI above strike.
- RTI below strike.
- RTI crossing strike.
- Missing 5s/15s/30s/60s history.
- Acceleration pairs: 5v30, 15v60, 5v60.

PLV parity cases must cover:

- Trade exactly at level.
- Trade within `±2c`.
- Trade above level.
- Trade below level.
- Signal taker side.
- Opposite taker side.
- Empty window.
- 30s/60s/300s/900s windows.

### DoD

- Parity files are committed with the bundle.
- Local C++ can run parity tests without accessing the full raw dataset.
- Parity test failure points to feature name and row id.

## Local C++ Work

This work should happen in the local C++ development environment, not directly on this VM.

## Chunk 6: Add Bundle/Schema Guardrails

Goal: prevent silent bad deployments.

Current risk:

- C++ feature lookup is name-based, but unknown model features can silently remain NaN.
- That makes it possible to load a model that research trained with features live does not compute.

Required changes:

- Add a supported-feature registry.
- During model load, compare model `feature_names` against supported features.
- Fail startup if required model features are unsupported.
- Print all unsupported names.
- Add a bundle version/model id to startup logs.

Files likely involved:

```text
src/strategy/ml_feature_index.hpp
src/ml/model_inference.cpp
tests/test_ml_feature_contract.cpp
```

### DoD

- Current old bundle still loads.
- `candidate_rti_plv_v1` refuses to load until new features are implemented.
- A test with an intentionally unknown feature fails.
- Startup logs show model id, feature count, and risk policy.

## Chunk 7: Implement RTI Dynamics Live

Goal: make C++ compute RTI dynamic features exactly like Python.

Research definition:

```text
current_dist = rti_vs_strike_bps
side_sign = +1 for YES, -1 for NO
prior_rti = current_rti - rti_change_{window}s
prior_dist = (prior_rti - strike) * 10000 / strike
dist_change = current_dist - prior_dist
toward = abs(prior_dist) - abs(current_dist)
side_momentum = dist_change * side_sign
```

Required C++ outputs:

- Current side-adjusted distance.
- Absolute distance.
- Near-strike flags.
- Distance change by window.
- Toward/away from strike by window.
- Side momentum by window.
- Velocity by window.
- Crossed-strike flags.
- Acceleration pairs.

Implementation notes:

- Use existing `rti_change_seconds(5/15/30/60)` as the source for prior RTI where possible.
- Preserve NaN behavior if RTI or strike is unavailable.
- Do not treat missing history as zero.
- Keep the computation in `fill_ml_features()` or a small helper called by it.

Files likely involved:

```text
src/strategy/ml_feature_index.hpp
src/strategy/strategy_thread.cpp
src/ml/ml_decision.hpp
tests/
```

### DoD

- C++ RTI parity cases match Python within tolerance.
- NaN/missing behavior matches the exported golden cases.
- No dynamic allocation is introduced on the hot path.
- Live `ml_feature_vectors` logs include these feature values when the model uses them.

## Chunk 8: Implement PLV Trade-Flow Live

Goal: make C++ compute price-level public trade-flow features exactly like research.

Current issue:

- `TradeFlowState` stores broad aggregate windows.
- It does not compute price-level volume.
- It is cleared on market roll.

Required changes:

- Keep asset-level public trade history for at least 900 seconds.
- Do not clear PLV history on 15-minute market roll.
- Still reset market-specific/orderbook state on roll.
- Add `LevelFlowFeatures` or equivalent.
- Compute PLV features by candidate side and candidate level.

Research semantics:

- Candidate side price:
  - YES candidate uses trade YES price.
  - NO candidate uses trade NO price.
- Candidate level is in cents.
- `above` means side price greater than candidate level.
- `below` means side price less than candidate level.
- `at` means side price equals candidate level.
- `near` means side price within `±2c`.
- Signal taker:
  - YES candidate: trade taker side is YES.
  - NO candidate: trade taker side is NO.

Recommended first implementation:

- Use the existing ring of public trades.
- Add a bounded scan for the 900s history per candidate.
- Measure latency.
- If too slow under threshold cascades, replace with per-window/per-price-bin aggregates.

Production implementation:

- Maintain per-window arrays over price bins `0..100`.
- Maintain signal/opp taker volume by price bin.
- Update arrays on trade add and window expiry.
- Query at/near/above/below in bounded O(100) or better.

Files likely involved:

```text
src/ml/trade_flow_state.hpp
src/ml/trade_flow_state.cpp
src/strategy/ml_feature_index.hpp
src/strategy/strategy_thread.cpp
```

### DoD

- PLV parity cases match Python.
- Public trade history survives market roll.
- Hot path does not allocate.
- Candidate evaluation latency remains acceptable during threshold cascades.
- Feature logs prove live PLV values are not stuck at zero.

## Chunk 9: Keep Depth Out Of V1

Goal: avoid training/deploying sparse depth features before data quality supports them.

Action:

- Do not add `depth_*` features to `candidate_rti_plv_v1`.
- Keep any future continuous depth logging work separate.

If depth is later enabled:

- Add continuous top-of-book/depth snapshots to historical data.
- Train a new model id, e.g. `candidate_rti_plv_depth_v1`.
- Implement live depth parity from `DenseBook`.
- Add depth parity cases.

### DoD

- `candidate_rti_plv_v1/feature_schema.json` contains no `depth_*` features.
- Local C++ does not need depth changes for this release.

## Chunk 10: Make Execution Mode Configurable

Goal: separate model validation from execution policy.

Current behavior:

- Buy fill immediately triggers maker sell at `fill + 2c`, capped at `98c`.

Required modes:

```text
hold_to_settlement
profit_take
```

`hold_to_settlement`:

- Submit buy order when candidate is approved.
- Do not submit a profit-take exit after fill.
- Let remaining position settle.

`profit_take`:

- Submit maker exit after fill.
- Default price: `fill + profit_take_mD`.
- Default cap: `profit_take_cap_mD`.
- Current-equivalent values: `profit_take_mD=20`, `profit_take_cap_mD=980`.

Config source:

- Prefer `risk_policy.json`.
- Allow env override only for emergency testing.

Suggested JSON:

```json
{
  "execution_mode": "hold_to_settlement",
  "profit_take_mD": 20,
  "profit_take_cap_mD": 980
}
```

Files likely involved:

```text
src/strategy/strategy_thread.cpp
src/strategy/strategy_thread.hpp
src/ml/risk_policy.hpp
src/ml/risk_policy.cpp
```

### DoD

- No hardcoded profit-take constant controls live behavior.
- Hold-to-settlement sends no exit order after buy fill.
- Profit-take mode reproduces current behavior.
- Entry logs include execution mode and profit-take settings.
- Replay can compare both modes.

## Chunk 11: Preserve Research-Style Signal Independence

Goal: make live execution match the research assumption that each recorded threshold-crossing candidate is an independent decision opportunity.

Current live behavior:

- Each side/level has one slot.
- Once a crossing is evaluated, the crossing stores `ml_evaluated=true`.
- A rejected crossing is usually not evaluated again later just because time-to-expiry changed.
- An approved crossing can block the same side/level while its slot is active.
- If profit-taking fully closes the slot, the code clears the crossing and allows re-cross/re-evaluation.

Research behavior:

- Research rows are recorded crossing events.
- If the data has two crossing rows for the same side/level in one market, research can score both independently.
- Research does not continuously re-score the same crossing every tick while price remains above the threshold.
- Research PnL generally assumes each accepted candidate is independently tradable unless the simulator explicitly adds slot/position constraints.

Required migration decision:

- For clean research parity, live should not let an old position silently suppress later recorded crossing opportunities unless the research replay simulator applies the same rule.
- The preferred v1 is to define a canonical signal policy and use it in both research replay and C++:

```text
one decision per recorded crossing event
same side/level may trade again only after a true re-cross event
slot blocking must be represented in research replay if enabled live
```

Implementation options:

1. Strict independent-candidate mode:
   - Every newly recorded crossing event gets one ML evaluation.
   - Slot state does not suppress ML scoring.
   - Risk/exposure may still suppress order submission.
   - Use this when validating model quality.

2. Slot-constrained execution mode:
   - Keep current slot blocking.
   - Research replay must simulate the same slot blocking before reporting PnL.
   - Use this when validating deployable execution PnL.

Default recommendation:

```text
model validation: strict independent-candidate mode
live execution validation: slot-constrained replay and live mode must match
```

Required logging:

- `crossing_event_id`
- side
- level
- TTE
- whether ML was evaluated
- whether order was blocked by active slot
- whether order was blocked by exposure/risk
- prior active slot state

### DoD

- Migration doc and model contract state whether a bundle was validated with independent-candidate or slot-constrained semantics.
- C++ logs every skipped candidate caused by active slot state.
- Research replay can reproduce the selected live slot policy.
- Reported PnL clearly separates model-candidate PnL from executable slot-constrained PnL.

## Chunk 12: Extend Risk Policy Support

Goal: express the deployable policy correctly in local C++.

Required v1 fields:

- `min_meta_p`
- `min_ml_edge`
- `max_asset_side`
- `one_contract`
- `disable_recovery`
- `disable_aggression`
- `execution_mode`
- `profit_take_mD`
- `profit_take_cap_mD`

Fields to warn/fail on if non-null and unsupported:

- `max_asset_side_tte`
- per-asset caps
- per-side caps
- external toxic model path

### DoD

- Unsupported non-null policy fields cause startup failure or loud warning.
- Null unsupported fields are allowed.
- `max_asset_side` remains rolling, not lifetime.
- Risk-policy startup log prints all active knobs.

## Chunk 13: Golden Prediction And Replay Tests

Goal: prove local C++ is running the same model/policy as research.

Required local tests:

1. Bundle load test.
2. Feature schema support test.
3. RTI dynamics parity test.
4. PLV parity test.
5. Golden prediction test.
6. Replay test with orders disabled.

Golden prediction test:

- Load `golden_rows.csv`.
- Recompute features using C++ path where possible.
- Score base model.
- Score meta model.
- Compare expected `p_yes`, `meta_p_good`, `ml_edge`, and final decision.

Replay test:

- Use recent logs.
- Disable live order submission.
- Feed recorded market/trade/RTI events through the strategy path.
- Compare approve/skip distributions to Python/research expectations.

### DoD

- Tests pass locally before any live deploy.
- Any failed feature parity shows feature name and candidate id.
- Replay output includes counts by reason:
  - `approved`
  - `below_bucket_threshold`
  - `below_meta_threshold`
  - `below_candidate_v2_edge`
  - `toxic_gate`
  - `rti_missing`
  - `trade_flow_stale`
  - `bbo_stale`
  - `asset_side_cap`
  - `exposure_cap`

## Chunk 14: Deployment And Rollout

Goal: ship in a way that makes live results interpretable.

### Pre-Deploy Checklist

- Bundle committed to modelling repo.
- Bundle copied to local/prod model directory.
- C++ supports every model feature.
- Golden prediction test passes.
- Replay test passes.
- `CRYPTO_ML_MODEL_DIR` points to the new immutable bundle.
- `CRYPTO_ML_ONE_CONTRACT=1`.
- Recovery disabled.
- Aggression disabled.
- Execution mode explicitly set.

### First Live Run

Recommended first run:

```text
execution_mode=hold_to_settlement
one_contract=true
disable_recovery=true
disable_aggression=true
max_asset_side=90
```

Run long enough to cover multiple assets and market regimes. A couple of hours is not enough to conclude edge quality.

### Monitoring

Track:

- Approved trades by asset.
- Approved trades by side.
- Approved trades by TTE bucket.
- Skip reasons.
- Feature missingness.
- RTI freshness.
- Trade-flow freshness.
- PLV nonzero rates.
- Fill rate.
- REST submission latency.
- Order ack latency.
- PnL by settlement and by early exit if profit-taking is enabled.

### Rollback

Keep this as rollback bundle:

```text
models/candidate_trade_features_v1_52k
```

Do not mutate released bundles. Roll forward by creating a new model directory.

### DoD

- Live run can be explained from logs.
- Feature distributions are comparable to research.
- Losses can be attributed to model decision, feature mismatch, or execution behavior.
- Rollback path is tested before scaling size.

## Key Risks

### Feature Mismatch

Highest risk. If C++ does not compute RTI/PLV exactly like research, live results are not meaningful.

Mitigation:

- Feature schema guardrails.
- Parity cases.
- Golden predictions.
- Live feature vector logging.

### Execution Mismatch

Research mostly used candidate-price-to-settlement PnL. Current live execution profit-takes. That changes the strategy.

Mitigation:

- Configurable execution mode.
- First validation in hold-to-settlement mode.
- Profit-taking treated as a separate execution experiment.

### Overfitting

RTI/PLV challenger may improve historical metrics without improving forward performance.

Mitigation:

- Compare against 52k static bundle.
- Penalize worst fold/drawdown.
- Run forward validation before scaling.

### Latency

PLV computation can get expensive during threshold cascades.

Mitigation:

- Start with bounded implementation.
- Measure.
- Move to price-bin aggregates if needed.

### XRP/RTI Quality

XRP previously had RTI quality issues. A model that depends on RTI dynamics can fail badly if live RTI is missing/synthetic.

Mitigation:

- Asset-level feature missingness logs.
- Optional per-asset enable/disable.
- Do not scale XRP until RTI parity is proven.

## Suggested Work Order

1. Freeze feature allowlist and export interface in modelling repo.
2. Export `candidate_rti_plv_v1` without depth.
3. Generate RTI/PLV parity cases.
4. Add C++ schema guardrails.
5. Implement RTI dynamics live.
6. Implement PLV live.
7. Add configurable execution mode.
8. Make signal independence/slot constraints explicit in both research replay and C++.
9. Extend risk-policy parsing.
10. Run local parity/golden/replay tests.
11. Deploy one-contract hold-to-settlement probe.
12. Compare live feature distributions and PnL.
13. Only then test profit-taking and larger caps.

## Definition Of Done For The Whole Migration

The migration is complete only when:

- `candidate_rti_plv_v1` exists as an immutable bundle in the modelling repo.
- Local C++ can load the bundle without unsupported features.
- C++ RTI dynamics match Python.
- C++ PLV features match Python.
- Golden prediction tests pass.
- Execution mode is configurable.
- Signal independence versus slot-constrained execution is explicit and replayable.
- First live run uses one contract, no recovery, no aggression.
- Logs are sufficient to diagnose every approved and skipped candidate.
- Rollback to `candidate_trade_features_v1_52k` is available.
