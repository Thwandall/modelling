# Research Ledger

This document records the modelling path that led to the current live Kalshi
crypto bundle. It separates trained models from policy layers, because many of
the headline PnL numbers came from wrappers around model predictions rather than
from new model files.

All PnL values are in cents and assume one contract per accepted trade unless
noted otherwise.

## Data Pipeline

The research table was built from point-in-time candidate rows. Each row is a
possible trade at a threshold crossing, not a market-level aggregate.

Core inputs:

- Kalshi market ticker, side, level, time-to-expiry bucket, and outcome.
- Kalshi top-of-book and order book features at or before candidate time.
- RTI replica features: RTI value, distance to strike, RTI momentum, and venue
  quality fields.
- Coinbase/reference market features.
- Kalshi public trade-flow features in the trade-feature runs.

Important data-quality rules:

- Joins must be point-in-time. Research uses prior/as-of joins so later RTI,
  book, or trade information does not leak into a candidate row.
- Train/test splits are chronological, not random.
- Duplicate threshold opportunities inside the same ticker are correlated and
  must not be split randomly across train and test.
- Rows with missing or synthetic RTI are dangerous for live inference. Recent
  live logs showed XRP RTI as synthetic; that asset should be treated carefully
  until RTI quality is fixed.
- Execution PnL in the research is candidate-price/settlement based. It does not
  fully model passive fill probability, queue priority, spread crossing, or
  profit-taking.

## Baseline Model Search

Initial boosted-tree tests compared LightGBM and XGBoost on several quality
slices.

| Dataset | Model | Test AUC | Trades | PnL | PnL/Trade | Max DD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `common` | LightGBM | 0.9437 | 4,946 | +5,772 | +1.17 | -6,383 |
| `common` | XGBoost | 0.9472 | 5,464 | -712 | -0.13 | -11,931 |
| `quality_ab` | LightGBM | 0.9650 | 5,320 | +12,924 | +2.43 | -3,044 |
| `quality_ab` | XGBoost | 0.9652 | 614 | -3,216 | -5.24 | -3,305 |
| `quality_a` | LightGBM | 0.9430 | 2,540 | +7,619 | +3.00 | -3,423 |
| `quality_a` | XGBoost | 0.9452 | 1,763 | +3,174 | +1.80 | -3,603 |

Decision:

- LightGBM was kept as the primary model. XGBoost was a comparison, not a
  complementary signal.
- `quality_ab` was the main early slice because it had strong AUC and materially
  better PnL than the alternatives.

## Rolling Base Model

The next validation used rolling chronological folds:

- Tune window: 24h.
- Test window: 6h.
- Step: 6h.
- Base model trained only on history before the tune/test window.
- Base probabilities calibrated on the trailing tune window.
- Per-asset/TTE edge thresholds selected on the tune window.

The conservative v2 base policy produced:

| Policy | Folds | Trades | PnL | PnL/Trade | Positive Folds | Negative Folds | Worst Fold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Conservative v2 base | 29 | 13,973 | +17,859 | +1.28 | 18 | 8 | -4,486 |

This base model was profitable historically but had large bad folds, especially
around April 26-27.

## Meta Filter

The meta filter is a second LightGBM model trained on trailing candidate trades.
It predicts whether a base-approved candidate is likely to be good.

Research process per fold:

1. Generate base candidates on the tune window.
2. Label tune candidates as profitable/non-profitable.
3. Train meta LightGBM on tune candidates.
4. Select a meta probability threshold from a grid on the tune window.
5. Apply the selected threshold to the next 6h test window.

This produced:

| Policy | Folds | Trades | PnL | PnL/Trade | Positive Folds | Negative Folds | Worst Fold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Meta-filter only | 29 | 11,024 | +32,078 | +2.91 | 20 | 6 | -3,336 |

Important live nuance:

- The research policy used fold-specific meta thresholds, often around `0.45`
  to `0.85`.
- A static live `min_meta_p=0.0` does not reproduce that. It scores the meta
  model but almost never blocks trades.

## Toxic Gates

Toxic gates are hardcoded rules applied after meta approval. They are not a
trained toxic model.

The current gates block selected RTI/momentum patterns:

- NO trades when RTI is above strike and 30s or 60s RTI momentum is positive.
- ETH NO trades in that same upward RTI/momentum state.
- ETH NO trades in selected TTE buckets when RTI is above strike.

Historical result:

| Policy | Folds | Trades | PnL | PnL/Trade | Positive Folds | Negative Folds | Worst Fold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Candidate v1: meta + toxic gates | 29 | 10,609 | +35,912 | +3.39 | 20 | 6 | -3,336 |

Known weakness:

- These toxic gates were discovered after inspecting failures, so their measured
  lift is at risk of overfitting.
- They mostly cover NO-side RTI/momentum toxicity. They do not solve fast
  YES-side adverse selection, threshold cascades, or nose-dive events.

## Candidate v2 Knobs

Candidate v2 knobs tuned post-model controls on frozen candidate v1 predictions.
No base or meta model was retrained.

Selected historical knobs:

```json
{
  "min_meta_p": 0.0,
  "min_ml_edge": 0.04,
  "max_asset_side": 120
}
```

Result:

| Policy | Scope | Trades | PnL | PnL/Trade | Worst Fold | Negative Folds |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Candidate v1 baseline | Design | 5,825 | +22,196 | +3.81 | -49 | 1 |
| Candidate v1 baseline | Holdout | 4,784 | +13,716 | +2.87 | -3,336 | 5 |
| Candidate v2 knobs | Design | 3,578 | +21,828 | +6.10 | 0 | 0 |
| Candidate v2 knobs | Holdout | 2,661 | +15,780 | +5.93 | -821 | 4 |
| Candidate v2 knobs | All folds | 6,239 | +37,608 | +6.03 | -821 | 4 |

Interpretation:

- This mostly reduced exposure and improved fold robustness.
- The static `min_meta_p=0.0` was selected in that specific knob search because
  the input predictions had already gone through the rolling meta threshold.
- In live C++, `min_meta_p=0.0` is not equivalent unless rolling meta threshold
  behavior is separately implemented.

## Trade-Feature Model Stack

The next major improvement added Kalshi public trade-flow features.

Examples:

- Trade volume over short windows.
- Signed YES/NO pressure.
- Trade VWAP by side.
- Entry price versus recent trade VWAP.
- Trade price momentum.
- Average trade size.

Historical rolling validation:

| Policy | Folds | Trades | PnL | PnL/Trade | Positive Folds | Negative Folds | Worst Fold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Trade-feature base | 30 | 15,367 | +46,804 | +3.05 | 25 | 2 | -3,236 |
| Trade-feature meta/toxic | 30 | 12,708 | +54,232 | +4.27 | 24 | 3 | -3,458 |

This is the source of the `+54,232c` number. It used rolling meta thresholds,
not a static `min_meta_p=0.0` live threshold.

## 52k Static Live Approximation

Because live C++ currently loads static model files and static risk knobs, the
deployable approximation to the 54k rolling policy is:

```json
{
  "min_meta_p": 0.8,
  "min_ml_edge": 0.0,
  "max_asset_side": 180
}
```

Historical knob-study result:

| Policy | Scope | Trades | PnL | PnL/Trade | Worst Fold | Negative Folds |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Trade-feature meta/toxic baseline | Design | 6,745 | +30,008 | +4.45 | -435 | 1 |
| Trade-feature meta/toxic baseline | Holdout | 5,963 | +24,224 | +4.06 | -3,458 | 2 |
| 52k static policy | Design | 6,126 | +29,289 | +4.78 | 0 | 0 |
| 52k static policy | Holdout | 5,661 | +23,622 | +4.17 | -3,502 | 2 |
| 52k static policy | All folds | 11,787 | +52,911 | +4.49 | -3,502 | 2 |

This policy is released as:

```text
models/candidate_trade_features_v1_52k
```

## Forward Tests

Forward testing exposed instability in the base model and showed that risk
wrappers matter.

Initial forward window:

| Policy | Trades | PnL | PnL/Trade | Win Rate | Max DD |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base | 37,764 | -78,289 | -2.07 | 42.7% | -107,205 |
| Meta/toxic | 16,990 | -19,141 | -1.13 | 94.8% | -66,166 |
| Candidate v2 knobs | 553 | +3,596 | +6.50 | 96.2% | -875 |

Latest post-freeze forward window:

| Window | Policy | Trades | PnL | PnL/Trade | Win Rate | Max DD |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Post-freeze all | Base | 63,518 | -34,990 | -0.55 | 53.5% | -107,205 |
| Post-freeze all | Meta/toxic | 34,938 | +35,113 | +1.01 | 97.2% | -66,166 |
| Post-freeze all | Candidate v2 knobs | 862 | +4,617 | +5.36 | 95.9% | -1,221 |
| Incremental after previous | Base | 25,754 | +43,299 | +1.68 | 69.4% | -3,311 |
| Incremental after previous | Meta/toxic | 17,948 | +54,254 | +3.02 | 99.5% | -1,221 |
| Incremental after previous | Candidate v2 knobs | 309 | +1,021 | +3.30 | 95.5% | -1,221 |

Interpretation:

- The base model was not stable enough alone.
- Meta/toxic recovered materially in the larger forward window, but the first
  short forward window was poor.
- Candidate v2 style knobs reduced drawdown and exposure, but can leave a lot
  of theoretical PnL unused.

## Regime Governors And Advanced Risk

Additional risk research tried online/bayesian/regime governors. These were
diagnostic/research policies, not exported model bundles.

Historical governor validation on candidate v2 predictions:

| Config | Folds | Trades | PnL | PnL/Trade | Positive Folds | Negative Folds | Worst Fold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Candidate v2 knobs | 26 | 6,182 | +37,037 | +5.99 | 21 | 4 | -821 |
| Meta/toxic | 26 | 10,533 | +35,191 | +3.34 | 19 | 6 | -3,336 |
| Governor w100 g2 | 26 | 5,543 | +17,677 | +3.19 | 17 | 8 | -2,117 |
| Governor default | 26 | 5,161 | +14,982 | +2.90 | 17 | 8 | -1,851 |

Trade-feature governor validation:

| Config | Folds | Trades | PnL | PnL/Trade | Positive Folds | Negative Folds | Worst Fold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Trade-feature meta/toxic | 27 | 12,624 | +53,879 | +4.27 | 23 | 3 | -3,458 |
| Trade-feature candidate v2 knobs | 27 | 7,455 | +46,300 | +6.21 | 25 | 1 | -1,374 |
| Governor w100 g2 | 27 | 7,231 | +34,177 | +4.73 | 22 | 4 | -1,501 |
| Governor default | 27 | 6,902 | +29,984 | +4.34 | 19 | 7 | -1,714 |

Advanced forward risk research:

| Variant | Trades | PnL | PnL/Trade | Win Rate | Max DD |
| --- | ---: | ---: | ---: | ---: | ---: |
| Candidate v2 knobs | 553 | +3,596 | +6.50 | 96.2% | -875 |
| Candidate v2 plus regime kill | 353 | +1,029 | +2.92 | 94.1% | -875 |
| Meta/toxic | 16,990 | -19,141 | -1.13 | 94.8% | -66,166 |
| Large-loss model threshold 0.300 | 16,990 | -19,141 | -1.13 | 94.8% | -66,166 |
| Online bucket lower-bound > 0 | 13,663 | -22,266 | -1.63 | 94.5% | -66,166 |
| Bayesian bucket lower-bound > 0 | 13,136 | -22,361 | -1.70 | 94.2% | -65,642 |
| Reliability-bin lower-bound > 0 | 10,766 | -24,849 | -2.31 | 94.0% | -58,350 |

Conclusion:

- The tested governors did not beat the simpler candidate v2 exposure controls.
- They remain useful diagnostic work, but they were not selected for live.

## Kelly Sizing

Kelly experiments were sizing/risk overlays, not separate prediction models.
They were used to ask whether stake size should vary with estimated edge. The
live policy currently uses `one_contract=true`, so Kelly sizing is not active.

## Live Deployment Mapping

The current live C++ bot loads:

- `base_model.txt`
- `meta_model.txt`
- `base_calibration.json`
- `thresholds.json`
- `risk_policy.json`

Current selected live-static policy:

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

What this does:

- Requires the meta model to score a candidate above `0.8`.
- Removes the additional `4c` live edge floor, relying on tuned bucket
  thresholds and the meta filter instead.
- Allows up to 180 accepted trades per asset/side rolling cap in C++.
- Keeps live sizing at one contract.
- Prevents recovery/aggression logic from changing the tested one-entry
  semantics.

What this does not do:

- It does not retrain the base model online.
- It does not retrain the meta model every 6h.
- It does not implement fold-specific rolling meta threshold selection.
- It does not include a trained adverse-selection/nose-dive model.
- It does not fully model passive fill probability or queue position.

## Known Gaps

The biggest unresolved modelling gaps are:

- Fast adverse selection and nose dives. The current toxic gates are too narrow.
- Live execution mismatch. Research assumes candidate-price entry and settlement
  PnL; live may post passive orders below the current ask and miss fills.
- Profit-taking mismatch. Some live exits differ from settlement-hold research.
- Data quality by asset, especially XRP RTI synthetic state observed in recent
  logs.
- Fill logging gaps, including missing order IDs in `fills.ndjson`.
- Need for a dedicated short-horizon toxic-flow model using spread widening,
  book depletion, threshold velocity, and taker-flow bursts.

