# Data Backup Manifest

Created before server resize on 2026-05-06.

This repository stores modelling code, model artifacts, and small reports. It does
not store the full raw market data because the largest inputs are multi-GB and Git
LFS is not installed on this machine.

## Committed In This Repo

- Modelling scripts under `scripts/`
- Model bundles under `models/`
- Per-asset PLV challenger reports under `reports/per_asset_challenger_plv_may05/`

## Not Committed To Git

These paths should be preserved by the server resize if the disk is retained. They
should be copied to object storage, another server, or a local machine for a true
backup.

| Path | Size | Notes |
| --- | ---: | --- |
| `/root/crypto_data/kalshi_public_trades` | 3.5G | Public trade backfill cache, 3506 files |
| `/root/crypto_data/KX*` | many dirs | Raw per-market live collection directories, 12426 dirs at manifest time |
| `/root/crypto_data/ml_features_quality_ab_plus_level_flow_backfilled.csv.gz` | 62M | Main PLV-enriched feature table |
| `/dev/shm/ml_features_quality_ab_plus_level_flow_rti_dynamics.parquet` | 48M | Temporary RTI-dynamics Parquet, lost on reboot unless copied |
| `/opt/kalshi/crypto/models` | 2.4M | Deployed C++ model bundle directory |
| `/opt/kalshi/crypto/.env` | 4K | Runtime config/secrets; do not commit raw |

## Checksums

```text
254f511ead6e1ada2bf8ab532299e177348ff053d12093c896e6cb5b326112ba  /root/crypto_data/ml_features_quality_ab_plus_level_flow_backfilled.csv.gz
5a9fd377df6f2a249f6c3c2568bea20ae84834435cc4e3a59eb58087bcb25765  /dev/shm/ml_features_quality_ab_plus_level_flow_rti_dynamics.parquet
```

## Recommended Backup Commands

Use one of these after enough persistent disk is available.

```bash
mkdir -p /root/crypto_data/backups
tar -C /root/crypto_data -czf /root/crypto_data/backups/kalshi_public_trades_2026-05-06.tar.gz kalshi_public_trades
tar -C /root/crypto_data -czf /root/crypto_data/backups/market_dirs_2026-05-06.tar.gz KX*
tar -C /opt/kalshi/crypto -czf /root/crypto_data/backups/crypto_bot_runtime_2026-05-06.tar.gz models .env
cp /dev/shm/ml_features_quality_ab_plus_level_flow_rti_dynamics.parquet /root/crypto_data/backups/
```

If copying off-box:

```bash
rsync -ah --progress /root/crypto_data/kalshi_public_trades user@host:/backup/crypto_data/
rsync -ah --progress /root/crypto_data/KX* user@host:/backup/crypto_data/markets/
rsync -ah --progress /root/crypto_data/ml_features_quality_ab_plus_level_flow_backfilled.csv.gz user@host:/backup/crypto_data/
rsync -ah --progress /opt/kalshi/crypto/models user@host:/backup/crypto_bot/
```

## Notes

- `/dev/shm` is memory-backed temporary storage. Files there will disappear after
  reboot or resize.
- Do not commit raw `.env` files. Commit a redacted template instead if needed.
- Normal GitHub rejects individual files over 100 MB and is a poor fit for
  multi-GB raw datasets. Use Git LFS, object storage, or an external backup target
  for the raw data.
