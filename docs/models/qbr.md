# QBR

## Metrics

| metric | value |
|---|---|
| `importance_top` | qbr_epa:178749.4844, pass_epa:77651.4844, spread:18683.5137, sack_epa:3441.6594, pen_epa:2503.3069 |


## Provenance

| metric | value |
|---|---|
| `features` | qbr_epa, sack_epa, pass_epa, rush_epa, pen_epa, spread |
| `hyperparameters` | {} |
| `training_seasons` | n/a |
| `trained_date` | 2026-06-17 |
| `xgboost_version` | 3.2.0 |


## Notes

- QBR correlation/RMSE vs ESPN QBR is integration-only (needs the ESPN QBR frame).

- Calibration + log-loss/Brier require a warmed --cache; run the integration report for those.
