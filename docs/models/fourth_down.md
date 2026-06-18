# Fourth-Down Yards

## Metrics

| metric | value |
|---|---|
| `importance_top` | yards_to_goal:20.6836, distance:15.3457, down:4.9701, posteam_total:2.4187, posteam_spread:2.2404 |


## Provenance

| metric | value |
|---|---|
| `features` | down, distance, yards_to_goal, posteam_total, posteam_spread |
| `hyperparameters` | {"booster":"gbtree","objective":"multi:softprob","eval_metric":"mlogloss","num_class":76,"eta":0.07,"gamma":4.325037e-09,"subsample":0.5385424,"colsample_bytree":0.6666667,"max_depth":4,"min_child_weight":7} |
| `training_seasons` | n/a |
| `trained_date` | 2026-06-18 |
| `xgboost_version` | 3.2.0 |


## Notes

- Calibration + log-loss/Brier require a warmed --cache; run the integration report for those.
