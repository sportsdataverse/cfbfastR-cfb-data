# CPOE

## Metrics

| metric | value |
|---|---|
| `importance_top` | down:210.8008, passing_down:201.0792, score_diff:101.7535, is_home:61.1751, yards_to_goal:60.7263, period:56.0416, distance:52.0364, seconds_remaining:47.1557 |
| `loso_log_loss` | 0.6765 |
| `loso_brier` | 0.2418 |


## Provenance

| metric | value |
|---|---|
| `features` | down, distance, yards_to_goal, score_diff, seconds_remaining, is_home, period, passing_down |
| `hyperparameters` | {"objective":"binary:logistic","eta":0.025,"gamma":5,"subsample":0.8,"colsample_bytree":0.8,"max_depth":4,"min_child_weight":6,"eval_metric":"logloss"} |
| `training_seasons` | n/a |
| `trained_date` | 2026-06-18 |
| `xgboost_version` | 3.2.0 |
