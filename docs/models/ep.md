# Expected Points (EP)

## Metrics

| metric | value |
|---|---|
| `importance_top` | TimeSecsRem:5.1218, yards_to_goal:4.9005, pos_score_diff_start:4.3265, down_4:4.1989, down_1:2.8056, down_3:2.6575, down_2:2.5157, distance:2.0548 |


## Provenance

| metric | value |
|---|---|
| `features` | TimeSecsRem, yards_to_goal, distance, down_1, down_2, down_3, down_4, pos_score_diff_start |
| `hyperparameters` | {} |
| `training_seasons` | n/a |
| `trained_date` | 2026-06-17 |
| `xgboost_version` | 3.2.0 |


## Notes

- Calibration + log-loss/Brier require a warmed --cache; run the integration report for those.
