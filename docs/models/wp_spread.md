# Win Probability (spread)

## Metrics

| metric | value |
|---|---|
| `importance_top` | ExpScoreDiff_Time_Ratio:3677.0161, pos_score_diff_start:2021.6655, spread_time:1185.6589, period:355.2285, pos_team_receives_2H_kickoff:212.8283, is_home:211.9621, adj_TimeSecsRem:166.3599, yards_to_goal:108.4043 |


## Provenance

| metric | value |
|---|---|
| `features` | pos_team_receives_2H_kickoff, spread_time, TimeSecsRem, adj_TimeSecsRem, ExpScoreDiff_Time_Ratio, pos_score_diff_start, down, distance, yards_to_goal, is_home, pos_team_timeouts_rem_before, def_pos_team_timeouts_rem_before, period |
| `hyperparameters` | {} |
| `training_seasons` | n/a |
| `trained_date` | 2026-06-17 |
| `xgboost_version` | 3.2.0 |


## Notes

- Calibration + log-loss/Brier require a warmed --cache; run the integration report for those.
