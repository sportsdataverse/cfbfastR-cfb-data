# Win Probability (naive)

## Overview

The naive Win Probability model answers *given only the game state, with no betting-market information, how likely is the possession team to win?* It is the spread model's sibling — identical except it drops the spread signal — and is the right surface when a pregame spread is unavailable or when you explicitly want a market-free WP.


## Recipe & lineage

A 12-feature XGBoost **binary:logistic** model, **65 trees**. The feature set is exactly the spread model's **minus `spread_time`**; everything else (game clock, score differential, field position, down/distance, timeouts, period, 2H-kickoff possession) is shared. Far fewer trees than the spread model (65 vs 760) because without the spread there is less structured signal to fit.


## Metrics

| metric | value |
|---|---|
| `n` | 2219607 |
| `logloss` | 0.4002 |
| `brier` | 0.1329 |
| `corr_vs_spread` | 0.941 |
| `q1_abs_div` | 0.1314 |
| `q4_abs_div` | 0.0185 |


## Discussion

No LOSO OOF parquet is shipped for the naive variant, so calibration here is computed on a full-history prediction pass over `pbp_full.parquet` (the naive feature matrix is deterministic from game state). The naive WP **correlates ~0.94 with the spread WP** and, as expected, the two **diverge most in the first quarter** — when the pregame spread carries the most information and the game state carries the least. By the fourth quarter the mean absolute gap between naive and spread WP collapses (Q1 ~0.13 vs Q4 ~0.02), because `spread_time` has decayed away and both models are reading the same near-final game state.


## Limitations

Because it ignores the market, the naive model is *less sharp* early in games: its log-loss and Brier are worse than the spread model's (it has strictly less information). It is the correct tool only when you want a spread-free WP or lack a spread; for forecasting accuracy when a spread exists, prefer the spread model. The calibration here is in-sample (resubstitution) rather than LOSO, so read it as a fit check rather than an out-of-sample guarantee.


## Provenance

| metric | value |
|---|---|
| `features` | pos_team_receives_2H_kickoff, TimeSecsRem, adj_TimeSecsRem, ExpScoreDiff_Time_Ratio, pos_score_diff_start, down, distance, yards_to_goal, is_home, pos_team_timeouts_rem_before, def_pos_team_timeouts_rem_before, period |
| `hyperparameters` | {} |
| `training_seasons` | n/a |
| `trained_date` | 2026-06-22 |
| `xgboost_version` | 3.2.0 |


## Notes

- Naive-WP calibration is in-sample (no LOSO OOF is shipped). It shares the spread recipe minus `spread_time`; metrics are a full-corpus prediction pass.
