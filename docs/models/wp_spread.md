# Win Probability (spread)

## Overview

The spread-aware Win Probability model estimates the probability that the team in possession wins the game, given game state **and the pregame point spread**. It produces the `vegas_wp`-style surface; consecutive-play differences define **Win Probability Added (WPA)**.


## Recipe & lineage

A 13-feature XGBoost **binary:logistic** model, **760 trees**, a faithful port of the `cfbscrapR-wpa.ipynb` recipe. The signature feature is `spread_time = pos_team_spread * exp(-4 * elapsed_share)` — the pregame spread decayed toward zero as the game clock runs out, so its influence vanishes by the fourth quarter. Other features: `TimeSecsRem`, `adj_TimeSecsRem`, `ExpScoreDiff_Time_Ratio`, `pos_score_diff_start`, `down`, `distance`, `yards_to_goal`, `is_home`, both teams' remaining timeouts, `period`, and `pos_team_receives_2H_kickoff`.


## Metrics

| metric | value |
|---|---|
| `n` | 2219607 |
| `logloss` | 0.3616 |
| `brier` | 0.1182 |
| `auc` | 0.9159 |
| `weighted_cal_err_pooled` | 0.0147 |


## Figures

![](figures/wp_spread_calibration.png)


## Discussion

LOSO pooled: `logloss` 0.3616, Brier 0.1182, AUC 0.9159, weighted calibration error 0.0147. The AUC near 0.92 and the sub-0.015 weighted calibration error mean the predicted win probabilities are both discriminating and well-calibrated across the whole probability range. The calibration figure facets by quarter so you can confirm calibration holds late in games (where WP is most actionable).


## Limitations

WPA — the first difference of WP — is intrinsically noisy: small per-play WP movements are dominated by model variance, so single-play WPA should be read as a directional signal, not a precise quantity. The spread input is a pregame number; the model does not re-estimate a live spread. Overtime and end-of-half edge cases are handled by the construction pipeline upstream, not by the model head.


## Provenance

| metric | value |
|---|---|
| `features` | pos_team_receives_2H_kickoff, spread_time, TimeSecsRem, adj_TimeSecsRem, ExpScoreDiff_Time_Ratio, pos_score_diff_start, down, distance, yards_to_goal, is_home, pos_team_timeouts_rem_before, def_pos_team_timeouts_rem_before, period |
| `hyperparameters` | {} |
| `training_seasons` | n/a |
| `trained_date` | 2026-06-17 |
| `xgboost_version` | 3.2.0 |
