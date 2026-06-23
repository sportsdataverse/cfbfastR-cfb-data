# CPOE

## Overview

The Completion Percentage Over Expected (CPOE) model estimates the probability a given pass attempt is completed (`cp`) from pre-throw game state. CPOE is the **percentage-point residual** `100 * (complete_pass - cp)`: positive when a passer completes throws a league-average passer would not. It is the CFB analogue of the nflfastR CP/CPOE surface.


## Model features

**8 features**, all known before the throw; the binary label is `complete_pass`.

| Feature | Type | What it encodes |
|---|---|---|
| `down` | numeric | Current down. |
| `distance` | numeric | Yards to go. |
| `yards_to_goal` | numeric | Field position. |
| `score_diff` | numeric | Possession-team score differential. |
| `seconds_remaining` | numeric | Clock context. |
| `is_home` | binary | Home-field indicator. |
| `period` | numeric | Quarter. |
| `passing_down` | binary | Whether the situation is an obvious passing down. |



## Recipe & lineage

An 8-feature XGBoost **binary:logistic** completion model. Lineage is the nflfastR CP recipe adapted to CFB game state; the residual `cp` is subtracted from the binary completion outcome and scaled to percentage points.


## The model

**Algorithm.** XGBoost, `objective=binary:logistic`. The predicted `cp` is the completion probability; **CPOE = `100 * (complete_pass - cp)`** on a percentage-point scale.

**Evaluation.** No season-held-out OOF is shipped here, so calibration is deferred to the model card (see Provenance). The intended check is the cfbscrapR probability binning recipe on a CPOE OOF once produced.


## Metrics

| metric | value |
|---|---|
| `importance_top` | score_diff:10.5265, is_home:10.0154, passing_down:7.8418, down:7.3835, distance:7.0049, yards_to_goal:6.9462, seconds_remaining:6.8857, period:6.6005 |


## Calibration Results


## Discussion

No CPOE leave-one-season-out OOF parquet is shipped in this artifacts set, so a per-play completion-probability calibration figure is **not** rendered here — the card is prose + provenance only. When a CPOE OOF (predicted `cp` + actual `complete_pass` + `period`) is produced, the same cfbscrapR binning recipe used for WP applies directly: bin `cp` into 0.05 buckets and compare to the empirical completion rate.


## Feature importance

Completion probability is driven primarily by `distance` / `yards_to_goal` (throw depth proxies) and `passing_down`; the clock/score context contributes a smaller game-script correction.


## Limitations

CPOE is blind to receiver separation, pressure, and air-yards charting we do not have, so it captures the *game-state-explainable* part of completion probability only. Without a shipped OOF parquet, the calibration claim here is deferred to the model card rather than shown out-of-sample.


## Provenance

| metric | value |
|---|---|
| `features` | down, distance, yards_to_goal, score_diff, seconds_remaining, is_home, period, passing_down |
| `hyperparameters` | {"objective":"binary:logistic","eta":0.025,"gamma":5,"subsample":0.8,"colsample_bytree":0.8,"max_depth":4,"min_child_weight":6,"eval_metric":"logloss"} |
| `training_seasons` | n/a |
| `trained_date` | 2026-06-17 |
| `xgboost_version` | 3.2.0 |


## Notes

- No CPOE LOSO OOF parquet is shipped, so a per-play completion-probability calibration figure is not rendered here; see Provenance for the model card. CPOE is the percentage-point residual 100*(complete_pass - cp).
