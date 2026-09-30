# xQBR retrain with a publish gate — pre-registration

**Status:** PRE-REGISTRATION, committed 2026-09-30 before any candidate is scored on the
holdout. Results go in `models/qbr/qbr.gate.json` (written by the code) and the
registry row, never edited into this section.

Owner decision (2026-09-30): retrain on the served input definition, test dropping the
spread feature, and add a publish gate: the new model must beat the current one against
ESPN QBR out of sample.

## 0. Why

The shipped `qbr_model.ubj` (sha256 `2abfde68…`, trained 2026-08-02 on 2004–2025)
was fitted on a feature recipe that the served path never computes:

- **QB runs were not in training.** Training grouped plays by `passer_player_name`,
  and rush plays carry no passer. So `rush_epa` was missing on every training row,
  and the booster has zero splits on it. Serving (`CFBPlayProcess.create_box_score`)
  groups on `coalesce(passer, rusher)`, so a QB's runs do reach the served features.
- **Overtime and penalty plays were handled differently.** Training used the EP/WP
  frame, which drops every overtime game, drops plays outside downs 1–4, and keeps
  non-scrimmage penalty plays. Serving keeps overtime, filters to `scrimmage_play`,
  and has no down filter.
- **The spread feature.** On identical play-by-play, a big favourite's QB scores about
  16 points more than an underdog's (`/mnt/sdv_repos/tmp/xqbr-probe/model_shape.txt`).
- **There was no gate and no persisted label.** Training re-scraped ESPN and joined on
  names that were looked up one athlete at a time.

Known before registration: on 2026 weeks 1–4 the incumbent tracks ESPN raw QBR with
r 0.859, RMSE 14.35 and bias −3.68 over 549 QB-games (probe, `quality_report.txt`).
Every 2025 value is in its training set.

## 1. Data (fixed)

- **Features = the served rows.** They are read from the published `adv_passing` box
  score, which is the `advBoxScore.pass` block that sdv-py writes. That is the exact frame
  GOP and every `adv_passing` consumer score, so train and serve agree by construction,
  not by a re-implementation. Columns: `qbr_epa sack_epa pass_epa rush_epa pen_epa spread
  era0..era3`. NaN or null means "no such plays" and is kept as XGBoost missing, as served.
- **Labels** are ESPN game-level raw QBR (`QBR`), taken from the committed
  `models/qbr/espn_qbr_labels.parquet` (provenance: `models/qbr/README.md`).
- **The join** is on `(game_id, athlete_id)`, both Int64, with dtypes asserted before
  the join. The athlete id is the per-(game, team, passer name) mode of the pbp
  `passer_player_id`. A feature row whose id repeats within a game is dropped as
  ambiguous and counted.

## 2. Split (frozen)

- **Holdout H:** ESPN-labelled QB-games in season 2026, season type 2, weeks 1–4, using
  ESPN's week from the label file. H is frozen now. Changing it requires a new
  pre-registration.
- **Train:** every labelled QB-game in seasons 2004–2025, regular season and postseason.
  Train and H are disjoint by season.
- The partition (game_id, athlete_id, split) is written beside the artifact at fit time
  and committed.

## 3. Arms (fixed; no tuning)

| arm | features |
|---|---|
| `incumbent` | shipped `qbr_model.ubj`, scored as served |
| `spread` | retrained, all 10 served features |
| `no_spread` | retrained, the 9 features without `spread` |

Both retrained arms use the shipped recipe unchanged: `QBR_PARAMS`, 45 rounds, and
XGBoost's default seed. No hyperparameter is searched, and nothing is re-run with
different settings after H has been scored. If a bug is found after scoring, fixing it
and re-running is allowed, but the re-run must be recorded in the gate record and in the
registry.

## 4. Gate (the correctness contract)

- **Metric:** squared error against ESPN raw QBR, per QB-game, paired across arms on
  identical rows of H.
- **Statistic:** Δ = mean(SE_arm − SE_incumbent).
- **Interval:** a 95% percentile bootstrap, B = 2000, seed 0, **clustered by game_id**:
  both teams' QBs in a game are resampled together.
- **Pass** requires Δ's 95% upper bound < 0, meaning the arm is strictly better than
  the incumbent, AND |H| ≥ 500 matched QB-games (549 observed at registration), AND
  the label→feature match rate on H ≥ 0.95 (0.998 observed at registration).
- The gate is enforced in code. `train-qbr` writes no model when no arm passes, and the
  artifact publisher refuses a QBR model whose sha256 has no passing gate record. The
  sdv-py bundle carries the same record, and a test checks it.

## 5. Choice rule

1. `no_spread` ships if it passes **and** its holdout RMSE is at most 0.30 QBR points
   worse than `spread`'s, or `spread` fails.
2. Otherwise `spread` ships if it passes. The owner then discloses the spread input.
3. Otherwise nothing ships and the incumbent stays.

The 0.30-point tolerance is about 2% of the incumbent's RMSE. "Costs little" is fixed
here, before scoring.

## 6. Reported, not gating

For each arm on H: RMSE, MAE, r and bias against raw QBR, and r against Total QBR
(`TQBR`). For the shipped arm: leave-one-season-out out-of-fold predictions on
2004–2025. Also the spread effect at identical pbp, and the 2025 Duke / Darian Mensah
hook (game 401754593), labelled in-sample.

## 7. Addendum, 2026-09-30, after scoring (model review), not part of the registration

- **Scored on H before registration, and not stated above.** The probe that led to this
  retrain (`/mnt/sdv_repos/tmp/xqbr-probe/quality_report.txt`) also scored a linear
  baseline, `50.70 + 59.91 * qbr_epa` fit on 2024. On the H QB-games with at least 14
  dropbacks it had RMSE 13.14, against the incumbent's 13.66 on the same rows. No arm
  was changed because of it, and it is not a gate arm.
- **Join losses (review I3).** 379 feature rows over 2004–2026 were dropped as ambiguous:
  one athlete under two box-score name spellings in a game. 9 of the 550 H labels did not
  join. Five of those are starters split across spellings, e.g. "Noah Kim" / "N. Kim".
  Four are a single athlete with no pbp passer id. The early seasons join worse. The
  counts are now written into the gate record (`join`,
  `holdout.train_match_rate_by_season`). The spelling split is a serving bug in sdv-py
  `create_box_score`, which groups passers by name; it is reported separately. The paired
  comparison is unaffected, but H's RMSE leaves out the rows serving scores worst.
- **Re-run.** `train-qbr` was re-run after review to add those diagnostics. The chosen arm,
  every holdout metric and the candidate sha256 must be unchanged, and a changed sha256
  would void the record. The re-run's record replaces the first one, and the registry
  notes it.
- **H is now spent for selection.** It chose between two arms, so a later gate on the same
  H is not clean. Another retrain needs a new pre-registration with new training seasons
  and a new holdout, e.g. train through 2026 and hold out 2027 weeks 1–4. `TRAIN_SEASONS`
  and `HOLDOUT` are frozen constants for exactly that reason.
