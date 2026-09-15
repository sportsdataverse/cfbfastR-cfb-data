# Matchup-feature oracle fixtures (2025 season)

Golden output of the R matchup pipeline, captured **2026-09-15** under R 4.6.1 +
cfbfastR 2.0.0 (the GitHub dev build 3.0.0.9000, installed the same day, loads the identical
frame: 293,200 rows / 283,172 after the distinct) by `ops/oneoff/20260915_matchup_oracle_capture/capture_oracle.R` (the R units run verbatim over
`cfbfastR::load_cfb_pbp(2025)` after `distinct(game_id, id_play, game_play_number)` →
283,172 plays, 1,657 games). The Python port must reproduce these; nothing here is
synthetic.

Ids: `game_id` Int32, `id_play` / `game_play_number` / `drive_id` Float64 exactly as the
`cfbfastR_cfb_pbp` release parquet ships them (R numerics). `drive_id` = `<game_id><2-digit
drive number>`.

| File | Rows | What | Source |
|---|---|---|---|
| `pbp_input_2025_sample.parquet` | 9,996 × 48 | INPUT: the 56 sample games (every 30th game id, `sample_game_ids.json`) from the 2025 `cfbfastR_cfb_pbp` release, 48 columns the units read | sdv-py `load_cfb_pbp_r([2025])`, 2026-09-15 |
| `play_flags_wepa_2025_sample.csv.gz` | 9,680 × 70 | per play: `EPA, rush, pass, off_wepa, def_wepa, rp_prediction, rroe` + the 60 situational flags (`<name>_weight`; the R `def_` copies are identical and omitted) | R capture, sample games |
| `drives_2025_sample.csv` | 1,225 × 19 | per drive after the `start_yards_to_goal > 40` / timeouts-non-null filter: first-play snapshot, `scoring_opp`, `scoring_opp_prediction`, `scoring_opp_oe`, `prev_drive_result` | R capture, drives of the sample games (season-wide frame filtered by game) |
| `wepa_weights.json` | 120 | the shipped WEPA weights (`off_*` 60 + `def_*` 60), by name | snapshot `models/final_wepa_weights_6_2_24.RDS` |
| `scoring_opp_coef.json` | 6 | logistic glm coefficients + formula/family; rank 6/6 | snapshot `models/scoring_opp_mod.RDS` |
| `rush_expect_coef.json` | 13 | logistic glm coefficients; rank 12/13 — `score_diff` aliased (null) and dropped by R's `predict` | snapshot `models/rp_mod_3.rds` |
| `rp_features_sample.csv` | 5,000 × 15 | rush/pass plays with the 12 rush-expectation features + R's `rp` prediction — the shipped applier is gated against it in `tests/cfb_matchup/test_glm_parity.py` (max diff 1e-15) | R capture, `set.seed(2025)` sample |
| `team_features_full_2025.csv` | 136 × 36 | per FBS team, FULL-season 2025 features (38-col family; the pipeline's prior-season priors file) | snapshot `data/prev_season_epa_data_2025.csv` (June 2026 run) |
| `team_features_asof_2025.csv` | 1,740 × 44 | per FBS team-game, the loop's as-of features + game meta — `capture_team_features.R`: the loop FUNCTION verbatim, but iterating the games present in the release (the source iterates the CFBD schedule; for 2025 the two differ by one bowl with no plays, dropped from the line anyway) | R capture 2026-09-15 |
| `team_features_full_2025_r.csv` | 136 × 44 | the same loop with the pipeline's synthetic future game (whole season) | R capture 2026-09-15 |
| `cfbd_games_elo_2024.parquet`, `cfbd_games_elo_2025.parquet` | 3,801 / 3,831 × 23 | CFBD `/games` rows with pregame / postgame ELO, divisions, points, notes | CFBD API 2026-09-15 |
| `cfbd_lines_2025.parquet` | 3,345 × 6 | CFBD `/lines` rows, one per (game, provider) | CFBD API 2026-09-15 |
| `team_features_full_2024.csv` | 229 × 36 | per team, FULL-season 2024 features — the priors file the source's 2025 run read (`prev_*` and the week-1 fill of every delivered 2025 row) | snapshot `data/prev_season_epa_data_2024.csv` (written by the source's 2025 preseason run, June 2025) |
| `team_pace_full.csv` | 3,046 × 6 | per (season, team) full-season pace 2014–2025 (the source's `tools/build_pace_full.R`; the 2024 rows fill the delivered 2025 openers) | snapshot `data/team_pace_full.csv` (June 2026 build) |
| `pace_hist_2025.csv` | 3,314 × 7 | per (season, team, game_id) as-of-date pace (off/def sec-per-play mean + median) | snapshot `data/pace_hist_2014_2025.csv`, season 2025 slice |
| `matchup_line_2025.csv` | 773 × 268 | the delivered matchup line, season 2025 (bowls excluded, CFP kept) | snapshot `output/cfb_data_2026_week_1.csv`, season 2025 slice |

Full-season files (too large to commit) live in `python/.cache/matchup/` and drive the
`integration`-marked tests: `cfbfastR_cfb_pbp_2025.parquet` (293,200 rows before dedupe),
`play_flags_wepa_2025.csv.gz` (283,172 rows), `drives_2025.csv.gz` (36,151 rows).

Regenerate: `Rscript ops/oneoff/20260915_matchup_oracle_capture/capture_oracle.R <snapshot_dir> <out_dir>` then the
sampling step in `ops/oneoff/20260915_matchup_oracle_capture/make_fixtures.py`. Re-run the oracle live before
treating any delta as a port regression — the release parquet is republished in-season.

Known oracle quirks (ported faithfully, documented in the module):
- flags: R `ifelse(NA, 1, 0)` is NA; `%in%` and `grepl` on NA are FALSE.
- `qb_rush_weight` is `rush == 1 & position_rush != "QB"` — a NON-QB rush flag despite the name.
- `early_down_rush_weight` has no rush condition (`down <= 2`).
- `prev_drive_result` lags over the season-wide, key-sorted drive frame, so a game's first
  drive sees the previous game's last drive.

Trainer parity frames (not committed; `python/.cache/matchup/`): `scoring_opp_training_frame.parquet`
(262,089 rows) and `rush_expect_training_frame.parquet` (1,304,773 rows) are the `model$data`
slots of the source's fitted objects, extracted by
`ops/oneoff/20260915_matchup_oracle_capture/dump_training_frames.R`; their sha256 is recorded
in `python/cfb_model_build/cfb_matchup/artifacts/*_meta.json`.

WEPA weight search: `wepa_search_evals.csv` (500 × 122) is the source's evaluated candidate
table — its 500 random weight vectors with the in-sample `r2` / `sd_err` each scored on the
source's 2014–2023 corpus (June 2024; NOT reproducible from the current release, informational
only; the shipped weights are row 302, the argmax). `wepa_scores_2025.csv` (6 rows) is the
source's `evaluate_weights` run verbatim over the 2025 release for candidates 1, 2, 3, 250,
500 and the shipped weights — the exact-parity oracle for the Python scorer
(`ops/oneoff/20260915_matchup_oracle_capture/capture_wepa_scores.R`, R 4.6.1, 2026-09-15;
games and scores from `cfbd_games_elo_2025.parquet`).

Side inputs (2026-09-15, `ops/oneoff/20260915_matchup_oracle_capture/capture_side_inputs.py 2025`,
CFBD API, tidied by `cfb_data_build.matchup_side`): `cfbd_talent_2025.parquet` (134 rows,
`/talent?year=2025`), `cfbd_weather_2025.parquet` (3,235 games, `/games/weather?year=2025&seasonType=both`,
the 10 line columns), `cfbd_teams_2025.parquet` (679 schools, `/teams?year=2025`: identity block +
home venue inline), `cfbd_coaches_thru_2025.parquet` (4,496 coach-school-seasons,
`/coaches?minYear=1990&maxYear=2025`). Oracle for all four = the side / meta / weather columns of
`matchup_line_2025.csv`; bars and the source-version columns that are NOT compared (colours, logo
CDN, corrected venue rows) are recorded in `tests/cfb_data_build/test_matchup_side.py`.
