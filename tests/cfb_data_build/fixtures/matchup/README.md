# Matchup-feature oracle fixtures (2025 season)

Golden output of the R matchup pipeline, captured **2026-09-15** under R 4.6.1 +
cfbfastR 2.0.0 by `ops/oneoff/20260915_matchup_oracle_capture/capture_oracle.R` (the R units run verbatim over
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
| `rp_features_sample.csv` | 5,000 × 15 | rush/pass plays with the 12 rush-expectation features + R's `rp` prediction (refit-parity oracle) | R capture, `set.seed(2025)` sample |
| `team_features_full_2025.csv` | 136 × 36 | per FBS team, FULL-season 2025 features (38-col family; the pipeline's prior-season priors file) | snapshot `data/prev_season_epa_data_2025.csv` |
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
