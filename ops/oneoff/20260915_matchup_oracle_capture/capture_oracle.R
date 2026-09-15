# Oracle capture for the matchup-pipeline port: run the R units (play tagging,
# WEPA product, rush-expectation predict, drive frame + scoring_opp predict)
# over the 2025 season exactly as the canonical pipeline does, and write the
# per-play / per-drive results as CSV for the Python parity tests.
# Run: Rscript capture_oracle.R <snapshot_dir> <out_dir>
suppressPackageStartupMessages({library(dplyr); library(purrr); library(tidyr); library(stringr); library(readr); library(jsonlite)})
args <- commandArgs(trailingOnly = TRUE)
SNAP <- args[1]; OUT <- args[2]; dir.create(OUT, showWarnings = FALSE, recursive = TRUE)
TARGET_SEASON <- 2025
MODEL_WEIGHTS_FILE <- file.path(SNAP, "models/final_wepa_weights_6_2_24.RDS")
rp_mod_3 <- read_rds(file.path(SNAP, "models/rp_mod_3.rds"))
test_scoring_opp_model <- read_rds(file.path(SNAP, "models/scoring_opp_mod.RDS"))
write_json(as.list(read_rds(MODEL_WEIGHTS_FILE)), file.path(OUT, "wepa_weights.json"), digits = NA, auto_unbox = TRUE)
cat("weights written\n")

current_pbp <- cfbfastR::load_cfb_pbp(TARGET_SEASON) %>%
  dplyr::distinct(game_id, id_play, game_play_number, .keep_all = TRUE) %>%
  dplyr::mutate(clock_minutes = as.numeric(clock_minutes), clock_seconds = as.numeric(clock_seconds),
                adj_TimeSecsRem = as.numeric(adj_TimeSecsRem))
cat("pbp rows:", nrow(current_pbp), "\n")
current_pbp_rp <- current_pbp %>% dplyr::filter(.data$rush == 1 | .data$pass == 1)
current_pbp_rp$rp <- suppressWarnings(predict(rp_mod_3, newdata = current_pbp_rp, type = "response"))

# ---- verbatim from the canonical pipeline: play tagging (A) + WEPA (B) ----
system.time({
  current_pbp_tagged <- current_pbp %>%
    mutate(
      passing_epa = ifelse(pass == 1, 1, 0),
      passing_ex_int_epa = ifelse(
        pass == 1 & !(grepl("Interception", play_type)),
        1,
        0
      ),
      rush_epa = ifelse(rush == 1, 1, 0),
      pos_rush_epa = ifelse(rush == 1 & EPA > 0, 1, 0),
      neg_rush_epa = ifelse(rush == 1 & EPA <= 0, 1, 0),
      short_yd_rush_epa = ifelse(rush == 1 & distance < 5, 1, 0),
      long_yd_rush_epa = ifelse(rush == 1 & distance >= 5, 1, 0),
      qb_rush_epa = ifelse(rush == 1 & position_rush != "QB", 1, 0),
      completed_passing_epa = ifelse(
        play_type %in% c("Pass Reception", "Passing Touchdown"),
        1,
        0
      ),
      incompleted_passing_epa = ifelse(play_type == "Pass Incompletion", 1, 0),
      off_holding_epa = ifelse(penalty_detail == "Offensive Holding", 1, 0),
      defensive_pi_epa = ifelse(
        penalty_detail == "Pass Interference" & EPA > 0,
        1,
        0
      ),
      false_start_epa = ifelse(penalty_detail == "False Start", 1, 0),
      roughing_epa = ifelse(penalty_detail == "Roughing the Passer", 1, 0),
      defensive_holding_epa = ifelse(
        penalty_detail == "Defensive Holding",
        1,
        0
      ),
      offensive_unnecessary_roughness_epa = ifelse(
        penalty_detail %in% c("Unnecessary Roughness") & EPA < 0,
        1,
        0
      ),
      offensive_unsportsmanlike_epa = ifelse(
        penalty_detail %in% c("Unsportsmanlike Conduct") & EPA < 0,
        1,
        0
      ),
      defensive_unnecessary_roughness_epa = ifelse(
        penalty_detail %in% c("Unnecessary Roughness") & EPA >= 0,
        1,
        0
      ),
      defensive_unsportsmanlike_epa = ifelse(
        penalty_detail %in% c("Unsportsmanlike Conduct") & EPA >= 0,
        1,
        0
      ),
      offensive_pi_epa = ifelse(
        penalty_detail == "Pass Interference" & EPA <= 0,
        1,
        0
      ),
      off_hold_or_false_start_epa = ifelse(
        penalty_detail %in% c("Offensive Holding", "False Start"),
        1,
        0
      ),
      sack_epa = ifelse(sack == 1, 1, 0),
      fumble_epa = ifelse(fumble_vec == 1, 1, 0),
      sack_fumble_epa = ifelse(sack == 1 & fumble_vec == 1, 1, 0),
      non_sack_fumble_epa = ifelse(sack == 0 & fumble_vec == 1, 1, 0),
      non_fumble_sack_epa = ifelse(sack == 1 & fumble_vec == 0, 1, 0),
      int_epa = ifelse(pass == 1 & (grepl("Interception", play_type)), 1, 0),
      return_td_epa = ifelse(
        (grepl("Return", play_type) | grepl("Recovery", play_type)) &
          grepl("Touchdown", play_type),
        1,
        0
      ),
      punt_epa = ifelse(grepl("Punt", play_type), 1, 0),
      blocked_punt_epa = ifelse(grepl("Blocked Punt", play_type), 1, 0),
      fg_epa = ifelse(grepl("Field Goal", play_type), 1, 0),
      kickoff_epa = ifelse(grepl("Kickoff", play_type), 1, 0),
      first_down_epa = ifelse(down == 1, 1, 0),
      second_down_epa = ifelse(down == 2, 1, 0),
      third_down_epa = ifelse(down == 3, 1, 0),
      fourth_down_epa = ifelse((rush == 1 | pass == 1) & down == 4, 1, 0),
      third_down_ex_sack_int_epa = ifelse(
        down == 3 &
          !(grepl("Interception", play_type)) &
          !(grepl("Sack", play_type)),
        1,
        0
      ),
      third_down_pos_epa = ifelse(down == 3 & EPA > 0, 1, 0),
      third_down_long_ex_sack_int_epa = ifelse(
        down == 3 &
          distance > 5 &
          !(grepl("Interception", play_type)) &
          !(grepl("Sack", play_type)),
        1,
        0
      ),
      first_down_rush_epa = ifelse(down == 1 & rush == 1, 1, 0),
      second_down_rush_epa = ifelse(down == 2 & rush == 1, 1, 0),
      third_down_rush_epa = ifelse(down == 3 & rush == 1, 1, 0),
      fourth_down_rush_epa = ifelse(down == 4 & rush == 1, 1, 0),
      first_down_pass_epa = ifelse(down == 1 & pass == 1, 1, 0),
      second_down_pass_epa = ifelse(down == 2 & pass == 1, 1, 0),
      third_down_pass_epa = ifelse(down == 3 & pass == 1, 1, 0),
      fourth_down_pass_epa = ifelse(down == 4 & pass == 1, 1, 0),
      neutral_second_down_rush_epa = ifelse(
        down == 2 & rush == 1 & wp_before > 0.05 & wp_after < 0.95,
        1,
        0
      ),
      early_down_rush_epa = ifelse(down <= 2, 1, 0),
      early_down_sack_epa = ifelse(
        down <= 2 & sack == 1 & fumble_vec == 0,
        1,
        0
      ),
      red_zone_epa = ifelse(yards_to_goal <= 20, 1, 0),
      goal_to_go_epa = ifelse(Goal_To_Go == TRUE, 1, 0),
      goalline_epa = ifelse(yards_to_goal <= 3, 1, 0),
      plus_territory_epa = ifelse(yards_to_goal <= 50, 1, 0),
      low_wp_epa = ifelse((wp_before <= 0.05 | wp_before >= 0.95), 1, 0),
      garbage_time_epa = ifelse(
        (score_diff >= 28 & period == 1) |
          (score_diff >= 24 & period == 2) |
          (score_diff >= 21 & period == 3) |
          (score_diff >= 16 & period == 4),
        1,
        0
      ),
      asym_low_wp_epa = ifelse((wp_before <= 0.2 | wp_before >= 0.95), 1, 0),
      asym_garbage_time_epa = ifelse(
        (wp_before <= 0.2 | wp_before >= 0.95) & period == 4,
        1,
        0
      ),
      offense_home_epa = ifelse(pos_team == home, 1, 0),
      offense_away_epa = ifelse(pos_team == away, 1, 0)
    ) %>%
    rename_with(
      ~ paste0(str_remove(.x, "_epa"), "_weight"),
      ends_with("_epa", ignore.case = FALSE)
    ) %>%
    mutate(across(ends_with("_weight"), ~.x, .names = "def_{.col}")) %>%
    rename_with(
      ~ paste0("off_", .x),
      (ends_with("_weight", ignore.case = FALSE) &
        !starts_with("def_", ignore.case = FALSE))
    ) %>%
    select(ends_with("_weight"))
})

# ---------- WEPA (weights * features) ----------
model_weights <- read_rds(MODEL_WEIGHTS_FILE)

current_wepa <- current_pbp_tagged %>%
  map2(model_weights, `*`) %>%
  bind_rows() %>%
  ungroup() %>%
  mutate(
    across(ends_with("_weight"), ~ as.numeric(replace_na(.x, 0) + 1)),
    epa = current_pbp %>% pull(EPA),
    off_wepa = pmap_dbl(pick(starts_with("off_"), epa), prod),
    def_wepa = pmap_dbl(pick(starts_with("def_"), epa), prod)
  ) %>%
  select(off_wepa, def_wepa)

# ---- (C) apply on the full frame, as the pipeline does ----
current_pbp_aug <- current_pbp %>% dplyr::bind_cols(current_wepa)
rp_pred <- suppressWarnings(predict.glm(rp_mod_3, newdata = current_pbp_aug, type = "response"))
stopifnot(length(rp_pred) == nrow(current_pbp_aug))
current_pbp_aug <- current_pbp_aug %>% dplyr::mutate(rp_prediction = as.numeric(rp_pred), rroe = rush - rp_prediction)

flags <- current_pbp_tagged %>% dplyr::select(dplyr::starts_with("off_")) %>%
  dplyr::rename_with(~ stringr::str_remove(.x, "^off_"))
out_plays <- dplyr::bind_cols(
  current_pbp_aug %>% dplyr::select(game_id, id_play, game_play_number, EPA, rush, pass, off_wepa, def_wepa, rp_prediction, rroe),
  flags
)
readr::write_csv(out_plays, file.path(OUT, "play_flags_wepa_2025.csv.gz"), na = "")
cat("plays written:", nrow(out_plays), ncol(out_plays), "\n")

# ---- (D) drive frame + scoring_opp, verbatim logic, season-wide (no per-team filter) ----
mk_gsr <- function(df) dplyr::mutate(df, gsr = as.numeric(adj_TimeSecsRem))
team_all_plays <- current_pbp_aug %>% dplyr::filter(!is.na(ppa)) %>% mk_gsr()
team_drive <- team_all_plays %>%
  dplyr::mutate(scoring_opp_ind_no_td = ifelse(down == 1 & yards_to_goal < 40, 1, 0)) %>%
  dplyr::group_by(season, start_date, drive_id, pos_team, def_pos_team) %>%
  dplyr::summarise(dplyr::across(c(new_drive_pts, drive_start_period, TimeSecsRem, adj_TimeSecsRem, yards_to_goal,
                                   drive_end_yards_to_goal, pos_team_timeouts, def_pos_team_timeouts, drive_result), ~ .[1]),
                   scoring_opp_ind_no_td = ifelse(sum(scoring_opp_ind_no_td) > 0, 1, 0), .groups = "drop") %>%
  dplyr::rename(start_yards_to_goal = yards_to_goal, half_secs_rem = TimeSecsRem, game_secs_rem = adj_TimeSecsRem) %>%
  dplyr::filter(start_yards_to_goal > 40, !is.na(pos_team_timeouts), !is.na(def_pos_team_timeouts)) %>%
  dplyr::mutate(prev_drive_result = dplyr::lag(drive_result),
                scoring_opp = dplyr::case_when(new_drive_pts >= 6 ~ 1, scoring_opp_ind_no_td == 1 ~ 1, TRUE ~ 0),
                scoring_opp_prediction = suppressWarnings(predict.glm(test_scoring_opp_model, ., type = "response")),
                scoring_opp_oe = scoring_opp - scoring_opp_prediction)
readr::write_csv(team_drive, file.path(OUT, "drives_2025.csv.gz"), na = "")
cat("drives written:", nrow(team_drive), "\n")

# ---- (C refit oracle) a 5k-play sample with the 12 rp features + R prediction ----
set.seed(2025)
rp_cols <- c("pos_team_score","def_pos_team_score","score_diff","half","period","TimeSecsRem","down","distance","yards_to_goal","ep_before","wp_before","rz_play","rush")
samp <- current_pbp_rp %>% dplyr::select(game_id, id_play, dplyr::all_of(rp_cols), rp) %>% tidyr::drop_na() %>% dplyr::slice_sample(n = 5000)
readr::write_csv(samp, file.path(OUT, "rp_features_sample.csv"), na = "")
cat("rp sample written:", nrow(samp), "\n")
cat("CAPTURE-COMPLETE\n")
