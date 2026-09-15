# Oracle capture, part 2: the per-team as-of aggregation (unit F) run verbatim
# over the 2025 season. The loop function and mk_gsr are extracted from the
# canonical pipeline unchanged; only its inputs are rebuilt here from the pbp
# (played games -> games_cur) and the FBS team list of the prior-season prep
# file (cur_szn_team). Writes team_features_asof_2025.csv (one row per FBS
# team-game, features from plays strictly before the game's date; the first
# game uses same-day regular-season plays, as the pipeline does) and
# team_features_full_2025_r.csv (synthetic future game -> whole season).
# Run: Rscript capture_team_features.R <snapshot_dir> <oracle_dir>
suppressPackageStartupMessages({library(dplyr); library(purrr); library(tidyr); library(stringr); library(readr); library(lubridate)})
args <- commandArgs(trailingOnly = TRUE)
SNAP <- args[1]; OUT <- args[2]
TARGET_SEASON <- 2025
MODEL_WEIGHTS_FILE <- file.path(SNAP, "models/final_wepa_weights_6_2_24.RDS")
rp_mod_3 <- read_rds(file.path(SNAP, "models/rp_mod_3.rds"))
test_scoring_opp_model <- read_rds(file.path(SNAP, "models/scoring_opp_mod.RDS"))
current_pbp <- cfbfastR::load_cfb_pbp(TARGET_SEASON) %>%
  dplyr::distinct(game_id, id_play, game_play_number, .keep_all = TRUE) %>%
  dplyr::mutate(clock_minutes = as.numeric(clock_minutes), clock_seconds = as.numeric(clock_seconds),
                adj_TimeSecsRem = as.numeric(adj_TimeSecsRem))
cat("pbp rows:", nrow(current_pbp), "\n")
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
current_pbp_aug <- current_pbp %>% dplyr::bind_cols(current_wepa)
rp_pred <- suppressWarnings(predict.glm(rp_mod_3, newdata = current_pbp_aug, type = "response"))
current_pbp_aug <- current_pbp_aug %>% dplyr::mutate(rp_prediction = as.numeric(rp_pred), rroe = rush - rp_prediction)
rm(current_pbp_tagged, current_wepa); invisible(gc())

# games_cur: played games, from the pbp itself (same fields the loop reads)
games_cur <- current_pbp %>%
  dplyr::distinct(game_id, season, week, season_type, start_date, home, away) %>%
  dplyr::transmute(game_id, season, week, season_type,
                   start_date = as.Date(lubridate::as_datetime(start_date)),
                   home_team = home, away_team = away)
cat("games:", nrow(games_cur), "\n")
fbs <- readr::read_csv(file.path(SNAP, "data/prev_season_epa_data_2025.csv"), show_col_types = FALSE)$team
cur_szn_team <- tibble::tibble(season = TARGET_SEASON, team = fbs)
cat("FBS teams:", nrow(cur_szn_team), "\n")

# ---- verbatim: mk_gsr + the per-team loop function ----
mk_gsr <- function(df) {
  # Prefer cfbfastR's adjusted seconds if present
  if ("adj_TimeSecsRem" %in% names(df)) {
    return(dplyr::mutate(df, gsr = as.numeric(adj_TimeSecsRem)))
  }

  # Otherwise compute from period + clock.{minutes,seconds} (or "clock" MM:SS)
  have_min <- "clock_minutes" %in% names(df)
  have_sec <- "clock_seconds" %in% names(df)
  have_str <- "clock" %in% names(df)

  # if ((!have_min || !have_sec) && have_str) {
  #   mins <- suppressWarnings(as.numeric(sub(":.*$", "", df$clock)))
  #   secs <- suppressWarnings(as.numeric(sub("^.*:", "", df$clock)))
  #   if (!have_min) df$clock.minutes <- mins
  #   if (!have_sec) df$clock.seconds <- secs
  #   have_min <- TRUE; have_sec <- TRUE
  # }

  dplyr::mutate(
    df,
    period = suppressWarnings(as.integer(period)),
    clock.minutes = if (have_min) {
      suppressWarnings(as.numeric(clock_minutes))
    } else {
      NA_real_
    },
    clock.seconds = if (have_sec) {
      suppressWarnings(as.numeric(clock_seconds))
    } else {
      NA_real_
    },
    sec_in_period = clock.minutes * 60 + clock.seconds,
    # 15-min regulation quarters; OT -> NA (dropped in pace calc)
    gsr = dplyr::if_else(
      !is.na(sec_in_period) & !is.na(period) & period <= 4L,
      (4L - period) * 900 + sec_in_period,
      NA_real_
    )
  )
}
calc_cur_epa_play_and_drive_data <- function(tm_row) {
  tryCatch(
    {
      team <- cur_szn_team$team[tm_row]
      cur_season <- TARGET_SEASON
      message(paste(
        "Calculating current EPA/play and drive data for",
        team,
        "in",
        cur_season
      ))
      team_games <- games_cur %>%
        dplyr::filter(
          season == {{ cur_season }},
          (home_team == {{ team }} | away_team == {{ team }})
        ) %>%
        dplyr::select(game_id, start_date) %>%
        dplyr::arrange(start_date)

      results <- list()

      for (i in seq_len(nrow(team_games))) {
        game <- team_games[i, ]
        game_date <- game$start_date
        game_id <- game$game_id

        if (i == 1) {
          team_all_plays <- current_pbp_aug %>%
            dplyr::filter(
              season == {{ cur_season }},
              start_date == {{ game_date }},
              (home == {{ team }} | away == {{ team }}),
              !is.na(ppa),
              season_type == "regular"
            ) %>%
            mk_gsr()
        } else {
          team_all_plays <- current_pbp_aug %>%
            dplyr::filter(
              season == {{ cur_season }},
              start_date < {{ game_date }},
              (home == {{ team }} | away == {{ team }}),
              !is.na(ppa)
            ) %>%
            mk_gsr()
        }

        if (nrow(team_all_plays) == 0) {
          result <- tibble::tibble(
            game_id = {{ game_id }},
            team = {{ team }},
            off_plays_per_game = NA,
            off_3d_per_game = NA,
            off_epa = NA,
            off_pass_epa = NA,
            off_rush_epa = NA,
            off_rroe = NA,
            off_1st_down_rush_rate = NA,
            off_scoring_opp_rate_oe = NA,
            off_pts_per_scoring_opp = NA,
            off_starting_fp = NA,
            off_wepa = NA,
            off_success_rate = NA,
            off_early_success_rate = NA,
            off_late_success_rate = NA,
            off_pass_success_rate = NA,
            off_rush_success_rate = NA,
            off_3rd_down_pct = NA,
            def_plays_per_game = NA,
            def_3d_per_game = NA,
            def_epa = NA,
            def_pass_epa = NA,
            def_rush_epa = NA,
            def_rroe = NA,
            def_1st_down_rush_rate = NA,
            def_scoring_opp_rate_oe = NA,
            def_pts_per_scoring_opp = NA,
            def_starting_fp = NA,
            def_wepa = NA,
            def_success_rate = NA,
            def_early_success_rate = NA,
            def_late_success_rate = NA,
            def_pass_success_rate = NA,
            def_rush_success_rate = NA,
            def_3rd_down_pct = NA,
            # pace (sec/play)
            off_sec_per_play_mean = NA_real_,
            off_sec_per_play_median = NA_real_,
            def_sec_per_play_mean = NA_real_,
            def_sec_per_play_median = NA_real_
          )
        } else {
          team_drive <- team_all_plays %>%
            dplyr::mutate(
              scoring_opp_ind_no_td = ifelse(down == 1 & yards_to_goal < 40, 1, 0)
            ) %>%
            dplyr::group_by(
              season,
              start_date,
              drive_id,
              pos_team,
              def_pos_team
            ) %>%
            dplyr::summarise(
              dplyr::across(
                c(
                  new_drive_pts,
                  drive_start_period,
                  TimeSecsRem,
                  adj_TimeSecsRem,
                  yards_to_goal,
                  drive_end_yards_to_goal,
                  pos_team_timeouts,
                  def_pos_team_timeouts,
                  drive_result
                ),
                ~ .[1]
              ),
              scoring_opp_ind_no_td = ifelse(sum(scoring_opp_ind_no_td) > 0, 1, 0),
              .groups = "drop"
            ) %>%
            dplyr::rename(
              start_yards_to_goal = yards_to_goal,
              half_secs_rem = TimeSecsRem,
              game_secs_rem = adj_TimeSecsRem
            ) %>%
            dplyr::filter(
              start_yards_to_goal > 40,
              !is.na(pos_team_timeouts),
              !is.na(def_pos_team_timeouts)
            ) %>%
            dplyr::mutate(
              prev_drive_result = dplyr::lag(drive_result),
              scoring_opp = dplyr::case_when(
                new_drive_pts >= 6 ~ 1,
                scoring_opp_ind_no_td == 1 ~ 1,
                TRUE ~ 0
              ),
              # test_scoring_opp_model is likewise rank-deficient; silence the benign
              # non-estimable note (predictions for estimable cases are correct).
              scoring_opp_prediction = suppressWarnings(predict.glm(
                test_scoring_opp_model,
                .,
                type = "response"
              )),
              scoring_opp_oe = scoring_opp - scoring_opp_prediction
            )

          # --- PACE (improved): offense & defense with proper filtering ---

          # Offensive pace calculation
          off_pace <- team_all_plays %>%
            dplyr::filter(offense_play == {{ team }}, rush == 1 | pass == 1) %>%
            dplyr::group_by(game_id, drive_id) %>%
            dplyr::arrange(
              game_id,
              drive_id,
              dplyr::desc(gsr),
              .by_group = TRUE
            ) %>%
            dplyr::mutate(
              prev_drive_id = dplyr::lag(drive_id),
              prev_gsr = dplyr::lag(gsr),
              # Only calculate time between plays in the same drive
              sec_since_prev = dplyr::case_when(
                is.na(prev_drive_id) ~ NA_real_, # First play of game
                drive_id != prev_drive_id ~ NA_real_, # First play of new drive
                is.na(prev_gsr) | is.na(gsr) ~ NA_real_, # Missing time data
                TRUE ~ pmax(prev_gsr - gsr, 0) # Normal calculation
              )
            ) %>%
            dplyr::ungroup()

          # Defensive pace calculation (same logic)
          def_pace <- team_all_plays %>%
            dplyr::filter(defense_play == {{ team }}, rush == 1 | pass == 1) %>%
            dplyr::group_by(game_id, drive_id) %>%
            dplyr::arrange(
              game_id,
              drive_id,
              dplyr::desc(gsr),
              .by_group = TRUE
            ) %>%
            dplyr::mutate(
              prev_drive_id = dplyr::lag(drive_id),
              prev_gsr = dplyr::lag(gsr),
              sec_since_prev = dplyr::case_when(
                is.na(prev_drive_id) ~ NA_real_,
                drive_id != prev_drive_id ~ NA_real_,
                is.na(prev_gsr) | is.na(gsr) ~ NA_real_,
                TRUE ~ pmax(prev_gsr - gsr, 0)
              )
            ) %>%
            dplyr::ungroup()

          # Calculate summary statistics
          off_sec_per_play_mean <- mean(off_pace$sec_since_prev, na.rm = TRUE)
          off_sec_per_play_median <- median(
            off_pace$sec_since_prev,
            na.rm = TRUE
          )
          def_sec_per_play_mean <- mean(def_pace$sec_since_prev, na.rm = TRUE, )
          def_sec_per_play_median <- median(
            def_pace$sec_since_prev,
            na.rm = TRUE
          )
          # message(paste("Pace for team", {{team}}, "- Offense (mean/median):",
          #               round({{off_sec_per_play_mean}}, 2), "/",
          #               round({{off_sec_per_play_median}}, 2),
          #               "; Defense (mean/median):",
          #               round({{def_sec_per_play_mean}}, 2), "/",
          #               round({{def_sec_per_play_median}}, 2),
          #               "; Off plays:", nrow({{off_pace}}),
          #               "; Def plays:", nrow({{def_pace}})))
          # Additional validation
          if (
            {{ off_sec_per_play_mean }} > 50 || {{ def_sec_per_play_mean }} > 50
          ) {
            message(paste(
              "Note: unusually high pace values for team",
              {{ team }},
              "- check data quality (row is kept, not discarded)"
            ))
          }

          result <- tibble::tibble(
            game_id = {{ game_id }},
            team = {{ team }},
            off_plays_per_game = team_all_plays %>%
              dplyr::filter(offense_play == {{ team }}) %>%
              dplyr::group_by(game_id) %>%
              dplyr::summarise(plays = dplyr::n(), .groups = "drop") %>%
              dplyr::pull(plays) %>%
              stats::median(na.rm = TRUE),
            off_3d_per_game = team_all_plays %>%
              dplyr::filter(offense_play == {{ team }}, down == 3) %>%
              dplyr::group_by(game_id) %>%
              dplyr::summarise(plays = dplyr::n(), .groups = "drop") %>%
              dplyr::pull(plays) %>%
              stats::median(na.rm = TRUE),
            off_epa = team_all_plays %>%
              dplyr::filter(offense_play == {{ team }}) %>%
              dplyr::pull(ppa) %>%
              mean(na.rm = TRUE),
            off_pass_epa = team_all_plays %>%
              dplyr::filter(offense_play == {{ team }}, pass == 1) %>%
              dplyr::pull(ppa) %>%
              mean(na.rm = TRUE),
            off_rush_epa = team_all_plays %>%
              dplyr::filter(offense_play == {{ team }}, rush == 1) %>%
              dplyr::pull(ppa) %>%
              mean(na.rm = TRUE),
            off_rroe = team_all_plays %>%
              dplyr::filter(offense_play == {{ team }}) %>%
              dplyr::pull(rroe) %>%
              mean(na.rm = TRUE),
            off_1st_down_rush_rate = team_all_plays %>%
              dplyr::filter(offense_play == {{ team }}, down == 1) %>%
              dplyr::pull(rush) %>%
              mean(na.rm = TRUE),
            off_scoring_opp_rate_oe = team_drive %>%
              dplyr::filter(pos_team == {{ team }}) %>%
              dplyr::pull(scoring_opp_oe) %>%
              mean(na.rm = TRUE),
            off_pts_per_scoring_opp = team_drive %>%
              dplyr::filter(pos_team == {{ team }}, scoring_opp == 1) %>%
              dplyr::pull(new_drive_pts) %>%
              mean(na.rm = TRUE),
            off_starting_fp = team_drive %>%
              dplyr::filter(
                pos_team == {{ team }},
                prev_drive_result == "PUNT"
              ) %>%
              dplyr::pull(start_yards_to_goal) %>%
              mean(na.rm = TRUE),
            off_wepa = team_all_plays %>%
              dplyr::filter(offense_play == {{ team }}) %>%
              dplyr::pull(off_wepa) %>%
              mean(na.rm = TRUE),
            off_success_rate = team_all_plays %>%
              dplyr::filter(offense_play == {{ team }}) %>%
              dplyr::pull(success) %>%
              mean(na.rm = TRUE),
            off_early_success_rate = team_all_plays %>%
              dplyr::filter(offense_play == {{ team }}, down <= 2) %>%
              dplyr::pull(success) %>%
              mean(na.rm = TRUE),
            off_late_success_rate = team_all_plays %>%
              dplyr::filter(offense_play == {{ team }}, down == 3) %>%
              dplyr::pull(success) %>%
              mean(na.rm = TRUE),
            off_pass_success_rate = team_all_plays %>%
              dplyr::filter(offense_play == {{ team }}, pass == 1) %>%
              dplyr::pull(success) %>%
              mean(na.rm = TRUE),
            off_rush_success_rate = team_all_plays %>%
              dplyr::filter(offense_play == {{ team }}, rush == 1) %>%
              dplyr::pull(success) %>%
              mean(na.rm = TRUE),
            off_3rd_down_pct = team_all_plays %>%
              dplyr::filter(offense_play == {{ team }}, down <= 3) %>%
              dplyr::summarise(pct = mean(down == 3, na.rm = TRUE)) %>%
              dplyr::pull(pct),
            def_plays_per_game = team_all_plays %>%
              dplyr::filter(defense_play == {{ team }}) %>%
              dplyr::group_by(game_id) %>%
              dplyr::summarise(plays = dplyr::n(), .groups = "drop") %>%
              dplyr::pull(plays) %>%
              stats::median(na.rm = TRUE),
            def_3d_per_game = team_all_plays %>%
              dplyr::filter(defense_play == {{ team }}, down == 3) %>%
              dplyr::group_by(game_id) %>%
              dplyr::summarise(plays = dplyr::n(), .groups = "drop") %>%
              dplyr::pull(plays) %>%
              stats::median(na.rm = TRUE),
            def_epa = team_all_plays %>%
              dplyr::filter(defense_play == {{ team }}) %>%
              dplyr::pull(ppa) %>%
              mean(na.rm = TRUE),
            def_pass_epa = team_all_plays %>%
              dplyr::filter(defense_play == {{ team }}, pass == 1) %>%
              dplyr::pull(ppa) %>%
              mean(na.rm = TRUE),
            def_rush_epa = team_all_plays %>%
              dplyr::filter(defense_play == {{ team }}, rush == 1) %>%
              dplyr::pull(ppa) %>%
              mean(na.rm = TRUE),
            def_rroe = team_all_plays %>%
              dplyr::filter(defense_play == {{ team }}) %>%
              dplyr::pull(rroe) %>%
              mean(na.rm = TRUE),
            def_1st_down_rush_rate = team_all_plays %>%
              dplyr::filter(defense_play == {{ team }}, down == 1) %>%
              dplyr::pull(rush) %>%
              mean(na.rm = TRUE),
            def_scoring_opp_rate_oe = team_drive %>%
              dplyr::filter(def_pos_team == {{ team }}) %>%
              dplyr::pull(scoring_opp_oe) %>%
              mean(na.rm = TRUE),
            def_pts_per_scoring_opp = team_drive %>%
              dplyr::filter(def_pos_team == {{ team }}, scoring_opp == 1) %>%
              dplyr::pull(new_drive_pts) %>%
              mean(na.rm = TRUE),
            def_starting_fp = team_drive %>%
              dplyr::filter(
                def_pos_team == {{ team }},
                prev_drive_result == "PUNT"
              ) %>%
              dplyr::pull(start_yards_to_goal) %>%
              mean(na.rm = TRUE),
            def_wepa = team_all_plays %>%
              dplyr::filter(defense_play == {{ team }}) %>%
              dplyr::pull(def_wepa) %>%
              mean(na.rm = TRUE),
            def_success_rate = team_all_plays %>%
              dplyr::filter(defense_play == {{ team }}) %>%
              dplyr::pull(success) %>%
              mean(na.rm = TRUE),
            def_early_success_rate = team_all_plays %>%
              dplyr::filter(defense_play == {{ team }}, down <= 2) %>%
              dplyr::pull(success) %>%
              mean(na.rm = TRUE),
            def_late_success_rate = team_all_plays %>%
              dplyr::filter(defense_play == {{ team }}, down == 3) %>%
              dplyr::pull(success) %>%
              mean(na.rm = TRUE),
            def_pass_success_rate = team_all_plays %>%
              dplyr::filter(defense_play == {{ team }}, pass == 1) %>%
              dplyr::pull(success) %>%
              mean(na.rm = TRUE),
            def_rush_success_rate = team_all_plays %>%
              dplyr::filter(defense_play == {{ team }}, rush == 1) %>%
              dplyr::pull(success) %>%
              mean(na.rm = TRUE),
            def_3rd_down_pct = team_all_plays %>%
              dplyr::filter(defense_play == {{ team }}, down <= 3) %>%
              dplyr::summarise(pct = mean(down == 3, na.rm = TRUE)) %>%
              dplyr::pull(pct),
            # pace outputs
            off_sec_per_play_mean = {{ off_sec_per_play_mean }},
            off_sec_per_play_median = {{ off_sec_per_play_median }},
            def_sec_per_play_mean = {{ def_sec_per_play_mean }},
            def_sec_per_play_median = {{ def_sec_per_play_median }}
          )
        }

        results[[i]] <- result
      }

      dplyr::bind_rows(results)
    },
    # NOTE: deliberately no `warning =` handler. A tryCatch warning handler
    # unwinds the WHOLE function on the first warning, so a benign data-quality
    # warning (e.g. the high-pace note above) used to discard the team's entire
    # EPA result -> all-NA output. This was the Michigan State / Eastern /
    # Western Michigan bug. Let warnings print normally; the result completes.
    error = function(e) {
      message(paste("Error calculating EPA/drive data for team", tm_row, ":", e))
    }
  )
}

run_all <- function() {
  if (requireNamespace("furrr", quietly = TRUE) && requireNamespace("future", quietly = TRUE)) {
    options(future.globals.maxSize = 3 * 1024^3)
    future::plan(future::multisession, workers = min(6L, max(1L, parallel::detectCores() - 2L)))
    on.exit(future::plan(future::sequential), add = TRUE)
    furrr::future_map_dfr(seq_len(nrow(cur_szn_team)), purrr::possibly(calc_cur_epa_play_and_drive_data), .options = furrr::furrr_options(seed = TRUE))
  } else {
    purrr::map_dfr(seq_len(nrow(cur_szn_team)), purrr::possibly(calc_cur_epa_play_and_drive_data))
  }
}
asof <- run_all()
asof <- asof %>% dplyr::left_join(games_cur %>% dplyr::select(game_id, season, week, season_type, start_date), by = "game_id")
readr::write_csv(asof, file.path(OUT, "team_features_asof_2025.csv"), na = "")
cat("as-of rows:", nrow(asof), "\n")

synth <- tibble::tibble(game_id = -seq_len(nrow(cur_szn_team)), season = TARGET_SEASON, week = 99L, season_type = "synthetic",
                        start_date = as.Date("2026-02-15"), home_team = cur_szn_team$team, away_team = "SYNTH_FULL_SEASON")
games_cur <- dplyr::bind_rows(games_cur, synth)
full <- run_all() %>% dplyr::filter(game_id < 0) %>% dplyr::mutate(season = TARGET_SEASON) %>% dplyr::distinct(team, .keep_all = TRUE)
readr::write_csv(full, file.path(OUT, "team_features_full_2025_r.csv"), na = "")
cat("full rows:", nrow(full), "\n")
cat("CAPTURE-F-COMPLETE\n")
