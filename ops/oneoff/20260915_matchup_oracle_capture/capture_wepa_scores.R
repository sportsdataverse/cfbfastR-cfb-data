# Oracle capture, part 3: the WEPA weight-search SCORER run verbatim over the
# 2025 season for a handful of candidate weight vectors (5 rows of the
# source's 500-candidate evaluation table + the shipped weights). Writes
# wepa_scores_2025.csv: candidate id, adj r2, sd_err -- the exact-parity
# oracle for the Python scorer (same plays, same games, same weights).
# Run: Rscript capture_wepa_scores.R <snapshot_dir> <fixture_dir> <out_dir>
suppressPackageStartupMessages({library(dplyr); library(purrr); library(tidyr); library(stringr); library(readr); library(tibble)})
args <- commandArgs(trailingOnly = TRUE)
SNAP <- args[1]; FIX <- args[2]; OUT <- args[3]
TARGET_SEASON <- 2025
current_pbp <- cfbfastR::load_cfb_pbp(TARGET_SEASON) %>%
  dplyr::distinct(game_id, id_play, game_play_number, .keep_all = TRUE)
all_hist_pbp <- current_pbp
cat("pbp rows:", nrow(all_hist_pbp), "\n")
# games with final scores: the CFBD /games fixture (same rows the pipeline's cfbd_game_info returns)
g <- arrow::read_parquet(file.path(FIX, "cfbd_games_elo_2025.parquet"))
all_hist_games <- g %>% dplyr::transmute(game_id, season, start_date = as.Date(substr(start_date, 1, 10)),
                                         home_team, away_team, home_points, away_points) %>%
  dplyr::filter(game_id %in% unique(all_hist_pbp$game_id), !is.na(home_points), !is.na(away_points))
cat("games:", nrow(all_hist_games), "\n")
# ---- verbatim: play tagging (A) -- produces all_hist_pbp_tagged ----
current_pbp_tagged <- NULL
system.time({
  all_hist_pbp_tagged <- all_hist_pbp %>%
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
cat("tagged cols:", ncol(all_hist_pbp_tagged), "\n")
evals <- readr::read_csv(file.path(FIX, "wepa_search_evals.csv"), show_col_types = FALSE)
wcols <- names(all_hist_pbp_tagged)
stopifnot(all(wcols %in% names(evals)))
shipped <- readr::read_rds(file.path(SNAP, "models/final_wepa_weights_6_2_24.RDS"))
cand_ids <- c(1, 2, 3, 250, 500)
mat <- rbind(as.matrix(evals[cand_ids, wcols]), as.matrix(shipped[wcols]))
rownames(mat) <- c(paste0("eval_", cand_ids), "shipped")
# ---- verbatim: evaluate_weights from tools/wepa_weight_training.R, with the
# all_hist_pbp `select(1:def_pos_team)` replaced by the columns it reads ----
evaluate_weights <- function(w_matrix_row) {
  pbp_desc <- all_hist_pbp %>% select(game_id, season, start_date, pos_team, def_pos_team) %>%
    dplyr::mutate(start_date = as.Date(substr(start_date, 1, 10)))
  wepa_calc_values <- all_hist_pbp_tagged %>%
    map2( mat[w_matrix_row,],`*`) %>%
    bind_rows() %>% ungroup() %>%
    mutate(across(ends_with("_weight"), ~ as.numeric(replace_na(.x,0) + 1)),
           epa = all_hist_pbp %>% pull(EPA),
           off_wepa = pmap_dbl(pick(starts_with("off_"),epa),prod),
           def_wepa = pmap_dbl(pick(starts_with("def_"),epa),prod)) %>%
    select(off_wepa, def_wepa)
  pbp_wepa <- bind_cols(pbp_desc,wepa_calc_values)
  summarize_per_game <- function(game_row) {
    h_team <- all_hist_games$home_team[game_row]
    a_team <- all_hist_games$away_team[game_row]
    cur_season <- all_hist_games$season[game_row]
    cur_date <- as.Date(all_hist_games$start_date[game_row])
    margin <- all_hist_games$home_points[game_row] - all_hist_games$away_points[game_row]
    home_all_plays <- pbp_wepa %>% filter(season == cur_season, start_date < cur_date, (pos_team == h_team | def_pos_team == h_team), !is.na(off_wepa))
    away_all_plays <- pbp_wepa %>% filter(season == cur_season, start_date < cur_date, (pos_team == a_team | def_pos_team == a_team), !is.na(off_wepa))
    tibble(
      home_off_wepa = home_all_plays %>% filter(pos_team == h_team) %>% pull(off_wepa) %>% mean(.,na.rm = T),
      home_def_wepa = home_all_plays %>% filter(def_pos_team == h_team) %>% pull(def_wepa) %>% mean(.,na.rm = T),
      away_off_wepa = away_all_plays %>% filter(pos_team == a_team) %>% pull(off_wepa) %>% mean(.,na.rm = T),
      away_def_wepa = away_all_plays %>% filter(def_pos_team == a_team) %>% pull(def_wepa) %>% mean(.,na.rm = T),
      mov_home_team = margin)
  }
  weight_eval_df <- 1:nrow(all_hist_games) %>% map(possibly(summarize_per_game)) %>% bind_rows()
  reg_mod <- lm(mov_home_team ~ ., data = (weight_eval_df %>% na.omit()))
  tibble(candidate = rownames(mat)[w_matrix_row], n_games = nrow(weight_eval_df %>% na.omit()),
         sd_err = summary(reg_mod)$sigma, r2 = summary(reg_mod)$adj.r.squared)
}
res <- purrr::map_dfr(seq_len(nrow(mat)), evaluate_weights)
print(res)
readr::write_csv(res, file.path(OUT, "wepa_scores_2025.csv"))
# the per-game frame for the shipped weights too, so the aggregation itself can be checked
cat("CAPTURE-WEPA-COMPLETE\n")
