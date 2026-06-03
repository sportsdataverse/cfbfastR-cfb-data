suppressPackageStartupMessages({
  library(dplyr); library(data.table); library(arrow); library(jsonlite); library(optparse); library(cli)
})
if (!exists("read_final_json")) source("R/_data_utils.R")

# ESPN team box: g$boxscore$teams[] — each has $team (id/abbrev/name) + $statistics
# (list of {name, displayValue, value, label}). Pivot stat name -> column (displayValue).
reshape_team_box <- function(g) {
  teams <- g$boxscore$teams
  if (is.null(teams) || length(teams) == 0) return(data.frame())
  rows <- lapply(teams, function(t) {
    stats <- t$statistics %||% list()
    vals <- lapply(stats, function(s) as.character(s$displayValue %||% s$value %||% NA))
    names(vals) <- vapply(stats, function(s) s$name %||% s$label %||% "stat", character(1))
    df <- if (length(vals)) as.data.frame(vals, stringsAsFactors = FALSE, check.names = FALSE) else data.frame()
    df$team_id <- as.integer(t$team$id %||% NA)
    df$team_abbreviation <- t$team$abbreviation %||% NA_character_
    df$team_name <- t$team$displayName %||% NA_character_
    df$home_away <- t$homeAway %||% NA_character_
    df
  })
  out <- as.data.frame(data.table::rbindlist(rows, fill = TRUE, use.names = TRUE), check.names = FALSE)
  out$game_id <- as.integer(g$id); out$season <- as.integer(g$season)
  out
}

build_team_box_season <- function(season, master = fetch_master_local(), live = TRUE, publish = TRUE) {
  build_season(season, "team_box", "team_box", "espn_cfb_team_box", reshape_team_box,
               master = master, live = live, publish = publish)
}

main <- function() {
  opt <- optparse::parse_args(optparse::OptionParser(option_list = list(
    optparse::make_option(c("-s", "--start_year"), type = "integer"),
    optparse::make_option(c("-e", "--end_year"), type = "integer"))))
  master <- fetch_master_local()
  for (y in opt$start_year:opt$end_year) build_team_box_season(y, master = master, live = TRUE)
}
if (sys.nframe() == 0L && length(commandArgs(trailingOnly = TRUE)) > 0L) main()
