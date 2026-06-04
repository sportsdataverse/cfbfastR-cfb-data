suppressPackageStartupMessages({
  library(data.table); library(arrow); library(jsonlite); library(optparse); library(cli)
})
if (!exists("read_final_json")) source("R/_data_utils.R")

# One game-meta row from g$header$competitions[[1]] + g$gameInfo + top-level identity.
reshape_schedule_row <- function(g) {
  comp <- tryCatch(g$header$competitions[[1]], error = function(e) NULL)
  if (is.null(comp)) return(data.frame())
  comps <- comp$competitors %||% list()
  side <- function(ha) {
    m <- Filter(function(c) identical(c$homeAway, ha), comps)
    if (length(m)) m[[1]] else NULL
  }
  home <- side("home"); away <- side("away")
  data.frame(
    game_id                = as.integer(g$id),
    season                 = as.integer(g$season),
    week                   = if (!is.null(g$week)) as.integer(g$week) else NA_integer_,
    season_type            = if (!is.null(g$season_type)) as.integer(g$season_type) else NA_integer_,
    game_date              = comp$date %||% NA_character_,
    neutral_site           = isTRUE(comp$neutralSite),
    conference_competition = isTRUE(comp$conferenceCompetition),
    home_id                = as.integer(home$team$id %||% NA),
    away_id                = as.integer(away$team$id %||% NA),
    home_team              = home$team$displayName %||% NA_character_,
    away_team              = away$team$displayName %||% NA_character_,
    home_abbreviation      = home$team$abbreviation %||% NA_character_,
    away_abbreviation      = away$team$abbreviation %||% NA_character_,
    home_score             = suppressWarnings(as.integer(home$score %||% NA)),
    away_score             = suppressWarnings(as.integer(away$score %||% NA)),
    home_winner            = isTRUE(home$winner),
    away_winner            = isTRUE(away$winner),
    venue                  = g$gameInfo$venue$fullName %||% NA_character_,
    attendance             = suppressWarnings(as.integer(g$gameInfo$attendance %||% NA)),
    status                 = comp$status$type$name %||% NA_character_,
    stringsAsFactors = FALSE
  )
}

build_schedules_season <- function(season, master = fetch_master_local(), live = TRUE, publish = TRUE) {
  build_season(season, "schedules", "cfb_schedule", "espn_cfb_schedules", reshape_schedule_row,
               master = master, live = live, publish = publish)
}

main <- function() {
  opt <- optparse::parse_args(optparse::OptionParser(option_list = list(
    optparse::make_option(c("-s", "--start_year"), type = "integer"),
    optparse::make_option(c("-e", "--end_year"), type = "integer"))))
  master <- fetch_master_local()
  for (y in opt$start_year:opt$end_year) build_schedules_season(y, master = master, live = TRUE)
}
if (sys.nframe() == 0L && length(commandArgs(trailingOnly = TRUE)) > 0L) main()
