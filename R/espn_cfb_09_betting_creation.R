suppressPackageStartupMessages({
  library(data.table); library(arrow); library(jsonlite); library(optparse); library(cli)
})
if (!exists("read_final_json")) source("R/_data_utils.R")

# One game-level betting row from the RESOLVED scalar odds fields of g$betting.
# The list payloads (odds/odds_full/pickcenter/predictor/against_the_spread/propbets) are
# kept only in the JSON, NOT in the rectangular release.
reshape_betting <- function(g) {
  b <- g$betting
  if (is.null(b)) return(data.frame())
  data.frame(
    game_id               = as.integer(g$id),
    season                = as.integer(g$season),
    week                  = if (!is.null(g$week)) as.integer(g$week) else NA_integer_,
    game_spread           = as.numeric(b$game_spread %||% NA),
    over_under            = as.numeric(b$over_under %||% NA),
    home_favorite         = as.logical(b$home_favorite %||% NA),
    home_team_spread      = as.numeric(b$home_team_spread %||% NA),
    game_spread_available = as.logical(b$game_spread_available %||% NA),
    odds_source           = as.character(b$odds_source %||% NA),
    stringsAsFactors = FALSE
  )
}

build_betting_season <- function(season, master = fetch_master_local(), live = TRUE, publish = TRUE) {
  build_season(season, "betting", "betting", "espn_cfb_betting", reshape_betting,
               master = master, live = live, publish = publish)
}

main <- function() {
  opt <- optparse::parse_args(optparse::OptionParser(option_list = list(
    optparse::make_option(c("-s", "--start_year"), type = "integer"),
    optparse::make_option(c("-e", "--end_year"), type = "integer"))))
  master <- fetch_master_local()
  for (y in opt$start_year:opt$end_year) build_betting_season(y, master = master, live = TRUE)
}
if (sys.nframe() == 0L && length(commandArgs(trailingOnly = TRUE)) > 0L) main()
