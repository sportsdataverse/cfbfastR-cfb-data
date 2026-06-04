suppressPackageStartupMessages({
  library(data.table); library(arrow); library(jsonlite); library(optparse); library(cli)
})
if (!exists("read_final_json")) source("R/_data_utils.R")

# Season-level ESPN roster derived from the per-game `game_rosters` compilation
# (espn_cfb_07_game_rosters_creation.R, which 07 writes to disk before this runs).
# Drop the per-game circumstance fields and de-duplicate to one row per athlete per
# team per season, keeping the most-recent (latest game) attribute values.
GAME_ROSTER_GAME_COLS <- c("game_id", "week", "starter", "did_not_play", "order",
                           "home_away", "winner")

derive_rosters <- function(gr) {
  if (is.null(gr) || !nrow(gr)) return(data.frame())
  dt <- data.table::as.data.table(gr)
  if ("game_id" %in% names(dt)) data.table::setorderv(dt, "game_id")
  keys <- intersect(c("season", "team_id", "athlete_id"), names(dt))
  if (length(keys)) dt <- dt[, .SD[.N], by = keys]
  drop <- intersect(GAME_ROSTER_GAME_COLS, names(dt))
  if (length(drop)) dt[, (drop) := NULL]
  as.data.frame(dt, check.names = FALSE)
}

build_rosters_season <- function(season, live = TRUE, publish = TRUE) {
  gr_path <- file.path("cfb", "game_rosters", "parquet", sprintf("game_rosters_%d.parquet", season))
  gr <- if (file.exists(gr_path)) {
    as.data.frame(arrow::read_parquet(gr_path), check.names = FALSE)
  } else {
    cli::cli_alert_warning("rosters {season}: no game_rosters parquet at {gr_path} (run 07 first)")
    data.frame()
  }
  df <- derive_rosters(gr)
  cli::cli_alert_info("rosters {season}: {nrow(df)} athlete-team rows (from {nrow(gr)} game-roster rows)")
  write_dataset(df, "rosters", season, "rosters")
  if (live && publish && nrow(df) > 0) publish_dataset("rosters", season, "rosters", "espn_cfb_rosters")
  invisible(df)
}

main <- function() {
  opt <- optparse::parse_args(optparse::OptionParser(option_list = list(
    optparse::make_option(c("-s", "--start_year"), type = "integer"),
    optparse::make_option(c("-e", "--end_year"), type = "integer"))))
  for (y in opt$start_year:opt$end_year) build_rosters_season(y, live = TRUE)
}
if (sys.nframe() == 0L && length(commandArgs(trailingOnly = TRUE)) > 0L) main()
