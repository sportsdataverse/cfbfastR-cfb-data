suppressPackageStartupMessages({
  library(data.table); library(arrow); library(jsonlite); library(optparse); library(cli)
})
if (!exists("read_final_json")) source("R/_data_utils.R")

# g$game_rosters: list of flat per-athlete dicts (athlete_id, names, position, jersey, team_*, ...).
# One row per rostered athlete *per game* (the season-long game-roster compilation).
reshape_game_rosters <- function(g) flat_block_df(g$game_rosters, g)

build_game_rosters_season <- function(season, master = fetch_master_local(), live = TRUE, publish = TRUE) {
  build_season(season, "game_rosters", "game_rosters", "espn_cfb_game_rosters", reshape_game_rosters,
               master = master, live = live, publish = publish)
}

main <- function() {
  opt <- optparse::parse_args(optparse::OptionParser(option_list = list(
    optparse::make_option(c("-s", "--start_year"), type = "integer"),
    optparse::make_option(c("-e", "--end_year"), type = "integer"))))
  master <- fetch_master_local()
  for (y in opt$start_year:opt$end_year) build_game_rosters_season(y, master = master, live = TRUE)
}
if (sys.nframe() == 0L && length(commandArgs(trailingOnly = TRUE)) > 0L) main()
