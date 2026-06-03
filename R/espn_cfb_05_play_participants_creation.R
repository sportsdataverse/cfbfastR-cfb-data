suppressPackageStartupMessages({
  library(data.table); library(arrow); library(jsonlite); library(optparse); library(cli)
})
if (!exists("read_final_json")) source("R/_data_utils.R")

# g$play_participants: list of flat per-play dicts (game_id, play_id, *_player_name/_id/_names/_ids).
reshape_play_participants <- function(g) flat_block_df(g$play_participants, g)

build_play_participants_season <- function(season, master = fetch_master_local(),
                                           live = TRUE, publish = TRUE) {
  build_season(season, "play_participants", "play_participants", "espn_cfb_play_participants",
               reshape_play_participants, master = master, live = live, publish = publish)
}

main <- function() {
  opt <- optparse::parse_args(optparse::OptionParser(option_list = list(
    optparse::make_option(c("-s", "--start_year"), type = "integer"),
    optparse::make_option(c("-e", "--end_year"), type = "integer"))))
  master <- fetch_master_local()
  for (y in opt$start_year:opt$end_year) build_play_participants_season(y, master = master, live = TRUE)
}
if (sys.nframe() == 0L && length(commandArgs(trailingOnly = TRUE)) > 0L) main()
