suppressPackageStartupMessages({
  library(data.table); library(arrow); library(jsonlite); library(optparse); library(cli)
})
if (!exists("read_final_json")) source("R/_data_utils.R")

# g$injuries: list of per-team/per-athlete injury dicts (from the summary allowlist).
# Frequently EMPTY for CFB -> returns a 0-row data.frame; the expected columns
# (team_id, athlete_id, status, type/detail) appear when ESPN populates the block.
reshape_injuries <- function(g) flat_block_df(g$injuries, g)

build_injuries_season <- function(season, master = fetch_master_local(), live = TRUE, publish = TRUE) {
  build_season(season, "injuries", "injuries", "espn_cfb_injuries", reshape_injuries,
               master = master, live = live, publish = publish)
}

main <- function() {
  opt <- optparse::parse_args(optparse::OptionParser(option_list = list(
    optparse::make_option(c("-s", "--start_year"), type = "integer"),
    optparse::make_option(c("-e", "--end_year"), type = "integer"))))
  master <- fetch_master_local()
  for (y in opt$start_year:opt$end_year) build_injuries_season(y, master = master, live = TRUE)
}
if (sys.nframe() == 0L && length(commandArgs(trailingOnly = TRUE)) > 0L) main()
