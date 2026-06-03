suppressPackageStartupMessages({
  library(data.table); library(arrow); library(jsonlite); library(optparse); library(cli)
})
if (!exists("read_final_json")) source("R/_data_utils.R")

# g$power_index is the ESPN core-API FPI wrapper {count, pageIndex, ..., items:[...]}.
# Flatten items to one row per entry; empty (pre-~2015 / no FPI) -> 0-row data.frame.
reshape_power_index <- function(g) {
  pidx <- g$power_index
  if (is.null(pidx) || length(pidx) == 0) return(data.frame())
  items <- if (!is.null(pidx$items)) pidx$items else pidx
  if (is.null(items) || length(items) == 0) return(data.frame())
  flat_block_df(items, g)
}

build_power_index_season <- function(season, master = fetch_master_local(), live = TRUE, publish = TRUE) {
  build_season(season, "power_index", "power_index", "espn_cfb_power_index", reshape_power_index,
               master = master, live = live, publish = publish)
}

main <- function() {
  opt <- optparse::parse_args(optparse::OptionParser(option_list = list(
    optparse::make_option(c("-s", "--start_year"), type = "integer"),
    optparse::make_option(c("-e", "--end_year"), type = "integer"))))
  master <- fetch_master_local()
  for (y in opt$start_year:opt$end_year) build_power_index_season(y, master = master, live = TRUE)
}
if (sys.nframe() == 0L && length(commandArgs(trailingOnly = TRUE)) > 0L) main()
