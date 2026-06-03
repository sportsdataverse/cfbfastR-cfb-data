suppressPackageStartupMessages({
  library(data.table); library(arrow); library(jsonlite); library(optparse); library(cli)
})
if (!exists("read_final_json")) source("R/_data_utils.R")

# Long-form per-(team, period) scoring from g$team_box_extra[[team_id]]$linescores.
# Present only when team_box_extra exists (recent seasons); empty otherwise.
reshape_linescores <- function(g) {
  tbe <- g$team_box_extra
  if (is.null(tbe) || length(tbe) == 0) return(data.frame())
  rows <- list()
  for (tid in names(tbe)) {
    ls <- tbe[[tid]]$linescores
    if (is.null(ls) || length(ls) == 0) next
    for (i in seq_along(ls)) {
      rows[[length(rows) + 1L]] <- data.frame(
        team_id = as.integer(tid),
        period  = as.integer(i),
        value   = as.character(ls[[i]]$displayValue %||% ls[[i]]$value %||% NA),
        stringsAsFactors = FALSE)
    }
  }
  if (length(rows) == 0) return(data.frame())
  df <- do.call(rbind, rows)
  df$game_id <- as.integer(g$id); df$season <- as.integer(g$season)
  df
}

build_linescores_season <- function(season, master = fetch_master_local(), live = TRUE, publish = TRUE) {
  build_season(season, "linescores", "linescores", "espn_cfb_linescores", reshape_linescores,
               master = master, live = live, publish = publish)
}

main <- function() {
  opt <- optparse::parse_args(optparse::OptionParser(option_list = list(
    optparse::make_option(c("-s", "--start_year"), type = "integer"),
    optparse::make_option(c("-e", "--end_year"), type = "integer"))))
  master <- fetch_master_local()
  for (y in opt$start_year:opt$end_year) build_linescores_season(y, master = master, live = TRUE)
}
if (sys.nframe() == 0L && length(commandArgs(trailingOnly = TRUE)) > 0L) main()
