suppressPackageStartupMessages({
  library(data.table); library(arrow); library(jsonlite); library(optparse); library(cli)
})
if (!exists("read_final_json")) source("R/_data_utils.R")

# advBoxScore section -> output-dataset name.
.ADV_MAP <- c(team = "adv_team", pass = "adv_passing", rush = "adv_rushing",
              receiver = "adv_receiving", defensive = "adv_defensive",
              turnover = "adv_turnover", drives = "adv_drives",
              situational = "adv_situational")
# Phase-H1 sdv-py expansion sections (tolerated when present).
.ADV_EXTRA <- c(defensive_players = "adv_defensive_players", specialists = "adv_specialists")

# g$advBoxScore -> named list of stamped data.frames, one per section.
reshape_adv_box <- function(g) {
  abx <- g$advBoxScore
  out <- list()
  for (sec in names(.ADV_MAP)) {
    out[[.ADV_MAP[[sec]]]] <- flat_block_df(if (is.null(abx)) NULL else abx[[sec]], g)
  }
  for (sec in names(.ADV_EXTRA)) {
    if (!is.null(abx) && !is.null(abx[[sec]])) {
      out[[.ADV_EXTRA[[sec]]]] <- flat_block_df(abx[[sec]], g)
    }
  }
  out
}

# Multi-frame season driver: each advBox section becomes its own dataset/tag.
build_adv_box_season <- function(season, master = fetch_master_local(), live = TRUE, publish = TRUE) {
  ids <- season_game_ids_from_master(master, season)
  cli::cli_alert_info("adv_box {season}: {length(ids)} games")
  acc <- list()
  for (gid in ids) {
    g <- if (live) fetch_final(gid) else NULL
    if (is.null(g)) next
    secs <- tryCatch(reshape_adv_box(g),
                     error = function(e) { cli::cli_alert_warning("adv {gid}: {conditionMessage(e)}"); NULL })
    if (is.null(secs)) next
    for (nm in names(secs)) acc[[nm]] <- c(acc[[nm]], list(secs[[nm]]))
  }
  for (nm in names(acc)) {
    df <- bind_games(acc[[nm]])
    write_dataset(df, nm, season, nm)
    if (live && publish && nrow(df) > 0) {
      publish_dataset(nm, season, nm, sprintf("espn_cfb_%s", nm))
    }
  }
  invisible(acc)
}

main <- function() {
  opt <- optparse::parse_args(optparse::OptionParser(option_list = list(
    optparse::make_option(c("-s", "--start_year"), type = "integer"),
    optparse::make_option(c("-e", "--end_year"), type = "integer"))))
  master <- fetch_master_local()
  for (y in opt$start_year:opt$end_year) build_adv_box_season(y, master = master, live = TRUE)
}
if (sys.nframe() == 0L && length(commandArgs(trailingOnly = TRUE)) > 0L) main()
