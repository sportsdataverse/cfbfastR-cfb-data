suppressPackageStartupMessages({
  library(dplyr); library(data.table); library(arrow); library(jsonlite); library(optparse); library(cli)
})
if (!exists("read_final_json")) source("R/_data_utils.R")

# ESPN player box: g$boxscore$players[] — per team a list of stat-category groups, each with
# $name (passing/rushing/...), $keys (column labels), $athletes[] ($athlete + $stats aligned to keys).
# One row per athlete per category; category $keys become columns (filled NA across categories).
reshape_player_box <- function(g) {
  groups <- g$boxscore$players
  if (is.null(groups) || length(groups) == 0) return(data.frame())
  rows <- list()
  for (grp in groups) {
    tid <- as.integer(grp$team$id %||% NA)
    for (cat in grp$statistics %||% list()) {
      keys <- unlist(cat$keys)
      cname <- cat$name %||% NA_character_
      for (a in cat$athletes %||% list()) {
        stats <- unlist(a$stats)
        vals <- as.list(as.character(stats))
        if (length(keys) == length(vals) && length(keys) > 0) {
          names(vals) <- keys
        } else if (length(vals) > 0) {
          names(vals) <- paste0("stat_", seq_along(vals))
        }
        df <- if (length(vals)) as.data.frame(vals, stringsAsFactors = FALSE, check.names = FALSE) else data.frame()
        df$category <- cname
        df$athlete_id <- as.integer(a$athlete$id %||% NA)
        df$athlete_name <- a$athlete$displayName %||% NA_character_
        df$jersey <- a$athlete$jersey %||% NA_character_
        df$team_id <- tid
        rows[[length(rows) + 1L]] <- df
      }
    }
  }
  if (length(rows) == 0) return(data.frame())
  out <- as.data.frame(data.table::rbindlist(rows, fill = TRUE, use.names = TRUE), check.names = FALSE)
  out$game_id <- as.integer(g$id); out$season <- as.integer(g$season)
  out
}

build_player_box_season <- function(season, master = fetch_master_local(), live = TRUE, publish = TRUE) {
  build_season(season, "player_box", "player_box", "espn_cfb_player_box", reshape_player_box,
               master = master, live = live, publish = publish)
}

main <- function() {
  opt <- optparse::parse_args(optparse::OptionParser(option_list = list(
    optparse::make_option(c("-s", "--start_year"), type = "integer"),
    optparse::make_option(c("-e", "--end_year"), type = "integer"))))
  master <- fetch_master_local()
  for (y in opt$start_year:opt$end_year) build_player_box_season(y, master = master, live = TRUE)
}
if (sys.nframe() == 0L && length(commandArgs(trailingOnly = TRUE)) > 0L) main()
