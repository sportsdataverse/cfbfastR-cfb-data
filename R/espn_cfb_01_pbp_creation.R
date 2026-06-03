suppressPackageStartupMessages({
  library(dplyr); library(data.table); library(arrow)
  library(jsonlite); library(optparse); library(cli)
})
if (!exists("read_final_json")) source("R/_data_utils.R")

# Convert one enriched play dict to a 1-row data.frame: scalars kept, JSON null -> NA,
# multi-value (lists like teamParticipants/sack_players) boxed into a list-column.
.play_to_row <- function(p) {
  cells <- lapply(p, function(v) {
    if (is.null(v) || length(v) == 0) return(NA)
    if (length(v) == 1) return(v[[1]])
    I(list(v))
  })
  as.data.frame(cells, stringsAsFactors = FALSE, check.names = FALSE)
}

# plays (list of ~380-key enriched dicts) -> one row per play, stamped with identity.
reshape_pbp <- function(g) {
  plays <- g$plays
  if (is.null(plays) || length(plays) == 0) return(data.frame())
  df <- as.data.frame(
    data.table::rbindlist(lapply(plays, .play_to_row), fill = TRUE, use.names = TRUE),
    check.names = FALSE)
  df$game_id <- as.integer(g$id)
  df$season  <- as.integer(g$season)
  if (!is.null(g$week)) df$week <- as.integer(g$week)
  df
}

# Apply cfbfastR's canonical column ordering/tiering when the installed cfbfastR exposes
# it (the refactor/pbp-epa-wpa-modular branch's .pbp_apply_output_schema). Until that ships,
# this is a graceful pass-through (the full enriched frame is released as-is).
conform_pbp <- function(df, output = "default") {
  if (nrow(df) == 0) return(df)
  fn <- tryCatch(getFromNamespace(".pbp_apply_output_schema", "cfbfastR"),
                 error = function(e) NULL)
  if (!is.null(fn)) {
    return(tryCatch(fn(df, output = output), error = function(e) {
      cli::cli_alert_warning("conform_pbp: schema fn errored ({conditionMessage(e)}); passing through"); df
    }))
  }
  df
}

build_pbp_season <- function(season, master = fetch_master_local(), live = TRUE, publish = TRUE) {
  build_season(season, dataset = "pbp", stem = "play_by_play", tag = "espn_cfb_pbp",
               reshape_fn = function(g) conform_pbp(reshape_pbp(g)),
               master = master, live = live, publish = publish)
}

main <- function() {
  opt <- optparse::parse_args(optparse::OptionParser(option_list = list(
    optparse::make_option(c("-s", "--start_year"), type = "integer"),
    optparse::make_option(c("-e", "--end_year"), type = "integer"))))
  master <- fetch_master_local()
  for (y in opt$start_year:opt$end_year) build_pbp_season(y, master = master, live = TRUE)
}

if (sys.nframe() == 0L && length(commandArgs(trailingOnly = TRUE)) > 0L) main()
