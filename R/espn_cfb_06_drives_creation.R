suppressPackageStartupMessages({
  library(data.table); library(arrow); library(jsonlite); library(optparse); library(cli)
})
if (!exists("read_final_json")) source("R/_data_utils.R")

# One ESPN drive object -> 1-row data.frame (drive-level fields; nested plays[] dropped).
.drive_to_row <- function(d) {
  row <- list(
    drive_id             = d$id,
    team_id              = if (!is.null(d$team$id)) as.integer(d$team$id) else NA_integer_,
    result               = d$result,
    display_result       = d$displayResult,
    short_display_result = d$shortDisplayResult,
    description          = d$description,
    yards                = d$yards,
    offensive_plays      = d$offensivePlays,
    is_score             = d$isScore,
    start_period         = d$start$period$number,
    start_yard_line      = d$start$yardLine,
    start_clock          = d$start$clock$displayValue,
    start_text           = d$start$text,
    end_period           = d$end$period$number,
    end_yard_line        = d$end$yardLine,
    end_clock            = d$end$clock$displayValue,
    time_elapsed         = d$timeElapsed$displayValue,
    n_plays              = length(d$plays)
  )
  as.data.frame(lapply(row, function(v) if (is.null(v) || length(v) == 0) NA else v[[1]]),
                stringsAsFactors = FALSE, check.names = FALSE)
}

# g$drives is {previous: [...], current: [...]} (ESPN); occasionally a bare list of drives,
# or empty {}. Unroll to one row per drive.
reshape_drives <- function(g) {
  dv <- g$drives
  if (is.null(dv) || length(dv) == 0) return(data.frame())
  if (!is.null(names(dv)) && any(c("previous", "current") %in% names(dv))) {
    all_drives <- c(dv$previous, dv$current)
  } else {
    all_drives <- dv  # bare list of drives keyed by team index, or flat list
    all_drives <- unlist(all_drives, recursive = FALSE)
  }
  all_drives <- Filter(function(d) is.list(d) && !is.null(d$id), all_drives)
  if (length(all_drives) == 0) return(data.frame())
  df <- as.data.frame(
    data.table::rbindlist(lapply(all_drives, .drive_to_row), fill = TRUE, use.names = TRUE),
    check.names = FALSE)
  df$game_id <- as.integer(g$id); df$season <- as.integer(g$season)
  df
}

build_drives_season <- function(season, master = fetch_master_local(), live = TRUE, publish = TRUE) {
  build_season(season, "drives", "drives", "espn_cfb_drives", reshape_drives,
               master = master, live = live, publish = publish)
}

main <- function() {
  opt <- optparse::parse_args(optparse::OptionParser(option_list = list(
    optparse::make_option(c("-s", "--start_year"), type = "integer"),
    optparse::make_option(c("-e", "--end_year"), type = "integer"))))
  master <- fetch_master_local()
  for (y in opt$start_year:opt$end_year) build_drives_season(y, master = master, live = TRUE)
}
if (sys.nframe() == 0L && length(commandArgs(trailingOnly = TRUE)) > 0L) main()
