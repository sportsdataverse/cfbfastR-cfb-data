# Shared data readers for the presentation scripts (playoff / postweek /
# previews / trends). They consume the season tables the team-summaries creator
# (R/espn_cfb_15_team_summaries_creation.R) produces as gzipped CSV.
#
# The repo's cfb/ tree is gitignored, so the canonical copy of each dataset lives
# in the sportsdataverse-data GitHub release (tags espn_cfb_team_summaries /
# espn_cfb_passing / espn_cfb_rushing / espn_cfb_receiving / espn_cfb_percentiles).
# Each reader is local-first (uses a freshly-created cfb/ file when present, e.g.
# in dev or same-run) and falls back to downloading the release asset (the CI
# path, where a fresh checkout has no cfb/ data).
suppressPackageStartupMessages({
    library(readr)
})

SDV_DATA_RELEASE_BASE <- "https://github.com/sportsdataverse/sportsdataverse-data/releases/download"

summary_local_path <- function(dataset, stem, season) {
    file.path("cfb", dataset, "csv", sprintf("%s_%d.csv.gz", stem, as.integer(season)))
}

read_summary_dataset <- function(dataset, stem, tag, season) {
    season <- as.integer(season)
    local <- summary_local_path(dataset, stem, season)
    if (file.exists(local)) {
        return(readr::read_csv(local, show_col_types = FALSE))
    }
    url <- sprintf("%s/%s/%s_%d.csv.gz", SDV_DATA_RELEASE_BASE, tag, stem, season)
    tmp <- tempfile(fileext = ".csv.gz")
    utils::download.file(url, tmp, mode = "wb", quiet = TRUE)
    readr::read_csv(tmp, show_col_types = FALSE)
}

read_team_summaries <- function(season) {
    read_summary_dataset("team_summaries", "cfb_team_summaries", "espn_cfb_team_summaries", season)
}

read_percentiles <- function(season) {
    read_summary_dataset("percentiles", "cfb_percentiles", "espn_cfb_percentiles", season)
}

read_passing <- function(season) {
    read_summary_dataset("passing", "cfb_passing", "espn_cfb_passing", season)
}

read_rushing <- function(season) {
    read_summary_dataset("rushing", "cfb_rushing", "espn_cfb_rushing", season)
}

read_receiving <- function(season) {
    read_summary_dataset("receiving", "cfb_receiving", "espn_cfb_receiving", season)
}
