# Shared helpers for cfbfastR-cfb-data creation scripts.
# Pure-ish reshape + IO helpers; network isolated to fetch_* so reshape fns stay testable.

RAW_BASE <- "https://raw.githubusercontent.com/sportsdataverse/cfbfastR-cfb-raw/main/cfb"
PUBLISH_REPOS <- c("sportsdataverse/sportsdataverse-data")

read_final_json <- function(path_or_url) {
  jsonlite::fromJSON(path_or_url, simplifyVector = FALSE)
}

# Read the -raw schedule master parquet and return game_ids for a season.
season_game_ids_from_master <- function(master_path_or_url, season) {
  df <- arrow::read_parquet(master_path_or_url)
  ids <- df$game_id[df$season == season]
  unique(as.integer(ids[!is.na(ids)]))
}

# Download the -raw schedule master parquet to a temp file; return its path.
fetch_master_local <- function() {
  dest <- tempfile(fileext = ".parquet")
  curl::curl_download(paste0(RAW_BASE, "/cfb_schedule_master.parquet"), dest, quiet = TRUE)
  dest
}

final_url <- function(game_id) sprintf("%s/json/final/%s.json", RAW_BASE, game_id)

# Fetch + parse one final JSON with retry/backoff; NULL on persistent failure (logged).
fetch_final <- function(game_id, tries = 3L) {
  for (i in seq_len(tries)) {
    out <- tryCatch(read_final_json(final_url(game_id)), error = function(e) e)
    if (!inherits(out, "error")) return(out)
    Sys.sleep(min(2^(i - 1), 5))
  }
  cli::cli_alert_warning("fetch_final failed for {game_id}")
  NULL
}

# One flat dict -> 1-row data.frame: scalars kept, JSON null -> NA, multi-value -> list-col.
dict_to_row <- function(p) {
  cells <- lapply(p, function(v) {
    if (is.null(v) || length(v) == 0) return(NA)
    if (length(v) == 1) return(v[[1]])
    I(list(v))
  })
  as.data.frame(cells, stringsAsFactors = FALSE, check.names = FALSE)
}

# A list-of-flat-dicts block (play_participants, rosters, injuries, ...) -> stamped data.frame.
flat_block_df <- function(block, g) {
  if (is.null(block) || length(block) == 0) return(data.frame())
  df <- as.data.frame(
    data.table::rbindlist(lapply(block, dict_to_row), fill = TRUE, use.names = TRUE),
    check.names = FALSE)
  df$game_id <- as.integer(g$id); df$season <- as.integer(g$season)
  if (!is.null(g$week)) df$week <- as.integer(g$week)
  df
}

# Bind a list of per-game data.frames (drift-safe).
bind_games <- function(frames) {
  frames <- Filter(function(x) !is.null(x) && nrow(x) > 0, frames)
  if (length(frames) == 0) return(data.frame())
  as.data.frame(data.table::rbindlist(frames, fill = TRUE, use.names = TRUE))
}

# JSON-encode any list-columns (nested participant/name lists) to character so arrow/csv
# can serialize them. Keeps all data, parquet/csv/rds share one schema.
stringify_list_cols <- function(df) {
  for (nm in names(df)) {
    col <- df[[nm]]
    if (is.list(col) && !is.data.frame(col)) {
      df[[nm]] <- vapply(col, function(x) {
        if (is.null(x) || length(x) == 0 || (length(x) == 1 && is.na(x[[1]]))) return(NA_character_)
        tryCatch(as.character(jsonlite::toJSON(x, auto_unbox = TRUE, null = "null", na = "null")),
                 error = function(e) paste(unlist(x), collapse = "|"))
      }, character(1))
    }
  }
  df
}

# Write parquet + rds + gzipped csv under cfb/{dataset}/ and append a manifest row.
write_dataset <- function(df, dataset, season, stem) {
  if (is.null(df) || nrow(df) == 0) {
    cli::cli_alert_info("{dataset} {season}: 0 rows, skipping write")
    return(invisible(NULL))
  }
  df <- stringify_list_cols(df)
  base <- file.path("cfb", dataset)
  for (sub in c("parquet", "rds", "csv")) dir.create(file.path(base, sub), recursive = TRUE, showWarnings = FALSE)
  arrow::write_parquet(df, file.path(base, "parquet", sprintf("%s_%d.parquet", stem, season)))
  saveRDS(df, file.path(base, "rds", sprintf("%s_%d.rds", stem, season)))
  readr::write_csv(df, file.path(base, "csv", sprintf("%s_%d.csv.gz", stem, season)))
  .append_manifest(dataset, season, nrow(df))
  invisible(df)
}

.append_manifest <- function(dataset, season, row_count) {
  f <- file.path("cfb", dataset, sprintf("cfb_%s_in_data_repo.csv", dataset))
  row <- data.frame(season = as.integer(season), row_count = as.integer(row_count),
                    generated_at_utc = format(Sys.time(), tz = "UTC", usetz = TRUE),
                    stringsAsFactors = FALSE)
  if (file.exists(f)) {
    old <- readr::read_csv(f, show_col_types = FALSE)
    row <- dplyr::bind_rows(old[old$season != season, , drop = FALSE], row)
  }
  row <- row[order(row$season), , drop = FALSE]
  readr::write_csv(row, f)
}

# Upload one file to BOTH publish repos under a release tag (idempotent overwrite).
# piggyback memoises pb_releases() (a cachem cache that can persist a stale "no releases"
# result, even one created earlier in the same workflow run by releases_init) -> bust it
# first so a just-created release is found.
pb_upload_both <- function(file, tag, repos = PUBLISH_REPOS, token = Sys.getenv("GITHUB_PAT")) {
  try(memoise::forget(piggyback::pb_releases), silent = TRUE)
  for (repo in repos) {
    tryCatch(
      piggyback::pb_upload(file = file, repo = repo, tag = tag, overwrite = TRUE, .token = token),
      error = function(e) cli::cli_alert_danger("pb_upload {repo}@{tag} {basename(file)}: {conditionMessage(e)}")
    )
  }
}

# Publish all three formats for a dataset+season to both repos.
publish_dataset <- function(dataset, season, stem, tag) {
  base <- file.path("cfb", dataset)
  specs <- list(
    list(sub = "parquet", fn = sprintf("%s_%d.parquet", stem, season)),
    list(sub = "rds",     fn = sprintf("%s_%d.rds", stem, season)),
    list(sub = "csv",     fn = sprintf("%s_%d.csv.gz", stem, season))
  )
  for (s in specs) {
    f <- file.path(base, s$sub, s$fn)
    if (file.exists(f)) pb_upload_both(f, tag)
  }
}

# Generic per-season driver: enumerate -> fetch -> reshape(each) -> bind -> write -> publish.
build_season <- function(season, dataset, stem, tag, reshape_fn,
                         master = fetch_master_local(), live = TRUE, publish = TRUE) {
  ids <- season_game_ids_from_master(master, season)
  cli::cli_alert_info("{dataset} {season}: {length(ids)} games")
  frames <- lapply(ids, function(gid) {
    g <- if (live) fetch_final(gid) else NULL
    if (is.null(g)) return(NULL)
    tryCatch(reshape_fn(g), error = function(e) {
      cli::cli_alert_warning("{dataset} {gid}: {conditionMessage(e)}"); NULL
    })
  })
  df <- bind_games(frames)
  write_dataset(df, dataset, season, stem)
  if (live && publish && nrow(df) > 0) publish_dataset(dataset, season, stem, tag)
  invisible(df)
}
