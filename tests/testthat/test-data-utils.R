source(file.path("..", "..", "R", "_data_utils.R"))

FIX <- testthat::test_path("fixtures", "final_401628455.json")

test_that("read_final_json parses a game and returns a named list", {
  g <- read_final_json(FIX)
  expect_true(is.list(g))
  expect_true(all(c("plays", "advBoxScore", "betting", "id", "season") %in% names(g)))
})

test_that("write_dataset writes parquet + rds + csv and a manifest row", {
  tmp <- withr::local_tempdir()
  withr::local_dir(tmp)
  df <- data.frame(game_id = c(1L, 2L), x = c(10, 20))
  write_dataset(df, dataset = "demo", season = 2024, stem = "demo")
  expect_true(file.exists("cfb/demo/parquet/demo_2024.parquet"))
  expect_true(file.exists("cfb/demo/rds/demo_2024.rds"))
  expect_true(file.exists("cfb/demo/csv/demo_2024.csv.gz"))
  m <- readr::read_csv("cfb/demo/cfb_demo_in_data_repo.csv", show_col_types = FALSE)
  expect_equal(m$season, 2024L)
  expect_equal(m$row_count, 2L)
})

test_that("season_game_ids_from_master filters the master to a season", {
  tmp <- withr::local_tempdir()
  master <- file.path(tmp, "m.parquet")
  arrow::write_parquet(
    data.frame(game_id = c(1L, 2L, 3L), season = c(2023L, 2024L, 2024L)), master)
  ids <- season_game_ids_from_master(master, 2024)
  expect_setequal(ids, c(2L, 3L))
})

test_that("write_dataset serializes list-columns (nested data) to parquet", {
  tmp <- withr::local_tempdir()
  withr::local_dir(tmp)
  df <- data.frame(game_id = 1L, x = 2)
  df$participants <- I(list(list(list(id = 1, role = "rusher"), list(id = 2, role = "tackler"))))
  write_dataset(df, dataset = "nested", season = 2024, stem = "nested")
  back <- arrow::read_parquet("cfb/nested/parquet/nested_2024.parquet")
  expect_equal(nrow(back), 1L)
  expect_type(back$participants, "character")     # list-col JSON-encoded
  expect_match(back$participants, "rusher")
})

test_that("bind_games drops NULL/empty frames and unions columns", {
  out <- bind_games(list(
    data.frame(a = 1, b = 2),
    NULL,
    data.frame(a = 3, c = 4)
  ))
  expect_equal(nrow(out), 2L)
  expect_true(all(c("a", "b", "c") %in% names(out)))
})

# --- pb_upload_both: a failed upload must NOT be silent -----------------------
# Regression for the 2026-07-16 team_summaries re-backfill, which exited 0 with two
# parquet silently un-uploaded (the release kept 5-week-old bytes). The upload error
# was caught and logged, so every caller -- including the daily publish cron -- read
# that as success. These are source()d globals rather than package functions, so the
# seams are stubbed by rebinding where pb_upload_both actually resolves them.

# Swap a binding in `env` for the duration of `code`, then restore it.
with_stub <- function(name, fn, env, code) {
  had <- exists(name, envir = env, inherits = FALSE)
  orig <- if (had) get(name, envir = env) else NULL
  locked <- environmentIsLocked(env) && had && bindingIsLocked(name, env)
  if (locked) unlockBinding(name, env)
  assign(name, fn, envir = env)
  on.exit({
    if (had) assign(name, orig, envir = env) else rm(list = name, envir = env)
    if (locked) lockBinding(name, env)
  }, add = TRUE)
  force(code)
}

test_that("pb_upload_both retries a transient upload failure, then succeeds", {
  attempts <- 0L
  res <- with_stub(".ensure_release_visible", function(...) TRUE, globalenv(),
    with_stub("pb_upload", function(...) {
      attempts <<- attempts + 1L
      # the exact transient seen in the re-backfill: a rate-limited HTML error page
      if (attempts < 2L) stop('Unexpected content type "text/html".')
      TRUE
    }, asNamespace("piggyback"),
      pb_upload_both("f.parquet", "some_tag", repos = "o/r", token = "t", wait = 0)
    )
  )

  expect_true(res)
  expect_identical(attempts, 2L)   # failed once, succeeded on the retry
})

test_that("pb_upload_both RAISES when the upload never succeeds (no silent success)", {
  expect_error(
    with_stub(".ensure_release_visible", function(...) TRUE, globalenv(),
      with_stub("pb_upload", function(...) stop("boom"), asNamespace("piggyback"),
        pb_upload_both("f.parquet", "some_tag", repos = "o/r", token = "t", tries = 2L, wait = 0)
      )
    ),
    "FAILED after 2 attempts"
  )
})

test_that("pb_upload_both RAISES when the release never becomes visible", {
  expect_error(
    with_stub(".ensure_release_visible", function(...) FALSE, globalenv(),
      pb_upload_both("f.parquet", "some_tag", repos = "o/r", token = "t", wait = 0)
    ),
    "never became visible"
  )
})
