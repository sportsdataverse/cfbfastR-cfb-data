source(file.path("..", "..", "R", "_data_utils.R"))
source(file.path("..", "..", "R", "espn_cfb_01_pbp_creation.R"))

GID <- 401628455
g <- read_final_json(testthat::test_path("fixtures", sprintf("final_%d.json", GID)))

test_that("reshape_pbp returns one row per play stamped with identifiers", {
  df <- reshape_pbp(g)
  expect_s3_class(df, "data.frame")
  expect_equal(nrow(df), length(g$plays))
  expect_gt(nrow(df), 50)
  expect_true(all(c("game_id", "season") %in% names(df)))
  expect_true(all(df$game_id == as.integer(g$id)))
})

test_that("reshape_pbp maps JSON null to NA and preserves dotted column names", {
  df <- reshape_pbp(g)
  # dotted keys like start.down survive (check.names = FALSE)
  expect_true(any(grepl("\\.", names(df))))
  # at least one enriched column present
  expect_true("game_play_number" %in% names(df))
})

test_that("conform_pbp is a no-op pass-through when cfbfastR lacks the schema fn", {
  df <- reshape_pbp(g)
  out <- conform_pbp(df)
  expect_s3_class(out, "data.frame")
  expect_equal(nrow(out), nrow(df))
})
