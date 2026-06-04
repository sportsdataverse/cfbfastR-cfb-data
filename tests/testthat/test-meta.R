source(file.path("..", "..", "R", "_data_utils.R"))
for (f in c("09_betting", "10_schedules", "11_linescores", "12_power_index")) {
  source(file.path("..", "..", "R", sprintf("espn_cfb_%s_creation.R", f)))
}
g <- read_final_json(testthat::test_path("fixtures", "final_401628455.json"))

test_that("reshape_betting -> one self-describing game row", {
  df <- reshape_betting(g)
  expect_equal(nrow(df), 1L)
  expect_true(all(c("game_id", "season", "game_spread", "over_under", "odds_source") %in% names(df)))
  expect_equal(df$game_id, as.integer(g$id))
})

test_that("reshape_schedule_row -> one game-meta row with team ids", {
  df <- reshape_schedule_row(g)
  expect_equal(nrow(df), 1L)
  expect_true(all(c("game_id", "season", "week", "home_id", "away_id") %in% names(df)))
  expect_false(is.na(df$home_id))
  expect_false(is.na(df$away_id))
})

test_that("reshape_linescores -> long per-(team, period) rows when present", {
  df <- reshape_linescores(g)
  expect_s3_class(df, "data.frame")
  if (nrow(df) > 0) {
    expect_true(all(c("team_id", "period", "value", "game_id", "season") %in% names(df)))
    expect_true(all(df$game_id == as.integer(g$id)))
  }
})

test_that("reshape_power_index degrades to a data.frame", {
  df <- reshape_power_index(g)
  expect_s3_class(df, "data.frame")
})
