source(file.path("..", "..", "R", "_data_utils.R"))
for (f in c("05_play_participants", "06_drives", "07_game_rosters", "08_rosters", "14_injuries")) {
  source(file.path("..", "..", "R", sprintf("espn_cfb_%s_creation.R", f)))
}
g <- read_final_json(testthat::test_path("fixtures", "final_401628455.json"))

test_that("reshape_play_participants -> one row per play, stamped", {
  df <- reshape_play_participants(g)
  expect_s3_class(df, "data.frame")
  expect_equal(nrow(df), length(g$play_participants))
  expect_true(all(c("play_id", "game_id", "season") %in% names(df)))
  expect_true(all(df$game_id == as.integer(g$id)))
})

test_that("reshape_drives -> one row per drive with drive-level fields", {
  df <- reshape_drives(g)
  expect_s3_class(df, "data.frame")
  expect_gt(nrow(df), 0)
  expect_true(all(c("drive_id", "team_id", "result", "game_id", "season") %in% names(df)))
  expect_true(all(df$game_id == as.integer(g$id)))
})

test_that("reshape_game_rosters -> one row per athlete per game, stamped", {
  df <- reshape_game_rosters(g)
  expect_s3_class(df, "data.frame")
  expect_equal(nrow(df), length(g$game_rosters))
  expect_true(all(c("athlete_id", "team_id", "game_id", "season") %in% names(df)))
})

test_that("derive_rosters -> season roster: deduped, game-level fields dropped", {
  gr <- reshape_game_rosters(g)
  df <- derive_rosters(gr)
  expect_s3_class(df, "data.frame")
  # one game -> each athlete-team appears once, so the row count is preserved
  expect_equal(nrow(df), length(g$game_rosters))
  # season-identity fields kept; per-game circumstance fields dropped
  expect_true(all(c("athlete_id", "team_id", "season") %in% names(df)))
  expect_false(any(c("game_id", "week", "starter", "did_not_play") %in% names(df)))
  # de-dup key is unique
  key <- paste(df$season, df$team_id, df$athlete_id)
  expect_equal(anyDuplicated(key), 0L)
})

test_that("reshape_injuries degrades to a data.frame when empty", {
  df <- reshape_injuries(g)
  expect_s3_class(df, "data.frame")  # 0 rows for this fixture (CFB injuries usually empty)
})
