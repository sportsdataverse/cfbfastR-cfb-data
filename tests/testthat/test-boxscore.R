source(file.path("..", "..", "R", "_data_utils.R"))
source(file.path("..", "..", "R", "espn_cfb_02_team_box_creation.R"))
source(file.path("..", "..", "R", "espn_cfb_03_player_box_creation.R"))

g <- read_final_json(testthat::test_path("fixtures", "final_401628455.json"))

test_that("reshape_team_box returns one row per team, stamped + pivoted stats", {
  df <- reshape_team_box(g)
  expect_s3_class(df, "data.frame")
  expect_equal(nrow(df), length(g$boxscore$teams))
  expect_true(all(c("team_id", "home_away", "game_id", "season") %in% names(df)))
  expect_true("totalYards" %in% names(df))
  expect_true(all(df$game_id == as.integer(g$id)))
})

test_that("reshape_player_box returns athlete rows with category + stamped ids", {
  df <- reshape_player_box(g)
  expect_s3_class(df, "data.frame")
  expect_gt(nrow(df), 0)
  expect_true(all(c("category", "athlete_id", "team_id", "game_id", "season") %in% names(df)))
  expect_true(all(df$game_id == as.integer(g$id)))
  expect_true("passing" %in% df$category)
})
