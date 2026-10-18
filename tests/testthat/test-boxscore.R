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

test_that("reshape_player_box names passing stats when ESPN omits QBR (game-on-paper-app#328)", {
  # six passing keys (ending adjQBR) but five stats: the trailing key is the one dropped
  keys <- list("completions/passingAttempts", "passingYards", "yardsPerPassAttempt",
               "passingTouchdowns", "interceptions", "adjQBR")
  g2 <- list(id = 401858206, season = 2026, boxscore = list(players = list(list(
    team = list(id = "2390"),
    statistics = list(list(name = "passing", keys = keys, athletes = list(list(
      athlete = list(id = "5121169", displayName = "Darian Mensah"),
      stats = list("27/30", "401", "13.4", "5", "0")
    )))))
  )))
  df <- reshape_player_box(g2)
  expect_equal(df[["completions/passingAttempts"]], "27/30")
  expect_equal(df[["interceptions"]], "0")
  expect_false(any(startsWith(names(df), "stat_")))
})
