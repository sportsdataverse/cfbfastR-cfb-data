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

test_that("bind_games drops NULL/empty frames and unions columns", {
  out <- bind_games(list(
    data.frame(a = 1, b = 2),
    NULL,
    data.frame(a = 3, c = 4)
  ))
  expect_equal(nrow(out), 2L)
  expect_true(all(c("a", "b", "c") %in% names(out)))
})
