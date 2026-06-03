source(file.path("..", "..", "R", "_data_utils.R"))
source(file.path("..", "..", "R", "espn_cfb_04_adv_box_creation.R"))
g <- read_final_json(testthat::test_path("fixtures", "final_401628455.json"))

test_that("reshape_adv_box returns the 8 core sections as stamped data.frames", {
  out <- reshape_adv_box(g)
  expect_true(is.list(out))
  core <- c("adv_team", "adv_passing", "adv_rushing", "adv_receiving",
            "adv_defensive", "adv_turnover", "adv_drives", "adv_situational")
  for (nm in core) {
    expect_true(nm %in% names(out), info = nm)
    expect_s3_class(out[[nm]], "data.frame")
    if (nrow(out[[nm]]) > 0) {
      expect_true(all(c("game_id", "season") %in% names(out[[nm]])), info = nm)
      expect_true(all(out[[nm]]$game_id == as.integer(g$id)), info = nm)
    }
  }
})

test_that("adv_team and adv_passing have rows for a real game", {
  out <- reshape_adv_box(g)
  expect_gt(nrow(out$adv_team), 0)
  expect_gt(nrow(out$adv_passing), 0)
})
