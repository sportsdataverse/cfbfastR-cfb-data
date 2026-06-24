# Regression tests for the ID-column-type + name-matching conventions.
#
# Join keys (game_id / team_id / athlete_id) are only as correct as the *class*
# agreement on both sides. These pin the R manifestations of the bug class that
# historically surfaced only downstream (the cfb_line_odds game_id/team_id fills,
# crosswalk joins producing wrong/empty matches):
#   - leading-zero loss when an id that can start with 0 is coerced to numeric,
#   - dplyr 1.1+ erroring (not silently mis-joining) on incompatible join-key classes,
#   - case-sensitive player/team-name matching.
# Mirrors the sdv-py `tests/test_id_conventions.py` contract for the R stack.

test_that("leading-zero ids must stay character (numeric coercion is the trap)", {
  raw <- "007"                       # e.g. a zero-padded code
  expect_equal(as.character(as.integer(raw)), "7")   # the trap: leading zero lost
  expect_equal(raw, "007")                           # convention: keep ids as character
})

test_that("dplyr refuses to join incompatible-class keys (catch it, don't mis-join)", {
  skip_if_not_installed("dplyr")
  left  <- data.frame(game_id = c("1", "2", "3"), x = c(10, 20, 30), stringsAsFactors = FALSE)
  right <- data.frame(game_id = c(1L, 2L, 3L),     y = c(1, 2, 3))   # integer key

  # dplyr 1.1+ errors on character-vs-integer join keys rather than silently dropping rows.
  expect_error(dplyr::inner_join(left, right, by = "game_id"))

  # Pin one canonical class at the boundary, then the join is total.
  right_ok <- dplyr::mutate(right, game_id = as.character(game_id))
  joined <- dplyr::inner_join(left, right_ok, by = "game_id")
  expect_equal(nrow(joined), 3L)
})

test_that("player/team-name matching folds case", {
  skip_if_not_installed("stringr")
  names <- c("Travis Hunter", "travis hunter", "T. HUNTER")
  hits <- stringr::str_detect(names, stringr::regex("hunter", ignore_case = TRUE))
  expect_true(all(hits))
})
