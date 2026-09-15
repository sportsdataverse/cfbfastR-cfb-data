# Extract the training frames embedded in the two fitted glm objects so the
# Run: Rscript dump_training_frames.R <snapshot_dir> <repo>/python/.cache/matchup
# Python trainers can be gated against R's coefficients on the SAME rows.
suppressPackageStartupMessages(library(dplyr))
args <- commandArgs(trailingOnly = TRUE); snap <- args[1]; out <- args[2]  # <snapshot_dir> <python/.cache/matchup>
has_arrow <- requireNamespace("arrow", quietly = TRUE)
write_frame <- function(df, stem) {
  if (has_arrow) { arrow::write_parquet(df, file.path(out, paste0(stem, ".parquet"))) }
  else { readr::write_csv(df, file.path(out, paste0(stem, ".csv.gz")), na = "") }
  cat(stem, ":", nrow(df), "x", ncol(df), if (has_arrow) "parquet" else "csv.gz", "\n")
}
m <- readRDS(file.path(snap, "models/scoring_opp_mod.RDS"))
d <- m$data; cat("scoring_opp training frame cols:", paste(names(d), collapse = ","), "\n")
write_frame(d, "scoring_opp_training_frame")
rm(m, d); invisible(gc())
m <- readRDS(file.path(snap, "models/rp_mod_3.rds"))
d <- m$data; cat("rush_expect training frame cols:", paste(names(d), collapse = ","), " nobs:", nobs(m), "\n")
keep <- intersect(names(d), c("game_id","id_play","year","season","week","pos_team_score","def_pos_team_score","score_diff","half","period","TimeSecsRem","down","distance","yards_to_goal","ep_before","wp_before","rz_play","rush"))
write_frame(d[, keep], "rush_expect_training_frame")
cat("DUMP-COMPLETE\n")
