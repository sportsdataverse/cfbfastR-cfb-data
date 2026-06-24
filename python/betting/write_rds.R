#!/usr/bin/env Rscript
# Mirror the rebuilt cfb_line_odds parquet to the legacy .rds the cfbfastR-data
# betting release ships (rds/cfb_lines_odds.rds). Run after betting.build_line_odds.
#
#   Rscript betting/write_rds.R            # default base = ../../cfbfastR-data/betting
#   Rscript betting/write_rds.R <betting_base_dir>
suppressMessages(library(arrow))

args <- commandArgs(trailingOnly = TRUE)
base <- if (length(args) >= 1) args[[1]] else "../../cfbfastR-data/betting"

pq  <- file.path(base, "parquet", "cfb_line_odds.parquet")
out <- file.path(base, "rds", "cfb_lines_odds.rds")

df <- as.data.frame(arrow::read_parquet(pq))
dir.create(dirname(out), showWarnings = FALSE, recursive = TRUE)
saveRDS(df, out)  # default gzip compression, matching the prior release
cat(sprintf("wrote %s: %d rows x %d cols (seasons %d-%d)\n",
            out, nrow(df), ncol(df), min(df$season, na.rm = TRUE), max(df$season, na.rm = TRUE)))
