#!/usr/bin/env Rscript
# Publish the Python-built model-PBP parquet as the espn_cfb_model_pbp dataset.
# Converts to parquet/rds/gzipped-csv parity via the shared write_dataset() writer.
# Does NOT re-run any Python model code; reads the pre-built parquet from --parquet.
suppressPackageStartupMessages({
  library(arrow); library(optparse); library(cli)
})
# NB: guard on publish_dataset (unique to _data_utils.R), NOT write_dataset —
# library(arrow) above exports its own write_dataset(), so exists("write_dataset")
# is always TRUE and would skip sourcing, leaving arrow's fn to shadow ours.
if (!exists("publish_dataset")) source("R/_data_utils.R")

opt <- optparse::parse_args(optparse::OptionParser(option_list = list(
  optparse::make_option(c("-p", "--parquet"), type = "character",
                        help = "Path to the Python-built model-PBP parquet file"),
  optparse::make_option(c("-s", "--season"),  type = "integer",
                        help = "CFB season year (e.g. 2024)"))))

if (is.null(opt$parquet) || is.null(opt$season)) {
  cli::cli_abort("Both --parquet <path> and --season <year> are required.")
}

cli::cli_alert_info("Reading model-PBP parquet: {opt$parquet}")
df <- as.data.frame(arrow::read_parquet(opt$parquet))
cli::cli_alert_info("model_pbp {opt$season}: {nrow(df)} rows, {ncol(df)} cols")

write_dataset(df, "model_pbp", opt$season, "model_pbp")

if (identical(Sys.getenv("CFB_PUBLISH"), "1")) {
  publish_dataset("model_pbp", opt$season, "model_pbp", "espn_cfb_model_pbp")
  cli::cli_alert_success("Published model_pbp {opt$season} -> espn_cfb_model_pbp")
} else {
  cli::cli_alert_info("Wrote model_pbp {opt$season} locally (set CFB_PUBLISH=1 to upload)")
}
