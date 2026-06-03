# Print a per-dataset season x row_count summary to stdout and append a markdown table to
# $GITHUB_STEP_SUMMARY (when set). Reads the cfb/*/cfb_*_in_data_repo.csv manifests.
suppressPackageStartupMessages({ library(readr); library(optparse); library(cli) })

opt <- optparse::parse_args(optparse::OptionParser(option_list = list(
  optparse::make_option(c("-s", "--start_year"), type = "integer", default = NA),
  optparse::make_option(c("-e", "--end_year"), type = "integer", default = NA))))

manifests <- list.files("cfb", pattern = "^cfb_.*_in_data_repo\\.csv$",
                        recursive = TRUE, full.names = TRUE)

rows <- list()
for (m in manifests) {
  df <- tryCatch(readr::read_csv(m, show_col_types = FALSE), error = function(e) NULL)
  if (is.null(df) || nrow(df) == 0) next
  ds <- sub("^cfb_(.*)_in_data_repo\\.csv$", "\\1", basename(m))
  for (i in seq_len(nrow(df))) {
    rows[[length(rows) + 1L]] <- data.frame(dataset = ds, season = df$season[i],
                                            row_count = df$row_count[i], stringsAsFactors = FALSE)
  }
}

if (length(rows) == 0) {
  cli::cli_alert_info("run_summary: no manifests found yet")
  quit(save = "no", status = 0)
}
summary_df <- do.call(rbind, rows)
summary_df <- summary_df[order(summary_df$dataset, summary_df$season), ]

cli::cli_h2("CFB data summary")
print(summary_df, row.names = FALSE)

step <- Sys.getenv("GITHUB_STEP_SUMMARY")
if (nzchar(step)) {
  con <- file(step, open = "a")
  writeLines("## CFB data summary", con)
  writeLines("", con)
  writeLines("| dataset | season | row_count |", con)
  writeLines("| --- | --- | --- |", con)
  for (i in seq_len(nrow(summary_df))) {
    writeLines(sprintf("| %s | %s | %s |", summary_df$dataset[i],
                       summary_df$season[i], summary_df$row_count[i]), con)
  }
  close(con)
}
