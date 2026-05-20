#!/usr/bin/env Rscript
# Drive the v0.2 MSstats functions (designSampleSize, linear summarization,
# quantile normalization, MSstatsContrastMatrix) for R-parity testing.
#
# Usage:
#   Rscript r_reference_driver2.R <msstats_long_tsv> <out_dir>
#
# Outputs (in out_dir):
#   design_numsample.tsv  designSampleSize(..., numSample=TRUE) result
#   linear_run_level.tsv  dataProcess(summaryMethod='linear') ProteinLevelData
#   quantile_run_med.tsv  per-run median after quantile normalization
#   contrast_matrix.tsv   MSstatsContrastMatrix("pairwise", conditions)

suppressPackageStartupMessages({
  library(MSstats)
  library(data.table)
})

args <- commandArgs(trailingOnly = TRUE)
in_tsv  <- args[[1]]
out_dir <- args[[2]]
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

input <- read.table(in_tsv, sep = "\t", header = TRUE, stringsAsFactors = FALSE,
                    check.names = FALSE, na.strings = c("NA", ""))
input$Intensity <- as.numeric(input$Intensity)
for (col in c("ProteinName","PeptideSequence","FragmentIon","ProductCharge",
              "IsotopeLabelType","Condition","BioReplicate","Run")) {
  input[[col]] <- as.character(input[[col]])
}
input$PrecursorCharge <- as.integer(input$PrecursorCharge)

# --- TMP processing + groupComparison + designSampleSize -----------------
processed <- MSstats::dataProcess(
  input, normalization = "equalizeMedians", summaryMethod = "TMP",
  censoredInt = "NA", MBimpute = FALSE, use_log_file = FALSE)

groups <- sort(unique(input$Condition))
contrast.matrix <- matrix(c(-1, 1), nrow = 1)
colnames(contrast.matrix) <- groups
rownames(contrast.matrix) <- paste0(groups[2], "-", groups[1])
gc <- MSstats::groupComparison(contrast.matrix = contrast.matrix,
                               data = processed, use_log_file = FALSE)

ds <- MSstats::designSampleSize(
  data = gc$FittedModel, desiredFC = c(1.25, 1.5), FDR = 0.05,
  numSample = TRUE, power = 0.9, use_log_file = FALSE)
write.table(as.data.frame(ds), file.path(out_dir, "design_numsample.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)

# --- linear summarization ------------------------------------------------
processed_lin <- MSstats::dataProcess(
  input, normalization = "equalizeMedians", summaryMethod = "linear",
  censoredInt = "NA", MBimpute = FALSE, use_log_file = FALSE)
runlvl_lin <- as.data.frame(processed_lin$ProteinLevelData)
write.table(runlvl_lin, file.path(out_dir, "linear_run_level.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)

# --- quantile normalization ---------------------------------------------
# preprocessCore::normalize.quantiles can fail with a pthread error on some
# HPC environments — guard so the rest of the driver still completes.
q_ok <- tryCatch({
  processed_q <- MSstats::dataProcess(
    input, normalization = "quantile", summaryMethod = "TMP",
    censoredInt = "NA", MBimpute = FALSE, use_log_file = FALSE)
  featq <- as.data.frame(processed_q$FeatureLevelData)
  run_med_q <- aggregate(featq$ABUNDANCE, by = list(RUN = featq$RUN),
                         FUN = function(x) median(x, na.rm = TRUE))
  colnames(run_med_q) <- c("RUN", "median")
  write.table(run_med_q, file.path(out_dir, "quantile_run_med.tsv"),
              sep = "\t", quote = FALSE, row.names = FALSE)
  TRUE
}, error = function(e) {
  message("quantile normalization skipped: ", conditionMessage(e))
  FALSE
})

# --- contrast matrix -----------------------------------------------------
cm <- MSstats::MSstatsContrastMatrix("pairwise", groups)
cm_df <- as.data.frame(cm)
cm_df$label <- rownames(cm)
write.table(cm_df, file.path(out_dir, "contrast_matrix.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)

cat("R MSstats v0.2 driver done\n")
