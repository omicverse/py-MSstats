#!/usr/bin/env Rscript
# Drive MSstats::dataProcess + MSstats::groupComparison on a TSV input
# dumped by the Python side, and write the per-protein summary table and
# per-protein contrast results back out.
#
# Usage:
#   Rscript r_reference_driver.R <msstats_long_tsv> <out_dir>
#
# Outputs (in out_dir):
#   processed.tsv     dataProcess()$FeatureLevelData (RunlevelData)
#   feature_level.tsv dataProcess()$FeatureLevelData
#   run_level.tsv     dataProcess()$ProteinLevelData (one row / (protein, run))
#   group_comp.tsv    groupComparison()$ComparisonResult
#   run_medians.tsv   post-normalization per-run median of ABUNDANCE
#                     (sanity: should be constant across runs)
#   timing.tsv        wall-clock for dataProcess / groupComparison

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

# Ensure column types match MSstats expectations.
input$Intensity <- as.numeric(input$Intensity)
# MSstats expects character/factor for these.
for (col in c("ProteinName","PeptideSequence","FragmentIon","ProductCharge",
              "IsotopeLabelType","Condition","BioReplicate","Run")) {
  input[[col]] <- as.character(input[[col]])
}
input$PrecursorCharge <- as.integer(input$PrecursorCharge)

t0 <- proc.time()[[3]]
processed <- MSstats::dataProcess(
  input,
  normalization     = "equalizeMedians",
  summaryMethod     = "TMP",
  censoredInt       = "NA",
  MBimpute          = FALSE,
  use_log_file      = FALSE
)
t1 <- proc.time()[[3]]

# MSstats post-4.x returns a list with $FeatureLevelData and $ProteinLevelData.
feat <- as.data.frame(processed$FeatureLevelData)
runlvl <- as.data.frame(processed$ProteinLevelData)

write.table(feat, file.path(out_dir, "feature_level.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)
write.table(runlvl, file.path(out_dir, "run_level.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)

# Sanity: per-run median of (post-normalization) ABUNDANCE in feature-level data
run_med <- aggregate(feat$ABUNDANCE, by = list(RUN = feat$RUN),
                     FUN = function(x) median(x, na.rm = TRUE))
colnames(run_med) <- c("RUN","median")
write.table(run_med, file.path(out_dir, "run_medians.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)

# Build a 2-group contrast: group1 - group0 (matches Python smoke setup).
groups <- sort(unique(input$Condition))
if (length(groups) != 2) {
  stop(sprintf("Expected exactly two conditions, got %d: %s",
               length(groups), paste(groups, collapse = ", ")))
}
contrast.matrix <- matrix(c(-1, 1), nrow = 1)
colnames(contrast.matrix) <- groups
rownames(contrast.matrix) <- paste0(groups[2], "-", groups[1])

t2 <- proc.time()[[3]]
gc <- MSstats::groupComparison(
  contrast.matrix = contrast.matrix,
  data            = processed,
  use_log_file    = FALSE
)
t3 <- proc.time()[[3]]
gc_res <- as.data.frame(gc$ComparisonResult)
write.table(gc_res, file.path(out_dir, "group_comp.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)

write.table(data.frame(dataProcess = t1 - t0,
                       groupComparison = t3 - t2),
            file.path(out_dir, "timing.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)
cat("R MSstats pipeline done\n")
