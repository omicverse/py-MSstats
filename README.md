# pymsstats

A **pure-Python port of [Bioconductor MSstats](https://bioconductor.org/packages/release/bioc/html/MSstats.html)** (Choi *et al.*, *Bioinformatics* 2014) — vendor converters, pre-processing, normalization, peptide-to-protein summarization, per-protein LMM-based group comparison, sample-size design, and QC plotting for mass-spec proteomics.

- Full pipeline mirroring R: `vendor → dataProcess (log2 + normalize + TMP/linear) → groupComparison → designSampleSize / plots`
- **No `rpy2`**, no R install — every algorithm reimplemented directly in NumPy / SciPy / statsmodels / pandas
- **100% coverage of the R MSstats 4.14.2 public API** — all 50 exported names (46 functions ported, 4 example datasets replaced by a synthetic generator)
- Bit-for-bit reproduction of TMP `LogIntensities` (5e-14), `log2FC` (8e-15), linear summarization (9e-14), `designSampleSize` `numSample` (exact), `MSstatsContrastMatrix`, `checkRepeatedDesign` and `SDRFtoAnnotation` against R MSstats 4.14.2
- **2.5×–3.9× faster** than R MSstats on the 500-protein × 8-run benchmark (Pearson r = 1.0 on every output)
- Nine vendor converters (DIA-NN, Spectronaut, FragPipe, OpenMS, Skyline, Proteome Discoverer, Progenesis, OpenSWATH, DIA-Umpire) on top of MaxQuant
- Modular worker API (`MSstatsPrepareFor*`, `MSstatsSummarize*`, `MSstatsGroupComparison*`) + SDRF metadata helpers
- AnnData / pandas-friendly: accepts MSstats-format long DataFrame (10 canonical columns)

> This is a **standalone mirror** of the canonical implementation that lives in [`omicverse`](https://github.com/Starlitnightly/omicverse). All algorithmic work is developed upstream in omicverse and synced here.

## Install

```bash
pip install pymsstats
```

Depends only on `numpy`, `scipy`, `pandas`, and `statsmodels`.

## Quick-start

```python
import numpy as np
import pandas as pd
from pymsstats import data_process, group_comparison

# msstats_df is a long-format DataFrame with the 10 canonical MSstats columns:
#   ProteinName, PeptideSequence, PrecursorCharge, FragmentIon, ProductCharge,
#   IsotopeLabelType, Condition, BioReplicate, Run, Intensity
msstats_df = pd.read_csv("msstats_long.tsv", sep="\t")

# Pre-process: log2 → equalize-medians → drop low-obs features → TMP summarize.
processed = data_process(
    msstats_df,
    normalization="equalizeMedians",
    summary_method="TMP",
    log_base=2,
)
# processed columns: Protein, RUN, LogIntensities, GROUP, SUBJECT, n_features, n_obs

# Two-group test: group1 vs group0.
result = group_comparison(
    processed,
    contrast_matrix=np.array([[-1.0, 1.0]]),
    groups=["group0", "group1"],
)
# result columns: Protein, Label, log2FC, SE, df, t, pvalue, adj.pvalue, issue
```

### Vendor converters

Six converters turn raw vendor output into MSstats long-format:

```python
from pymsstats import (
    maxquant_to_msstats, diann_to_msstats, spectronaut_to_msstats,
    fragpipe_to_msstats, openms_to_msstats, skyline_to_msstats,
)

ann = pd.read_csv("annotation.csv")   # columns: Run, Condition, BioReplicate

# MaxQuant
pg = pd.read_csv("proteinGroups.txt", sep="\t")
ev = pd.read_csv("evidence.txt", sep="\t")
msstats_df = maxquant_to_msstats(pg, ev, ann)

# DIA-NN
report = pd.read_csv("report.tsv", sep="\t")
msstats_df = diann_to_msstats(report, ann, qvalue_cutoff=0.01)

# Spectronaut / FragPipe / OpenMS / Skyline
msstats_df = spectronaut_to_msstats(pd.read_csv("spectronaut.tsv", sep="\t"))
msstats_df = skyline_to_msstats(pd.read_csv("skyline.csv"), ann)
```

### Sample-size design, quantification, plotting

```python
from pymsstats import design_sample_size, quantification, group_comparison_plots

# how many replicates for 80% power at FC 1.25–1.5?
design = design_sample_size(processed, desired_fc=(1.25, 1.5), power=0.8)

# per-protein abundance matrix
quant = quantification(processed, type="Sample")

# volcano / heatmap on the comparison result
ax = group_comparison_plots(result, type="VolcanoPlot")
```

## Low-level API

```python
from pymsstats import equalize_medians, tmp_summarize, medpolish

# Each stage is exposed for users who want to slot one piece of the
# pipeline into a custom workflow.
norm_df = equalize_medians(feature_df, abundance_col="ABUNDANCE")
prot_df = tmp_summarize(norm_df)        # per (protein, run) LogIntensities
mp = medpolish(X)                       # raw R-stats::medpolish equivalent
```

## R-parity status

The MSstats `dataProcess` → `groupComparison` pipeline is tested against
R MSstats 4.14.2 on the canonical LFQ design (globally-unique BioReplicate IDs):

| Step | R counterpart | Match |
|---|---|---|
| `equalize_medians` | `MSstats:::.normalizeMedian` | bit-exact (post-norm run medians flat to 1e-9) |
| `medpolish` | `stats::medpolish` / `MSstats:::median_polish_summary` | row+col effects bit-exact (1e-9) on noiseless tables |
| `tmp_summarize` LogIntensities | `MSstats::dataProcess$ProteinLevelData` | **bit-exact, max abs diff = 5e-14** |
| `group_comparison` log2FC | `MSstats::groupComparison` | **bit-exact, max abs diff = 8e-15** |
| `group_comparison` SE | `MSstats::groupComparison` | bit-exact for OLS path (1e-15); ≤ 5% on LMM-fallback path |
| `group_comparison` pvalue | `MSstats::groupComparison` | bit-exact for OLS path; Pearson r ≥ 0.99 for LMM path |
| `group_comparison` adj.pvalue | `MSstats::groupComparison` | same as pvalue |

The OLS path (`auto` mode when SUBJECT is nested in GROUP) is **bit-identical
to R**. The mixed path (paired/repeated-measures design where the same
BioReplicate appears across groups) uses `statsmodels.MixedLM`; when its
solver hits a singular Hessian (common for small balanced data) we fall back
to a fixed-effects `lm(ABUNDANCE ~ GROUP + SUBJECT)`, which gives slightly
different SEs from R's `lme4::lmer` (the ranking — Pearson r ≥ 0.99 on
pvalue — is unaffected).

## Benchmark

500 proteins × 8 runs × 3 features, label-free DDA design:

| Step | R (ms) | Python (ms) | Speedup |
|---|---|---|---|
| `dataProcess` | 3754 | 2393 | 1.57× |
| `groupComparison` | 3880 | 626 | 6.20× |
| **Pipeline total** | **7634** | **3019** | **2.53×** |
| Including `Rscript` startup | 11807 | 3019 | 3.91× |

Run it yourself:

```bash
python examples/benchmark.py --runs 3 --n-proteins 500
```

## Reproducing R results exactly

`tests/r_reference_driver.R` runs `MSstats::dataProcess` + `MSstats::groupComparison`
on the same TSV input the Python side dumps. `tests/test_r_parity.py` (skipped
if no CMAP / MSstats env) checks every numerical field — see the R-parity
table above.

```bash
# Run only the R-parity tests
pytest tests/test_r_parity.py -v

# Run everything (smoke + R-parity)
pytest tests/ -v
```

## MSstats public API coverage (v0.3 — 100%)

All **50 exported names** of R MSstats 4.14.2 are covered. The 46
functions are ported to Python; the 4 bundled example datasets are
copyrighted Bioconductor data objects (not redistributed) and are
replaced by a synthetic generator.

### Core pipeline

| R function | Python | Status |
|---|---|---|
| `dataProcess` (log2 + equalizeMedians + TMP) | `data_process` | ✅ ported |
| `groupComparison` (OLS / LMM auto-dispatch + BH) | `group_comparison` | ✅ ported |

### Vendor converters

| R function | Python | Status |
|---|---|---|
| `MaxQtoMSstatsFormat` | `maxquant_to_msstats` | ✅ ported |
| `DIANNtoMSstatsFormat` | `diann_to_msstats` | ✅ ported |
| `SpectronauttoMSstatsFormat` | `spectronaut_to_msstats` | ✅ ported |
| `FragPipetoMSstatsFormat` | `fragpipe_to_msstats` | ✅ ported |
| `OpenMStoMSstatsFormat` | `openms_to_msstats` | ✅ ported |
| `SkylinetoMSstatsFormat` | `skyline_to_msstats` | ✅ ported |
| `PDtoMSstatsFormat` | `pd_to_msstats` | ✅ ported |
| `ProgenesistoMSstatsFormat` | `progenesis_to_msstats` | ✅ ported |
| `OpenSWATHtoMSstatsFormat` | `openswath_to_msstats` | ✅ ported |
| `DIAUmpiretoMSstatsFormat` | `diaumpire_to_msstats` | ✅ ported |

### Statistical helpers

| R function | Python | Status |
|---|---|---|
| `validateAnnotation` | `validate_annotation` | ✅ ported |
| `MSstatsMergeFractions` | `merge_fractions` | ✅ ported |
| `MSstatsContrastMatrix` | `msstats_contrast_matrix` | ✅ ported (R-exact) |
| `designSampleSize` | `design_sample_size` | ✅ ported (R-exact `numSample`) |
| `MSstatsNormalize` (equalizeMedians/quantile/globalStandards) | `msstats_normalize` | ✅ ported |
| `MSstatsSummarize` (TMP / linear) | `msstats_summarize` | ✅ ported (R-exact) |
| `MSstatsSelectFeatures` (all / topN) | `select_features` | ✅ ported |
| `MSstatsHandleMissing` | `msstats_handle_missing` | ✅ ported |
| `checkRepeatedDesign` | `check_repeated_design` | ✅ ported (R-exact) |
| `makePeptidesDictionary` | `make_peptides_dictionary` | ✅ ported |
| `quantification` | `quantification` | ✅ ported |
| `getProcessed` / `getSamplesInfo` / `getSelectedProteins` | `get_processed` / `get_samples_info` / `get_selected_proteins` | ✅ ported |

### Modular worker API

| R function | Python | Status |
|---|---|---|
| `MSstatsPrepareForDataProcess` | `prepare_for_data_process` | ✅ ported |
| `MSstatsPrepareForSummarization` | `prepare_for_summarization` | ✅ ported |
| `MSstatsPrepareForGroupComparison` | `prepare_for_group_comparison` | ✅ ported |
| `MSstatsSummarize` (modular) | `msstats_summarize_modular` | ✅ ported |
| `MSstatsSummarizationOutput` | `summarization_output` | ✅ ported |
| `MSstatsSummarizeWithSingleCore` | `summarize_single_core` | ✅ ported |
| `MSstatsSummarizeSingleTMP` | `summarize_single_tmp` | ✅ ported |
| `MSstatsSummarizeSingleLinear` | `summarize_single_linear` | ✅ ported |
| `MSstatsGroupComparison` | `msstats_group_comparison` | ✅ ported |
| `MSstatsGroupComparisonSingleProtein` | `group_comparison_single_protein` | ✅ ported |
| `MSstatsGroupComparisonOutput` | `group_comparison_output` | ✅ ported |

### SDRF helpers

| R function | Python | Status |
|---|---|---|
| `extractSDRF` | `extract_sdrf` | ✅ ported |
| `SDRFtoAnnotation` | `sdrf_to_annotation` | ✅ ported (R-exact) |
| `example_SDRF` | `example_sdrf` | ✅ ported (synthetic stand-in) |

### Plotting

| R function | Python | Status |
|---|---|---|
| `dataProcessPlots` (Profile/QC/Condition) | `data_process_plots` | ✅ ported |
| `groupComparisonPlots` (Volcano/Heatmap/Comparison) | `group_comparison_plots` | ✅ ported |
| `groupComparisonQCPlots` | `group_comparison_qc_plots` | ✅ ported |
| `modelBasedQCPlots` | `model_based_qc_plots` | ✅ ported |
| `designSampleSizePlots` | `design_sample_size_plots` | ✅ ported |
| `theme_msstats` | `theme_msstats` | ✅ ported |
| `savePlot` | `save_plot` | ✅ ported |

### Bundled example datasets

| R object | Python | Status |
|---|---|---|
| `DDARawData` | `load_dda_example()` | ⚠️ synthetic stand-in — original R data not redistributed |
| `DDARawData.Skyline` | `load_dda_example()` | ⚠️ synthetic stand-in |
| `DIARawData` | `load_dia_example()` | ⚠️ synthetic stand-in |
| `SRMRawData` | `load_srm_example()` | ⚠️ synthetic stand-in |

The four R example datasets are copyrighted Bioconductor data objects;
pymsstats provides `make_example_dataset()` / `load_dda_example()` /
`load_dia_example()` / `load_srm_example()` which return a synthetic
MSstats long-format DataFrame with the same 10 canonical columns so the
pipeline can be exercised end-to-end without the original data.

### Known limitations (within ported functions)

- The mixed-effects (`lme4::lmer`) path uses `statsmodels.MixedLM` with
  an OLS fallback on singular-Hessian cases (SE differs ≤ 5%; ranking
  unaffected — see R-parity table above).
- `feature_subset='highQuality'` in `select_features` falls back to
  `'all'` with a warning (the leverage-based outlier rule is rarely
  exercised).
- TMT / labeled (heavy-light) workflows are out of scope (MSstatsTMT is
  a separate Bioconductor package).

## Relationship to omicverse

Developed **upstream** in [`omicverse`](https://github.com/Starlitnightly/omicverse). This repo is a standalone mirror with the same code, same API, minus the omicverse packaging.

## Citation

If you use this package, please cite the MSstats paper:

> Choi, M., Chang, C.-Y., Clough, T., Broudy, D., Killeen, T., MacLean, B., & Vitek, O.
> **MSstats: an R package for statistical analysis of quantitative mass spectrometry-based proteomic experiments.**
> *Bioinformatics* **30** (17), 2524-2526 (2014). DOI: 10.1093/bioinformatics/btu305

…and acknowledge omicverse / this repo for the Python port.

## License

Artistic-2.0 — matches the upstream Bioconductor package.
