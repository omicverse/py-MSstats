"""
pymsstats: Pure-Python port of Bioconductor MSstats (Choi et al. 2014).

Covers the canonical label-free DDA pipeline plus a comprehensive set of
the MSstats public API — vendor converters, statistical helpers, and
QC plotting.

Core pipeline
-------------
* :func:`data_process` — log2 + per-run median normalization + Tukey
  median polish summarization (R ``MSstats::dataProcess``).
* :func:`group_comparison` — per-protein OLS / mixed-effects Wald test
  for a contrast matrix, with Benjamini-Hochberg correction
  (``MSstats::groupComparison``).

Vendor converters (:mod:`pymsstats.io`)
---------------------------------------
* :func:`maxquant_to_msstats`, :func:`diann_to_msstats`,
  :func:`spectronaut_to_msstats`, :func:`fragpipe_to_msstats`,
  :func:`openms_to_msstats`, :func:`skyline_to_msstats`.
* :func:`validate_annotation`, :func:`merge_fractions`.

Statistical helpers
-------------------
* :func:`msstats_contrast_matrix` — build a contrast matrix from a string
  spec or pair list (``MSstats::MSstatsContrastMatrix``).
* :func:`design_sample_size` — power / sample-size calculation
  (``MSstats::designSampleSize``).
* :func:`msstats_normalize`, :func:`quantile_normalize`,
  :func:`normalize_global_standards` — standalone normalization.
* :func:`msstats_summarize`, :func:`linear_summarize` — alternative
  peptide-to-protein summarization.
* :func:`select_features` — top-N / feature filtering.
* :func:`msstats_handle_missing` — censored-value flagging.
* :func:`quantification` — per-protein abundance table.
* :func:`get_processed`, :func:`get_samples_info`,
  :func:`get_selected_proteins` — accessors.

Plotting (:mod:`pymsstats.plotting`)
------------------------------------
* :func:`theme_msstats`, :func:`data_process_plots`,
  :func:`group_comparison_plots`, :func:`group_comparison_qc_plots`,
  :func:`model_based_qc_plots`.

Quick-start
-----------

>>> from pymsstats import data_process, group_comparison, msstats_contrast_matrix
>>> processed = data_process(msstats_long_df)
>>> C = msstats_contrast_matrix("groupB-groupA", ["groupA", "groupB"])
>>> result = group_comparison(processed, contrast_matrix=C)
"""
from __future__ import annotations

from .contrasts import msstats_contrast_matrix
from .design import design_sample_size
from .group_comparison import group_comparison
from .imputation import msstats_handle_missing
from .io import (
    CANONICAL_COLS,
    diann_to_msstats,
    fragpipe_to_msstats,
    maxquant_to_msstats,
    merge_fractions,
    openms_to_msstats,
    skyline_to_msstats,
    spectronaut_to_msstats,
    validate_annotation,
)
from .normalization import (
    equalize_medians,
    msstats_normalize,
    normalize_global_standards,
    quantile_normalize,
)
from .pipeline import data_process
from .preprocess import select_features
from .summarization import (
    get_processed,
    get_samples_info,
    get_selected_proteins,
    linear_summarize,
    medpolish,
    msstats_summarize,
    quantification,
    tmp_summarize,
)

__version__ = "0.2.0"

# plotting functions are lazy-importable (matplotlib optional)
from .plotting import (  # noqa: E402
    data_process_plots,
    group_comparison_plots,
    group_comparison_qc_plots,
    model_based_qc_plots,
    theme_msstats,
)

__all__ = [
    # core pipeline
    "data_process",
    "group_comparison",
    # vendor converters
    "maxquant_to_msstats",
    "diann_to_msstats",
    "spectronaut_to_msstats",
    "fragpipe_to_msstats",
    "openms_to_msstats",
    "skyline_to_msstats",
    "validate_annotation",
    "merge_fractions",
    # statistical helpers
    "msstats_contrast_matrix",
    "design_sample_size",
    "msstats_normalize",
    "quantile_normalize",
    "normalize_global_standards",
    "equalize_medians",
    "msstats_summarize",
    "linear_summarize",
    "tmp_summarize",
    "medpolish",
    "select_features",
    "msstats_handle_missing",
    "quantification",
    "get_processed",
    "get_samples_info",
    "get_selected_proteins",
    # plotting
    "theme_msstats",
    "data_process_plots",
    "group_comparison_plots",
    "group_comparison_qc_plots",
    "model_based_qc_plots",
    # misc
    "CANONICAL_COLS",
]
