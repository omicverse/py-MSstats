"""
pymsstats: Pure-Python port of Bioconductor MSstats (Choi et al. 2014).

Covers the full public API of R MSstats 4.14.2 — the canonical label-free
DDA pipeline, every vendor converter, the modular worker API, SDRF
helpers, statistical helpers, and QC plotting.

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
  :func:`openms_to_msstats`, :func:`skyline_to_msstats`,
  :func:`pd_to_msstats`, :func:`progenesis_to_msstats`,
  :func:`openswath_to_msstats`, :func:`diaumpire_to_msstats`.
* :func:`validate_annotation`, :func:`merge_fractions`.

Modular worker API (:mod:`pymsstats.workers`)
---------------------------------------------
* :func:`prepare_for_data_process`, :func:`prepare_for_summarization`,
  :func:`prepare_for_group_comparison`, :func:`msstats_summarize_modular`,
  :func:`summarization_output`, :func:`msstats_group_comparison`,
  :func:`group_comparison_output`, :func:`group_comparison_single_protein`,
  :func:`summarize_single_linear`, :func:`summarize_single_tmp`,
  :func:`summarize_single_core`.

SDRF helpers (:mod:`pymsstats.sdrf`)
------------------------------------
* :func:`extract_sdrf`, :func:`sdrf_to_annotation`, :func:`example_sdrf`.

Statistical helpers
-------------------
* :func:`msstats_contrast_matrix`, :func:`design_sample_size`,
  :func:`design_sample_size_plots`.
* :func:`msstats_normalize`, :func:`quantile_normalize`,
  :func:`normalize_global_standards`, :func:`equalize_medians`.
* :func:`msstats_summarize`, :func:`linear_summarize`,
  :func:`tmp_summarize`, :func:`medpolish`.
* :func:`select_features`, :func:`check_repeated_design`,
  :func:`make_peptides_dictionary`, :func:`msstats_handle_missing`.
* :func:`quantification`, :func:`get_processed`, :func:`get_samples_info`,
  :func:`get_selected_proteins`.

Plotting (:mod:`pymsstats.plotting`)
------------------------------------
* :func:`theme_msstats`, :func:`data_process_plots`,
  :func:`group_comparison_plots`, :func:`group_comparison_qc_plots`,
  :func:`model_based_qc_plots`, :func:`save_plot`.

Example datasets (:mod:`pymsstats.datasets`)
--------------------------------------------
* :func:`load_dda_example`, :func:`load_dia_example`,
  :func:`load_srm_example`, :func:`make_example_dataset`.

Quick-start
-----------

>>> from pymsstats import data_process, group_comparison, msstats_contrast_matrix
>>> processed = data_process(msstats_long_df)
>>> C = msstats_contrast_matrix("groupB-groupA", ["groupA", "groupB"])
>>> result = group_comparison(processed, contrast_matrix=C)
"""
from __future__ import annotations

from .contrasts import msstats_contrast_matrix
from .datasets import (
    load_dda_example,
    load_dia_example,
    load_srm_example,
    make_example_dataset,
)
from .design import design_sample_size, design_sample_size_plots
from .group_comparison import group_comparison
from .imputation import msstats_handle_missing
from .io import (
    CANONICAL_COLS,
    diann_to_msstats,
    diaumpire_to_msstats,
    fragpipe_to_msstats,
    maxquant_to_msstats,
    merge_fractions,
    openms_to_msstats,
    openswath_to_msstats,
    pd_to_msstats,
    progenesis_to_msstats,
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
from .preprocess import (
    check_repeated_design,
    make_peptides_dictionary,
    select_features,
)
from .sdrf import example_sdrf, extract_sdrf, sdrf_to_annotation
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
from .workers import (
    group_comparison_output,
    group_comparison_single_protein,
    msstats_group_comparison,
    msstats_summarize_modular,
    prepare_for_data_process,
    prepare_for_group_comparison,
    prepare_for_summarization,
    summarization_output,
    summarize_single_core,
    summarize_single_linear,
    summarize_single_tmp,
)

__version__ = "0.3.0"

# plotting functions are lazy-importable (matplotlib optional)
from .plotting import (  # noqa: E402
    data_process_plots,
    group_comparison_plots,
    group_comparison_qc_plots,
    model_based_qc_plots,
    save_plot,
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
    "pd_to_msstats",
    "progenesis_to_msstats",
    "openswath_to_msstats",
    "diaumpire_to_msstats",
    "validate_annotation",
    "merge_fractions",
    # statistical helpers
    "msstats_contrast_matrix",
    "design_sample_size",
    "design_sample_size_plots",
    "msstats_normalize",
    "quantile_normalize",
    "normalize_global_standards",
    "equalize_medians",
    "msstats_summarize",
    "linear_summarize",
    "tmp_summarize",
    "medpolish",
    "select_features",
    "check_repeated_design",
    "make_peptides_dictionary",
    "msstats_handle_missing",
    "quantification",
    "get_processed",
    "get_samples_info",
    "get_selected_proteins",
    # modular worker API
    "prepare_for_data_process",
    "prepare_for_summarization",
    "prepare_for_group_comparison",
    "msstats_summarize_modular",
    "summarization_output",
    "msstats_group_comparison",
    "group_comparison_output",
    "group_comparison_single_protein",
    "summarize_single_linear",
    "summarize_single_tmp",
    "summarize_single_core",
    # SDRF helpers
    "extract_sdrf",
    "sdrf_to_annotation",
    "example_sdrf",
    # plotting
    "theme_msstats",
    "data_process_plots",
    "group_comparison_plots",
    "group_comparison_qc_plots",
    "model_based_qc_plots",
    "save_plot",
    # example datasets
    "make_example_dataset",
    "load_dda_example",
    "load_dia_example",
    "load_srm_example",
    # misc
    "CANONICAL_COLS",
    "__version__",
]
