"""
pymsstats: Pure-Python port of Bioconductor MSstats (Choi et al. 2014).

Minimal MVP covering the canonical label-free DDA pipeline:

* :func:`data_process` — log2 + per-run median normalization + Tukey
  median polish summarization (R default in ``MSstats::dataProcess``).
* :func:`group_comparison` — per-protein OLS / mixed-effects Wald test
  for a contrast matrix, with Benjamini-Hochberg correction (mirrors
  ``MSstats::groupComparison``).
* :func:`maxquant_to_msstats` — minimal MaxQuant ``proteinGroups`` +
  ``evidence`` → MSstats long-format converter (the most-common case
  only; full converter deferred to v0.2).

Quick-start
-----------

>>> from pymsstats import data_process, group_comparison
>>> processed = data_process(msstats_long_df)
>>> result = group_comparison(processed, contrast_matrix=np.array([[-1, 1]]))
>>> result.head()

The functions :func:`equalize_medians`, :func:`tmp_summarize`,
:func:`medpolish` are exposed for users who want to slot one stage of
the pipeline into a custom workflow.
"""
from __future__ import annotations

from .group_comparison import group_comparison
from .normalization import equalize_medians
from .pipeline import data_process, maxquant_to_msstats, CANONICAL_COLS
from .summarization import medpolish, tmp_summarize

__version__ = "0.1.0"

__all__ = [
    "data_process",
    "group_comparison",
    "maxquant_to_msstats",
    "equalize_medians",
    "tmp_summarize",
    "medpolish",
    "CANONICAL_COLS",
]
