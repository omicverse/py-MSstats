"""
Per-run median normalization — ``MSstats:::.normalizeMedian``.

For each run, compute the median log-intensity (over all features, only
on rows with `LABEL == "L"` in label-free DDA), and shift every feature
of that run so its run-median equals the **grand median** across runs.

Reference R algorithm (paraphrased, label-free single-fraction):

```R
ABUNDANCE_RUN      <- median(ABUNDANCE, by=RUN)        # one value / run
ABUNDANCE_FRACTION <- median(ABUNDANCE_RUN)            # grand median
ABUNDANCE_new      <- ABUNDANCE - ABUNDANCE_RUN + ABUNDANCE_FRACTION
```

After this step every run has the same median log-intensity.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def equalize_medians(
    df: pd.DataFrame,
    *,
    abundance_col: str = "ABUNDANCE",
    run_col: str = "RUN",
    label_col: str = "LABEL",
    label_value: str = "L",
    fraction_col: str | None = None,
) -> pd.DataFrame:
    """Shift every run so its median log-intensity equals the grand median.

    Mirrors ``MSstats:::.normalizeMedian`` for single-label (``LABEL=='L'``)
    label-free workflows. For label-reference (SILAC / TMT) the run
    median is computed on the heavy channel — set ``label_value="H"``.

    Parameters
    ----------
    df
        Long-format MSstats table with at least the abundance, run,
        and label columns (NaN-allowed in abundance).
    abundance_col, run_col, label_col
        Column names (defaults match MSstats's internal post-log2 columns:
        ``ABUNDANCE`` / ``RUN`` / ``LABEL``).
    label_value
        Which label to use for the per-run median (``'L'`` for label-free
        and SILAC light, ``'H'`` if heavy is the internal standard).
    fraction_col
        If not None, do the equalization within each fraction (mirrors
        ``MSstats`` multi-fraction designs). Defaults to a single-fraction
        run (column not required).

    Returns
    -------
    pd.DataFrame
        A *copy* of ``df`` with the abundance column shifted in place.
        An auxiliary ``ABUNDANCE_RUN`` column is NOT added (the R column
        is dropped at the end of ``.normalizeMedian``).
    """
    out = df.copy()
    abundance = out[abundance_col].astype(float)
    runs = out[run_col].astype(str)
    labels = out[label_col].astype(str)

    if fraction_col is None:
        fractions = pd.Series(["_one_"] * len(out), index=out.index)
    else:
        fractions = out[fraction_col].astype(str)

    # 1) per-(run, fraction) median over the rows with label == label_value
    label_mask = labels == label_value
    # Use only label_value rows for the median; in label-free LFQ all are 'L'.
    grp = pd.DataFrame({
        "RUN": runs, "FRACTION": fractions, "ABUNDANCE": abundance,
        "_use": label_mask,
    })
    run_medians = (
        grp.loc[grp["_use"], :]
           .groupby(["RUN", "FRACTION"], dropna=False)["ABUNDANCE"]
           .median()
    )
    # 2) per-fraction grand median (median of run medians within a fraction)
    fraction_grand = run_medians.groupby(level="FRACTION").median()
    # 3) shift = grand - run-median
    shift = fraction_grand.reindex(run_medians.index, level="FRACTION") - run_medians
    # 4) apply shift
    key = list(zip(runs, fractions))
    shift_per_row = pd.Series([shift.get(k, 0.0) for k in key], index=out.index)
    out[abundance_col] = abundance + shift_per_row.fillna(0.0)
    return out
