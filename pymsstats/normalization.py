"""
Per-run normalization for MSstats — ports of ``MSstats:::.normalizeMedian``,
``MSstats:::.normalizeQuantile`` and ``MSstats:::.normalizeGlobalStandards``,
plus a public :func:`msstats_normalize` dispatch matching
``MSstats::MSstatsNormalize``.

For ``equalizeMedians``: for each run, compute the median log-intensity
(over all features, only on rows with ``LABEL == "L"`` in label-free DDA),
and shift every feature of that run so its run-median equals the **grand
median** across runs.

For ``quantile``: classical Bolstad quantile normalization across the
``feature × run`` matrix — replace every column with the mean of its
sorted values (taken across runs) at the same rank.

For ``globalStandards``: shift each run so the mean abundance of a
user-supplied set of standard peptides/proteins is constant.

After ``equalizeMedians`` every run has the same median log-intensity.
"""
from __future__ import annotations

from typing import Iterable

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


# -----------------------------------------------------------------------------
# Quantile normalization
# -----------------------------------------------------------------------------
def quantile_normalize(
    df: pd.DataFrame,
    *,
    abundance_col: str = "ABUNDANCE",
    run_col: str = "RUN",
    feature_col: str = "FEATURE",
    label_col: str = "LABEL",
    label_value: str = "L",
) -> pd.DataFrame:
    """Bolstad quantile normalization on the feature×run abundance matrix.

    Mirrors ``MSstats:::.normalizeQuantile`` for the single-label LFQ case.
    Forms a ``feature × run`` wide matrix of log2 abundances, sorts every
    column, replaces each column with the row mean of those sorted values,
    then unsorts back to the original positions.

    NaNs are preserved (they don't participate in the sort/mean — they
    remain NaN in the output).

    Parameters
    ----------
    df
        Long-format feature-level table.
    abundance_col, run_col, feature_col, label_col, label_value
        Column names. Only rows with ``label_col == label_value`` are
        normalized (label-free → all ``'L'``).

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with the abundance column replaced for the
        label-value rows.
    """
    out = df.copy()
    if label_col in out.columns:
        target = out[label_col].astype(str) == label_value
    else:
        target = pd.Series([True] * len(out), index=out.index)
    sub = out.loc[target].copy()
    if len(sub) == 0:
        return out
    wide = sub.pivot_table(
        index=feature_col, columns=run_col,
        values=abundance_col, aggfunc="mean", dropna=False,
    )
    M = wide.to_numpy(dtype=float)
    nrows, ncols = M.shape
    if ncols < 2:
        return out
    # Standard Bolstad quantile normalization, NaN-aware:
    #   1) sort each column → sorted ranks
    #   2) compute the across-column mean at each rank (nanmean for ties/NaN)
    #   3) put back via the inverse of the sorted order
    sorted_idx = np.argsort(M, axis=0, kind="mergesort")
    sorted_vals = np.take_along_axis(M, sorted_idx, axis=0)
    rank_means = np.nanmean(sorted_vals, axis=1)
    # Place the per-rank mean back at the original positions
    out_M = np.empty_like(M)
    for j in range(ncols):
        out_M[sorted_idx[:, j], j] = rank_means
    # restore NaNs
    out_M[np.isnan(M)] = np.nan
    out_wide = pd.DataFrame(out_M, index=wide.index, columns=wide.columns)
    # melt back into long-format with the original row order
    sub["_orig_idx"] = sub.index
    melted = out_wide.reset_index().melt(
        id_vars=feature_col, var_name=run_col, value_name="_new_abundance",
    )
    sub_renamed = sub.merge(melted, on=[feature_col, run_col], how="left")
    sub_renamed.set_index("_orig_idx", inplace=True)
    out.loc[sub_renamed.index, abundance_col] = sub_renamed["_new_abundance"]
    return out


# -----------------------------------------------------------------------------
# Global standards normalization
# -----------------------------------------------------------------------------
def normalize_global_standards(
    df: pd.DataFrame,
    standards: Iterable[str],
    *,
    abundance_col: str = "ABUNDANCE",
    run_col: str = "RUN",
    protein_col: str = "PROTEIN",
    peptide_col: str = "PeptideSequence",
    label_col: str = "LABEL",
    label_value: str = "L",
) -> pd.DataFrame:
    """Shift each run by the global-standards mean.

    Mirrors ``MSstats:::.normalizeGlobalStandards``: per run, average the
    log2 abundances of all rows belonging to the user-supplied
    ``standards`` (matched against protein name OR peptide sequence),
    then subtract that mean and add the median-of-means so every run has
    the same mean standard abundance.
    """
    standards = list(standards)
    out = df.copy()
    mask = (out[protein_col].astype(str).isin(standards))
    if peptide_col in out.columns:
        mask = mask | (out[peptide_col].astype(str).isin(standards))
    if not mask.any():
        raise ValueError(
            f"none of the standards {standards} found in the data"
        )
    label_target = (
        (out[label_col].astype(str) == label_value)
        if label_col in out.columns else pd.Series(True, index=out.index)
    )
    standard_rows = out.loc[mask & label_target]
    mean_by_run = (
        standard_rows.groupby(run_col)[abundance_col]
        .mean()
    )
    median_of_means = float(mean_by_run.median())
    shift_per_run = (median_of_means - mean_by_run)
    # apply only on the label_value rows
    shift_series = out[run_col].map(shift_per_run).fillna(0.0)
    out.loc[label_target, abundance_col] = (
        pd.to_numeric(out.loc[label_target, abundance_col], errors="coerce")
        + shift_series.loc[label_target]
    )
    return out


# -----------------------------------------------------------------------------
# Dispatch
# -----------------------------------------------------------------------------
def msstats_normalize(
    df: pd.DataFrame,
    *,
    method: str = "equalizeMedians",
    standards: Iterable[str] | None = None,
    abundance_col: str = "ABUNDANCE",
    run_col: str = "RUN",
    feature_col: str = "FEATURE",
    label_col: str = "LABEL",
    label_value: str = "L",
) -> pd.DataFrame:
    """Dispatch on ``method`` — port of ``MSstats::MSstatsNormalize``.

    Methods (case-insensitive):

    * ``'equalizeMedians'`` (default) → :func:`equalize_medians`
    * ``'quantile'`` → :func:`quantile_normalize`
    * ``'globalStandards'`` → :func:`normalize_global_standards`
      (requires ``standards`` argument)
    * ``'none'`` / ``False`` → no-op
    """
    m = str(method).upper() if not isinstance(method, bool) else "NONE"
    if m in ("NONE", "FALSE"):
        return df.copy()
    if m == "EQUALIZEMEDIANS":
        return equalize_medians(df, abundance_col=abundance_col,
                                run_col=run_col, label_col=label_col,
                                label_value=label_value)
    if m == "QUANTILE":
        return quantile_normalize(df, abundance_col=abundance_col,
                                  run_col=run_col, feature_col=feature_col,
                                  label_col=label_col, label_value=label_value)
    if m == "GLOBALSTANDARDS":
        if not standards:
            raise ValueError(
                "method='globalStandards' requires the standards argument"
            )
        return normalize_global_standards(
            df, standards,
            abundance_col=abundance_col, run_col=run_col,
            label_col=label_col, label_value=label_value,
        )
    raise ValueError(f"unknown normalization method={method!r}")


__all__ = [
    "equalize_medians", "quantile_normalize",
    "normalize_global_standards", "msstats_normalize",
]
