"""Censored-value handling — port of ``MSstats::MSstatsHandleMissing``.

The R function flags **censored** observations (intensities below an
LOD-style cutoff). For TMP summarization with ``MBimpute=TRUE`` it sets
a per-feature boolean ``censored`` column based on either:

* the missing symbol (``"NA"`` or ``"0"``), or
* a quantile-based cutoff
  ``cutoff = Q1 - (Q[censoredCutoff] - Q3) / IQR * IQR`` on the post-log2
  intensity distribution (restricted to ``LABEL == "L"`` rows).

For label-free LFQ in v0.2 we restrict to the cutoff="0.999" R default
behaviour; the cutoff is computed on the in-data log2 intensities.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def msstats_handle_missing(
    data: pd.DataFrame,
    *,
    summary_method: str = "TMP",
    impute: bool = True,
    missing_symbol: str | None = "NA",
    censored_cutoff: float | None = 0.999,
    intensity_col: str = "INTENSITY",
    abundance_col: str = "ABUNDANCE",
    label_col: str = "LABEL",
) -> pd.DataFrame:
    """Flag censored / missing values per the MSstats rule.

    Parameters
    ----------
    data
        Feature-level long table after log2 + (optional) normalization.
        Must have at least ``INTENSITY, ABUNDANCE`` columns.
    summary_method
        Only ``"TMP"`` participates in the censored flagging — other
        methods leave the column as all-False.
    impute, missing_symbol, censored_cutoff
        See the R counterpart.

    Returns
    -------
    pd.DataFrame
        Copy of ``data`` with a new boolean ``censored`` column.
    """
    out = data.copy()
    out["censored"] = False

    if (summary_method != "TMP") or (not impute) or missing_symbol is None:
        return out

    intensity = pd.to_numeric(out[intensity_col], errors="coerce")
    abundance = pd.to_numeric(out[abundance_col], errors="coerce")
    label = out[label_col].astype(str) if label_col in out.columns \
        else pd.Series(["L"] * len(out), index=out.index)
    is_L = (label == "L")

    if censored_cutoff is not None:
        ok = intensity.notna() & (intensity > 1) & is_L
        if ok.any():
            q = np.nanquantile(
                abundance[ok], [0.01, 0.25, 0.5, 0.75, float(censored_cutoff)],
            )
            iqr = q[3] - q[1]
            multiplier = (q[4] - q[3]) / iqr if iqr > 0 else 0.0
            cutoff_lower = q[1] - multiplier * iqr
            censored_mask = (
                intensity.notna() & is_L & (abundance < cutoff_lower)
            )
            if cutoff_lower <= 0 and missing_symbol == "0":
                censored_mask |= (abundance.notna() & (abundance <= 0))
            if missing_symbol == "NA":
                censored_mask |= intensity.isna()
            out["censored"] = censored_mask.fillna(False)
    else:
        if missing_symbol == "0":
            out["censored"] = (
                is_L & intensity.notna()
                & ((intensity == 1) | (abundance <= 0))
            )
        elif missing_symbol == "NA":
            out["censored"] = is_L & abundance.isna()

    # heavy-label rows are never censored
    if label_col in out.columns:
        out.loc[out[label_col].astype(str) == "H", "censored"] = False
    return out


__all__ = ["msstats_handle_missing"]
