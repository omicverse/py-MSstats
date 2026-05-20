"""Feature-selection helpers — port of ``MSstats::MSstatsSelectFeatures``.

Supported strategies:

* ``"all"`` — keep all features (no filtering, R default).
* ``"top3"`` / ``"topN"`` — pick the top-N most intense features per
  protein, ranked by mean log2 intensity across runs
  (mirrors ``MSstats:::.selectTopFeatures``).
* ``"highQuality"`` — flag uninformative features and outliers (not
  ported here — the R helper uses a leverage / studentized-residual
  rule that is rarely exercised in practice; we return ``method="all"``
  behaviour with a warning).
"""
from __future__ import annotations

import warnings
from typing import Iterable

import numpy as np
import pandas as pd


def select_features(
    data: pd.DataFrame,
    *,
    feature_subset: str = "all",
    top_n: int = 3,
    min_feature_count: int = 2,
    protein_col: str = "PROTEIN",
    feature_col: str = "FEATURE",
    abundance_col: str = "ABUNDANCE",
) -> pd.DataFrame:
    """Mark features for removal (column ``remove`` added).

    Parameters
    ----------
    data
        Feature-level long table with at least ``PROTEIN, FEATURE,
        ABUNDANCE`` columns (post log2).
    feature_subset
        ``'all'`` (default), ``'top3'``, ``'topN'``, or ``'highQuality'``.
    top_n
        Number of features per protein to keep when ``feature_subset``
        is ``'top3'`` (forces ``top_n=3``) or ``'topN'`` (uses the
        provided ``top_n``).
    min_feature_count
        Used only by ``'highQuality'`` (not implemented — kept for API
        compatibility).
    protein_col, feature_col, abundance_col
        Column names.

    Returns
    -------
    pd.DataFrame
        Copy of ``data`` with a boolean ``remove`` column. Rows whose
        feature did NOT pass the filter have ``remove == True``.
    """
    out = data.copy()
    if feature_subset == "all":
        out["remove"] = False
        return out
    if feature_subset == "highQuality":
        warnings.warn(
            "feature_subset='highQuality' is not yet ported — falling back "
            "to 'all'. The R selectHighQualityFeatures helper is rarely used.",
            stacklevel=2,
        )
        out["remove"] = False
        return out
    if feature_subset == "top3":
        top_n = 3
    elif feature_subset != "topN":
        raise ValueError(f"unknown feature_subset={feature_subset!r}")

    mean_by_feature = (
        out.loc[pd.to_numeric(out[abundance_col], errors="coerce") > 0]
        .groupby([protein_col, feature_col], dropna=False)[abundance_col]
        .mean()
        .rename("MeanAbundance")
        .reset_index()
    )
    # rank ascending on the *negative* mean → ranks 1..N best to worst
    mean_by_feature["feature_rank"] = (
        mean_by_feature.groupby(protein_col)["MeanAbundance"]
        .rank(method="min", ascending=False)
    )
    kept = mean_by_feature.loc[mean_by_feature["feature_rank"] <= top_n,
                               [protein_col, feature_col]]
    kept_set = set(map(tuple, kept.values.tolist()))
    key = list(zip(out[protein_col], out[feature_col]))
    out["remove"] = [k not in kept_set for k in key]
    return out


__all__ = ["select_features"]
