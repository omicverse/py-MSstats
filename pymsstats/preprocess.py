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


def check_repeated_design(
    data: pd.DataFrame,
    *,
    group_col: str = "GROUP",
    subject_col: str = "SUBJECT",
) -> bool:
    """Detect a repeated-measures (time-course) design.

    Port of ``MSstats::checkRepeatedDesign``. The R function builds the
    ``SUBJECT × GROUP`` contingency table from the protein-level data and
    returns ``True`` if any subject appears in more than one group
    (a time-course / repeated-measures design); ``False`` for the
    standard case-control design where each subject is in one group.

    Parameters
    ----------
    data
        Protein-level long table (output of :func:`pymsstats.data_process`)
        or a feature-level table — anything with ``GROUP`` and ``SUBJECT``
        columns. The accessor also accepts a dict / object with a
        ``ProteinLevelData`` key (mirrors the R ``summarization_output``
        argument).
    group_col, subject_col
        Column names.

    Returns
    -------
    bool
        ``True`` if the design is repeated-measures.
    """
    if isinstance(data, dict) and "ProteinLevelData" in data:
        data = data["ProteinLevelData"]
    elif hasattr(data, "ProteinLevelData"):
        data = data.ProteinLevelData
    sub = data[[subject_col, group_col]].drop_duplicates()
    n_groups_per_subject = sub.groupby(subject_col)[group_col].nunique()
    return bool((n_groups_per_subject > 1).any())


def make_peptides_dictionary(
    data: pd.DataFrame,
    annotation: pd.DataFrame | None = None,
    *,
    protein_col: str = "ProteinName",
    peptide_col: str = "PeptideSequence",
) -> pd.DataFrame:
    """Build a peptide → protein lookup table.

    Port of ``MSstats::makePeptidesDictionary``. R uses this dictionary
    (peptide ↔ protein) to support normalization that needs to map
    peptides back to proteins (e.g. global-standards normalization).

    Parameters
    ----------
    data
        MSstats long-format table with at least the protein and peptide
        columns. The ``annotation`` argument is accepted for R-signature
        compatibility but not required.
    annotation
        Optional — ignored (R uses it only to attach run metadata).
    protein_col, peptide_col
        Column names. Both the vendor-style spellings (``ProteinName`` /
        ``PeptideSequence``) and the processed-table spellings
        (``PROTEIN`` / ``PEPTIDE``) are auto-detected.

    Returns
    -------
    pd.DataFrame
        Unique ``(PeptideSequence, ProteinName)`` rows.
    """
    df = data
    pcol = protein_col if protein_col in df.columns else (
        "PROTEIN" if "PROTEIN" in df.columns else protein_col
    )
    seqcol = peptide_col if peptide_col in df.columns else (
        "PEPTIDE" if "PEPTIDE" in df.columns else peptide_col
    )
    out = (
        df[[seqcol, pcol]]
        .drop_duplicates()
        .rename(columns={seqcol: "PeptideSequence", pcol: "ProteinName"})
        .reset_index(drop=True)
    )
    return out


__all__ = [
    "select_features",
    "check_repeated_design",
    "make_peptides_dictionary",
]
