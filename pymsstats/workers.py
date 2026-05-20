"""Modular worker API — ports of the MSstats 4.x "building block" functions.

MSstats 4.x exposes the internal steps of :func:`dataProcess` /
:func:`groupComparison` as separate public functions so power users can
swap individual stages. These are *not* new algorithms — they are the
exact steps the high-level functions already run, exposed individually:

* ``MSstatsPrepareForDataProcess`` — log-transform + zero/NA handling
  (:func:`prepare_for_data_process`).
* ``MSstatsPrepareForSummarization`` — feature selection + censored
  flagging (:func:`prepare_for_summarization`).
* ``MSstatsSummarize`` / ``MSstatsSummarizeWithSingleCore`` /
  ``MSstatsSummarizeSingleLinear`` / ``MSstatsSummarizeSingleTMP`` —
  per-protein peptide → protein summarization.
* ``MSstatsSummarizationOutput`` — assemble the protein-level table.
* ``MSstatsPrepareForGroupComparison`` — split a processed table into a
  per-protein list ready for comparison.
* ``MSstatsGroupComparison`` / ``MSstatsGroupComparisonSingleProtein`` /
  ``MSstatsGroupComparisonOutput`` — the per-protein Wald test and its
  output assembly.

This module re-exposes the existing :mod:`pymsstats` machinery under the
modular names with the same R-style semantics.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .group_comparison import (
    _bh,
    _detect_subject_design,
    _fit_one_protein_lm,
    _fit_one_protein_mixed,
    group_comparison,
)
from .imputation import msstats_handle_missing
from .preprocess import select_features
from .summarization import linear_summarize, msstats_summarize, tmp_summarize


# -----------------------------------------------------------------------------
# dataProcess building blocks
# -----------------------------------------------------------------------------
def prepare_for_data_process(
    data: pd.DataFrame,
    *,
    log_base: int = 2,
    fix_missing: Optional[str] = None,
) -> pd.DataFrame:
    """Prepare a raw feature table for summarization.

    Port of ``MSstatsPrepareForDataProcess``. Adds the internal MSstats
    columns: maps zero / negative / NA intensities to NaN, log-transforms
    the intensity, and attaches the ``PROTEIN, FEATURE, RUN, GROUP,
    SUBJECT, LABEL`` aliases used by the rest of the pipeline.

    Parameters
    ----------
    data
        MSstats long-format table (10 canonical columns).
    log_base
        Logarithm base (R default 2).
    fix_missing
        ``None`` (default) or ``'zero_to_na'`` / ``'na_to_zero'`` —
        the R ``fix_missing`` argument for vendor-specific missing-value
        conventions. ``'na_to_zero'`` turns NaN intensities into 0
        *before* the log step (so they become missing again).

    Returns
    -------
    pd.DataFrame
        Copy of ``data`` with ``INTENSITY, ABUNDANCE, PROTEIN, FEATURE,
        RUN, GROUP, SUBJECT, LABEL`` columns added.
    """
    df = data.copy()
    intensity = pd.to_numeric(df["Intensity"], errors="coerce")
    if fix_missing == "na_to_zero":
        intensity = intensity.fillna(0.0)
    elif fix_missing == "zero_to_na":
        intensity = intensity.where(intensity != 0, np.nan)
    intensity = intensity.where(intensity > 0, np.nan)
    df["INTENSITY"] = intensity
    with np.errstate(divide="ignore"):
        if log_base == 2:
            df["ABUNDANCE"] = np.log2(intensity)
        else:
            df["ABUNDANCE"] = np.log(intensity) / np.log(log_base)
    df["PROTEIN"] = df["ProteinName"].astype(str)
    df["RUN"] = df["Run"].astype(str)
    df["LABEL"] = df.get("IsotopeLabelType", "L").astype(str) \
        if "IsotopeLabelType" in df.columns else "L"
    df["GROUP"] = df["Condition"].astype(str)
    df["SUBJECT"] = df["BioReplicate"].astype(str)
    df["FEATURE"] = (
        df["PeptideSequence"].astype(str) + "_"
        + df["PrecursorCharge"].astype(str) + "_"
        + df["FragmentIon"].astype(str) + "_"
        + df["ProductCharge"].astype(str)
    )
    return df


def prepare_for_summarization(
    data: pd.DataFrame,
    *,
    method: str = "TMP",
    impute: bool = True,
    censored_symbol: Optional[str] = "NA",
    remove_uninformative_feature_outlier: bool = False,
) -> pd.DataFrame:
    """Prepare a log-transformed table for summarization.

    Port of ``MSstatsPrepareForSummarization``. Runs feature selection
    (optionally flagging uninformative features) and censored-value
    flagging — the two preprocessing steps that sit between
    normalization and summarization.

    Parameters
    ----------
    data
        Output of :func:`prepare_for_data_process` (must have
        ``PROTEIN, FEATURE, ABUNDANCE, INTENSITY`` columns).
    method
        Summarization method (``'TMP'`` or ``'linear'``) — only used to
        decide whether censored flagging applies.
    impute
        Whether censored values will be imputed downstream.
    censored_symbol
        ``'NA'`` (default) or ``'0'`` — the missing-value symbol.
    remove_uninformative_feature_outlier
        When True, run :func:`select_features` with the high-quality
        filter; otherwise keep all features.

    Returns
    -------
    pd.DataFrame
        Copy of ``data`` with a boolean ``remove`` column (feature
        selection) and a boolean ``censored`` column.
    """
    feature_subset = (
        "highQuality" if remove_uninformative_feature_outlier else "all"
    )
    out = select_features(data, feature_subset=feature_subset)
    out = msstats_handle_missing(
        out,
        summary_method=method,
        impute=impute,
        missing_symbol=censored_symbol,
    )
    return out


def summarize_single_tmp(
    single_protein: pd.DataFrame,
    *,
    impute: bool = True,
    censored_symbol: Optional[str] = "NA",
    remove50missing: bool = False,
    protein_col: str = "PROTEIN",
    run_col: str = "RUN",
    feature_col: str = "FEATURE",
    abundance_col: str = "ABUNDANCE",
) -> pd.DataFrame:
    """TMP-summarize a single protein.

    Port of ``MSstatsSummarizeSingleTMP``. Runs Tukey median polish on
    one protein's feature × run matrix.

    Parameters
    ----------
    single_protein
        Feature-level long table for ONE protein.
    impute, censored_symbol, remove50missing
        Accepted for R-signature compatibility (the censored handling is
        applied upstream by :func:`prepare_for_summarization`).
    protein_col, run_col, feature_col, abundance_col
        Column names.

    Returns
    -------
    pd.DataFrame
        Per-run summary (``Protein, RUN, LogIntensities, n_features,
        n_obs``).
    """
    return tmp_summarize(
        single_protein,
        protein_col=protein_col, run_col=run_col,
        feature_col=feature_col, abundance_col=abundance_col,
    )


def summarize_single_linear(
    single_protein: pd.DataFrame,
    *,
    equal_variances: bool = True,
    protein_col: str = "PROTEIN",
    run_col: str = "RUN",
    feature_col: str = "FEATURE",
    abundance_col: str = "ABUNDANCE",
) -> pd.DataFrame:
    """Linear-model summarize a single protein.

    Port of ``MSstatsSummarizeSingleLinear``. Fits ``ABUNDANCE ~ RUN +
    FEATURE`` for one protein and reports per-run estimates.

    Returns the same schema as :func:`summarize_single_tmp`.
    """
    return linear_summarize(
        single_protein,
        protein_col=protein_col, run_col=run_col,
        feature_col=feature_col, abundance_col=abundance_col,
        equal_variance=equal_variances,
    )


def summarize_single_core(
    data: pd.DataFrame,
    *,
    method: str = "TMP",
    impute: bool = True,
    censored_symbol: Optional[str] = "NA",
    remove50missing: bool = False,
    equal_variance: bool = True,
    protein_col: str = "PROTEIN",
    run_col: str = "RUN",
    feature_col: str = "FEATURE",
    abundance_col: str = "ABUNDANCE",
) -> pd.DataFrame:
    """Summarize every protein on a single core.

    Port of ``MSstatsSummarizeWithSingleCore``. Loops over proteins and
    summarizes each with the chosen method. (pymsstats is already
    single-threaded; the parallel ``numberOfCores`` path of R is not
    needed.)

    Returns
    -------
    pd.DataFrame
        Per-(protein, run) summary table.
    """
    return msstats_summarize(
        data, method=method,
        protein_col=protein_col, run_col=run_col,
        feature_col=feature_col, abundance_col=abundance_col,
        equal_variance=equal_variance,
    )


def msstats_summarize_modular(
    data: pd.DataFrame,
    *,
    method: str = "TMP",
    impute: bool = True,
    censored_symbol: Optional[str] = "NA",
    remove50missing: bool = False,
    equal_variance: bool = True,
    protein_col: str = "PROTEIN",
    run_col: str = "RUN",
    feature_col: str = "FEATURE",
    abundance_col: str = "ABUNDANCE",
) -> pd.DataFrame:
    """Summarize peptide-level data to protein level.

    Port of ``MSstatsSummarize`` (the modular dispatch). Identical to
    :func:`summarize_single_core` — kept as a separate name to match the
    R API exactly.
    """
    return summarize_single_core(
        data, method=method, impute=impute, censored_symbol=censored_symbol,
        remove50missing=remove50missing, equal_variance=equal_variance,
        protein_col=protein_col, run_col=run_col,
        feature_col=feature_col, abundance_col=abundance_col,
    )


def summarization_output(
    data: pd.DataFrame,
    summarized: pd.DataFrame,
    processed: Optional[pd.DataFrame] = None,
    *,
    method: str = "TMP",
    impute: bool = True,
    censored_symbol: Optional[str] = "NA",
    run_col: str = "RUN",
) -> dict:
    """Assemble the summarization output object.

    Port of ``MSstatsSummarizationOutput``. R returns a list with
    ``FeatureLevelData`` and ``ProteinLevelData`` slots; this Python port
    returns the equivalent dict, attaching the per-run ``GROUP`` /
    ``SUBJECT`` metadata to the protein-level table.

    Parameters
    ----------
    data
        The feature-level table fed into summarization (must carry the
        ``RUN, GROUP, SUBJECT`` aliases).
    summarized
        Output of :func:`msstats_summarize_modular`.
    processed
        Optional processed feature table to keep alongside the result.
    method, impute, censored_symbol
        Accepted for R-signature compatibility.
    run_col
        Run column name in ``data``.

    Returns
    -------
    dict
        ``{'FeatureLevelData': ..., 'ProteinLevelData': ...}``.
    """
    out = summarized.copy()
    if {"GROUP", "SUBJECT"}.issubset(data.columns):
        annot = (
            data[[run_col, "GROUP", "SUBJECT"]]
            .drop_duplicates(subset=[run_col])
            .set_index(run_col)
        )
        if "GROUP" not in out.columns:
            out["GROUP"] = out["RUN"].map(annot["GROUP"])
        if "SUBJECT" not in out.columns:
            out["SUBJECT"] = out["RUN"].map(annot["SUBJECT"])
    return {
        "FeatureLevelData": data if processed is None else processed,
        "ProteinLevelData": out,
    }


# -----------------------------------------------------------------------------
# groupComparison building blocks
# -----------------------------------------------------------------------------
def prepare_for_group_comparison(
    summarization_output,
    *,
    protein_col: str = "Protein",
) -> dict:
    """Split a protein-level table into a per-protein dict.

    Port of ``MSstatsPrepareForGroupComparison``. R turns the
    summarization output into a list with one element per protein, ready
    for the per-protein Wald test.

    Parameters
    ----------
    summarization_output
        Either the dict returned by :func:`summarization_output`, or a
        protein-level DataFrame directly (output of
        :func:`pymsstats.data_process`).
    protein_col
        Protein column name.

    Returns
    -------
    dict
        ``{protein_name: per_protein_DataFrame}``.
    """
    if isinstance(summarization_output, dict):
        df = summarization_output["ProteinLevelData"]
    elif hasattr(summarization_output, "ProteinLevelData"):
        df = summarization_output.ProteinLevelData
    else:
        df = summarization_output
    return {
        str(prot): sub.reset_index(drop=True)
        for prot, sub in df.groupby(protein_col, sort=False)
    }


def group_comparison_single_protein(
    protein_data: pd.DataFrame,
    contrast,
    *,
    groups: Optional[Sequence[str]] = None,
    repeated: Optional[bool] = None,
    abundance_col: str = "LogIntensities",
    group_col: str = "GROUP",
    subject_col: str = "SUBJECT",
    run_col: str = "RUN",
) -> pd.DataFrame:
    """Run the Wald test for a single protein.

    Port of ``MSstatsGroupComparisonSingleProtein``. Fits the per-protein
    LM / LMM and tests every row of the contrast matrix.

    Parameters
    ----------
    protein_data
        Protein-level long table for ONE protein (``LogIntensities,
        GROUP, SUBJECT, RUN``).
    contrast
        Contrast matrix — a ``(n_contrasts, n_groups)`` array or labelled
        DataFrame.
    groups
        Ordered group names matching the contrast columns. Inferred from
        the contrast DataFrame columns or from the data if None.
    repeated
        ``None`` (auto-detect), ``True`` (mixed model), or ``False``
        (fixed effects). Mirrors the R ``repeated`` argument.
    abundance_col, group_col, subject_col, run_col
        Column names.

    Returns
    -------
    pd.DataFrame
        Columns ``Protein, Label, log2FC, SE, df, t, pvalue, issue`` —
        one row per contrast (no adj.pvalue: that is computed across
        proteins by :func:`group_comparison_output`).
    """
    df = protein_data.copy()
    df["GROUP"] = df[group_col].astype(str)
    df["ABUNDANCE"] = df[abundance_col].astype(float)
    if subject_col in df.columns:
        df["SUBJECT"] = df[subject_col].astype(str)
    else:
        df["SUBJECT"] = df[run_col].astype(str)

    if isinstance(contrast, pd.DataFrame):
        contrast_groups = list(contrast.columns)
        labels = list(contrast.index.astype(str))
        C = contrast.to_numpy(dtype=float)
    else:
        C = np.atleast_2d(np.asarray(contrast, dtype=float))
        contrast_groups = (
            list(groups) if groups is not None
            else sorted(df["GROUP"].unique())
        )
        labels = [f"contrast_{i}" for i in range(C.shape[0])]

    if repeated is None:
        mode = _detect_subject_design(df)
    elif repeated:
        mode = "mixed"
    else:
        mode = "lm"
    fit_fn = _fit_one_protein_mixed if mode == "mixed" else _fit_one_protein_lm

    prot_name = (
        str(df["Protein"].iloc[0]) if "Protein" in df.columns and len(df)
        else "protein"
    )
    rows = []
    for label, c_row in zip(labels, C):
        r = fit_fn(df, contrast_groups, c_row)
        rows.append({
            "Protein": prot_name, "Label": label,
            "log2FC": r["log2FC"], "SE": r["SE"], "df": r["df"],
            "t": r["t"], "pvalue": r["pvalue"], "issue": r["issue"],
        })
    return pd.DataFrame(rows)


def msstats_group_comparison(
    prepared,
    contrast,
    *,
    groups: Optional[Sequence[str]] = None,
    repeated: Optional[bool] = None,
    save_fitted_models: bool = False,
    samples_info: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Run the Wald test for every protein.

    Port of ``MSstatsGroupComparison``. Loops over the per-protein dict
    (from :func:`prepare_for_group_comparison`) and tests each protein.

    Parameters
    ----------
    prepared
        Either the per-protein dict from
        :func:`prepare_for_group_comparison`, or a protein-level
        DataFrame (which is split automatically).
    contrast
        Contrast matrix (array or labelled DataFrame).
    groups, repeated
        See :func:`group_comparison_single_protein`.
    save_fitted_models, samples_info
        Accepted for R-signature compatibility (unused — pymsstats does
        not retain fitted-model objects).

    Returns
    -------
    pd.DataFrame
        Per-(protein, contrast) results WITHOUT adj.pvalue. Pass the
        result to :func:`group_comparison_output` for the BH correction.
    """
    if not isinstance(prepared, dict):
        prepared = prepare_for_group_comparison(prepared)
    parts = [
        group_comparison_single_protein(
            sub, contrast, groups=groups, repeated=repeated,
        )
        for sub in prepared.values()
    ]
    if not parts:
        return pd.DataFrame(
            columns=["Protein", "Label", "log2FC", "SE", "df",
                     "t", "pvalue", "issue"]
        )
    return pd.concat(parts, ignore_index=True)


def group_comparison_output(
    input: pd.DataFrame,
    summarization_output=None,
    *,
    log_base: int = 2,
) -> pd.DataFrame:
    """Assemble the final group-comparison table.

    Port of ``MSstatsGroupComparisonOutput``. Adds the Benjamini-Hochberg
    ``adj.pvalue`` (computed per contrast across all proteins) to the raw
    per-protein test results.

    Parameters
    ----------
    input
        Output of :func:`msstats_group_comparison` (per-protein results).
    summarization_output
        Accepted for R-signature compatibility (unused).
    log_base
        Accepted for R-signature compatibility.

    Returns
    -------
    pd.DataFrame
        Columns ``Protein, Label, log2FC, SE, df, t, pvalue,
        adj.pvalue, issue``.
    """
    res = input.copy()
    res["adj.pvalue"] = np.nan
    for label, idx in res.groupby("Label").groups.items():
        res.loc[idx, "adj.pvalue"] = _bh(res.loc[idx, "pvalue"].to_numpy())
    return res[["Protein", "Label", "log2FC", "SE", "df", "t",
                "pvalue", "adj.pvalue", "issue"]]


__all__ = [
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
]
