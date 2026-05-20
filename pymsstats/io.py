"""Vendor → MSstats long-format converters.

Ports of ``MaxQtoMSstatsFormat``, ``DIANNtoMSstatsFormat``,
``SpectronauttoMSstatsFormat``, ``FragPipetoMSstatsFormat``,
``OpenMStoMSstatsFormat``, ``SkylinetoMSstatsFormat`` plus a couple of
small annotation helpers.

The R converters live across ``MSstats`` + ``MSstatsConvert`` and call a
nested helper chain (``MSstatsImport`` → vendor ``clean`` → ``MSstatsClean``
→ ``MSstatsPreprocess`` → ``MSstatsBalancedDesign``). The Python ports here
collapse that into a single pandas pipeline:

1. select the per-vendor relevant columns and rename to MSstats names;
2. apply per-vendor filters (Q-value cutoffs, decoys, NH3/H2O fragments,
   truncated peaks, Spectronaut PG-Q, …);
3. drop shared peptides (``useUniquePeptide=True``) by default;
4. aggregate multi-PSM features (``summaryforMultipleRows``, R default
   ``max`` for DIA-NN/Spectronaut, ``sum`` for Skyline);
5. drop features below the ``min_obs`` measurement count, and proteins
   with a single feature when requested;
6. attach the user-supplied annotation table (``Run, Condition,
   BioReplicate``) so every row has a Condition/BioReplicate.
"""
from __future__ import annotations

from typing import Callable, Iterable, Optional

import numpy as np
import pandas as pd


CANONICAL_COLS = [
    "ProteinName", "PeptideSequence", "PrecursorCharge",
    "FragmentIon", "ProductCharge", "IsotopeLabelType",
    "Condition", "BioReplicate", "Run", "Intensity",
]


# -----------------------------------------------------------------------------
# Annotation helpers
# -----------------------------------------------------------------------------
def validate_annotation(
    annotation: pd.DataFrame,
    raw_df: Optional[pd.DataFrame] = None,
    *,
    required: Iterable[str] = ("Run", "Condition", "BioReplicate"),
) -> pd.DataFrame:
    """Validate an MSstats annotation table.

    R counterpart: ``MSstats::validateAnnotation``.

    Checks that the table has the required columns, the Conditions are
    not all the same (would make differential testing impossible), and
    that each BioReplicate is assigned to exactly one Condition
    (the standard group-comparison design).

    Parameters
    ----------
    annotation
        Long table with at least ``Run, Condition, BioReplicate``.
    raw_df
        Optional vendor input — if supplied, also check that every Run
        in ``raw_df`` has a matching annotation row.
    required
        Required column names. Defaults to ``('Run', 'Condition',
        'BioReplicate')``.

    Returns
    -------
    pd.DataFrame
        A *copy* of ``annotation`` with the required columns cast to str.
    """
    missing = [c for c in required if c not in annotation.columns]
    if missing:
        raise ValueError(
            f"annotation is missing required columns: {missing}\n"
            f"Expected at least: {list(required)}"
        )
    ann = annotation.copy()
    for c in required:
        ann[c] = ann[c].astype(str)
    if ann["Condition"].nunique() < 2:
        raise ValueError(
            "annotation has fewer than 2 distinct Conditions — "
            "MSstats requires at least two conditions for differential testing."
        )
    # Each BioReplicate → at most one Condition (group-comparison design)
    n_cond_per_biorep = ann.groupby("BioReplicate")["Condition"].nunique()
    if (n_cond_per_biorep > 1).any():
        bad = n_cond_per_biorep[n_cond_per_biorep > 1].index.tolist()
        raise ValueError(
            "In group-comparison design each BioReplicate must map to "
            f"exactly one Condition. Offending BioReplicates: {bad}"
        )
    if raw_df is not None and "Run" in raw_df.columns:
        miss_runs = set(raw_df["Run"].astype(str)) - set(ann["Run"])
        if miss_runs:
            raise ValueError(
                f"runs in raw data missing from annotation: {sorted(miss_runs)}"
            )
    return ann


def merge_fractions(
    df: pd.DataFrame,
    *,
    fraction_col: str = "Fraction",
    run_col: str = "Run",
    feature_cols: Iterable[str] = (
        "ProteinName", "PeptideSequence", "PrecursorCharge",
        "FragmentIon", "ProductCharge", "IsotopeLabelType",
    ),
    intensity_col: str = "Intensity",
    annot_cols: Iterable[str] = ("Condition", "BioReplicate"),
) -> pd.DataFrame:
    """Sum intensities across fractions of the same (condition, bioreplicate).

    R counterpart: ``MSstats::MSstatsMergeFractions``.

    For a multi-fraction design each ``(Condition, BioReplicate)`` is run
    in several fractions; the canonical MSstats step is to sum the
    feature intensity across all fractions of a sample. The resulting
    "Run" identifier becomes ``Condition_BioReplicate_merged``.

    If the input table has no ``Fraction`` column or only one unique
    fraction, the table is returned unchanged.

    Parameters
    ----------
    df
        MSstats long-format table.
    fraction_col, run_col, feature_cols, intensity_col, annot_cols
        Column names.

    Returns
    -------
    pd.DataFrame
        Long-format table with one row per (Condition, BioReplicate, feature).
    """
    if fraction_col not in df.columns:
        return df.copy()
    if df[fraction_col].nunique() <= 1:
        return df.copy()
    out = df.copy()
    out[run_col] = (
        out["Condition"].astype(str) + "_"
        + out["BioReplicate"].astype(str) + "_merged"
    )
    group_cols = list(feature_cols) + [run_col] + list(annot_cols)
    out = (
        out.groupby(group_cols, dropna=False, as_index=False)[intensity_col]
        .sum(min_count=1)
    )
    return out[[*feature_cols, *annot_cols, run_col, intensity_col]]


# -----------------------------------------------------------------------------
# Generic vendor-pipeline scaffolding
# -----------------------------------------------------------------------------
def _drop_shared_peptides(
    df: pd.DataFrame,
    *,
    peptide_cols: Iterable[str] = ("PeptideSequence", "PrecursorCharge"),
    protein_col: str = "ProteinName",
) -> pd.DataFrame:
    """Remove peptides that map to more than one protein."""
    grp = (
        df.groupby(list(peptide_cols), dropna=False)[protein_col]
        .nunique()
    )
    shared = grp[grp > 1].index
    if len(shared) == 0:
        return df
    mask = ~df.set_index(list(peptide_cols)).index.isin(shared)
    return df.loc[mask].reset_index(drop=True)


def _drop_single_feature_proteins(
    df: pd.DataFrame,
    *,
    feature_cols: Iterable[str] = (
        "PeptideSequence", "PrecursorCharge", "FragmentIon", "ProductCharge",
    ),
    protein_col: str = "ProteinName",
) -> pd.DataFrame:
    """Drop proteins quantified by a single (peptide_, charge, fragment, _)."""
    feat = (
        df[[protein_col, *feature_cols]].drop_duplicates()
    )
    n_feat = feat.groupby(protein_col).size()
    keep = n_feat[n_feat >= 2].index
    return df.loc[df[protein_col].isin(keep)].reset_index(drop=True)


def _drop_few_measurements(
    df: pd.DataFrame,
    *,
    feature_cols: Iterable[str] = (
        "PeptideSequence", "PrecursorCharge", "FragmentIon", "ProductCharge",
    ),
    protein_col: str = "ProteinName",
    intensity_col: str = "Intensity",
    min_obs: int = 2,
) -> pd.DataFrame:
    """Drop features observed (non-NaN, > 0) in fewer than ``min_obs`` runs."""
    has_int = df[intensity_col].notna() & (
        pd.to_numeric(df[intensity_col], errors="coerce") > 0
    )
    n_obs = (
        df.loc[has_int].groupby([protein_col, *feature_cols], dropna=False)
        .size()
    )
    keep_idx = set(n_obs[n_obs >= min_obs].index)
    key = list(zip(*[df[c] for c in [protein_col, *feature_cols]]))
    mask = [k in keep_idx for k in key]
    return df.loc[mask].reset_index(drop=True)


def _aggregate_psms(
    df: pd.DataFrame,
    *,
    feature_cols: Iterable[str] = (
        "ProteinName", "PeptideSequence", "PrecursorCharge",
        "FragmentIon", "ProductCharge", "IsotopeLabelType",
    ),
    run_col: str = "Run",
    intensity_col: str = "Intensity",
    summary_fn: str | Callable = "max",
    annot_cols: Iterable[str] = ("Condition", "BioReplicate"),
) -> pd.DataFrame:
    """Aggregate multi-PSM rows of the same feature/run.

    Mirrors ``MSstatsConvert::MSstatsPreprocess`` step
    ``summarize_multiple_psms`` — ``max`` for DIA-NN / Spectronaut /
    FragPipe / OpenMS, ``sum`` for Skyline. Custom callables (e.g.
    ``np.nansum``) are passed through.
    """
    group_cols = [*feature_cols, run_col]
    extra = [c for c in annot_cols if c in df.columns and c not in group_cols]
    group_cols = group_cols + extra
    if isinstance(summary_fn, str):
        if summary_fn == "max":
            fn: Callable = np.nanmax
        elif summary_fn == "sum":
            fn = np.nansum
        elif summary_fn == "mean":
            fn = np.nanmean
        elif summary_fn == "median":
            fn = np.nanmedian
        else:
            raise ValueError(f"unknown summary_fn={summary_fn!r}")
    else:
        fn = summary_fn
    df = df.copy()
    df[intensity_col] = pd.to_numeric(df[intensity_col], errors="coerce")
    out = (
        df.groupby(group_cols, dropna=False, as_index=False)[intensity_col]
        .agg(lambda x: fn(x.values) if x.notna().any() else np.nan)
    )
    return out


def _finalize_msstats_long(
    df: pd.DataFrame,
    *,
    annotation: Optional[pd.DataFrame] = None,
    isotope_label: str = "L",
) -> pd.DataFrame:
    """Final pass: enforce CANONICAL_COLS order, fill defaults, attach annotation."""
    out = df.copy()
    for col, default in (
        ("FragmentIon", "NA"),
        ("ProductCharge", "NA"),
        ("IsotopeLabelType", isotope_label),
    ):
        if col not in out.columns:
            out[col] = default
    if annotation is not None:
        ann = annotation.copy()
        ann["Run"] = ann["Run"].astype(str)
        out["Run"] = out["Run"].astype(str)
        drop_cols = [c for c in ("Condition", "BioReplicate")
                     if c in out.columns and c in ann.columns]
        if drop_cols:
            out = out.drop(columns=drop_cols)
        out = out.merge(ann, on="Run", how="left")
        if out["Condition"].isna().any():
            missing = out.loc[out["Condition"].isna(), "Run"].unique().tolist()
            raise ValueError(f"runs missing annotation rows: {missing}")
    for col in CANONICAL_COLS:
        if col not in out.columns:
            out[col] = "NA" if col in (
                "FragmentIon", "ProductCharge",
            ) else np.nan
    return out[CANONICAL_COLS].reset_index(drop=True)


# -----------------------------------------------------------------------------
# DIA-NN
# -----------------------------------------------------------------------------
DIANN_FRAGMENT_QUANT_COL_DEFAULT = "Fragment.Quant.Corrected"


def diann_to_msstats(
    report: pd.DataFrame,
    annotation: Optional[pd.DataFrame] = None,
    *,
    global_qvalue_cutoff: float = 0.01,
    qvalue_cutoff: float = 0.01,
    pg_qvalue_cutoff: float = 0.01,
    use_unique_peptide: bool = True,
    remove_few_measurements: bool = True,
    remove_oxidation_m_peptides: bool = True,
    remove_protein_with_1_feature: bool = True,
    mbr: bool = True,
    quantification_column: str = DIANN_FRAGMENT_QUANT_COL_DEFAULT,
) -> pd.DataFrame:
    """DIA-NN ``report.tsv`` → MSstats long-format.

    Port of ``DIANNtoMSstatsFormat``.

    Parameters
    ----------
    report
        DIA-NN ``report.tsv`` parsed as a DataFrame. Expected columns:
        ``Protein.Names``, ``Stripped.Sequence``, ``Modified.Sequence``,
        ``Precursor.Charge``, ``Q.Value``, ``Run`` plus the quantification
        column (default ``Fragment.Quant.Corrected``) and either
        (``Lib.Q.Value``, ``Lib.PG.Q.Value``) for MBR mode or
        (``Global.Q.Value``, ``Global.PG.Q.Value``) otherwise.
        Some report column names ship with dots (``.``), some with no
        separator (``ProteinNames``); both spellings are accepted.
    annotation
        Optional table (``Run, Condition, BioReplicate``). When None, a
        single-condition dummy annotation is built (use only for tests).
    global_qvalue_cutoff, qvalue_cutoff, pg_qvalue_cutoff
        Q-value thresholds. The MBR-mode default applies the latter two
        to ``Lib.Q.Value`` / ``Lib.PG.Q.Value``; non-MBR to ``Global.*``.
    use_unique_peptide, remove_few_measurements,
    remove_oxidation_m_peptides, remove_protein_with_1_feature, mbr
        See R counterpart.
    quantification_column
        Which DIA-NN column to use as intensity (default
        ``Fragment.Quant.Corrected``; ``Precursor.Quantity`` is common too).

    Returns
    -------
    pd.DataFrame
        MSstats long-format (10 canonical columns).
    """
    df = report.copy()
    # accept both dotted and non-dotted column spellings
    rename_map = {
        "Protein.Names": "ProteinNames",
        "Stripped.Sequence": "StrippedSequence",
        "Modified.Sequence": "ModifiedSequence",
        "Precursor.Charge": "PrecursorCharge",
        "Q.Value": "QValue",
        "Global.Q.Value": "GlobalQValue",
        "Global.PG.Q.Value": "GlobalPGQValue",
        "Lib.Q.Value": "LibQValue",
        "Lib.PG.Q.Value": "LibPGQValue",
        "Fragment.Quant.Corrected": "FragmentQuantCorrected",
        "Fragment.Quant.Raw": "FragmentQuantRaw",
        "Precursor.Quantity": "PrecursorQuantity",
        "PG.Quantity": "PGQuantity",
        "Fragment.Info": "FragmentInfo",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    # the internal quant column name
    qcol = quantification_column.replace(".", "")
    if qcol not in df.columns and quantification_column in df.columns:
        qcol = quantification_column
    # Filter on global Q
    if "QValue" in df.columns:
        df["QValue"] = pd.to_numeric(df["QValue"], errors="coerce")
        df = df.loc[df["QValue"] < global_qvalue_cutoff]
    if mbr:
        for col, thr in (("LibPGQValue", pg_qvalue_cutoff),
                         ("LibQValue", qvalue_cutoff)):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                df = df.loc[df[col] < thr]
    else:
        for col, thr in (("GlobalPGQValue", pg_qvalue_cutoff),
                         ("GlobalQValue", qvalue_cutoff)):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                df = df.loc[df[col] < thr]
    # decoys
    if "ProteinNames" in df.columns:
        df = df.loc[~df["ProteinNames"].astype(str).str.contains("DECOY|Decoys",
                                                                 regex=True)]
    # Oxidation-M peptides
    if remove_oxidation_m_peptides and "ModifiedSequence" in df.columns:
        df = df.loc[~df["ModifiedSequence"].astype(str).str.contains(
            r"\(UniMod:35\)|\(ox\)", regex=True)]
    # Build per-fragment rows from semicolon-separated lists if present
    if qcol in df.columns and df[qcol].astype(str).str.contains(";").any():
        df = df.assign(
            **{qcol: df[qcol].astype(str).str.split(";")}
        ).explode(qcol)
        if "FragmentInfo" in df.columns:
            df["FragmentInfo"] = df["FragmentInfo"].astype(str)
    df = df.rename(columns={
        "ProteinNames": "ProteinName",
        "ModifiedSequence": "PeptideSequence",
        qcol: "Intensity",
        "Run": "Run",
    })
    if "PeptideSequence" not in df.columns and "StrippedSequence" in df.columns:
        df["PeptideSequence"] = df["StrippedSequence"]
    if "FragmentInfo" in df.columns:
        df["FragmentIon"] = (
            df["FragmentInfo"].astype(str).str.replace(r"\^.*$", "", regex=True)
        )
    else:
        df["FragmentIon"] = "NA"
    df["ProductCharge"] = 1
    df["IsotopeLabelType"] = "L"
    df["Intensity"] = pd.to_numeric(df["Intensity"], errors="coerce")
    df = df.loc[df["Intensity"].notna()]
    # Skip NH3 / H2O fragments
    if "FragmentIon" in df.columns:
        df = df.loc[~df["FragmentIon"].astype(str).str.contains("NH3|H2O")]
    # required final columns
    keep = [c for c in (
        "ProteinName", "PeptideSequence", "PrecursorCharge",
        "FragmentIon", "ProductCharge", "IsotopeLabelType",
        "Run", "Intensity",
    ) if c in df.columns]
    df = df[keep].copy()
    df["PrecursorCharge"] = pd.to_numeric(
        df["PrecursorCharge"], errors="coerce").astype("Int64")

    if use_unique_peptide:
        df = _drop_shared_peptides(df)
    df = _aggregate_psms(
        df, feature_cols=("ProteinName", "PeptideSequence", "PrecursorCharge",
                          "FragmentIon", "ProductCharge", "IsotopeLabelType"),
        run_col="Run", summary_fn="max", annot_cols=(),
    )
    if remove_few_measurements:
        df = _drop_few_measurements(df, min_obs=2)
    if remove_protein_with_1_feature:
        df = _drop_single_feature_proteins(df)
    return _finalize_msstats_long(df, annotation=annotation)


# -----------------------------------------------------------------------------
# Spectronaut
# -----------------------------------------------------------------------------
def spectronaut_to_msstats(
    report: pd.DataFrame,
    annotation: Optional[pd.DataFrame] = None,
    *,
    intensity: str = "PeakArea",
    filter_with_qvalue: bool = False,
    qvalue_cutoff: float = 0.01,
    pg_qvalue_cutoff: float = 0.01,
    use_unique_peptide: bool = True,
    remove_few_measurements: bool = True,
    remove_protein_with_1_feature: bool = False,
    summary_for_multiple_rows: str = "max",
) -> pd.DataFrame:
    """Spectronaut report → MSstats long-format.

    Port of ``SpectronauttoMSstatsFormat``. Expected columns
    (Spectronaut "MSstats" export schema): ``PG.ProteinGroups``,
    ``EG.ModifiedSequence``, ``FG.Charge``, ``F.FrgIon``,
    (``F.Charge`` or ``F.FrgZ``), ``R.FileName``, ``R.Condition``,
    ``R.Replicate``, ``EG.Qvalue``, ``PG.Qvalue``, ``F.PeakArea``
    (or ``F.NormalizedPeakArea``), plus optional
    ``F.FrgLossType`` / ``F.ExcludedFromQuantification``.

    Both dotted (``PG.ProteinGroups``) and undotted (``PGProteinGroups``)
    column spellings are accepted.
    """
    df = report.copy()
    # rename to internal R-style (undotted)
    rename_map = {
        "PG.ProteinGroups":   "PGProteinGroups",
        "EG.ModifiedSequence": "EGModifiedSequence",
        "FG.Charge":          "FGCharge",
        "F.FrgIon":           "FFrgIon",
        "F.Charge":           "FCharge",
        "F.FrgZ":             "FFrgZ",
        "R.FileName":         "RFileName",
        "R.Condition":        "RCondition",
        "R.Replicate":        "RReplicate",
        "EG.Qvalue":          "EGQvalue",
        "PG.Qvalue":          "PGQvalue",
        "F.FrgLossType":      "FFrgLossType",
        "F.ExcludedFromQuantification": "FExcludedFromQuantification",
        f"F.{intensity}":     f"F{intensity}",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Spectronaut loss/exclusion filters
    if "FFrgLossType" in df.columns:
        df = df.loc[df["FFrgLossType"].astype(str) == "noloss"]
    if "FExcludedFromQuantification" in df.columns:
        col = df["FExcludedFromQuantification"]
        if col.dtype == bool:
            df = df.loc[~col]
        else:
            df = df.loc[col.astype(str).str.lower().isin({"false", "0", ""})]

    # Q-value filters
    if filter_with_qvalue:
        if "EGQvalue" in df.columns:
            df["EGQvalue"] = pd.to_numeric(df["EGQvalue"], errors="coerce")
            df = df.loc[df["EGQvalue"] < qvalue_cutoff]
        if "PGQvalue" in df.columns:
            df["PGQvalue"] = pd.to_numeric(df["PGQvalue"], errors="coerce")
            df = df.loc[df["PGQvalue"] < pg_qvalue_cutoff]

    # decoys
    if "PGProteinGroups" in df.columns:
        df = df.loc[~df["PGProteinGroups"].astype(str).str.contains(
            "DECOY|Decoys", regex=True)]

    f_charge_col = ("FCharge" if "FCharge" in df.columns
                    else ("FFrgZ" if "FFrgZ" in df.columns else None))
    intensity_col = f"F{intensity}"
    df = df.rename(columns={
        "PGProteinGroups":  "ProteinName",
        "EGModifiedSequence": "PeptideSequence",
        "FGCharge":         "PrecursorCharge",
        "FFrgIon":          "FragmentIon",
        f_charge_col or "FCharge": "ProductCharge",
        "RFileName":        "Run",
        intensity_col:      "Intensity",
        "RCondition":       "Condition",
        "RReplicate":       "BioReplicate",
    })
    df["IsotopeLabelType"] = "L"
    df["Intensity"] = pd.to_numeric(df["Intensity"], errors="coerce")
    # zero → NA (R rule)
    df.loc[df["Intensity"] == 0, "Intensity"] = np.nan
    keep = [c for c in (
        "ProteinName", "PeptideSequence", "PrecursorCharge",
        "FragmentIon", "ProductCharge", "IsotopeLabelType",
        "Condition", "BioReplicate", "Run", "Intensity",
    ) if c in df.columns]
    df = df[keep].copy()

    if use_unique_peptide:
        df = _drop_shared_peptides(df)
    df = _aggregate_psms(df, summary_fn=summary_for_multiple_rows)
    if remove_few_measurements:
        df = _drop_few_measurements(df, min_obs=2)
    if remove_protein_with_1_feature:
        df = _drop_single_feature_proteins(df)
    return _finalize_msstats_long(df, annotation=annotation)


# -----------------------------------------------------------------------------
# FragPipe — input is already MSstats long-format (just light cleaning)
# -----------------------------------------------------------------------------
def fragpipe_to_msstats(
    msstats_input: pd.DataFrame,
    annotation: Optional[pd.DataFrame] = None,
    *,
    use_unique_peptide: bool = True,
    remove_few_measurements: bool = True,
    remove_protein_with_1_feature: bool = False,
    summary_for_multiple_rows: str = "max",
) -> pd.DataFrame:
    """FragPipe ``msstats.csv`` (Philosopher MSstats export) → long-format.

    Port of ``FragPipetoMSstatsFormat``. FragPipe writes the file in the
    canonical MSstats layout already; the converter mostly:

    * drops shared peptides (when ``useUniquePeptide``);
    * aggregates multi-PSM rows of the same feature/run via ``max``;
    * drops single-feature proteins and low-measurement features.
    """
    df = msstats_input.copy()
    if "IsotopeLabelType" not in df.columns:
        df["IsotopeLabelType"] = "L"
    if "Intensity" in df.columns:
        df["Intensity"] = pd.to_numeric(df["Intensity"], errors="coerce")
        df.loc[df["Intensity"] == 0, "Intensity"] = np.nan
    if use_unique_peptide:
        df = _drop_shared_peptides(df)
    df = _aggregate_psms(df, summary_fn=summary_for_multiple_rows)
    if remove_few_measurements:
        df = _drop_few_measurements(df, min_obs=2)
    if remove_protein_with_1_feature:
        df = _drop_single_feature_proteins(df)
    return _finalize_msstats_long(df, annotation=annotation)


# -----------------------------------------------------------------------------
# OpenMS — input is already MSstats long-format
# -----------------------------------------------------------------------------
def openms_to_msstats(
    msstats_input: pd.DataFrame,
    annotation: Optional[pd.DataFrame] = None,
    *,
    use_unique_peptide: bool = True,
    remove_few_measurements: bool = True,
    remove_protein_with_1_feature: bool = False,
    summary_for_multiple_rows: str = "max",
) -> pd.DataFrame:
    """OpenMS MSstats export → long-format.

    Port of ``OpenMStoMSstatsFormat``. Accepts the canonical MSstats CSV
    written by OpenMS's MSstatsConverter (already MSstats long-format).
    """
    df = msstats_input.copy()
    if "Charge" in df.columns and "PrecursorCharge" not in df.columns:
        df = df.rename(columns={"Charge": "PrecursorCharge"})
    if "IsotopeLabelType" not in df.columns:
        df["IsotopeLabelType"] = "L"
    if "Intensity" in df.columns:
        df["Intensity"] = pd.to_numeric(df["Intensity"], errors="coerce")
        df.loc[df["Intensity"] == 0, "Intensity"] = np.nan
    df = df.drop_duplicates()
    if use_unique_peptide:
        df = _drop_shared_peptides(df)
    df = _aggregate_psms(df, summary_fn=summary_for_multiple_rows)
    if remove_few_measurements:
        df = _drop_few_measurements(df, min_obs=2)
    if remove_protein_with_1_feature:
        df = _drop_single_feature_proteins(df)
    return _finalize_msstats_long(df, annotation=annotation)


# -----------------------------------------------------------------------------
# Skyline
# -----------------------------------------------------------------------------
def skyline_to_msstats(
    report: pd.DataFrame,
    annotation: Optional[pd.DataFrame] = None,
    *,
    remove_irt: bool = True,
    filter_with_qvalue: bool = True,
    qvalue_cutoff: float = 0.01,
    use_unique_peptide: bool = True,
    remove_few_measurements: bool = True,
    remove_oxidation_m_peptides: bool = False,
    remove_protein_with_1_feature: bool = False,
) -> pd.DataFrame:
    """Skyline export → MSstats long-format.

    Port of ``SkylinetoMSstatsFormat``. Skyline's "MSstats Input" report
    already uses MSstats columns (with ``FileName`` → ``Run`` and ``Area``
    → ``Intensity``). The standard filters are:

    * remove iRT standards (``StandardType == 'iRT'``);
    * remove decoys (``ProteinName`` contains ``DECOY``);
    * remove truncated peaks (``Truncated == True``);
    * Q-value filter on ``DetectionQValue``.

    Multi-PSM rows of the same feature are aggregated by ``sum`` (R default).
    """
    df = report.copy()
    if "FileName" in df.columns and "Run" not in df.columns:
        df = df.rename(columns={"FileName": "Run"})
    if "Area" in df.columns and "Intensity" not in df.columns:
        df = df.rename(columns={"Area": "Intensity"})
    if "PeptideModifiedSequence" in df.columns:
        if "PeptideSequence" in df.columns:
            df = df.drop(columns=["PeptideSequence"])
        df = df.rename(columns={"PeptideModifiedSequence": "PeptideSequence"})

    df["Intensity"] = pd.to_numeric(df["Intensity"], errors="coerce")
    df.loc[df["Intensity"] == 0, "Intensity"] = np.nan
    df.loc[df["Intensity"] < 1, "Intensity"] = np.nan

    # iRT, decoys, truncated
    if remove_irt and "StandardType" in df.columns:
        df = df.loc[df["StandardType"].astype(str) != "iRT"]
    if "ProteinName" in df.columns:
        df = df.loc[~df["ProteinName"].astype(str).str.contains(
            "DECOY|Decoys", regex=True)]
    if "Truncated" in df.columns:
        is_trunc = df["Truncated"].astype(str).str.lower().isin(
            {"true", "1", "yes"}
        )
        df.loc[is_trunc, "Intensity"] = np.nan
    if remove_oxidation_m_peptides and "PeptideSequence" in df.columns:
        df = df.loc[~df["PeptideSequence"].astype(str).str.contains(r"\+16")]

    if filter_with_qvalue and "DetectionQValue" in df.columns:
        df["DetectionQValue"] = pd.to_numeric(
            df["DetectionQValue"], errors="coerce")
        df.loc[df["DetectionQValue"] >= qvalue_cutoff, "Intensity"] = np.nan

    if "IsotopeLabelType" not in df.columns:
        df["IsotopeLabelType"] = "L"

    keep = [c for c in (
        "ProteinName", "PeptideSequence", "PrecursorCharge",
        "FragmentIon", "ProductCharge", "IsotopeLabelType",
        "Condition", "BioReplicate", "Run", "Intensity",
    ) if c in df.columns]
    df = df[keep].copy()

    if use_unique_peptide:
        df = _drop_shared_peptides(df)
    df = _aggregate_psms(df, summary_fn="sum")
    if remove_few_measurements:
        df = _drop_few_measurements(df, min_obs=2)
    if remove_protein_with_1_feature:
        df = _drop_single_feature_proteins(df)
    return _finalize_msstats_long(df, annotation=annotation)


# -----------------------------------------------------------------------------
# Proteome Discoverer
# -----------------------------------------------------------------------------
def pd_to_msstats(
    input: pd.DataFrame,
    annotation: Optional[pd.DataFrame] = None,
    *,
    use_num_proteins_column: bool = False,
    use_unique_peptide: bool = True,
    summary_for_multiple_rows: str = "max",
    remove_few_measurements: bool = True,
    remove_oxidation_m_peptides: bool = False,
    remove_protein_with_1_peptide: bool = False,
    which_quantification: str = "Precursor.Area",
    which_proteinid: str = "Protein.Group.Accessions",
    which_sequence: str = "Sequence",
) -> pd.DataFrame:
    """Proteome Discoverer PSM export → MSstats long-format.

    Port of ``PDtoMSstatsFormat``. Proteome Discoverer writes a PSM
    table with a configurable protein-id / sequence / quantification
    column. The MSstats cleaner (``MSstatsConvert:::.cleanRawPDMSstats``):

    * selects ``which_proteinid, which_sequence, Modifications, Charge,
      SpectrumFile, which_quantification`` (+ optional ``Fraction``);
    * renames to ``ProteinName, PeptideSequence, Run, Intensity,
      PrecursorCharge``;
    * appends ``Modifications`` to the peptide sequence
      (``PeptideSequence_Modifications``);
    * when ``use_num_proteins_column`` and the ``X..Proteins`` (``#
      Proteins``) column is present, keeps only rows where it equals 1
      (unambiguous peptides).

    The remaining steps mirror the shared pipeline: drop shared
    peptides, aggregate multi-PSM rows by ``max``, drop low-measurement
    features and (optionally) single-peptide proteins.
    """
    df = input.copy()
    # PD column headers vary in punctuation ("Protein Group Accessions" vs
    # "Protein.Group.Accessions"); normalize both sides to a key with no
    # separators so the user's which_* arguments match (mirrors R's
    # MSstatsConvert:::.standardizeColnames).
    def _key(s):
        return "".join(ch for ch in str(s) if ch.isalnum()).lower()

    col_lookup = {_key(c): c for c in df.columns}

    def _resolve(name, *fallbacks):
        for cand in (name, *fallbacks):
            real = col_lookup.get(_key(cand))
            if real is not None:
                return real
        return None

    prot_col = _resolve(which_proteinid)
    seq_col = _resolve(which_sequence)
    quant_col = _resolve(which_quantification)
    charge_col = _resolve("Charge")
    run_col = _resolve("SpectrumFile", "Spectrum File", "Run")
    mods_col = _resolve("Modifications")
    rename_map = {}
    if prot_col is not None:
        rename_map[prot_col] = "ProteinName"
    if seq_col is not None:
        rename_map[seq_col] = "PeptideSequence"
    if quant_col is not None:
        rename_map[quant_col] = "Intensity"
    if charge_col is not None:
        rename_map[charge_col] = "PrecursorCharge"
    if run_col is not None:
        rename_map[run_col] = "Run"
    if mods_col is not None:
        rename_map[mods_col] = "Modifications"
    if use_num_proteins_column:
        for cand in ("# Proteins", "#Proteins", "X..Proteins", "Number.of.Proteins"):
            if cand in df.columns:
                df = df.loc[
                    pd.to_numeric(df[cand], errors="coerce") == 1
                ]
                break
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    if "Modifications" in df.columns and "PeptideSequence" in df.columns:
        df["PeptideSequence"] = (
            df["PeptideSequence"].astype(str) + "_"
            + df["Modifications"].fillna("").astype(str)
        )
    if remove_oxidation_m_peptides and "PeptideSequence" in df.columns:
        df = df.loc[~df["PeptideSequence"].astype(str).str.contains(
            "Oxidation", case=False, regex=False)]
    df["FragmentIon"] = "NA"
    df["ProductCharge"] = "NA"
    df["IsotopeLabelType"] = "L"
    if "Intensity" in df.columns:
        df["Intensity"] = pd.to_numeric(df["Intensity"], errors="coerce")
        df.loc[df["Intensity"] == 0, "Intensity"] = np.nan
    keep = [c for c in (
        "ProteinName", "PeptideSequence", "PrecursorCharge",
        "FragmentIon", "ProductCharge", "IsotopeLabelType",
        "Condition", "BioReplicate", "Run", "Intensity",
    ) if c in df.columns]
    df = df[keep].copy()
    if use_unique_peptide:
        df = _drop_shared_peptides(df)
    df = _aggregate_psms(df, summary_fn=summary_for_multiple_rows)
    if remove_few_measurements:
        df = _drop_few_measurements(df, min_obs=2)
    if remove_protein_with_1_peptide:
        df = _drop_single_feature_proteins(df)
    return _finalize_msstats_long(df, annotation=annotation)


# -----------------------------------------------------------------------------
# Progenesis QI
# -----------------------------------------------------------------------------
def progenesis_to_msstats(
    input: pd.DataFrame,
    annotation: Optional[pd.DataFrame] = None,
    *,
    use_unique_peptide: bool = True,
    summary_for_multiple_rows: str = "max",
    remove_few_measurements: bool = True,
    remove_oxidation_m_peptides: bool = False,
    remove_protein_with_1_peptide: bool = False,
) -> pd.DataFrame:
    """Progenesis QI export → MSstats long-format.

    Port of ``ProgenesistoMSstatsFormat``. Progenesis writes a *wide*
    "feature measurements" table (one raw-abundance column per run); the
    R cleaner melts that into long format. This Python port accepts the
    already-melted long table with columns ``ProteinName`` (or
    ``Protein`` / ``Accession``), ``Sequence`` (peptide), ``Charge`` (or
    ``Ions``), ``Run``, ``Intensity`` plus optional ``Modifications`` and
    ``Use.in.quantitation``. Rows with ``Use.in.quantitation == False``
    are dropped (the Progenesis-specific filter).
    """
    df = input.copy()
    for cand in ("Protein", "Accession"):
        if cand in df.columns and "ProteinName" not in df.columns:
            df = df.rename(columns={cand: "ProteinName"})
            break
    if "Ions" in df.columns and "Charge" not in df.columns:
        df = df.rename(columns={"Ions": "Charge"})
    df = df.rename(columns={
        "Sequence": "PeptideSequence",
        "Charge": "PrecursorCharge",
    })
    # Progenesis "Use in quantitation" filter
    for cand in ("Use.in.quantitation", "Useinquantitation",
                 "Use in quantitation"):
        if cand in df.columns:
            keep = df[cand].astype(str).str.lower().isin(
                {"true", "1", "yes", ""})
            df = df.loc[keep]
            break
    if "Modifications" in df.columns and "PeptideSequence" in df.columns:
        df["PeptideSequence"] = (
            df["PeptideSequence"].astype(str) + "_"
            + df["Modifications"].fillna("").astype(str)
        )
    if remove_oxidation_m_peptides and "PeptideSequence" in df.columns:
        df = df.loc[~df["PeptideSequence"].astype(str).str.contains(
            "Oxidation", case=False, regex=False)]
    # drop empty protein / peptide rows
    if "ProteinName" in df.columns:
        df = df.loc[df["ProteinName"].astype(str).str.strip() != ""]
    if "PeptideSequence" in df.columns:
        df = df.loc[df["PeptideSequence"].astype(str).str.strip() != ""]
    df["FragmentIon"] = "NA"
    df["ProductCharge"] = "NA"
    df["IsotopeLabelType"] = "L"
    if "Intensity" in df.columns:
        df["Intensity"] = pd.to_numeric(df["Intensity"], errors="coerce")
        df.loc[df["Intensity"] == 0, "Intensity"] = np.nan
    keep = [c for c in (
        "ProteinName", "PeptideSequence", "PrecursorCharge",
        "FragmentIon", "ProductCharge", "IsotopeLabelType",
        "Condition", "BioReplicate", "Run", "Intensity",
    ) if c in df.columns]
    df = df[keep].copy()
    if use_unique_peptide:
        df = _drop_shared_peptides(df)
    df = _aggregate_psms(df, summary_fn=summary_for_multiple_rows)
    if remove_few_measurements:
        df = _drop_few_measurements(df, min_obs=2)
    if remove_protein_with_1_peptide:
        df = _drop_single_feature_proteins(df)
    return _finalize_msstats_long(df, annotation=annotation)


# -----------------------------------------------------------------------------
# OpenSWATH
# -----------------------------------------------------------------------------
def openswath_to_msstats(
    input: pd.DataFrame,
    annotation: Optional[pd.DataFrame] = None,
    *,
    filter_with_mscore: bool = True,
    mscore_cutoff: float = 0.01,
    use_unique_peptide: bool = True,
    remove_few_measurements: bool = True,
    remove_protein_with_1_feature: bool = False,
    summary_for_multiple_rows: str = "max",
) -> pd.DataFrame:
    """OpenSWATH export → MSstats long-format.

    Port of ``OpenSWATHtoMSstatsFormat``. The OpenSWATH cleaner
    (``MSstatsConvert:::.cleanRawOpenSWATH``):

    * selects ``ProteinName, FullPeptideName, Charge, filename,
      aggr_Fragment_Annotation, aggr_Peak_Area, m_score, decoy``;
    * renames to ``ProteinName, PeptideSequence, PrecursorCharge, Run,
      FragmentIon, Intensity``;
    * explodes the semicolon-separated ``FragmentIon`` / ``Intensity``
      lists into per-fragment rows;
    * replaces ``:`` with ``_`` in peptide / fragment names;
    * sets ``Intensity < 1`` to NA;
    * drops decoys (``decoy == 1``) and (optionally) rows with
      ``m_score >= mscore_cutoff``.
    """
    df = input.copy()
    rename_map = {
        "FullPeptideName": "PeptideSequence",
        "Charge": "PrecursorCharge",
        "filename": "Run",
        "aggr_Fragment_Annotation": "FragmentIon",
        "aggr_Peak_Area": "Intensity",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    # decoy filter
    if "decoy" in df.columns:
        df = df.loc[pd.to_numeric(df["decoy"], errors="coerce").fillna(0) != 1]
    # m_score filter
    if filter_with_mscore and "m_score" in df.columns:
        ms = pd.to_numeric(df["m_score"], errors="coerce")
        df = df.loc[ms < mscore_cutoff]
    # explode semicolon-separated fragment/intensity lists
    for col in ("FragmentIon", "Intensity"):
        if col in df.columns:
            df[col] = df[col].astype(str)
    if ("FragmentIon" in df.columns
            and df["FragmentIon"].str.contains(";").any()):
        df = df.assign(
            FragmentIon=df["FragmentIon"].str.split(";"),
            Intensity=df["Intensity"].str.split(";"),
        )
        df = df.explode(["FragmentIon", "Intensity"])
    # ":" → "_" in peptide / fragment names
    for col in ("PeptideSequence", "FragmentIon"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(":", "_", regex=False)
    df["ProductCharge"] = "NA"
    df["IsotopeLabelType"] = "L"
    df["Intensity"] = pd.to_numeric(df["Intensity"], errors="coerce")
    df.loc[df["Intensity"] < 1, "Intensity"] = np.nan
    keep = [c for c in (
        "ProteinName", "PeptideSequence", "PrecursorCharge",
        "FragmentIon", "ProductCharge", "IsotopeLabelType",
        "Condition", "BioReplicate", "Run", "Intensity",
    ) if c in df.columns]
    df = df[keep].copy()
    if use_unique_peptide:
        df = _drop_shared_peptides(df)
    df = _aggregate_psms(df, summary_fn=summary_for_multiple_rows)
    if remove_few_measurements:
        df = _drop_few_measurements(df, min_obs=2)
    if remove_protein_with_1_feature:
        df = _drop_single_feature_proteins(df)
    return _finalize_msstats_long(df, annotation=annotation)


# -----------------------------------------------------------------------------
# DIA-Umpire
# -----------------------------------------------------------------------------
def diaumpire_to_msstats(
    raw_frag: pd.DataFrame,
    raw_pep: Optional[pd.DataFrame] = None,
    raw_pro: Optional[pd.DataFrame] = None,
    annotation: Optional[pd.DataFrame] = None,
    *,
    use_selected_frag: bool = True,
    use_selected_pep: bool = True,
    remove_few_measurements: bool = True,
    remove_protein_with_1_feature: bool = False,
    summary_for_multiple_rows: str = "max",
) -> pd.DataFrame:
    """DIA-Umpire export → MSstats long-format.

    Port of ``DIAUmpiretoMSstatsFormat``. DIA-Umpire writes three
    tables (fragment / peptide / protein); the converter joins them and
    keeps only "selected" fragments/peptides. This Python port accepts
    the fragment-level table already joined to protein / peptide
    identity (long format) with columns ``ProteinName``,
    ``PeptideSequence``, ``FragmentIon``, ``Run``, ``Intensity`` plus
    optional ``Selected_fragment`` / ``Selected_peptide`` flags.

    DIA-Umpire always removes shared peptides in R
    (``remove_shared_peptides = TRUE``).
    """
    df = raw_frag.copy()
    if use_selected_frag:
        for cand in ("Selected_fragment", "Selected.fragment",
                     "UseInQuant_frag"):
            if cand in df.columns:
                keep = df[cand].astype(str).str.lower().isin(
                    {"true", "1", "yes"})
                df = df.loc[keep]
                break
    if use_selected_pep:
        for cand in ("Selected_peptide", "Selected.peptide",
                     "UseInQuant_pep"):
            if cand in df.columns:
                keep = df[cand].astype(str).str.lower().isin(
                    {"true", "1", "yes"})
                df = df.loc[keep]
                break
    if "PrecursorCharge" not in df.columns:
        df["PrecursorCharge"] = "NA"
    df["ProductCharge"] = "NA"
    df["IsotopeLabelType"] = "L"
    if "Intensity" in df.columns:
        df["Intensity"] = pd.to_numeric(df["Intensity"], errors="coerce")
        df.loc[df["Intensity"] == 0, "Intensity"] = np.nan
    keep = [c for c in (
        "ProteinName", "PeptideSequence", "PrecursorCharge",
        "FragmentIon", "ProductCharge", "IsotopeLabelType",
        "Condition", "BioReplicate", "Run", "Intensity",
    ) if c in df.columns]
    df = df[keep].copy()
    # DIA-Umpire always removes shared peptides
    df = _drop_shared_peptides(df)
    df = _aggregate_psms(df, summary_fn=summary_for_multiple_rows)
    if remove_few_measurements:
        df = _drop_few_measurements(df, min_obs=2)
    if remove_protein_with_1_feature:
        df = _drop_single_feature_proteins(df)
    return _finalize_msstats_long(df, annotation=annotation)


# -----------------------------------------------------------------------------
# MaxQuant — re-exported (definition in pipeline.py for backwards-compat)
# -----------------------------------------------------------------------------
from .pipeline import maxquant_to_msstats   # noqa: E402,F401  (re-export)


__all__ = [
    "CANONICAL_COLS",
    "validate_annotation",
    "merge_fractions",
    "diann_to_msstats",
    "spectronaut_to_msstats",
    "fragpipe_to_msstats",
    "openms_to_msstats",
    "skyline_to_msstats",
    "maxquant_to_msstats",
    "pd_to_msstats",
    "progenesis_to_msstats",
    "openswath_to_msstats",
    "diaumpire_to_msstats",
]
