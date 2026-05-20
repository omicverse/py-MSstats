"""High-level :func:`data_process` pipeline — port of ``MSstats::dataProcess``.

The R reference does, in order:

* convert any zero / NA intensities to NaN
* log2-transform intensities (``log_base = 2``, R default)
* per-run median normalization (``equalizeMedians``)
* feature filtering: drop features observed in fewer than ``min_feature_obs`` runs
* TMP summarization to per-protein log abundance per run
* attach per-run metadata (``GROUP``, ``SUBJECT``) for downstream
  :func:`group_comparison`.

For v0.1 we expose the LFQ / single-label / no-fraction path. Imputation,
quantile normalization, label-reference workflows, and TMT / DIA-specific
post-processing are deferred to v0.2.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .normalization import equalize_medians
from .summarization import tmp_summarize


CANONICAL_COLS = [
    "ProteinName", "PeptideSequence", "PrecursorCharge",
    "FragmentIon", "ProductCharge", "IsotopeLabelType",
    "Condition", "BioReplicate", "Run", "Intensity",
]


def _validate_input(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in CANONICAL_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"input is missing required MSstats columns: {missing}\n"
            f"Expected: {CANONICAL_COLS}"
        )
    return df.copy()


def _make_feature_id(df: pd.DataFrame) -> pd.Series:
    """Build the MSstats ``FEATURE`` identifier.

    R combines ``PEPTIDE_CHARGE_FRAGMENT_PRODUCT`` after dropping NA-ish
    fields. We keep it simple: concatenate the four feature-level columns
    so each unique (peptide, precursor, fragment, product) combination is
    one feature.
    """
    return (
        df["PeptideSequence"].astype(str) + "_" +
        df["PrecursorCharge"].astype(str) + "_" +
        df["FragmentIon"].astype(str) + "_" +
        df["ProductCharge"].astype(str)
    )


def data_process(
    msstats_df: pd.DataFrame,
    *,
    normalization: str = "equalizeMedians",
    summary_method: str = "TMP",
    log_base: int = 2,
    min_feature_obs: int = 2,
    censored_int: str = "NA",
) -> pd.DataFrame:
    """Pre-process feature-level MSstats data → protein-level per-run abundances.

    Parameters
    ----------
    msstats_df
        Long-format MSstats table. Required columns:
        ``ProteinName, PeptideSequence, PrecursorCharge, FragmentIon,
        ProductCharge, IsotopeLabelType, Condition, BioReplicate, Run,
        Intensity``.
    normalization
        Only ``'equalizeMedians'`` (default) and ``'none'`` are supported
        in v0.1. ``'quantile'`` / ``'globalStandards'`` deferred to v0.2.
    summary_method
        Only ``'TMP'`` (default) is supported in v0.1. ``'linear'`` (the
        old MSstats linear-model summarization) deferred to v0.2.
    log_base
        Logarithm base (2 = default, matches R).
    min_feature_obs
        Drop features observed in fewer than this many runs (R default 2).
    censored_int
        ``'NA'`` (default) or ``'0'`` — value treated as missing in the
        intensity column.

    Returns
    -------
    pd.DataFrame
        Columns: ``Protein, RUN, LogIntensities, GROUP, SUBJECT,
        n_features, n_obs``. One row per ``(Protein, RUN)``.
    """
    if normalization not in ("equalizeMedians", "none"):
        raise NotImplementedError(
            f"normalization={normalization!r} is deferred to v0.2 "
            "(only 'equalizeMedians' and 'none' supported in v0.1)"
        )
    if summary_method != "TMP":
        raise NotImplementedError(
            f"summary_method={summary_method!r} is deferred to v0.2 "
            "(only 'TMP' supported in v0.1)"
        )

    df = _validate_input(msstats_df)

    # 1) replace zero / negative / NA intensities with NaN
    intensity = pd.to_numeric(df["Intensity"], errors="coerce")
    if censored_int == "0":
        intensity = intensity.where(intensity > 0, np.nan)
    else:
        intensity = intensity.where(intensity > 0, np.nan)
    df["INTENSITY"] = intensity

    # 2) log2 transform
    with np.errstate(divide="ignore"):
        if log_base == 2:
            df["ABUNDANCE"] = np.log2(df["INTENSITY"])
        else:
            df["ABUNDANCE"] = np.log(df["INTENSITY"]) / np.log(log_base)

    # 3) tag standard MSstats column aliases
    df["PROTEIN"] = df["ProteinName"].astype(str)
    df["RUN"] = df["Run"].astype(str)
    df["LABEL"] = df["IsotopeLabelType"].astype(str)
    df["GROUP"] = df["Condition"].astype(str)
    df["SUBJECT"] = df["BioReplicate"].astype(str)
    df["FEATURE"] = _make_feature_id(df)

    # 4) feature filtering — drop features with too few non-NaN runs
    obs_per_feature = (
        df.dropna(subset=["ABUNDANCE"]).groupby(["PROTEIN", "FEATURE"])
        .size()
    )
    keep_idx = obs_per_feature[obs_per_feature >= min_feature_obs].index
    keep_set = set(map(tuple, keep_idx.tolist()))
    mask = [
        (p, f) in keep_set for p, f in zip(df["PROTEIN"], df["FEATURE"])
    ]
    df = df.loc[mask].copy()

    # 5) normalization
    if normalization == "equalizeMedians":
        df = equalize_medians(
            df, abundance_col="ABUNDANCE", run_col="RUN", label_col="LABEL",
            label_value="L",
        )

    # 6) TMP summarization to per-(protein, run)
    summary = tmp_summarize(df)

    # 7) attach GROUP / SUBJECT via Run → (Condition, BioReplicate) lookup
    annot = (
        df[["RUN", "GROUP", "SUBJECT"]]
        .drop_duplicates(subset=["RUN"])
        .set_index("RUN")
    )
    summary["GROUP"] = summary["RUN"].map(annot["GROUP"])
    summary["SUBJECT"] = summary["RUN"].map(annot["SUBJECT"])
    return summary


# -----------------------------------------------------------------------------
# Optional MaxQuant converter (very stripped-down — full converter deferred).
# -----------------------------------------------------------------------------
def maxquant_to_msstats(
    protein_groups: pd.DataFrame,
    evidence: pd.DataFrame,
    annotation: pd.DataFrame,
    *,
    drop_reverse: bool = True,
    drop_contaminants: bool = True,
    use_unique_peptides: bool = True,
) -> pd.DataFrame:
    """Minimal MaxQuant → MSstats long-format converter.

    A simplified port of ``MaxQtoMSstatsFormat`` covering the most-common
    label-free DDA case. The full converter handles many MaxQuant edge
    cases that we defer to v0.2:

    * fractions, replicates and reference channels
    * matched-between-runs flag
    * label-incorporation rates (heavy / light)
    * peptide-feature aggregation modes other than ``'unique-peptides'``

    Parameters
    ----------
    protein_groups
        Parsed ``proteinGroups.txt`` (one row per protein group). Required
        columns: ``id``, ``Reverse``, ``Potential contaminant``, ``Only
        identified by site``, ``Protein IDs``.
    evidence
        Parsed ``evidence.txt`` (one row per evidence record). Required
        columns: ``Sequence``, ``Modified sequence``, ``Charge``, ``Raw file``,
        ``Intensity``, ``Protein group IDs``, optional ``Reverse``,
        ``Potential contaminant``.
    annotation
        Three-column DataFrame mapping each ``Raw file`` to a ``Condition``
        and ``BioReplicate`` (column names ``Run``, ``Condition``,
        ``BioReplicate``).

    Returns
    -------
    pd.DataFrame
        MSstats long-format with the canonical 10 columns.
    """
    pg = protein_groups.copy()
    ev = evidence.copy()
    ann = annotation.copy()

    if drop_reverse and "Reverse" in pg.columns:
        pg = pg.loc[pg["Reverse"].fillna("") != "+"]
    if drop_contaminants and "Potential contaminant" in pg.columns:
        pg = pg.loc[pg["Potential contaminant"].fillna("") != "+"]
    if "Only identified by site" in pg.columns:
        pg = pg.loc[pg["Only identified by site"].fillna("") != "+"]

    pg_id_to_name = dict(zip(pg["id"].astype(str), pg["Protein IDs"].astype(str)))
    valid_ids = set(pg_id_to_name.keys())

    if drop_reverse and "Reverse" in ev.columns:
        ev = ev.loc[ev["Reverse"].fillna("") != "+"]
    if drop_contaminants and "Potential contaminant" in ev.columns:
        ev = ev.loc[ev["Potential contaminant"].fillna("") != "+"]
    ev["Protein group IDs"] = ev["Protein group IDs"].astype(str)
    if use_unique_peptides:
        ev = ev.loc[~ev["Protein group IDs"].str.contains(";")]
    ev = ev.loc[ev["Protein group IDs"].isin(valid_ids)]

    ev["ProteinName"] = ev["Protein group IDs"].map(pg_id_to_name)
    ev["PeptideSequence"] = ev["Modified sequence"].astype(str) if (
        "Modified sequence" in ev.columns) else ev["Sequence"].astype(str)
    ev["PrecursorCharge"] = ev["Charge"].astype(int)
    ev["FragmentIon"] = "NA"
    ev["ProductCharge"] = "NA"
    ev["IsotopeLabelType"] = "L"
    ev["Run"] = ev["Raw file"].astype(str)

    ev = ev.merge(ann, on="Run", how="left")
    if ev["Condition"].isna().any():
        missing = ev.loc[ev["Condition"].isna(), "Run"].unique().tolist()
        raise ValueError(
            f"runs missing annotation rows: {missing}"
        )

    # Sum-aggregate evidence (PSM-level) → unique feature per run.
    agg = (
        ev.groupby(
            ["ProteinName", "PeptideSequence", "PrecursorCharge",
             "FragmentIon", "ProductCharge", "IsotopeLabelType",
             "Condition", "BioReplicate", "Run"],
            dropna=False,
        )["Intensity"].sum().reset_index()
    )
    return agg[CANONICAL_COLS]
