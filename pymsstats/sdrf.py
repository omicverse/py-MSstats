"""SDRF (Sample-Data Relationship Format) helpers.

Ports of ``MSstats::extractSDRF``, ``MSstats::SDRFtoAnnotation`` and the
bundled ``example_SDRF`` dataset.

SDRF is the ProteomeXchange / PRIDE metadata format — a tab-separated
table where each row is one MS run and columns carry experimental
metadata under bracketed headers such as ``comment[data file]``,
``characteristics[disease]`` and ``characteristics[biological
replicate]``. MSstats uses SDRF to derive the three-column annotation
table (``Run, Condition, BioReplicate``) consumed by the vendor
converters.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _standardize_colnames(names) -> list:
    """Mirror ``MSstatsConvert:::.standardizeColnames``.

    Replaces every non-alphanumeric run with a single dot, so
    ``comment[data file]`` → ``comment.data.file.``.
    """
    import re
    out = []
    for n in names:
        s = re.sub(r"[^A-Za-z0-9]+", ".", str(n))
        out.append(s)
    return out


def extract_sdrf(
    data: pd.DataFrame,
    *,
    run_name: str = "comment[data file]",
    condition_name: str = "characteristics[disease]",
    biological_replicate: str = "characteristics[biological replicate]",
    fraction: str | None = None,
    meta_data: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build an SDRF-style table from an MSstats annotation table.

    Port of ``MSstats::extractSDRF``. The R function takes a table with
    ``Condition, BioReplicate, Run`` (+ optional ``Fraction``), keeps the
    unique rows, and renames the columns to the SDRF header strings
    supplied by the caller. Optionally merges in a ``meta_data`` table
    keyed on the run name.

    Parameters
    ----------
    data
        Table with at least ``Condition, BioReplicate, Run`` columns
        (and ``Fraction`` if ``fraction`` is given).
    run_name, condition_name, biological_replicate, fraction
        SDRF header strings to rename the columns to.
    meta_data
        Optional extra metadata table to merge in (left/right join on the
        run-name column).

    Returns
    -------
    pd.DataFrame
        The SDRF table with the supplied bracketed header names.
    """
    extract_cols = ["Condition", "BioReplicate", "Run"]
    if fraction is not None:
        extract_cols = extract_cols + ["Fraction"]
    out = data[extract_cols].drop_duplicates().reset_index(drop=True)
    rename = {
        "Run": run_name,
        "Condition": condition_name,
        "BioReplicate": biological_replicate,
    }
    if fraction is not None:
        rename["Fraction"] = fraction
    out = out.rename(columns=rename)
    if meta_data is not None:
        out = out.merge(meta_data, how="outer", on=run_name)
    return out


def sdrf_to_annotation(
    data: pd.DataFrame,
    *,
    run_name: str = "comment[data file]",
    condition_name: str = "characteristics[disease]",
    biological_replicate: str = "characteristics[biological replicate]",
    fraction: str | None = None,
) -> pd.DataFrame:
    """Convert an SDRF table to an MSstats annotation table.

    Port of ``MSstats::SDRFtoAnnotation``. Column names are first
    standardized (brackets / spaces → dots, mirroring R's
    ``.standardizeColnames``), then the run / condition / biological
    replicate columns are selected and renamed to ``Run, Condition,
    BioReplicate`` (+ ``Fraction`` if requested).

    Parameters
    ----------
    data
        SDRF table (one row per MS run).
    run_name, condition_name, biological_replicate, fraction
        SDRF header strings identifying the columns to extract.

    Returns
    -------
    pd.DataFrame
        Annotation table with ``Run, Condition, BioReplicate``
        (+ ``Fraction``).
    """
    df = data.copy()
    extract_cols = [run_name, condition_name, biological_replicate]
    if fraction is not None:
        extract_cols = extract_cols + [fraction]
    std = _standardize_colnames(df.columns)
    df.columns = std
    std_extract = _standardize_colnames(extract_cols)
    missing = [c for c in std_extract if c not in df.columns]
    if missing:
        raise ValueError(
            "ERROR: one or more of the columns passed in the parameters "
            f"were not found in the data: {missing}. Available columns: "
            f"{list(df.columns)}"
        )
    out = df[std_extract].copy()
    new_names = ["Run", "Condition", "BioReplicate"]
    if fraction is not None:
        new_names = new_names + ["Fraction"]
    out.columns = new_names
    return out.reset_index(drop=True)


def example_sdrf(n_samples: int = 8, n_runs: int = 4) -> pd.DataFrame:
    """Return a small example SDRF DataFrame.

    Port of ``MSstats::example_SDRF`` — the R object is a large
    PRIDE-derived SDRF table that is not redistributed here; instead we
    synthesize a compact, structurally faithful SDRF table with the
    canonical bracketed headers so the SDRF helpers can be exercised.

    Parameters
    ----------
    n_samples
        Number of biological samples (default 8).
    n_runs
        Number of MS runs per sample (default 4).

    Returns
    -------
    pd.DataFrame
        An SDRF table with ``source name``, ``characteristics[*]`` and
        ``comment[*]`` columns.
    """
    rows = []
    for run in range(1, n_runs + 1):
        for s in range(1, n_samples + 1):
            disease = "disease" if s % 2 == 0 else "control"
            rows.append({
                "source name": f"Sample {s}",
                "characteristics[organism]": "Homo sapiens",
                "characteristics[disease]": disease,
                "characteristics[biological replicate]": s,
                "assay name": f"run {run}",
                "technology type": "proteomic profiling by mass spectrometry",
                "comment[fraction identifier]": 1,
                "comment[label]": "label free sample",
                "comment[data file]": f"sample{s}_run{run}.raw",
            })
    return pd.DataFrame(rows)


__all__ = ["extract_sdrf", "sdrf_to_annotation", "example_sdrf"]
