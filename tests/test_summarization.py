"""Tests for the summarization helpers — linear summary, quantification, accessors."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pymsstats import (
    data_process,
    get_samples_info,
    get_selected_proteins,
    linear_summarize,
    msstats_summarize,
    quantification,
    tmp_summarize,
)
from tests._synthetic import synthetic_msstats


def _feature_frame(seed=0, n_proteins=30):
    df, _ = synthetic_msstats(n_proteins=n_proteins, seed=seed)
    df = df.copy()
    df["ABUNDANCE"] = np.log2(df["Intensity"].astype(float))
    df["RUN"] = df["Run"]
    df["FEATURE"] = (
        df["PeptideSequence"].astype(str) + "_" + df["PrecursorCharge"].astype(str)
    )
    df["PROTEIN"] = df["ProteinName"]
    return df


def test_linear_summarize_schema():
    df = _feature_frame()
    out = linear_summarize(df)
    assert set(["Protein", "RUN", "LogIntensities"]).issubset(out.columns)
    assert len(out) > 0
    assert out["LogIntensities"].notna().all()


def test_linear_and_tmp_are_correlated():
    """Linear and TMP summaries should track each other closely on
    well-behaved synthetic data."""
    df = _feature_frame(seed=5, n_proteins=60)
    tmp = tmp_summarize(df).set_index(["Protein", "RUN"])["LogIntensities"]
    lin = linear_summarize(df).set_index(["Protein", "RUN"])["LogIntensities"]
    common = tmp.index.intersection(lin.index)
    r = float(np.corrcoef(tmp.loc[common], lin.loc[common])[0, 1])
    assert r > 0.95, f"TMP vs linear Pearson r = {r:.4f}"


def test_msstats_summarize_dispatch():
    df = _feature_frame()
    a = msstats_summarize(df, method="TMP")
    b = msstats_summarize(df, method="linear")
    assert len(a) > 0 and len(b) > 0
    with pytest.raises(ValueError):
        msstats_summarize(df, method="bogus")


def test_quantification_sample_matrix():
    df, _ = synthetic_msstats(n_proteins=20, seed=3)
    proc = data_process(df)
    q = quantification(proc, type="Sample", format="matrix")
    assert "Protein" in q.columns
    # one column per (group, subject) sample
    n_samples = proc[["GROUP", "SUBJECT"]].drop_duplicates().shape[0]
    assert q.shape[1] - 1 == n_samples


def test_quantification_group_long():
    df, _ = synthetic_msstats(n_proteins=20, seed=4)
    proc = data_process(df)
    q = quantification(proc, type="Group", format="long")
    assert set(["Protein", "Group", "LogIntensity"]).issubset(q.columns)


def test_get_samples_info():
    df, _ = synthetic_msstats(n_proteins=10, seed=6)
    proc = data_process(df)
    info = get_samples_info(proc)
    assert "NumRuns" in info.columns
    assert (info["NumRuns"] > 0).all()


def test_get_selected_proteins_by_name_and_index():
    all_p = ["A", "B", "C", "D"]
    assert get_selected_proteins(["B", "C"], all_p) == ["B", "C"]
    assert get_selected_proteins([1, 3], all_p) == ["A", "C"]
    with pytest.raises(ValueError):
        get_selected_proteins(["Z"], all_p)
