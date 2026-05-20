"""Tests for pymsstats.normalization — equalizeMedians, quantile, globalStandards."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pymsstats import (
    equalize_medians,
    msstats_normalize,
    normalize_global_standards,
    quantile_normalize,
)
from tests._synthetic import synthetic_msstats


def _feature_frame(seed=0):
    df, _ = synthetic_msstats(n_proteins=40, seed=seed)
    df = df.copy()
    df["ABUNDANCE"] = np.log2(df["Intensity"].astype(float))
    df["RUN"] = df["Run"]
    df["LABEL"] = df["IsotopeLabelType"]
    df["FEATURE"] = (
        df["PeptideSequence"].astype(str) + "_" + df["PrecursorCharge"].astype(str)
    )
    df["PROTEIN"] = df["ProteinName"]
    return df


def test_equalize_medians_flattens_run_medians():
    df = _feature_frame()
    rng = np.random.default_rng(7)
    off = {r: rng.normal(0, 2.0) for r in df["RUN"].unique()}
    df["ABUNDANCE"] = df["ABUNDANCE"] + df["RUN"].map(off)
    out = equalize_medians(df)
    med = out.groupby("RUN")["ABUNDANCE"].median()
    assert (med.max() - med.min()) < 1e-9


def test_quantile_normalize_makes_run_distributions_equal():
    df = _feature_frame()
    rng = np.random.default_rng(11)
    off = {r: rng.normal(0, 1.5) for r in df["RUN"].unique()}
    df["ABUNDANCE"] = df["ABUNDANCE"] + df["RUN"].map(off)
    out = quantile_normalize(df)
    # After quantile normalization the sorted distributions of every run
    # should be (near-)identical → run quantiles match.
    qs = (
        out.groupby("RUN")["ABUNDANCE"]
        .apply(lambda x: np.nanquantile(x.dropna(), [0.25, 0.5, 0.75]))
    )
    arr = np.vstack(qs.to_numpy())
    # spread across runs at each quantile should be tiny
    assert arr.std(axis=0).max() < 1e-6


def test_global_standards_normalization_runs():
    df = _feature_frame()
    standards = [df["PROTEIN"].iloc[0]]
    out = normalize_global_standards(df, standards)
    assert "ABUNDANCE" in out.columns
    assert len(out) == len(df)


def test_msstats_normalize_dispatch():
    df = _feature_frame()
    a = msstats_normalize(df, method="equalizeMedians")
    b = msstats_normalize(df, method="none")
    assert a.shape == df.shape
    np.testing.assert_array_equal(
        b["ABUNDANCE"].to_numpy(), df["ABUNDANCE"].to_numpy()
    )


def test_msstats_normalize_unknown_method_raises():
    df = _feature_frame()
    with pytest.raises(ValueError):
        msstats_normalize(df, method="bogus")
