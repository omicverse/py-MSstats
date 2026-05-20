"""Smoke tests for the vendor → MSstats converters."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pymsstats import (
    CANONICAL_COLS,
    diann_to_msstats,
    fragpipe_to_msstats,
    merge_fractions,
    openms_to_msstats,
    skyline_to_msstats,
    spectronaut_to_msstats,
    validate_annotation,
)
from tests._synthetic import (
    mock_diann_report,
    mock_msstats_long,
    mock_skyline_report,
    mock_spectronaut_report,
)


def _annotation(n_runs=4):
    return pd.DataFrame({
        "Run": [f"run{i}" for i in range(n_runs)],
        "Condition": ["A" if i < n_runs // 2 else "B" for i in range(n_runs)],
        "BioReplicate": [f"rep{i}" for i in range(n_runs)],
    })


def _check_long(out):
    assert list(out.columns) == CANONICAL_COLS
    assert len(out) > 0
    assert out["Intensity"].notna().any()
    assert out["Condition"].notna().all()


def test_diann_converter():
    diann = mock_diann_report()
    out = diann_to_msstats(diann, _annotation())
    _check_long(out)
    # 6 proteins x 3 peptides x 4 runs (max-aggregated per feature)
    assert out["ProteinName"].nunique() == 6


def test_diann_qvalue_filter_drops_rows():
    diann = mock_diann_report()
    diann.loc[diann["Run"] == "run0", "Lib.Q.Value"] = 0.9
    out = diann_to_msstats(diann, _annotation())
    # run0 rows filtered out
    assert "run0" not in set(out["Run"])


def test_spectronaut_converter():
    spec = mock_spectronaut_report()
    out = spectronaut_to_msstats(spec)  # annotation embedded in report
    _check_long(out)
    assert out["FragmentIon"].nunique() >= 2


def test_skyline_converter():
    sky = mock_skyline_report()
    out = skyline_to_msstats(sky)
    _check_long(out)


def test_skyline_drops_truncated():
    sky = mock_skyline_report()
    sky.loc[sky["FileName"] == "run0", "Truncated"] = "True"
    out = skyline_to_msstats(sky)
    # run0 intensities become NaN → feature few-measurement filter may drop
    assert (out.loc[out["Run"] == "run0", "Intensity"].isna().all()
            or "run0" not in set(out["Run"]))


def test_fragpipe_converter():
    fp = mock_msstats_long()
    out = fragpipe_to_msstats(fp)
    _check_long(out)


def test_openms_converter():
    om = mock_msstats_long()
    out = openms_to_msstats(om)
    _check_long(out)


def test_validate_annotation_ok():
    ann = _annotation()
    out = validate_annotation(ann)
    assert set(["Run", "Condition", "BioReplicate"]).issubset(out.columns)


def test_validate_annotation_single_condition_raises():
    ann = pd.DataFrame({
        "Run": ["r0", "r1"], "Condition": ["A", "A"],
        "BioReplicate": ["b0", "b1"],
    })
    with pytest.raises(ValueError, match="distinct Conditions"):
        validate_annotation(ann)


def test_merge_fractions_sums_intensities():
    df = mock_msstats_long()
    df["Fraction"] = np.where(df["Run"].isin(["run0", "run1"]), 1, 2)
    # remap so each (cond, biorep) appears in two fractions
    df["BioReplicate"] = df["Run"].map(
        {"run0": "s0", "run1": "s1", "run2": "s0", "run3": "s1"})
    df["Condition"] = df["Run"].map(
        {"run0": "A", "run1": "B", "run2": "A", "run3": "B"})
    merged = merge_fractions(df)
    assert "Run" in merged.columns
    assert merged["Run"].str.contains("merged").all()
    # one row per (cond, biorep, feature)
    assert len(merged) < len(df)


def test_merge_fractions_noop_without_fraction_col():
    df = mock_msstats_long()
    out = merge_fractions(df)
    assert len(out) == len(df)
