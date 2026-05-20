"""Tests for pymsstats.contrasts.msstats_contrast_matrix."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pymsstats import msstats_contrast_matrix


def test_single_string_contrast():
    C = msstats_contrast_matrix("groupB-groupA", ["groupA", "groupB"])
    assert C.values.tolist() == [[-1.0, 1.0]]
    assert list(C.columns) == ["groupA", "groupB"]


def test_vs_infix_contrast():
    C = msstats_contrast_matrix("groupB vs groupA", ["groupA", "groupB"])
    assert C.values.tolist() == [[-1.0, 1.0]]


def test_pairwise_contrast():
    C = msstats_contrast_matrix("pairwise", ["A", "B", "C"])
    assert C.shape == (3, 3)
    # every row should sum to ~0
    np.testing.assert_allclose(C.sum(axis=1).to_numpy(), 0.0, atol=1e-12)
    # row labels
    assert "A vs B" in list(C.index)


def test_multi_condition_per_side_averages():
    C = msstats_contrast_matrix(
        [(["B", "C"], ["A"])], ["A", "B", "C"]
    )
    row = C.iloc[0]
    assert row["A"] == -1.0
    assert abs(row["B"] - 0.5) < 1e-12
    assert abs(row["C"] - 0.5) < 1e-12


def test_list_of_strings():
    C = msstats_contrast_matrix(["B-A", "C-A"], ["A", "B", "C"])
    assert C.shape == (2, 3)
    assert C.loc["B vs A", "B"] == 1.0
    assert C.loc["B vs A", "A"] == -1.0


def test_ndarray_passthrough():
    arr = np.array([[-1.0, 1.0]])
    C = msstats_contrast_matrix(arr, ["A", "B"], labels=["B-A"])
    assert list(C.index) == ["B-A"]
    assert C.values.tolist() == [[-1.0, 1.0]]


def test_unknown_condition_raises():
    with pytest.raises(ValueError):
        msstats_contrast_matrix("X-A", ["A", "B"])
