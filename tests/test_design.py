"""Tests for pymsstats.design.design_sample_size."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pymsstats import data_process, design_sample_size
from tests._synthetic import synthetic_msstats


def test_design_sample_size_numsample_smoke():
    df, _ = synthetic_msstats(n_proteins=80, seed=0)
    proc = data_process(df)
    res = design_sample_size(proc, desired_fc=(1.25, 1.5),
                             fdr=0.05, num_sample=True, power=0.9)
    assert set(["desiredFC", "numSample", "FDR", "power", "CV"]).issubset(
        res.columns)
    assert len(res) > 1
    # min N must be positive integers
    assert (res["numSample"] > 0).all()
    # larger FC → fewer samples needed (monotone non-increasing)
    assert res["numSample"].iloc[0] >= res["numSample"].iloc[-1]


def test_design_sample_size_power_mode():
    df, _ = synthetic_msstats(n_proteins=80, seed=1)
    proc = data_process(df)
    res = design_sample_size(proc, desired_fc=(1.25, 1.5),
                             fdr=0.05, num_sample=10, power=True)
    assert (res["power"] >= 0).all() and (res["power"] <= 1).all()
    assert (res["numSample"] == 10).all()


def test_design_sample_size_single_fc():
    df, _ = synthetic_msstats(n_proteins=50, seed=2)
    proc = data_process(df)
    res = design_sample_size(proc, desired_fc=1.5)
    assert len(res) >= 1
    assert (res["numSample"] > 0).all()
