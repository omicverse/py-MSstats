"""Tests for pymsstats.plotting — each function returns an Axes, no crash."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from pymsstats import (
    data_process,
    data_process_plots,
    group_comparison,
    group_comparison_plots,
    group_comparison_qc_plots,
    model_based_qc_plots,
    msstats_contrast_matrix,
    theme_msstats,
)
from tests._synthetic import synthetic_msstats


@pytest.fixture(scope="module")
def processed_and_result():
    df, _ = synthetic_msstats(n_proteins=40, prop_de=0.3,
                              effect_size=2.0, seed=0)
    proc = data_process(df)
    C = msstats_contrast_matrix("group1-group0", ["group0", "group1"])
    res = group_comparison(proc, contrast_matrix=C)
    return proc, res


def _is_axes(obj):
    from matplotlib.axes import Axes
    return isinstance(obj, Axes)


def test_theme_msstats_returns_dict():
    cfg = theme_msstats("QCPlot")
    assert isinstance(cfg, dict)
    assert "axes.facecolor" in cfg


def test_data_process_plots_qc(processed_and_result):
    proc, _ = processed_and_result
    ax = data_process_plots(proc, type="QCPlot")
    assert _is_axes(ax)


def test_data_process_plots_profile(processed_and_result):
    proc, _ = processed_and_result
    prot = proc["Protein"].iloc[0]
    ax = data_process_plots(proc, type="ProfilePlot", protein=prot)
    assert _is_axes(ax)


def test_data_process_plots_condition(processed_and_result):
    proc, _ = processed_and_result
    ax = data_process_plots(proc, type="ConditionPlot")
    assert _is_axes(ax)


def test_group_comparison_volcano(processed_and_result):
    _, res = processed_and_result
    ax = group_comparison_plots(res, type="VolcanoPlot")
    assert _is_axes(ax)


def test_group_comparison_heatmap(processed_and_result):
    _, res = processed_and_result
    ax = group_comparison_plots(res, type="Heatmap")
    assert _is_axes(ax)


def test_group_comparison_comparison_plot(processed_and_result):
    _, res = processed_and_result
    ax = group_comparison_plots(res.head(12), type="ComparisonPlot")
    assert _is_axes(ax)


def test_group_comparison_qc_plots(processed_and_result):
    _, res = processed_and_result
    ax = group_comparison_qc_plots(res)
    assert _is_axes(ax)


def test_model_based_qc_plots(processed_and_result):
    proc, _ = processed_and_result
    ax = model_based_qc_plots(proc)
    assert _is_axes(ax)


def test_unknown_plot_type_raises(processed_and_result):
    proc, res = processed_and_result
    with pytest.raises(ValueError):
        data_process_plots(proc, type="Nonexistent")
    with pytest.raises(ValueError):
        group_comparison_plots(res, type="Nonexistent")
