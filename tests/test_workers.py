"""Tests for the modular worker API.

Verifies that the modular building blocks reproduce the high-level
:func:`data_process` / :func:`group_comparison` results.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pymsstats import (
    data_process,
    design_sample_size,
    group_comparison,
    group_comparison_output,
    group_comparison_single_protein,
    msstats_contrast_matrix,
    msstats_group_comparison,
    msstats_summarize_modular,
    prepare_for_data_process,
    prepare_for_group_comparison,
    prepare_for_summarization,
    summarization_output,
    summarize_single_core,
    summarize_single_linear,
    summarize_single_tmp,
)
from pymsstats.normalization import equalize_medians
from tests._synthetic import synthetic_msstats


def _processed(seed=0, n_proteins=20):
    long_df, _ = synthetic_msstats(
        n_proteins=n_proteins, n_per_group=4, seed=seed, effect_size=1.5)
    return data_process(long_df), long_df


# -----------------------------------------------------------------------------
# prepare_for_data_process
# -----------------------------------------------------------------------------
def test_prepare_for_data_process_adds_columns():
    _, long_df = _processed()
    prepped = prepare_for_data_process(long_df)
    for col in ("INTENSITY", "ABUNDANCE", "PROTEIN", "FEATURE",
                "RUN", "GROUP", "SUBJECT"):
        assert col in prepped.columns
    # log2 transform applied
    pos = prepped.loc[prepped["INTENSITY"] > 0]
    assert np.allclose(
        pos["ABUNDANCE"], np.log2(pos["INTENSITY"]), equal_nan=True)


# -----------------------------------------------------------------------------
# prepare_for_summarization
# -----------------------------------------------------------------------------
def test_prepare_for_summarization_adds_flags():
    _, long_df = _processed()
    prepped = prepare_for_data_process(long_df)
    out = prepare_for_summarization(prepped)
    assert "remove" in out.columns
    assert "censored" in out.columns


# -----------------------------------------------------------------------------
# modular summarization == data_process summarization
# -----------------------------------------------------------------------------
def test_modular_summarize_matches_data_process():
    processed, long_df = _processed()
    prepped = prepare_for_data_process(long_df)
    # min_feature_obs filter + normalization (mirror data_process)
    obs = (prepped.dropna(subset=["ABUNDANCE"])
           .groupby(["PROTEIN", "FEATURE"]).size())
    keep = set(obs[obs >= 2].index)
    mask = [(p, f) in keep
            for p, f in zip(prepped["PROTEIN"], prepped["FEATURE"])]
    prepped = prepped.loc[mask].copy()
    prepped = equalize_medians(prepped)
    summ = msstats_summarize_modular(prepped, method="TMP")
    merged = processed.merge(
        summ, on=["Protein", "RUN"], suffixes=("_dp", "_mod"))
    assert len(merged) > 0
    assert np.allclose(
        merged["LogIntensities_dp"], merged["LogIntensities_mod"],
        atol=1e-9)


def test_summarize_single_tmp_one_protein():
    _, long_df = _processed(n_proteins=3)
    prepped = prepare_for_data_process(long_df)
    one = prepped.loc[prepped["PROTEIN"] == prepped["PROTEIN"].iloc[0]]
    out = summarize_single_tmp(one)
    assert len(out) > 0
    assert out["Protein"].nunique() == 1


def test_summarize_single_linear_one_protein():
    _, long_df = _processed(n_proteins=3)
    prepped = prepare_for_data_process(long_df)
    one = prepped.loc[prepped["PROTEIN"] == prepped["PROTEIN"].iloc[0]]
    out = summarize_single_linear(one)
    assert len(out) > 0


def test_summarize_single_core_all_proteins():
    _, long_df = _processed(n_proteins=8)
    prepped = prepare_for_data_process(long_df)
    out = summarize_single_core(prepped, method="TMP")
    assert out["Protein"].nunique() == 8


# -----------------------------------------------------------------------------
# summarization_output
# -----------------------------------------------------------------------------
def test_summarization_output_dict():
    _, long_df = _processed(n_proteins=6)
    prepped = prepare_for_data_process(long_df)
    summ = msstats_summarize_modular(prepped, method="TMP")
    res = summarization_output(prepped, summ)
    assert set(res.keys()) == {"FeatureLevelData", "ProteinLevelData"}
    pld = res["ProteinLevelData"]
    assert "GROUP" in pld.columns
    assert "SUBJECT" in pld.columns


# -----------------------------------------------------------------------------
# modular group comparison == group_comparison
# -----------------------------------------------------------------------------
def test_modular_group_comparison_matches_high_level():
    processed, _ = _processed(n_proteins=20)
    C = msstats_contrast_matrix("group1-group0", ["group0", "group1"])
    ref = group_comparison(processed, contrast_matrix=C)

    prepared = prepare_for_group_comparison(processed)
    raw = msstats_group_comparison(prepared, C)
    out = group_comparison_output(raw)

    merged = ref.merge(out, on=["Protein", "Label"],
                       suffixes=("_ref", "_mod"))
    assert len(merged) == len(ref)
    ok = merged["log2FC_ref"].notna() & merged["log2FC_mod"].notna()
    assert np.allclose(
        merged.loc[ok, "log2FC_ref"], merged.loc[ok, "log2FC_mod"],
        atol=1e-9)
    ok_p = merged["pvalue_ref"].notna() & merged["pvalue_mod"].notna()
    assert np.allclose(
        merged.loc[ok_p, "pvalue_ref"], merged.loc[ok_p, "pvalue_mod"],
        atol=1e-9)
    ok_a = merged["adj.pvalue_ref"].notna() & merged["adj.pvalue_mod"].notna()
    assert np.allclose(
        merged.loc[ok_a, "adj.pvalue_ref"],
        merged.loc[ok_a, "adj.pvalue_mod"], atol=1e-9)


def test_group_comparison_single_protein():
    processed, _ = _processed(n_proteins=5)
    C = msstats_contrast_matrix("group1-group0", ["group0", "group1"])
    one = processed.loc[processed["Protein"] == processed["Protein"].iloc[0]]
    out = group_comparison_single_protein(one, C)
    assert len(out) == 1
    assert {"log2FC", "SE", "pvalue"}.issubset(out.columns)


def test_prepare_for_group_comparison_dict():
    processed, _ = _processed(n_proteins=7)
    d = prepare_for_group_comparison(processed)
    assert isinstance(d, dict)
    assert len(d) == 7
    for sub in d.values():
        assert "LogIntensities" in sub.columns


# -----------------------------------------------------------------------------
# design_sample_size_plots
# -----------------------------------------------------------------------------
def test_design_sample_size_plots_smoke():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from pymsstats import design_sample_size_plots
    processed, _ = _processed(n_proteins=15)
    res = design_sample_size(processed, desired_fc=(1.25, 1.75))
    ax = design_sample_size_plots(res)
    assert ax is not None


def test_save_plot_smoke(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pymsstats import save_plot
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    out = save_plot(fig, str(tmp_path / "test.png"))
    assert out.endswith(".png")
    import os
    assert os.path.exists(out)
