"""Basic sanity tests for the MSstats pipeline — no R needed."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pymsstats import (
    CANONICAL_COLS,
    data_process,
    equalize_medians,
    group_comparison,
    medpolish,
    tmp_summarize,
)


def generate_synthetic_msstats(
    n_proteins: int = 100,
    n_features_per_protein: int = 3,
    n_per_group: int = 4,
    seed: int = 0,
    prop_de: float = 0.1,
    effect_size: float = 1.0,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Synthetic MSstats long-format LFQ dataset.

    Returns the long-format DataFrame and a (n_proteins,) bool array of
    ground-truth DE flags.
    """
    rng = np.random.default_rng(seed)
    rows = []
    is_de = np.zeros(n_proteins, dtype=bool)
    for prot_i in range(n_proteins):
        base = rng.normal(20.0, 1.5)
        is_de_i = rng.uniform() < prop_de
        is_de[prot_i] = is_de_i
        delta = (rng.choice([-1, 1]) * effect_size) if is_de_i else 0.0
        for feat_i in range(n_features_per_protein):
            feat_offset = rng.normal(0, 0.5)
            for g in range(2):
                for s in range(n_per_group):
                    intensity = 2 ** rng.normal(
                        base + feat_offset + (delta if g == 1 else 0.0), 0.4,
                    )
                    rows.append({
                        "ProteinName":      f"prot_{prot_i:04d}",
                        "PeptideSequence":  f"PEP_{prot_i:04d}_{feat_i}",
                        "PrecursorCharge":  2,
                        "FragmentIon":      "NA",
                        "ProductCharge":    "NA",
                        "IsotopeLabelType": "L",
                        "Condition":        f"group{g}",
                        "BioReplicate":     f"rep{s}",
                        "Run":              f"Run_{g}_{s}",
                        "Intensity":        float(intensity),
                    })
    return pd.DataFrame(rows), is_de


def test_medpolish_known_table():
    """A 3×3 table where the row+col effects are explicit; medpolish should
    recover them."""
    # overall=10, row=(0, 2, -2), col=(0, 1, -1) → cell = overall + row + col.
    overall = 10.0
    row_eff = np.array([0.0, 2.0, -2.0])
    col_eff = np.array([0.0, 1.0, -1.0])
    X = overall + row_eff[:, None] + col_eff[None, :]
    mp = medpolish(X)
    assert abs(mp["overall"] - overall) < 1e-9
    np.testing.assert_allclose(mp["row"], row_eff, atol=1e-9)
    np.testing.assert_allclose(mp["col"], col_eff, atol=1e-9)
    np.testing.assert_allclose(mp["residuals"], np.zeros_like(X), atol=1e-9)


def test_equalize_medians_shifts_runs_to_grand_median():
    """After equalizeMedians every run should have the same median."""
    msstats_df, _ = generate_synthetic_msstats(seed=0)
    # Build the post-log2, ABUNDANCE-named frame the helper expects.
    df = msstats_df.copy()
    df["ABUNDANCE"] = np.log2(df["Intensity"].astype(float))
    df["RUN"] = df["Run"]
    df["LABEL"] = df["IsotopeLabelType"]
    # Spike-in artificial offsets so the runs DO start with different medians.
    rng = np.random.default_rng(99)
    run_offset = {r: rng.normal(0, 1.5) for r in df["RUN"].unique()}
    df["ABUNDANCE"] = df["ABUNDANCE"] + df["RUN"].map(run_offset)
    out = equalize_medians(df)
    medians = out.groupby("RUN")["ABUNDANCE"].median()
    # All run medians should be ~equal post-normalization.
    assert (medians.max() - medians.min()) < 1e-9


def test_data_process_returns_one_row_per_protein_per_run():
    msstats_df, _ = generate_synthetic_msstats(n_proteins=20, seed=2)
    proc = data_process(msstats_df)
    assert set(["Protein", "RUN", "LogIntensities",
                "GROUP", "SUBJECT"]).issubset(proc.columns)
    n_runs = msstats_df["Run"].nunique()
    n_prot = msstats_df["ProteinName"].nunique()
    # Up to n_prot * n_runs, possibly slightly less if some proteins had
    # all-NaN features.
    assert len(proc) <= n_prot * n_runs
    assert len(proc) >= 0.9 * n_prot * n_runs


def test_data_process_required_columns_validated():
    df = pd.DataFrame({"ProteinName": ["A"], "Intensity": [1.0]})
    with pytest.raises(ValueError, match="missing required"):
        data_process(df)


def test_group_comparison_finds_de_proteins():
    msstats_df, is_de = generate_synthetic_msstats(
        n_proteins=100, prop_de=0.15, effect_size=2.0, seed=3,
    )
    proc = data_process(msstats_df)
    res = group_comparison(
        proc, contrast_matrix=np.array([[-1.0, 1.0]]),
        groups=["group0", "group1"],
    )
    assert {"Protein", "Label", "log2FC", "SE", "df", "t",
            "pvalue", "adj.pvalue", "issue"}.issubset(res.columns)
    # The DE flag indexed by protein name.
    is_de_by_name = {
        f"prot_{i:04d}": v for i, v in enumerate(is_de)
    }
    res["is_de"] = res["Protein"].map(is_de_by_name)
    mean_p_de = res.loc[res["is_de"], "pvalue"].mean()
    mean_p_not = res.loc[~res["is_de"], "pvalue"].mean()
    assert mean_p_de < mean_p_not


def test_group_comparison_log2fc_sign_matches_contrast_direction():
    """For an all-DE simulation the per-protein log2FC mean magnitude
    should track the planted effect size (sign is random per protein)."""
    msstats_df, _ = generate_synthetic_msstats(
        n_proteins=30, n_features_per_protein=4, prop_de=1.0, effect_size=2.0,
        seed=7,
    )
    proc = data_process(msstats_df)
    res = group_comparison(
        proc, contrast_matrix=np.array([[-1.0, 1.0]]),
        groups=["group0", "group1"],
    )
    # Mean |log2FC| should be ~effect_size (here 2.0); individual feature
    # noise and inter-protein variability cause spread.
    assert res["log2FC"].abs().mean() > 1.0
    assert res["log2FC"].abs().mean() < 3.0


def test_group_comparison_handles_missing_runs():
    """Drop one run entirely and check the test still runs."""
    msstats_df, _ = generate_synthetic_msstats(n_proteins=30, seed=4)
    # Drop one run.
    msstats_df = msstats_df.loc[msstats_df["Run"] != "Run_0_0"].copy()
    proc = data_process(msstats_df)
    res = group_comparison(
        proc, contrast_matrix=np.array([[-1.0, 1.0]]),
        groups=["group0", "group1"],
    )
    # No exceptions; result has rows.
    assert len(res) >= 25


def test_canonical_cols_are_exported():
    """The CANONICAL_COLS list should match the MSstats schema."""
    assert CANONICAL_COLS == [
        "ProteinName", "PeptideSequence", "PrecursorCharge",
        "FragmentIon", "ProductCharge", "IsotopeLabelType",
        "Condition", "BioReplicate", "Run", "Intensity",
    ]
