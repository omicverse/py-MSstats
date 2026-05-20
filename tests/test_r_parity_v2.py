"""R-parity tests for the v0.2 functions.

Drives ``r_reference_driver2.R`` (designSampleSize, linear summarization,
quantile normalization, MSstatsContrastMatrix) and checks the Python
ports against R MSstats 4.14.2.

Skipped if the CMAP R env / MSstats package is unavailable.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pymsstats import (
    data_process,
    design_sample_size,
    msstats_contrast_matrix,
)
from pymsstats.normalization import equalize_medians
from pymsstats.pipeline import _make_feature_id
from pymsstats.summarization import linear_summarize
from tests._synthetic import synthetic_msstats


HERE = Path(__file__).parent
R_DRIVER2 = HERE / "r_reference_driver2.R"
CONDA_BIN = "/home/users/steorra/miniforge3/etc/profile.d/conda.sh"
CONDA_ENV = "/scratch/users/steorra/env/CMAP"


def _r_available() -> bool:
    if not Path(R_DRIVER2).exists():
        return False
    try:
        out = subprocess.run(
            ["bash", "-lc",
             f"source {CONDA_BIN} && conda activate {CONDA_ENV} "
             "&& Rscript -e 'suppressMessages(library(MSstats)); cat(\"OK\")'"],
            capture_output=True, text=True, timeout=90, check=False,
        )
        return out.returncode == 0 and "OK" in out.stdout
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _r_available(),
    reason="CMAP R env or MSstats not installed.",
)


@pytest.fixture(scope="module")
def r_outputs(tmp_path_factory):
    df, _ = synthetic_msstats(n_proteins=60, seed=0, unique_subjects=True)
    work = tmp_path_factory.mktemp("msstats_v2_parity")
    df.to_csv(work / "msstats_long.tsv", sep="\t", index=False, na_rep="NA")
    cmd = (
        f"source {CONDA_BIN} && conda activate {CONDA_ENV} && "
        f"Rscript {R_DRIVER2} {work / 'msstats_long.tsv'} {work / 'R_out'}"
    )
    subprocess.run(["bash", "-lc", cmd], check=True, capture_output=True,
                   text=True)
    return {"df": df, "R_out": work / "R_out"}


def test_design_sample_size_matches_R(r_outputs):
    proc = data_process(r_outputs["df"])
    res = design_sample_size(proc, desired_fc=(1.25, 1.5),
                             fdr=0.05, num_sample=True, power=0.9)
    r = pd.read_csv(r_outputs["R_out"] / "design_numsample.tsv", sep="\t")
    res = res.assign(fc=res["desiredFC"].round(3))
    r = r.assign(fc=r["desiredFC"].round(3))
    m = res.merge(r, on="fc", suffixes=("_py", "_r"))
    assert len(m) >= 8, f"too few common FC grid points ({len(m)})"
    pr = float(np.corrcoef(m["numSample_py"].astype(float),
                           m["numSample_r"].astype(float))[0, 1])
    assert pr > 0.9, f"design numSample Pearson r vs R = {pr:.4f}"
    # numSample should be near-identical (closed-form, same variance estimate)
    assert (m["numSample_py"] - m["numSample_r"]).abs().max() <= 2


def test_linear_summarization_matches_R(r_outputs):
    """Linear-model summarization should match R bit-for-bit."""
    df = r_outputs["df"].copy()
    df["INTENSITY"] = pd.to_numeric(df["Intensity"], errors="coerce")
    df["INTENSITY"] = df["INTENSITY"].where(df["INTENSITY"] > 0, np.nan)
    df["ABUNDANCE"] = np.log2(df["INTENSITY"])
    df["PROTEIN"] = df["ProteinName"].astype(str)
    df["RUN"] = df["Run"].astype(str)
    df["LABEL"] = df["IsotopeLabelType"].astype(str)
    df["FEATURE"] = _make_feature_id(df)
    obs = df.dropna(subset=["ABUNDANCE"]).groupby(["PROTEIN", "FEATURE"]).size()
    keep = set(map(tuple, obs[obs >= 2].index.tolist()))
    df = df.loc[[(p, f) in keep
                 for p, f in zip(df["PROTEIN"], df["FEATURE"])]].copy()
    df = equalize_medians(df, abundance_col="ABUNDANCE", run_col="RUN",
                          label_col="LABEL", label_value="L")
    lin = linear_summarize(df)
    r = pd.read_csv(r_outputs["R_out"] / "linear_run_level.tsv", sep="\t")
    r["RUN"] = r["originalRUN"].astype(str)
    m = lin.merge(r[["Protein", "RUN", "LogIntensities"]],
                  on=["Protein", "RUN"], suffixes=("_py", "_r"))
    assert len(m) > 0
    diff = (m["LogIntensities_py"] - m["LogIntensities_r"]).abs()
    pr = float(np.corrcoef(m["LogIntensities_py"],
                           m["LogIntensities_r"])[0, 1])
    assert pr > 0.99, f"linear summary Pearson r vs R = {pr:.5f}"
    assert diff.max() < 1e-6, (
        f"linear summary diverges from R (max abs diff = {diff.max():.3e})"
    )


def test_contrast_matrix_matches_R(r_outputs):
    df = r_outputs["df"]
    conditions = sorted(df["Condition"].unique())
    py = msstats_contrast_matrix("pairwise", conditions)
    r = pd.read_csv(r_outputs["R_out"] / "contrast_matrix.tsv", sep="\t")
    r = r.set_index("label")
    # align columns/rows
    common_rows = py.index.intersection(r.index)
    assert len(common_rows) == len(py.index), (
        f"contrast labels differ: py={list(py.index)} r={list(r.index)}"
    )
    for col in conditions:
        np.testing.assert_allclose(
            py.loc[common_rows, col].to_numpy(),
            r.loc[common_rows, col].to_numpy(),
            atol=1e-12,
        )


def test_quantile_normalization_matches_R(r_outputs):
    """Quantile normalization → run distributions should match R's
    (flat per-run quantiles). Skipped if R's preprocessCore failed."""
    q_file = r_outputs["R_out"] / "quantile_run_med.tsv"
    if not q_file.exists():
        pytest.skip("R quantile normalization unavailable (preprocessCore)")
    from pymsstats.normalization import quantile_normalize
    df = r_outputs["df"].copy()
    df["INTENSITY"] = pd.to_numeric(df["Intensity"], errors="coerce")
    df["INTENSITY"] = df["INTENSITY"].where(df["INTENSITY"] > 0, np.nan)
    df["ABUNDANCE"] = np.log2(df["INTENSITY"])
    df["RUN"] = df["Run"].astype(str)
    df["LABEL"] = df["IsotopeLabelType"].astype(str)
    df["FEATURE"] = _make_feature_id(df)
    norm = quantile_normalize(df)
    py_med = norm.groupby("RUN")["ABUNDANCE"].median()
    # After quantile normalization all run medians should be equal.
    assert (py_med.max() - py_med.min()) < 1e-6
    r_med = pd.read_csv(q_file, sep="\t")
    assert r_med["median"].std() < 1e-6
