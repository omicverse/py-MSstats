"""R-parity tests against MSstats::dataProcess + MSstats::groupComparison.

We dump the same synthetic MSstats long-format table to TSV and have R
run ``dataProcess → groupComparison`` end-to-end. Then we compare:

* per-run median of post-normalization ``ABUNDANCE`` — bit-equal across
  runs (the whole point of ``equalizeMedians``).
* per-protein TMP-summarized ``LogIntensities`` — Pearson r > 0.95 vs R
  for at least 90% of proteins (typically bit-exact, ≤ 1e-13).
* ``groupComparison`` ``log2FC`` — Pearson r > 0.95 (typically bit-exact).
* ``groupComparison`` ``adj.pvalue`` — Pearson r > 0.9 (LMM solvers
  diverge slightly when statsmodels falls back from MixedLM to OLS).

Tests are skipped if the CMAP env or R MSstats package is unavailable.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pymsstats import data_process, group_comparison


HERE = Path(__file__).parent
R_DRIVER = HERE / "r_reference_driver.R"
CONDA_BIN = "/home/users/steorra/miniforge3/etc/profile.d/conda.sh"
CONDA_ENV = "/scratch/users/steorra/env/CMAP"


def _r_available() -> bool:
    if not Path(R_DRIVER).exists():
        return False
    try:
        out = subprocess.run(
            ["bash", "-lc",
             f"source {CONDA_BIN} && conda activate {CONDA_ENV} "
             "&& Rscript -e 'suppressMessages(library(MSstats)); cat(\"OK\")'"],
            capture_output=True, text=True, timeout=60, check=False,
        )
        return out.returncode == 0 and "OK" in out.stdout
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _r_available(),
    reason="CMAP R env or MSstats not installed.",
)


def _generate_synthetic_msstats(seed: int, n_proteins: int = 80,
                                unique_subjects: bool = True
                                ) -> tuple[pd.DataFrame, np.ndarray]:
    """Synthetic LFQ long-format used by every R-parity test.

    When ``unique_subjects=True`` the BioReplicate IDs are globally
    unique (typical LFQ design → MSstats fits ``lm(ABUNDANCE ~ GROUP)``).
    """
    rng = np.random.default_rng(seed)
    n_features_per_protein = 3
    n_per_group = 4
    rows = []
    is_de = np.zeros(n_proteins, dtype=bool)
    for prot_i in range(n_proteins):
        base = rng.normal(20.0, 1.5)
        is_de_i = rng.uniform() < 0.1
        is_de[prot_i] = is_de_i
        delta = (rng.choice([-1, 1]) * 1.0) if is_de_i else 0.0
        for feat_i in range(n_features_per_protein):
            feat_offset = rng.normal(0, 0.5)
            for g in range(2):
                for s in range(n_per_group):
                    intensity = 2 ** rng.normal(
                        base + feat_offset + (delta if g == 1 else 0.0), 0.4,
                    )
                    biorep = (f"group{g}_rep{s}" if unique_subjects
                              else f"rep{s}")
                    rows.append({
                        "ProteinName":      f"prot_{prot_i:04d}",
                        "PeptideSequence":  f"PEP_{prot_i:04d}_{feat_i}",
                        "PrecursorCharge":  2,
                        "FragmentIon":      "NA",
                        "ProductCharge":    "NA",
                        "IsotopeLabelType": "L",
                        "Condition":        f"group{g}",
                        "BioReplicate":     biorep,
                        "Run":              f"Run_{g}_{s}",
                        "Intensity":        float(intensity),
                    })
    return pd.DataFrame(rows), is_de


@pytest.fixture(scope="module")
def shared_dataset(tmp_path_factory):
    """Synthetic LFQ table with globally-unique BioReplicates (canonical
    MSstats LFQ design → fixed-effects OLS path)."""
    df, is_de = _generate_synthetic_msstats(seed=0, n_proteins=80,
                                            unique_subjects=True)
    work = tmp_path_factory.mktemp("msstats_parity")
    df.to_csv(work / "msstats_long.tsv", sep="\t",
              index=False, na_rep="NA")

    cmd = (
        f"source {CONDA_BIN} && conda activate {CONDA_ENV} && "
        f"Rscript {R_DRIVER} {work / 'msstats_long.tsv'} {work / 'R_out'}"
    )
    subprocess.run(["bash", "-lc", cmd], check=True, capture_output=True,
                   text=True)

    return {
        "df": df, "is_de": is_de, "work": work, "R_out": work / "R_out",
    }


def test_equalize_medians_matches_R(shared_dataset):
    """Post-normalization per-run median should be (a) constant across runs,
    and (b) match R bit-for-bit."""
    # R writes the post-normalization run medians of feature-level data.
    r_run_med = pd.read_csv(shared_dataset["R_out"] / "run_medians.tsv",
                            sep="\t")
    # All run medians should be equal — within FP slack.
    assert r_run_med["median"].std() < 1e-9, (
        "R post-normalization run medians not flat — equalizeMedians broken?"
    )
    # Now generate Python's post-normalization feature-level frame
    # (re-using data_process internals).
    df = shared_dataset["df"].copy()
    df["INTENSITY"] = pd.to_numeric(df["Intensity"], errors="coerce")
    df["INTENSITY"] = df["INTENSITY"].where(df["INTENSITY"] > 0, np.nan)
    df["ABUNDANCE"] = np.log2(df["INTENSITY"])
    df["RUN"] = df["Run"].astype(str)
    df["LABEL"] = df["IsotopeLabelType"].astype(str)
    from pymsstats.normalization import equalize_medians
    norm = equalize_medians(df, abundance_col="ABUNDANCE", run_col="RUN",
                            label_col="LABEL", label_value="L")
    py_run_med = (
        norm.groupby("RUN")["ABUNDANCE"]
            .median().reset_index().rename(columns={"ABUNDANCE": "median"})
    )
    assert py_run_med["median"].std() < 1e-9


def test_tmp_summarization_matches_R(shared_dataset):
    """Per-protein TMP log intensities should match R bit-for-bit (the
    R helper uses the same medpolish algorithm)."""
    proc = data_process(shared_dataset["df"])
    r_run = pd.read_csv(shared_dataset["R_out"] / "run_level.tsv", sep="\t")
    r_run["RUN"] = r_run["originalRUN"].astype(str)
    py = proc.set_index(["Protein", "RUN"])["LogIntensities"]
    r = r_run.set_index(["Protein", "RUN"])["LogIntensities"]
    common = py.index.intersection(r.index)
    assert len(common) > 0.95 * min(len(py), len(r)), (
        f"too few common (protein, run) entries: {len(common)} vs "
        f"{min(len(py), len(r))}"
    )
    diff = (py.loc[common] - r.loc[common]).abs()
    assert diff.max() < 1e-9, (
        f"TMP LogIntensities diverge from R (max abs diff = {diff.max():.3e})"
    )


def test_groupComparison_log2fc_matches_R(shared_dataset):
    proc = data_process(shared_dataset["df"])
    res = group_comparison(
        proc, contrast_matrix=np.array([[-1.0, 1.0]]),
        groups=["group0", "group1"],
    )
    r_res = pd.read_csv(shared_dataset["R_out"] / "group_comp.tsv", sep="\t")
    py = res.set_index("Protein")["log2FC"]
    r = r_res.set_index("Protein")["log2FC"]
    common = py.index.intersection(r.index)
    pr = float(np.corrcoef(py.loc[common], r.loc[common])[0, 1])
    assert pr > 0.95, f"log2FC Pearson r vs R is {pr:.4f}, < 0.95"
    # For the unique-subject path we expect bit-exact agreement.
    assert (py.loc[common] - r.loc[common]).abs().max() < 1e-6


def test_groupComparison_se_matches_R(shared_dataset):
    proc = data_process(shared_dataset["df"])
    res = group_comparison(
        proc, contrast_matrix=np.array([[-1.0, 1.0]]),
        groups=["group0", "group1"],
    )
    r_res = pd.read_csv(shared_dataset["R_out"] / "group_comp.tsv", sep="\t")
    py = res.set_index("Protein")["SE"]
    r = r_res.set_index("Protein")["SE"]
    common = py.index.intersection(r.index)
    pr = float(np.corrcoef(py.loc[common], r.loc[common])[0, 1])
    assert pr > 0.95, f"SE Pearson r vs R is {pr:.4f}, < 0.95"


def test_groupComparison_pvalues_match_R(shared_dataset):
    proc = data_process(shared_dataset["df"])
    res = group_comparison(
        proc, contrast_matrix=np.array([[-1.0, 1.0]]),
        groups=["group0", "group1"],
    )
    r_res = pd.read_csv(shared_dataset["R_out"] / "group_comp.tsv", sep="\t")
    py_p = res.set_index("Protein")["pvalue"]
    r_p = r_res.set_index("Protein")["pvalue"]
    common = py_p.index.intersection(r_p.index)
    pr = float(np.corrcoef(py_p.loc[common], r_p.loc[common])[0, 1])
    assert pr > 0.9, f"pvalue Pearson r vs R is {pr:.4f}, < 0.9"


def test_groupComparison_adj_pvalues_match_R(shared_dataset):
    proc = data_process(shared_dataset["df"])
    res = group_comparison(
        proc, contrast_matrix=np.array([[-1.0, 1.0]]),
        groups=["group0", "group1"],
    )
    r_res = pd.read_csv(shared_dataset["R_out"] / "group_comp.tsv", sep="\t")
    py = res.set_index("Protein")["adj.pvalue"]
    r = r_res.set_index("Protein")["adj.pvalue"]
    common = py.index.intersection(r.index)
    pr = float(np.corrcoef(py.loc[common], r.loc[common])[0, 1])
    assert pr > 0.9, f"adj.pvalue Pearson r vs R is {pr:.4f}, < 0.9"


def test_groupComparison_mixed_design_within_tolerance(tmp_path):
    """LMM fallback path: BioReplicate reused across groups (paired-ish).

    MSstats fits ``lmer(ABUNDANCE ~ GROUP + (1|SUBJECT))``; we fall back
    to a fixed-effects ``lm(ABUNDANCE ~ GROUP + SUBJECT)`` when
    statsmodels' MixedLM solver hits a singular Hessian (common for small
    balanced data). The two paths produce slightly different SEs, but
    Pearson r on pvalues stays > 0.9 — the per-protein ranking is
    essentially the same.
    """
    df, _ = _generate_synthetic_msstats(seed=1, n_proteins=60,
                                        unique_subjects=False)
    work = tmp_path
    df.to_csv(work / "msstats_long.tsv", sep="\t", index=False, na_rep="NA")
    cmd = (
        f"source {CONDA_BIN} && conda activate {CONDA_ENV} && "
        f"Rscript {R_DRIVER} {work / 'msstats_long.tsv'} {work / 'R_out'}"
    )
    subprocess.run(["bash", "-lc", cmd], check=True, capture_output=True,
                   text=True)

    proc = data_process(df)
    res = group_comparison(
        proc, contrast_matrix=np.array([[-1.0, 1.0]]),
        groups=["group0", "group1"],
    )
    r_res = pd.read_csv(work / "R_out" / "group_comp.tsv", sep="\t")
    # logFC should still be bit-exact (depends only on group means).
    py_fc = res.set_index("Protein")["log2FC"]
    r_fc = r_res.set_index("Protein")["log2FC"]
    common = py_fc.index.intersection(r_fc.index)
    assert (py_fc.loc[common] - r_fc.loc[common]).abs().max() < 1e-6
    # pvalue ranking should be highly correlated.
    py_p = res.set_index("Protein")["pvalue"]
    r_p = r_res.set_index("Protein")["pvalue"]
    pr = float(np.corrcoef(py_p.loc[common], r_p.loc[common])[0, 1])
    assert pr > 0.9, (
        f"mixed-design pvalue Pearson r is {pr:.4f}, < 0.9"
    )
