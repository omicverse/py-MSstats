"""Shared synthetic-data generators for the pymsstats test-suite."""
from __future__ import annotations

import numpy as np
import pandas as pd


def synthetic_msstats(
    n_proteins: int = 60,
    n_features_per_protein: int = 3,
    n_per_group: int = 4,
    seed: int = 0,
    prop_de: float = 0.1,
    effect_size: float = 1.0,
    unique_subjects: bool = True,
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


def mock_diann_report(seed: int = 0, n_proteins: int = 6) -> pd.DataFrame:
    """Mock DIA-NN report.tsv-style long table."""
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_proteins):
        for pep in range(3):
            for run in range(4):
                rows.append({
                    "Protein.Names":      f"P{p:03d}",
                    "Stripped.Sequence":  f"PEP{p:03d}_{pep}",
                    "Modified.Sequence":  f"PEP{p:03d}_{pep}",
                    "Precursor.Charge":   2,
                    "Q.Value":            0.001,
                    "Lib.Q.Value":        0.001,
                    "Lib.PG.Q.Value":     0.001,
                    "Global.Q.Value":     0.001,
                    "Global.PG.Q.Value":  0.001,
                    "Run":                f"run{run}",
                    "Fragment.Quant.Corrected": float(rng.lognormal(15, 0.5)),
                })
    return pd.DataFrame(rows)


def mock_spectronaut_report(seed: int = 1, n_proteins: int = 6) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_proteins):
        for pep in range(3):
            for frag in range(2):
                for run in range(4):
                    rows.append({
                        "PG.ProteinGroups":   f"P{p:03d}",
                        "EG.ModifiedSequence": f"PEP{p:03d}_{pep}",
                        "FG.Charge":          2,
                        "F.FrgIon":           f"y{frag + 3}",
                        "F.Charge":           1,
                        "R.FileName":         f"run{run}",
                        "R.Condition":        "A" if run < 2 else "B",
                        "R.Replicate":        f"r{run}",
                        "EG.Qvalue":          0.001,
                        "PG.Qvalue":          0.001,
                        "F.FrgLossType":      "noloss",
                        "F.ExcludedFromQuantification": "False",
                        "F.PeakArea":         float(rng.lognormal(12, 0.5)),
                    })
    return pd.DataFrame(rows)


def mock_skyline_report(seed: int = 2, n_proteins: int = 6) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_proteins):
        for pep in range(3):
            for frag in range(2):
                for run in range(4):
                    rows.append({
                        "ProteinName":      f"P{p:03d}",
                        "PeptideModifiedSequence": f"PEP{p:03d}_{pep}",
                        "PrecursorCharge":  2,
                        "FragmentIon":      f"y{frag + 3}",
                        "ProductCharge":    1,
                        "IsotopeLabelType": "light",
                        "Condition":        "A" if run < 2 else "B",
                        "BioReplicate":     f"r{run}",
                        "FileName":         f"run{run}",
                        "Area":             float(rng.lognormal(13, 0.5)),
                        "Truncated":        "False",
                        "DetectionQValue":  0.001,
                        "StandardType":     "",
                    })
    return pd.DataFrame(rows)


def mock_msstats_long(seed: int = 3, n_proteins: int = 6) -> pd.DataFrame:
    """Mock MSstats-format CSV as produced by FragPipe / OpenMS exporters."""
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_proteins):
        for pep in range(3):
            for run in range(4):
                rows.append({
                    "ProteinName":      f"P{p:03d}",
                    "PeptideSequence":  f"PEP{p:03d}_{pep}",
                    "PrecursorCharge":  2,
                    "FragmentIon":      "NA",
                    "ProductCharge":    "NA",
                    "IsotopeLabelType": "L",
                    "Condition":        "A" if run < 2 else "B",
                    "BioReplicate":     f"r{run}",
                    "Run":              f"run{run}",
                    "Intensity":        float(rng.lognormal(14, 0.5)),
                })
    return pd.DataFrame(rows)
