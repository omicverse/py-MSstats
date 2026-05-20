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


def mock_pd_report(seed: int = 4, n_proteins: int = 6) -> pd.DataFrame:
    """Mock Proteome Discoverer PSM export."""
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_proteins):
        for pep in range(3):
            for run in range(4):
                rows.append({
                    "Protein Group Accessions": f"P{p:03d}",
                    "Sequence":                 f"PEP{p:03d}{pep}",
                    "Modifications":            "" if pep else "Carbamidomethyl",
                    "Charge":                   2,
                    "# Proteins":               1,
                    "SpectrumFile":             f"run{run}",
                    "Precursor Area":           float(rng.lognormal(15, 0.5)),
                })
    return pd.DataFrame(rows)


def mock_progenesis_report(seed: int = 5, n_proteins: int = 6) -> pd.DataFrame:
    """Mock (already-melted long) Progenesis QI export."""
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_proteins):
        for pep in range(3):
            for run in range(4):
                rows.append({
                    "Accession":         f"P{p:03d}",
                    "Sequence":          f"PEP{p:03d}{pep}",
                    "Modifications":     "",
                    "Charge":            2,
                    "Use in quantitation": "True",
                    "Run":               f"run{run}",
                    "Intensity":         float(rng.lognormal(13, 0.5)),
                })
    return pd.DataFrame(rows)


def mock_openswath_report(seed: int = 6, n_proteins: int = 6) -> pd.DataFrame:
    """Mock OpenSWATH export (semicolon-joined fragment lists)."""
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_proteins):
        for pep in range(3):
            for run in range(4):
                frags = ";".join(f"y{f}" for f in (3, 4, 5))
                ints = ";".join(
                    str(float(rng.lognormal(12, 0.5))) for _ in range(3)
                )
                rows.append({
                    "ProteinName":              f"P{p:03d}",
                    "FullPeptideName":          f"PEP{p:03d}{pep}",
                    "Charge":                   2,
                    "filename":                 f"run{run}",
                    "aggr_Fragment_Annotation": frags,
                    "aggr_Peak_Area":           ints,
                    "m_score":                  0.001,
                    "decoy":                    0,
                })
    return pd.DataFrame(rows)


def mock_diaumpire_report(seed: int = 7, n_proteins: int = 6) -> pd.DataFrame:
    """Mock DIA-Umpire fragment-level export."""
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_proteins):
        for pep in range(3):
            for frag in range(2):
                for run in range(4):
                    rows.append({
                        "ProteinName":       f"P{p:03d}",
                        "PeptideSequence":   f"PEP{p:03d}{pep}",
                        "FragmentIon":       f"y{frag + 3}",
                        "Selected_fragment": "True",
                        "Selected_peptide":  "True",
                        "Run":               f"run{run}",
                        "Intensity":         float(rng.lognormal(14, 0.5)),
                    })
    return pd.DataFrame(rows)


def mock_sdrf(n_samples: int = 6, n_runs: int = 2) -> pd.DataFrame:
    """Mock SDRF table with bracketed headers."""
    rows = []
    for run in range(1, n_runs + 1):
        for s in range(1, n_samples + 1):
            rows.append({
                "source name": f"Sample {s}",
                "characteristics[organism]": "Homo sapiens",
                "characteristics[disease]": (
                    "disease" if s % 2 == 0 else "control"),
                "characteristics[biological replicate]": s,
                "comment[fraction identifier]": 1,
                "comment[data file]": f"sample{s}_run{run}.raw",
            })
    return pd.DataFrame(rows)
