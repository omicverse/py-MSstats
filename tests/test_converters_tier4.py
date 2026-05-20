"""Smoke + column-mapping tests for the Tier-4 vendor converters.

Covers ``pd_to_msstats``, ``progenesis_to_msstats``,
``openswath_to_msstats`` and ``diaumpire_to_msstats``.
"""
from __future__ import annotations

import pandas as pd
import pytest

from pymsstats import (
    CANONICAL_COLS,
    diaumpire_to_msstats,
    openswath_to_msstats,
    pd_to_msstats,
    progenesis_to_msstats,
)
from tests._synthetic import (
    mock_diaumpire_report,
    mock_openswath_report,
    mock_pd_report,
    mock_progenesis_report,
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


# -----------------------------------------------------------------------------
# Proteome Discoverer
# -----------------------------------------------------------------------------
def test_pd_converter_smoke():
    out = pd_to_msstats(mock_pd_report(), _annotation())
    _check_long(out)
    assert out["ProteinName"].nunique() == 6


def test_pd_converter_column_mapping():
    """PD's protein-id / sequence / quant columns map to MSstats names."""
    rep = mock_pd_report()
    out = pd_to_msstats(rep, _annotation())
    # Modifications are appended to the peptide sequence
    assert out["PeptideSequence"].astype(str).str.contains("_").all()
    # Precursor Area becomes Intensity
    assert pd.to_numeric(out["Intensity"], errors="coerce").notna().any()


def test_pd_custom_quantification_column():
    rep = mock_pd_report().rename(columns={"Precursor Area": "Custom.Area"})
    out = pd_to_msstats(rep, _annotation(),
                        which_quantification="Custom.Area")
    _check_long(out)


# -----------------------------------------------------------------------------
# Progenesis
# -----------------------------------------------------------------------------
def test_progenesis_converter_smoke():
    out = progenesis_to_msstats(mock_progenesis_report(), _annotation())
    _check_long(out)
    assert out["ProteinName"].nunique() == 6


def test_progenesis_use_in_quantitation_filter():
    rep = mock_progenesis_report()
    rep.loc[rep["Run"] == "run0", "Use in quantitation"] = "False"
    out = progenesis_to_msstats(rep, _annotation())
    assert "run0" not in set(out["Run"])


# -----------------------------------------------------------------------------
# OpenSWATH
# -----------------------------------------------------------------------------
def test_openswath_converter_smoke():
    out = openswath_to_msstats(mock_openswath_report(), _annotation())
    _check_long(out)
    # semicolon-separated fragments are exploded into separate rows
    assert out["FragmentIon"].nunique() >= 3


def test_openswath_mscore_filter():
    rep = mock_openswath_report()
    rep.loc[rep["filename"] == "run0", "m_score"] = 0.9
    out = openswath_to_msstats(rep, _annotation())
    assert "run0" not in set(out["Run"])


def test_openswath_decoy_filter():
    rep = mock_openswath_report()
    rep.loc[rep["ProteinName"] == "P000", "decoy"] = 1
    out = openswath_to_msstats(rep, _annotation())
    assert "P000" not in set(out["ProteinName"])


# -----------------------------------------------------------------------------
# DIA-Umpire
# -----------------------------------------------------------------------------
def test_diaumpire_converter_smoke():
    out = diaumpire_to_msstats(mock_diaumpire_report(), annotation=_annotation())
    _check_long(out)
    assert out["ProteinName"].nunique() == 6


def test_diaumpire_selected_fragment_filter():
    rep = mock_diaumpire_report()
    rep.loc[rep["Run"] == "run0", "Selected_fragment"] = "False"
    out = diaumpire_to_msstats(rep, annotation=_annotation())
    assert "run0" not in set(out["Run"])
